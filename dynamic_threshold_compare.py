# -*- coding: utf-8 -*-
"""
动态阈值对比回测

对比固定阈值 vs 动态阈值在三只股票上的表现:
  002050: 交易次数需 > 3, 跑输持有需优于 -66.8%
  600038: 动态阈值亏损不应超过 -15%
  600416: 动态阈值不应恶化 > 3%
  
输出: output_v2/dynamic_threshold_results.json + threshold_analysis_report.md
"""
import sys, os, json, time, math
import numpy as np
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fast_compare import FastBacktester


STOCKS = ["002050", "600038", "600416"]
INITIAL_CASH = 100000


def run_comparison():
    """运行固定 vs 动态阈值对比"""
    results = {}
    
    for symbol in STOCKS:
        print(f"\n{'='*60}")
        print(f"  股票: {symbol}")
        print(f"{'='*60}")
        results[symbol] = {}
        
        # 固定阈值
        print(f"  [1/2] 固定阈值...")
        bt_fixed = FastBacktester(symbol, INITIAL_CASH, start_idx=210, use_dynamic=False)
        try:
            m_fixed = bt_fixed.run_backtest()
            if m_fixed:
                results[symbol]["fixed"] = {
                    "total_return": m_fixed["total_return"],
                    "max_drawdown": m_fixed["max_drawdown"],
                    "sharpe": m_fixed["sharpe"],
                    "trade_count": m_fixed["trade_count"],
                    "win_rate": m_fixed["win_rate"],
                    "hold_return": m_fixed["hold_return"],
                }
                print(f"    收益: {m_fixed['total_return']:+.2f}%  "
                      f"回撤: {m_fixed['max_drawdown']:.2f}%  "
                      f"交易: {m_fixed['trade_count']}次")
        except Exception as e:
            print(f"    错误: {e}")
            results[symbol]["fixed"] = {"error": str(e)}
        
        # 动态阈值
        print(f"  [2/2] 动态阈值...")
        bt_dyn = FastBacktester(symbol, INITIAL_CASH, start_idx=210, use_dynamic=True)
        try:
            m_dyn = bt_dyn.run_backtest()
            if m_dyn:
                results[symbol]["dynamic"] = {
                    "total_return": m_dyn["total_return"],
                    "max_drawdown": m_dyn["max_drawdown"],
                    "sharpe": m_dyn["sharpe"],
                    "trade_count": m_dyn["trade_count"],
                    "win_rate": m_dyn["win_rate"],
                    "hold_return": m_dyn["hold_return"],
                }
                results[symbol]["threshold_log"] = m_dyn.get("threshold_log", [])
                print(f"    收益: {m_dyn['total_return']:+.2f}%  "
                      f"回撤: {m_dyn['max_drawdown']:.2f}%  "
                      f"交易: {m_dyn['trade_count']}次")
                
                # 阈值范围
                if m_dyn.get("threshold_log"):
                    buys = [t["buy_threshold"] for t in m_dyn["threshold_log"]]
                    sells = [t["sell_threshold"] for t in m_dyn["threshold_log"]]
                    print(f"    买阈: [{min(buys):.1f}, {max(buys):.1f}]  "
                          f"卖阈: [{min(sells):.1f}, {max(sells):.1f}]")
        except Exception as e:
            print(f"    错误: {e}")
            results[symbol]["dynamic"] = {"error": str(e)}
        
        # 买入持有
        if results[symbol].get("fixed"):
            results[symbol]["buy_hold"] = {
                "return": results[symbol]["fixed"].get("hold_return", 0)
            }
    
    return results


def check_criteria(results):
    """验证是否符合验收标准"""
    checks = {}
    
    # 002050: 交易次数 > 3 且跑输优于 -66.8%
    r = results.get("002050", {})
    fixed = r.get("fixed", {})
    dyn = r.get("dynamic", {})
    
    if fixed and dyn:
        dyn_trades = int(dyn.get("trade_count", 0))
        dyn_ret = float(dyn.get("total_return", 0))
        bh_ret = float(r.get("buy_hold", {}).get("return", 0))
        dyn_gap = dyn_ret - bh_ret
        
        checks["002050_trades"] = {
            "pass": bool(dyn_trades > 3),
            "detail": f"动态{dyn_trades}次 > 固定3次" if dyn_trades > 3 else f"动态{dyn_trades}次不达标"
        }
        checks["002050_gap"] = {
            "pass": bool(dyn_gap > -66.8),
            "detail": f"跑输{dyn_gap:.1f}% vs -66.8%" if dyn_gap > -66.8 else f"跑输{dyn_gap:.1f}%劣于-66.8%"
        }
    
    # 600038: 亏损不超过 -15%
    r = results.get("600038", {})
    dyn = r.get("dynamic", {})
    if dyn:
        ret = dyn.get("total_return", 0)
        checks["600038_loss"] = {
            "pass": ret >= -15,
            "detail": f"亏损{ret:.1f}% <= -15%" if ret >= -15 else f"亏损{ret:.1f}% > -15%"
        }
    
    # 600416: 不恶化 > 3%
    r = results.get("600416", {})
    fixed = r.get("fixed", {})
    dyn = r.get("dynamic", {})
    if fixed and dyn:
        delta = dyn.get("total_return", 0) - fixed.get("total_return", 0)
        checks["600416_no_worse"] = {
            "pass": delta >= -3,
            "detail": f"收益差{delta:+.2f}% >= -3%" if delta >= -3 else f"收益差{delta:+.2f}% < -3%"
        }
    
    return checks


def generate_report(results, checks):
    """生成可读报告"""
    lines = []
    lines.append("# 动态阈值对比回测报告\n")
    lines.append(f"> 运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 样本: {', '.join(STOCKS)} | 初始资金: {INITIAL_CASH:,}\n")
    lines.append(f"> 回测起点: 210日 | 扫描间隔: 5日\n\n")
    
    # 一、汇总表
    lines.append("## 一、固定 vs 动态阈值对比\n\n")
    lines.append("| 股票 | 模式 | 收益 | 回撤 | 夏普 | 交易数 | 胜率 | vs持有 |\n")
    lines.append("|------|------|------|------|------|--------|------|--------|\n")
    
    for symbol in STOCKS:
        r = results.get(symbol, {})
        for mode in ["fixed", "dynamic"]:
            m = r.get(mode, {})
            if "error" not in m and m:
                ret = m.get("total_return", 0)
                dd = m.get("max_drawdown", 0)
                sh = m.get("sharpe", 0)
                tc = m.get("trade_count", 0)
                wr = m.get("win_rate", 0)
                bh = r.get("buy_hold", {}).get("return", 0)
                gap = ret - bh
                label = "固定" if mode == "fixed" else "动态"
                lines.append(f"| {symbol} | {label} | {ret:+.2f}% | {dd:.2f}% | "
                             f"{sh:.2f} | {tc} | {wr:.0f}% | {gap:+.1f}% |\n")
    
    # 二、阈值变化曲线
    lines.append("\n## 二、动态阈值变化曲线\n\n")
    for symbol in STOCKS:
        r = results.get(symbol, {})
        log = r.get("threshold_log", [])
        if log and len(log) > 0:
            lines.append(f"### {symbol}\n\n")
            buys = [t["buy_threshold"] for t in log]
            sells = [t["sell_threshold"] for t in log]
            lines.append(f"- 买阈范围: **[{min(buys):.1f}, {max(buys):.1f}]**\n")
            lines.append(f"- 卖阈范围: **[{min(sells):.1f}, {max(sells):.1f}]**\n")
            lines.append(f"- 扫描点数: {len(log)}\n")
            
            # 采样
            sample = log[::max(1, len(log)//10)]
            lines.append(f"\n采样(每{max(1, len(log)//10)}点):\n\n")
            lines.append("| 日期 | 买阈 | 卖阈 | 趋势强度 | 波动分位 | 说明 |\n")
            lines.append("|------|------|------|---------|---------|------|\n")
            for t in sample:
                lines.append(f"| {t['date']} | {t['buy_threshold']:.1f} | {t['sell_threshold']:.1f} | "
                             f"{t.get('trend_strength',0):.0f}% | {t.get('volatility_percentile',50):.0f} | "
                             f"{t.get('detail','')} |\n")
            lines.append("\n")
    
    # 三、验收结果
    lines.append("\n## 三、验收结果\n\n")
    all_pass = True
    for key, chk in checks.items():
        status = "PASS" if chk["pass"] else "FAIL"
        if not chk["pass"]:
            all_pass = False
        lines.append(f"- **{key}**: {status} — {chk['detail']}\n")
    
    lines.append(f"\n**综合**: {'全部通过' if all_pass else '存在未达标项'}\n")
    
    # 四、参数敏感性分析
    lines.append("\n## 四、参数敏感性分析\n\n")
    lines.append("当前参数:\n")
    lines.append("- 趋势调整系数: 0.3 (趋势每强1%, 买阈降0.3)\n")
    lines.append("- 波动调整系数: 0.1 (分位每高10, 买阈升1)\n")
    lines.append("- 阈值范围: 买[35,65], 卖[20,50]\n")
    lines.append("- EMA平滑: 5期\n\n")
    lines.append("建议调优方向:\n")
    lines.append("1. 若动态阈值交易仍不足: 降低base_buy(50→45) 或 增大趋势系数(0.3→0.5)\n")
    lines.append("2. 若回撤过大: 提高base_buy(50→55) 或 增大波动系数(0.1→0.15)\n")
    lines.append("3. 若阈值跳变大: 增大EMA平滑(5→10)\n")
    
    return "".join(lines)


def main():
    print("=" * 60)
    print("  动态阈值对比回测")
    print("=" * 60)
    
    results = run_comparison()
    checks = check_criteria(results)
    report = generate_report(results, checks)
    
    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), "output_v2")
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = os.path.join(out_dir, "dynamic_threshold_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": "动态阈值对比回测",
            "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stocks": results,
            "checks": checks,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  JSON: {json_path}")
    
    report_path = os.path.join(out_dir, "threshold_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  报告: {report_path}")
    
    print("\n" + report)
    
    return results


if __name__ == "__main__":
    main()
