# -*- coding: utf-8 -*-
"""
IC 衰减监控脚本

定期运行，对比最近截面 IC 与基准 IC，衰减超 50% 告警。
独立于 scanner 管线，可 cron 定时执行。

用法:
  python monitor_ic_decay.py                          # 检查输出到 stdout
  python monitor_ic_decay.py --output report.json     # 输出 JSON
  python monitor_ic_decay.py --run-quick-ic           # 运行快速 IC 后检查
  python monitor_ic_decay.py --alert-threshold 0.5    # 自定义衰减阈值
  python monitor_ic_decay.py --max-age-days 30        # 基准 IC 有效期
"""
import sys, os, json, time, math, io
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if sys.stdout.encoding.lower() != 'utf-8' else sys.stdout

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(SCRIPT_DIR, "output_v2", "factor_7_ic.json")
IC_CACHE_PATH = os.path.join(SCRIPT_DIR, "factor_ic_cache.json")
ALERT_LOG_PATH = os.path.join(SCRIPT_DIR, "ic_decay_alerts.json")
REPORT_PATH = os.path.join(SCRIPT_DIR, "output_v2", "ic_monitor_report.json")

FACTOR_NAMES_CN = {
    "wyckoff": "威科夫",
    "risk_reward": "盈亏比",
    "volume": "量能",
    "candlestick": "K线形态",
    "trend_momentum": "趋势动量",
    "relative_strength": "相对强度",
    "volatility": "波动率",
}

# 因子名称映射（缓存中文名 → 基线英文名）
# 用于 check_decay() 中匹配基线和当前 IC 数据
FACTOR_NAME_MAP = {
    "威科夫": "wyckoff",
    "盈亏比": "risk_reward",
    "量能": "volume",
    "K线": "candlestick",
    "板块": "sector",
    "趋势动量": "trend_momentum",
    "大盘": "macro",
    "相对强度": "relative_strength",
    # 以下可能是缓存中额外出现的因子——None 跳过 IC 对比
    "RSI": None,
    "Δ动量": None,
    "换手率": None,
    "强度": None,
    "互证": None,
}


def load_baseline():
    """读取基准 IC"""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("factors", {})


def load_current_ic():
    """读取当前 IC (从 factor_ic_cache.json)"""
    if not os.path.exists(IC_CACHE_PATH):
        return None
    with open(IC_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get("data", {})


def load_alerts():
    """读取历史告警"""
    if os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": []}


def save_alerts(alerts_obj):
    with open(ALERT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(alerts_obj, f, ensure_ascii=False, indent=2)


def decay_level(name, base_val, cur_val, threshold):
    """计算衰减程度
    
    返回 (decay_pct, status):
    - decay_pct: 负数=衰减, 正数=增强, 0=无变化
    - status: "严重衰减" | "衰减" | "稳定" | "增强" | "符号反转"
    """
    if base_val is None or cur_val is None:
        return (None, "无数据")
    
    if abs(base_val) < 0.001:
        # 基准 IC≈0, 检查是否变成更有意义的值
        if abs(cur_val) > 0.01:
            return (None, "增强(从零变有效)")
        return (None, "稳定(始终约零)")
    
    # 符号检查
    base_sign = 1 if base_val > 0 else -1
    cur_sign = 1 if cur_val > 0 else -1
    sign_flipped = base_sign != cur_sign and abs(base_val) > 0.005
    
    if sign_flipped:
        # 符号反转是严重问题
        decay = (abs(cur_val) - abs(base_val)) / abs(base_val)
        return (round(decay, 3), "严重-SIGN_FLIP")
    
    # 正常计算衰减
    decay = (abs(cur_val) - abs(base_val)) / abs(base_val)
    decay = round(decay, 3)
    
    # base_val 和 cur_val 符号相同，衰减=绝对值下降比例
    if decay <= -threshold:
        return (decay, f"严重衰减({-decay*100:.0f}%)")
    elif decay <= -0.3:
        return (decay, f"轻微衰减({-decay*100:.0f}%)")
    elif decay >= 0.3:
        return (decay, f"增强({decay*100:.0f}%)")
    else:
        return (decay, "稳定")


def check_decay(baseline, current, threshold=0.5):
    """对比基准与当前 IC
    
    Args:
        baseline: 基准 IC 数据
        current: 当前 IC 数据
        threshold: 衰减阈值 (0.5 = 50%)
    
    Returns: {factor: {base_ic, cur_ic, decay_pct, status, alert: bool}}
    """
    if not baseline:
        return {"error": "缺少基准 IC 数据"}
    if not current:
        return {"error": "缺少当前 IC 数据"}
    
    results = {}
    warnings = []  # 因子key匹配告警

    # 检查当前因子中是否存在基准中不存在的 key
    unmatched = set(current.keys()) - set(baseline.keys())
    if unmatched:
        print(f"  [警告] 以下因子在基准中不存在，将被跳过: {unmatched}")

    all_keys = set(baseline.keys()) | set(current.keys())
    
    for key in sorted(all_keys):
        # 名称映射：current 中的中文键映射到 baseline 的英文键
        mapped_key = FACTOR_NAME_MAP.get(key, key)
        if mapped_key is None:
            # 新因子（缓存中有但映射表中为 None），跳过 IC 对比
            continue
        
        # 告警: 当前key不在映射表中且不在baseline中 → 可能key不匹配
        if key not in FACTOR_NAME_MAP and mapped_key not in baseline:
            warnings.append(f"因子 '{key}' 不在名称映射表中，且未在基准IC中找到，可能key不匹配(中/英文命名冲突)")
            continue
        
        b = baseline.get(mapped_key, {})
        c = current.get(key, {})
        
        base_ic = b.get("ic_mean") if isinstance(b, dict) else None
        cur_ic = c.get("ic") if isinstance(c, dict) else None

        decay, status = decay_level(key, base_ic, cur_ic, threshold)
        alert = status not in ("稳定", "无数据", "稳定(始终约零)", "增强(从零变有效)")

        results[mapped_key] = {
            "name_cn": FACTOR_NAMES_CN.get(mapped_key, key),
            "mapped_key": mapped_key,
            "base_ic": round(base_ic, 4) if base_ic is not None else None,
            "cur_ic": round(cur_ic, 4) if cur_ic is not None else None,
            "decay_pct": decay,
            "status": status,
            "alert": alert,
        }
    
    if warnings:
        results["_warnings"] = warnings
    return results


def find_triggered_factors(results):
    """提取触发的因子"""
    return {k: v for k, v in results.items() 
            if isinstance(v, dict) and v.get("alert")}


def print_report(results, baseline_time=None, cur_time=None):
    """打印可读报告"""
    if "error" in results:
        print(f"  [错误] {results['error']}")
        return
    
    print(f"\n{'='*68}")
    print(f"  IC 衰减监控报告")
    print(f"{'='*68}")
    if baseline_time:
        print(f"  基准时间: {baseline_time}")
    if cur_time:
        print(f"  当前时间: {cur_time}")
    
    # 筛选告警
    alerts = find_triggered_factors(results)
    if alerts:
        al = len(alerts)
        print(f"\n  [告警] {al} 个因子出现异常:")
        for key, r in alerts.items():
            print(f"    * {r['name_cn']}({key}): {r['status']} (基准 {r['base_ic']})")
    
    # 详细表格
    # 显示key不匹配警告
    if "_warnings" in results:
        print(f"\n  [注意] 发现 {len(results['_warnings'])} 个key不匹配:")
        for w in results["_warnings"]:
            print(f"    * {w}")
    
    print(f"\n  {'因子':<14} {'基准IC':>8} {'当前IC':>8} {'变化%':>8} {'状态':<18} {'告警':>6}")
    print(f"  {'-'*66}")
    for key in sorted(results.keys()):
        if key.startswith("_"):
            continue
        r = results[key]
        base_str = f"{r['base_ic']:+.4f}" if r['base_ic'] is not None else "   -"
        cur_str = f"{r['cur_ic']:+.4f}" if r['cur_ic'] is not None else "   -"
        dec_str = f"{r['decay_pct']*100:+.0f}%" if r['decay_pct'] is not None else "   -"
        flag = "[!]" if r['alert'] else ""
        print(f"  {r['name_cn']+f'({key})':<14} {base_str:>8} {cur_str:>8} {dec_str:>8} {r['status']:<18} {flag:>6}")
    print()


def run_quick_ic():
    """运行一次快速 IC 检查（15截面）"""
    try:
        from factor_7_validate import run_all_factors
        run_all_factors()
        return True
    except Exception as e:
        print(f"  快速 IC 运行失败: {e}")
        return False


def save_report(results, baseline_time=None, cur_time=None, path=None):
    """保存报告到 JSON"""
    if path is None:
        path = REPORT_PATH
    
    alerts = find_triggered_factors(results)
    
    factor_data = {k: v for k, v in results.items() if not k.startswith("_")}
    report = {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_time": baseline_time,
        "current_time": cur_time,
        "n_alerts": len(alerts),
        "factors": factor_data,
    }
    if "_warnings" in results:
        report["key_warnings"] = results["_warnings"]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return path


def log_alerts(results):
    """记录告警到历史日志"""
    alerts = find_triggered_factors(results)
    if not alerts:
        return 0
    
    log = load_alerts()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for key, r in alerts.items():
        log["alerts"].append({
            "time": now,
            "factor": key,
            "name_cn": r["name_cn"],
            "status": r["status"],
            "base_ic": r["base_ic"],
            "cur_ic": r["cur_ic"],
            "decay_pct": r["decay_pct"],
        })
    
    # 保留最近 200 条
    if len(log["alerts"]) > 200:
        log["alerts"] = log["alerts"][-200:]
    
    save_alerts(log)
    return len(alerts)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="IC 衰减监控")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--run-quick-ic", action="store_true", help="运行快速 IC 后检查")
    parser.add_argument("--alert-threshold", type=float, default=0.5, help="衰减阈值 (默认 0.5)")
    parser.add_argument("--max-age-days", type=int, default=60, help="基准 IC 有效期 (天)")
    parser.add_argument("--quiet", action="store_true", help="静默模式(不打印)")
    args = parser.parse_args()
    
    # 检查基准 IC 时效
    baseline_time = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            bd = json.load(f)
        baseline_time = bd.get("run_time", "未知")
        try:
            bt = datetime.strptime(baseline_time, "%Y-%m-%d %H:%M:%S")
            age = (datetime.now() - bt).days
            if age > args.max_age_days:
                if not args.quiet:
                    print(f"  [提示] 基准 IC 已 {age} 天，超过 {args.max_age_days} 天有效期")
                    print(f"  建议运行: python monitor_ic_decay.py --run-quick-ic")
        except ValueError:
            pass
    
    # 可选: 运行快速 IC
    if args.run_quick_ic:
        if not args.quiet:
            print("  运行快速 IC...")
        run_quick_ic()
        if not args.quiet:
            print("  快速 IC 完成")
    
    # 加载数据
    baseline = load_baseline()
    current = load_current_ic()
    cur_time = None
    if current:
        cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not baseline:
        print("  错误: 找不到基准 IC 文件: {}".format(BASELINE_PATH))
        print("  请先运行 factor_7_validate.py 生成基准 IC")
        sys.exit(1)
    
    if not current:
        print("  错误: 找不到当前 IC 缓存: {}".format(IC_CACHE_PATH))
        print("  请先运行 scanner 管线积累 IC 数据, 或使用 --run-quick-ic")
        sys.exit(1)
    
    # 检查衰减
    results = check_decay(baseline, current, args.alert_threshold)
    
    if not args.quiet:
        print_report(results, baseline_time, cur_time)
    
    # 记录告警
    n_logged = log_alerts(results)
    if n_logged and not args.quiet:
        print(f"  已记录 {n_logged} 条告警到 {ALERT_LOG_PATH}")
    
    # 输出 JSON
    out_path = save_report(results, baseline_time, cur_time, args.output)
    if not args.quiet:
        print(f"  报告已保存: {out_path}")
    
    # 退出码: 有告警 → 1
    alerts = find_triggered_factors(results)
    if alerts:
        sys.exit(1)
    
    return results


if __name__ == "__main__":
    main()