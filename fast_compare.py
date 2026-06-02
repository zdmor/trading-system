# -*- coding: utf-8 -*-
"""
快速对比回测 v2 — 使用预缓存数据 + 简化的威科夫信号
绕过 StockScorer 的 Tushare API 调用，秒级完成
输出: output_v2/three_stock_compare_v2.json
"""
import sys, os, json, math, time
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import DataFetcher, Analyzer
from scanner import WyckoffAnalyzer
from data_providers import AkshareProvider
from dynamic_thresholds import get_dynamic_thresholds


class FastBacktester:
    """快速回测引擎 — 不调用 StockScorer/Tushare，用本地化威科夫信号"""

    def __init__(self, symbol, initial_cash=100000, start_idx=210, use_dynamic=False):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.start_idx = start_idx
        self.use_dynamic = use_dynamic
        self.df = None
        self.name = ""
        self.industry_name = ""
        self.equity_curve = []
        self.trades = []
        self.results = []
        self.threshold_log = []  # 记录每次扫描的阈值

    def fetch_data(self):
        fetcher = DataFetcher()
        df = fetcher.get_daily(self.symbol, days=500)
        if df is None or len(df) < self.start_idx + 10:
            raise ValueError(f"数据不足")
        self.name = fetcher.get_name(self.symbol)
        self.df = df.reset_index(drop=True)
        try:
            self.industry_name = AkshareProvider.get_stock_industry(self.symbol)
        except Exception:
            self.industry_name = ""
        return self

    def run_backtest(self):
        """一键运行：扫描 → 模拟交易 → 统计 → 记录 equity_curve"""
        self.fetch_data()
        self.results = []
        self.trades = []
        self.equity_curve = []

        n = len(self.df)
        step = 5
        total = (n - self.start_idx) // step

        # 准备技术指标
        df_analyzed = self.df.copy()
        df_analyzed = Analyzer.calc_ma(df_analyzed, [20, 50, 200])
        df_analyzed = Analyzer.calc_atr(df_analyzed)

        c = df_analyzed["close"].values.astype(float)
        h = df_analyzed["high"].values.astype(float)
        l = df_analyzed["low"].values.astype(float)
        v = df_analyzed["volume"].values.astype(float)
        o = df_analyzed["open"].values.astype(float)

        # 获取所有扫描日期的威科夫信号
        scan_results = []
        for idx in range(self.start_idx, n, step):
            try:
                date = self.df.iloc[idx]["date"]
                price = c[idx]

                # 分段看趋势
                slice_end = idx + 1
                slice_start = max(0, idx - 210)
                cs = c[slice_start:slice_end]
                hs = h[slice_start:slice_end]
                ls = l[slice_start:slice_end]
                vs = v[slice_start:slice_end]
                os = o[slice_start:slice_end]

                if len(cs) < 50:
                    continue

                # 趋势方向
                ma50 = np.mean(cs[-50:])
                ma200 = np.mean(cs[-200:]) if len(cs) >= 200 else ma50
                if cs[-1] > ma50 > ma200:
                    trend_dir = "多头"
                elif cs[-1] < ma50 < ma200:
                    trend_dir = "空头"
                else:
                    trend_dir = "震荡"

                # 威科夫信号
                sigs, _ = WyckoffAnalyzer.analyze_all(
                    cs.tolist(), hs.tolist(), ls.tolist(), os.tolist(), vs.tolist()
                )

                sig_name = sigs[0][0] if sigs else "-"
                sig_score = sigs[0][1] if sigs else 0

                # 简化评分 = 威科夫分数 + 基线 + 趋势加成 + 信号惩罚
                trend_bonus = 10 if trend_dir == "多头" else (-10 if trend_dir == "空头" else 0)
                sig_penalty = -25 if sig_name == "Upthrust" else (-10 if sig_name == "EVR" else 0)
                composite = max(0, min(100, sig_score * 0.7 + 30 + trend_bonus + sig_penalty))

                scan_results.append({
                    "idx": idx, "date": date, "price": price,
                    "trend": trend_dir, "signal": sig_name,
                    "signal_score": sig_score, "composite_score": composite,
                })
            except Exception:
                continue

        print(f"  有效扫描: {len(scan_results)}/{total} 次")

        self.results = scan_results

        # ====== 模拟交易 + 记录权益曲线 ======
        cash = self.initial_cash
        shares = 0
        buy_threshold = 50
        sell_threshold = 35
        self.threshold_log = []

        for rec in scan_results:
            score = rec["composite_score"]
            price = rec["price"]
            idx = rec["idx"]

            # 动态阈值: 用当前切片的 K 线数据重新计算阈值
            if self.use_dynamic:
                slice_s = max(0, idx - 400)
                cs_dyn = c[slice_s:idx + 1]
                hs_dyn = h[slice_s:idx + 1]
                ls_dyn = l[slice_s:idx + 1]
                vs_dyn = v[slice_s:idx + 1]
                thresh = get_dynamic_thresholds(cs_dyn, hs_dyn, ls_dyn, vs_dyn)
                buy_threshold = thresh["buy_threshold"]
                sell_threshold = thresh["sell_threshold"]
                # EMA 平滑与上一值
                if self.threshold_log:
                    prev_b = self.threshold_log[-1]["buy_threshold"]
                    prev_s = self.threshold_log[-1]["sell_threshold"]
                    buy_threshold = round(prev_b * 0.7 + buy_threshold * 0.3, 1)
                    sell_threshold = round(prev_s * 0.7 + sell_threshold * 0.3, 1)
                self.threshold_log.append({
                    "idx": idx, "date": str(rec["date"].date()),
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "trend_strength": thresh.get("trend_strength", 0),
                    "volatility_percentile": thresh.get("volatility_percentile", 50),
                    "detail": thresh.get("detail", ""),
                })

            # 买入 — 未持仓用70%现金，已持仓用30%（金字塔加仓）
            if score >= buy_threshold and cash > 0:
                buy_amt = cash * (0.3 if shares > 0 else 0.7)
                buy_sh = int(buy_amt / price / 100) * 100
                if buy_sh > 0:
                    cost = buy_sh * price
                    cash -= cost
                    shares += buy_sh
                    self.trades.append({
                        "date": rec["date"], "action": "买入",
                        "price": price, "shares": buy_sh, "amount": cost,
                        "score": score,
                    })

            # 卖出
            elif score < sell_threshold and shares > 0:
                proceeds = shares * price
                cash += proceeds
                self.trades.append({
                    "date": rec["date"], "action": "卖出",
                    "price": price, "shares": shares, "amount": proceeds,
                    "score": score,
                })
                shares = 0

            # ★ 关键修复：每次扫描后追加权益曲线
            equity = cash + shares * price
            self.equity_curve.append(equity)

        # ====== 计算指标 ======
        if not self.results:
            return None

        final_price = float(self.df.iloc[-1]["close"])
        final_value = cash + shares * final_price
        start_idx_val = self.df.iloc[self.start_idx]["close"]
        start_date = self.df.iloc[self.start_idx]["date"]
        end_date = self.df.iloc[-1]["date"]
        years = max((end_date - start_date).days / 365.25, 0.1)

        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
        hold_return = (final_price - start_idx_val) / start_idx_val * 100

        # 最大回撤
        max_dd = 0.0
        if len(self.equity_curve) > 1:
            peak = self.equity_curve[0]
            for val in self.equity_curve:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                if dd > max_dd:
                    max_dd = dd
        max_dd_pct = max_dd * 100

        # 交易统计
        buys = [t for t in self.trades if t["action"] == "买入"]
        sells = [t for t in self.trades if t["action"] == "卖出"]
        won = 0
        n_rounds = min(len(buys), len(sells))
        for i in range(n_rounds):
            if sells[i]["amount"] > buys[i]["amount"]:
                won += 1
        win_rate = won / n_rounds * 100 if n_rounds > 0 else 0
        trade_count = len(buys) + len(sells)

        # 夏普(简化)
        if len(self.equity_curve) >= 5:
            rets = []
            for i in range(1, len(self.equity_curve)):
                if self.equity_curve[i - 1] > 0:
                    rets.append(self.equity_curve[i] / self.equity_curve[i - 1] - 1)
            if rets:
                avg_ret = np.mean(rets)
                std_ret = np.std(rets)
                sharpe = (avg_ret / std_ret * math.sqrt(252 / 5)) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        return {
            "symbol": self.symbol,
            "name": self.name,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "years": round(years, 1),
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "max_drawdown": round(max_dd_pct, 2),
            "sharpe": round(sharpe, 2),
            "trade_count": trade_count,
            "win_rate": round(win_rate, 1),
            "hold_return": round(hold_return, 2),
            "final_value": round(final_value, 2),
            "equity_curve_len": len(self.equity_curve),
            "use_dynamic": self.use_dynamic,
            "threshold_log": self.threshold_log,
        }


def run_three_stock_compare():
    """三股对比回测"""
    stocks = ["600038", "002050", "600416"]
    cash = 100000
    results = {}

    for symbol in stocks:
        print(f"\n{'='*50}")
        print(f"回测: {symbol}")
        print(f"{'='*50}")

        bt = FastBacktester(symbol, cash, start_idx=210)
        try:
            m = bt.run_backtest()
            if m:
                results[symbol] = m
                print(f"  总收益: {m['total_return']:+.2f}%")
                print(f"  年化: {m['annual_return']:+.2f}%")
                print(f"  最大回撤: {m['max_drawdown']:.2f}%  ← {'✅ 非零' if m['max_drawdown'] > 0.1 else '❌ 为0!'}")
                print(f"  夏普: {m['sharpe']:.2f}")
                print(f"  交易: {m['trade_count']}次  胜率: {m['win_rate']:.1f}%")
                print(f"  买入持有: {m['hold_return']:+.2f}%")
                print(f"  权益曲线长度: {m['equity_curve_len']}")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    out_dir = os.path.join(os.path.dirname(__file__), 'output_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'three_stock_compare_v2.json')

    output = {
        "description": "三股对比回测v2 — 修复max_drawdown=0 bug后",
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "self_check": {
            "all_max_drawdown_non_zero": all(
                r.get("max_drawdown", 0) > 0.1 for r in results.values()
            ),
            "details": {
                s: f"max_drawdown={r.get('max_drawdown', 0):.2f}%" for s, r in results.items()
            }
        }
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"三股对比结果已保存: {out_path}")
    print(f"{'='*60}")

    # 表格输出
    print(f"\n{'股票':<8} {'总收益':>8} {'年化':>8} {'回撤':>8} {'夏普':>6} {'交易':>6} {'胜率':>6} {'持有':>8}")
    print(f"{'-'*66}")
    for s, m in results.items():
        print(f"  {s:<8} {m.get('total_return', 0):>+7.2f}% {m.get('annual_return', 0):>+7.2f}%"
              f" {m.get('max_drawdown', 0):>7.2f}% {m.get('sharpe', 0):>5.2f}"
              f" {m.get('trade_count', 0):>4}次 {m.get('win_rate', 0):>5.1f}%"
              f" {m.get('hold_return', 0):>+7.2f}%")

    # 自检
    print(f"\n--- 自检 ---")
    all_ok = all(r.get("max_drawdown", 0) > 0.1 for r in results.values())
    print(f"  {'✅' if all_ok else '❌'} 所有max_drawdown非零: {all_ok}")
    for s, r in results.items():
        print(f"  {s}: max_drawdown = {r.get('max_drawdown', 0):.2f}%")
    
    return results


if __name__ == "__main__":
    run_three_stock_compare()
