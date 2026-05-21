"""
回测系统
对个股运行历史模拟：Wyckoff 信号回溯 + 评分有效性 + 模拟交易
支持可调阈值和参数扫描
"""
import sys, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from main import DataFetcher, Analyzer, Strategy
from scanner import WyckoffAnalyzer
from scoring import StockScorer
from data_providers import AkshareProvider


class Backtester:
    """个股回测引擎 — 滑动窗口模拟分析管道"""

    SIG_NAMES = {
        "Spring": "弹簧", "弱Spring": "弱弹簧",
        "SOS": "强势信号", "LPS": "最后支撑", "弱LPS": "弱支撑",
        "Upthrust": "上冲回落", "弱Upthrust": "弱上冲",
        "EVR": "努力无结果", "Compression": "压缩蓄势",
        "Markup": "主升段",
    }

    def __init__(self, symbol, initial_cash=100000, start_idx=210, step=5,
                 buy_threshold=50, strong_buy=65, sell_threshold=40,
                 default_sector_score=55):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.start_idx = start_idx
        self.step = step
        self.buy_threshold = buy_threshold
        self.strong_buy = strong_buy
        self.sell_threshold = sell_threshold
        self.default_sector_score = default_sector_score

        self.df = None
        self.name = ""
        self.industry_name = ""
        self.market_df = None
        self.results = []
        self.trades = []

    def _market_trend(self, idx=None):
        """判断大盘趋势: 返回 'bull'/'bear'/'neutral' """
        if self.market_df is None or len(self.market_df) < 50:
            return "neutral"
        mdf = self.market_df
        if idx is not None:
            mdf = mdf.iloc[:idx + 1]
        if len(mdf) < 50:
            return "neutral"
        c = mdf["close"].values
        ma50 = np.mean(c[-50:]) if len(c) >= 50 else c[-1]
        ma200 = np.mean(c[-200:]) if len(c) >= 200 else c[-1]
        if c[-1] > ma50 > ma200:
            return "bull"
        if c[-1] < ma50 < ma200:
            return "bear"
        return "neutral"

    def _market_score(self, idx=None):
        """大盘评分(0-100)，用于模拟交易阈值调节"""
        trend = self._market_trend(idx)
        if trend == "bull":
            return 80
        if trend == "bear":
            return 20
        return 50

    # ─────────────────────── 数据 ───────────────────────

    def fetch_data(self):
        """获取全部历史数据并缓存行业，同时获取大盘指数"""
        fetcher = DataFetcher()
        df = fetcher.get_daily(self.symbol, days=500)
        if df is None or len(df) < self.start_idx + 10:
            raise ValueError(f"数据不足 ({len(df) if df is not None else 0}条)")
        self.name = fetcher.get_name(self.symbol)
        self.df = df.reset_index(drop=True)

        # 获取大盘指数 (SH 000001)
        try:
            import requests
            url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"
            r = requests.get(url, timeout=15)
            data = r.json()
            klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or data.get("data", {}).get("sh000001", {}).get("day", [])
            if klines and len(klines) >= 100:
                import pandas as pd
                rows = []
                for k in klines:
                    rows.append({"date": pd.Timestamp(k[0]), "open": float(k[1]), "close": float(k[2]),
                                 "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])})
                self.market_df = pd.DataFrame(rows).reset_index(drop=True)
        except Exception:
            self.market_df = None

        # 缓存行业名称（用于板块评分默认值）
        try:
            self.industry_name = AkshareProvider.get_stock_industry(self.symbol)
        except Exception:
            self.industry_name = ""
        return self

    def _slice(self, idx):
        return self.df.iloc[:idx + 1].copy()

    def _forward_return(self, idx, days):
        if idx + days >= len(self.df):
            return None
        buy = self.df.iloc[idx]["close"]
        sell = self.df.iloc[idx + days]["close"]
        return (sell - buy) / buy * 100

    # ─────────────────────── 单日分析 ───────────────────────

    def run_date(self, idx):
        df_slice = self._slice(idx)
        price = float(df_slice["close"].iloc[-1])
        date = df_slice["date"].iloc[-1]

        df_slice = Analyzer.calc_atr(df_slice)
        df_slice = Analyzer.calc_ma(df_slice, [20, 50, 200])
        supports, resistances = Analyzer.detect_levels(df_slice)
        levels = {"supports": supports, "resistances": resistances, "current": price}

        strategy = Strategy(df_slice, self.initial_cash)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()

        recent_lows = df_slice["low"].iloc[-60:].min()
        stop_price = round(recent_lows * 0.985, 2) if recent_lows > 0 else round(price * 0.93, 2)
        exit_prices = [r for r in resistances] if resistances else [round(price * 1.08, 2)]
        entry_check = strategy.entry_check(levels)

        trend_dir = trend["direction"]
        wyckoff_sigs = []
        phase_label = ""
        if len(df_slice) >= 50 and trend_dir != "未知":
            c = df_slice["close"].values.astype(float)
            h = df_slice["high"].values.astype(float)
            l = df_slice["low"].values.astype(float)
            v = df_slice["volume"].values.astype(float)
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, []
            )
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir
            )

        # 大盘切片（取到当前日期）
        market_slice = None
        market_trend_label = "未知"
        if self.market_df is not None:
            try:
                # 用日期对齐切片
                mask = self.market_df["date"] <= date
                market_slice = self.market_df[mask].copy()
                if len(market_slice) >= 50:
                    mc = market_slice["close"].values
                    ma50 = np.mean(mc[-50:])
                    ma200 = np.mean(mc[-200:]) if len(mc) >= 200 else ma50
                    if mc[-1] > ma50 > ma200:
                        market_trend_label = "多头"
                    elif mc[-1] < ma50 < ma200:
                        market_trend_label = "空头"
                    else:
                        market_trend_label = "震荡"
            except Exception:
                pass

        # 评分（板块用缓存的行业名 + 默认分）
        scorer = StockScorer(
            df=df_slice, price=price, trend=trend, vol=vol, levels=levels,
            stop_price=stop_price, exit_prices=exit_prices,
            wyckoff_signals=wyckoff_sigs, wyckoff_phase=phase_label,
            symbol=None, industry=self.industry_name,
            default_sector_score=self.default_sector_score,
            market_df=market_slice,
        )
        scoring = scorer.compute()

        # 评分映射到操作（适配可调阈值）
        cs = scoring["composite_score"]
        if cs >= self.strong_buy:
            action = "强加仓"
            factor = 1.0
        elif cs >= self.buy_threshold:
            action = "加仓"
            factor = 0.7
        elif cs >= self.sell_threshold:
            action = "持有"
            factor = 0.0
        elif cs >= 30:
            action = "减仓"
            factor = -0.5
        else:
            action = "离场"
            factor = -1.0

        best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")
        best_sig_name = self.SIG_NAMES.get(best_sig[0], best_sig[0])
        fwd = {d: self._forward_return(idx, d) for d in [5, 10, 20]}
        fwd_score = self._forward_return(idx, 10)

        return {
            "idx": idx, "date": date, "price": price,
            "trend": trend_dir, "phase": phase_label,
            "market_trend": market_trend_label,
            "signal_raw": best_sig[0], "signal": best_sig_name,
            "signal_score": best_sig[1],
            "entry_signal": entry_check.get("signal", "未知"),
            "composite_score": cs, "score_action": action,
            "forward_5d": fwd[5], "forward_10d": fwd[10],
            "forward_20d": fwd[20], "forward_score_10d": fwd_score,
        }

    # ─────────────────────── 全量回测 ───────────────────────

    def run_all(self):
        self.fetch_data()
        self.results = []
        n = len(self.df)
        total = (n - self.start_idx) // self.step
        c = 0
        for idx in range(self.start_idx, n, self.step):
            try:
                rec = self.run_date(idx)
                self.results.append(rec)
                c += 1
                print(f"\r  进度: {c}/{total}  {rec['date'].strftime('%m-%d')} 评分{rec['composite_score']:.0f}", end="", flush=True)
            except Exception:
                continue
        print()
        return self

    # ─────────────────────── 模拟交易 ───────────────────────

    def simulate_trades(self, results=None):
        """模拟交易，使用大盘趋势动态调整卖出阈值"""
        if results is None:
            results = self.results
        self.trades = []
        cash = self.initial_cash
        shares = 0

        for rec in results:
            price = rec["price"]
            score = rec["composite_score"]
            action = rec["score_action"]
            market_trend = rec.get("market_trend", "未知")

            # 动态卖出阈值：牛市更宽松(不易卖)，熊市更严格(易卖)
            sell_adj = 0
            if market_trend == "多头":
                sell_adj = -8   # 牛市: 卖出阈值降低8分 → 更不容易触发卖出
            elif market_trend == "空头":
                sell_adj = 5    # 熊市: 卖出阈值提高5分 → 更容易触发卖出
            effective_sell = self.sell_threshold + sell_adj

            # 买入: 评分 >= buy_threshold
            if score >= self.buy_threshold and cash > 0 and action in ("强加仓", "加仓"):
                buy_amt = cash * 0.7
                buy_sh = int(buy_amt / price / 100) * 100
                if buy_sh > 0:
                    cash -= buy_sh * price
                    shares += buy_sh
                    self.trades.append({
                        "date": rec["date"], "action": "买入",
                        "price": price, "shares": buy_sh,
                        "amount": buy_sh * price, "score": score,
                    })

            # 卖出: 评分 < 动态sell_threshold 且持有
            elif score < effective_sell and shares > 0 and action in ("减仓", "离场"):
                proceeds = shares * price
                cash += proceeds
                self.trades.append({
                    "date": rec["date"], "action": "卖出",
                    "price": price, "shares": shares,
                    "amount": proceeds, "score": score,
                })
                shares = 0

        final_price = self.df["close"].iloc[-1]
        return cash + shares * final_price

    # ─────────────────────── 指标计算 ───────────────────────

    def calc_metrics(self, results=None):
        if results is None:
            results = self.results
        if not results:
            return {}

        final_value = self.simulate_trades(results)
        final_price = self.df["close"].iloc[-1]
        start_date = self.df["date"].iloc[self.start_idx]
        end_date = self.df["date"].iloc[-1]
        years = max((end_date - start_date).days / 365.25, 0.1)

        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100

        # 买入持有
        hold_return = (final_price - self.df["close"].iloc[self.start_idx]) / self.df["close"].iloc[self.start_idx] * 100

        # 信号质量
        sig_stats = {}
        for rec in results:
            sig = rec["signal_raw"]
            if sig in ("-", "数据不足", ""):
                continue
            if sig not in sig_stats:
                sig_stats[sig] = {"cnt": 0, "5d": [], "10d": [], "20d": []}
            sig_stats[sig]["cnt"] += 1
            for k, dk in [("5d", 5), ("10d", 10), ("20d", 20)]:
                v = self._forward_return(rec["idx"], dk)
                if v is not None:
                    sig_stats[sig][k].append(v)

        # 评分有效性
        buckets = {}
        for rec in results:
            act = rec["score_action"]
            if act not in buckets:
                buckets[act] = []
            v = self._forward_return(rec["idx"], 10)
            if v is not None:
                buckets[act].append(v)

        # 交易统计
        buys = [t for t in self.trades if t["action"] == "买入"]
        sells = [t for t in self.trades if t["action"] == "卖出"]
        won = total_pnl = win_pnl = loss_pnl = 0
        n_trades = min(len(buys), len(sells))
        for i in range(n_trades):
            pnl = sells[i]["amount"] - buys[i]["amount"]
            total_pnl += pnl
            if pnl > 0:
                won += 1; win_pnl += pnl
            else:
                loss_pnl += abs(pnl)

        return {
            "total_return": round(total_return, 2),
            "annual_return": round(annual_return, 2),
            "hold_return": round(hold_return, 2),
            "final_value": round(final_value, 2),
            "trade_count": n_trades,
            "buy_trades": len(buys),
            "sell_trades": len(sells),
            "win_rate": round(won / n_trades * 100, 1) if n_trades > 0 else 0,
            "profit_factor": round(win_pnl / loss_pnl, 2) if loss_pnl > 0 else ("∞" if win_pnl > 0 else 0),
            "start_date": start_date, "end_date": end_date, "years": round(years, 1),
            "sig_stats": sig_stats, "score_buckets": buckets,
            "_win_pnl": win_pnl, "_loss_pnl": loss_pnl,
        }

    # ─────────────────────── 报告 ───────────────────────

    def report(self, results=None):
        m = self.calc_metrics(results)
        if not m:
            return m

        sep = "=" * 60
        dash = "-" * 60

        print(f"\n{sep}")
        print(f"  回测: {self.name} ({self.symbol})")
        print(f"  区间: {m['start_date'].strftime('%Y%m%d')}-{m['end_date'].strftime('%Y%m%d')} ({m['years']}年)")
        print(f"  参数: 买入>={self.buy_threshold} 强买>={self.strong_buy} 卖出<{self.sell_threshold}")
        print(f"  分析: {len(results or self.results)}次 步长{self.step}日")
        print(f"{sep}")

        print(f"\n{dash}\n  收益对比\n{dash}")
        print(f"  策略: {m['total_return']:>+8.2f}%  年化: {m['annual_return']:>+8.2f}%")
        print(f"  买入持有: {m['hold_return']:>+8.2f}%")
        print(f"  最终资产: {m['final_value']:>10,.2f}  交易: {m['trade_count']}次 胜率{m['win_rate']}%")
        if m.get("profit_factor") not in (0, "∞"):
            print(f"  盈亏比: {m['profit_factor']}")

        sigs = m["sig_stats"]
        if sigs:
            print(f"\n{dash}\n  信号质量\n{dash}")
            print(f"  {'信号':<14} {'次':>4} {'5日':>8} {'10日':>8} {'20日':>8} {'5日胜':>6} {'10日胜':>6} {'20日胜':>6}")
            print(f"  {'-'*66}")
            for s in ["SOS","Spring","LPS","Compression","弱Spring","弱LPS","EVR","Upthrust","弱Upthrust"]:
                if s not in sigs: continue
                st = sigs[s]
                nm = self.SIG_NAMES.get(s, s)
                def avg(arr): return f"{np.mean(arr):>+7.2f}%" if arr else "    -  "
                def wr(arr): return f"{sum(1 for x in arr if x>0)/len(arr)*100:.0f}%" if arr else "  -"
                print(f"  {nm:<14} {st['cnt']:>4} {avg(st['5d'])} {avg(st['10d'])} {avg(st['20d'])}  {wr(st['5d']):>6} {wr(st['10d']):>6} {wr(st['20d']):>6}")

        if m["score_buckets"]:
            print(f"\n{dash}\n  评分有效性 — 评分后10日收益\n{dash}")
            print(f"  {'评级':<10} {'样本':>6} {'平均收益':>10} {'胜率':>8}")
            print(f"  {'-'*38}")
            for act in ["强加仓","加仓","持有","减仓","离场"]:
                arr = m["score_buckets"].get(act, [])
                if arr:
                    print(f"  {act:<10} {len(arr):>6} {np.mean(arr):>+9.2f}% {sum(1 for x in arr if x>0)/len(arr)*100:>7.0f}%")

        if self.trades:
            print(f"\n{dash}\n  交易日志\n{dash}")
            print(f"  {'日期':<8} {'操作':<8} {'价格':>8} {'股数':>6} {'金额':>10} {'评分':>6}")
            print(f"  {'-'*48}")
            cum_pnl = 0
            buy_cost = 0
            buy_shares = 0
            for t in self.trades:
                if t["action"] == "买入":
                    buy_cost += t["amount"]
                    buy_shares += t["shares"]
                    print(f"  {t['date'].strftime('%m-%d')} {t['action']:<8} {t['price']:>8.2f} {t['shares']:>6} {t['amount']:>10,.0f} {t['score']:>5.0f}")
                elif t["action"] == "卖出" and buy_shares > 0:
                    pnl = t["amount"] - buy_cost
                    cum_pnl += pnl
                    arrow = "↑" if pnl >= 0 else "↓"
                    print(f"  {t['date'].strftime('%m-%d')} {t['action']:<8} {t['price']:>8.2f} {t['shares']:>6} {t['amount']:>10,.0f} {t['score']:>5.0f}  {arrow}{abs(pnl):>8,.0f}")
                    buy_cost = 0
                    buy_shares = 0

        print(f"\n{sep}\n  回测完毕\n{sep}\n")
        return m


# ─────────────────────── 参数扫描 ───────────────────────

def sweep(symbol="002050", cash=100000, buy_range=(50, 75, 5)):
    """扫描不同买入阈值，对比效果"""
    base = Backtester(symbol, cash)
    base.fetch_data()
    base.run_all()

    # 用同一组 results 模拟不同阈值
    results = base.results

    print(f"\n{'='*60}")
    print(f"  参数扫描: {base.name} ({symbol})")
    print(f"  基准区间: {len(results)}次分析")
    print(f"{'='*60}")
    print(f"  {'阈值':>6} {'操作':<10} {'收益':>10} {'年化':>8} {'交易':>6} {'胜率':>6} {'持有':>8}")
    print(f"  {'-'*58}")

    best = {"total_return": -999}
    for threshold in range(buy_range[0], buy_range[1] + 1, buy_range[2]):
        bt = Backtester(symbol, cash, buy_threshold=threshold,
                        strong_buy=threshold + 15)
        bt.df = base.df
        bt.name = base.name
        bt.industry_name = base.industry_name

        # 重新映射操作
        adj = []
        for r in results:
            cs = r["composite_score"]
            if cs >= bt.strong_buy:
                act = "强加仓"
            elif cs >= bt.buy_threshold:
                act = "加仓"
            elif cs >= bt.sell_threshold:
                act = "持有"
            elif cs >= 30:
                act = "减仓"
            else:
                act = "离场"
            r2 = dict(r)
            r2["score_action"] = act
            adj.append(r2)

        m = bt.calc_metrics(adj)
        arrow = ""
        if m["total_return"] > best["total_return"]:
            best = m
            arrow = " ← best"

        print(f"  {threshold:>6} {m['buy_trades']+m['sell_trades']//2 if m['buy_trades']>0 else 0:<6}笔"
              f" {m['total_return']:>+9.2f}% {m['annual_return']:>+7.2f}%"
              f" {m['trade_count']:>4}次 {m['win_rate']:>5.1f}%"
              f" {m['hold_return']:>+7.2f}%{arrow}")

    print(f"\n  最优解: 买入≥{best.get('buy_threshold', '?')} 收益{best.get('total_return', '?')}% 交易{best.get('trade_count', '?')}次 胜率{best.get('win_rate', '?')}%")
    print(f"  (买入持有: {best.get('hold_return', '?')}%)")
    print(f"{'='*60}\n")


# ─────────────────────── 命令行入口 ───────────────────────

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "002050"
    cash = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace(".", "").isdigit() else 100000
    mode = sys.argv[3] if len(sys.argv) > 3 else "single"

    if mode == "sweep":
        sweep(symbol, cash, buy_range=(50, 75, 5))
    else:
        bt = Backtester(symbol, cash,
                        buy_threshold=50, strong_buy=65, sell_threshold=40,
                        default_sector_score=55)
        bt.run_all()
        bt.report()
