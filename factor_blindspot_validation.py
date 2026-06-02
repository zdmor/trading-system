"""
盲区因子验证 — risk_reward / wyckoff / relative_strength
=============================================================
这3个因子合计占评分系统52%权重，但从未被IC框架验证过。
原因是它们无法用简化版OHLCV因子计算，需要完整的分析管道。

方法：跑200只股票的完整分析管道，提取原始因子值，对比前向收益。
"""
import sys, os, time, json
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from main import DataFetcher, Analyzer, Strategy
from scanner import WyckoffAnalyzer
from scoring import StockScorer
from data_providers import AkshareProvider

FORWARD_DAYS = [5, 10, 20]
STOCK_COUNT = 200
MIN_DATA_DAYS = 120

SIG_NAMES = {
    "Spring": "弹簧", "弱Spring": "弱弹簧",
    "SOS": "强势信号", "LPS": "最后支撑", "弱LPS": "弱支撑",
    "Upthrust": "上冲回落", "弱Upthrust": "弱上冲",
    "EVR": "努力无结果", "Compression": "压缩蓄势",
    "Markup": "主升段",
}


def get_stock_list(n=200):
    """获取候选股列表"""
    from scanner import Scanner
    s = Scanner()
    s.fetch_all_stocks()
    candidates = s.filter_candidates()
    codes = []
    seen = set()
    bj_prefixes = ("8", "4", "9")
    for c in candidates:
        code = c["code"]
        name = c.get("name", "")
        sym = code.split(".")[-1] if "." in code else code
        if sym.startswith(bj_prefixes) or "ST" in name or code in seen:
            continue
        seen.add(code)
        codes.append(code)
        if len(codes) >= n:
            break
    return codes


def analyze_single(raw_code, market_df=None):
    """对单只股票运行完整分析管道，提取盲区因子原始值"""
    try:
        # scanner 返回 sz.002050 格式，DataFetcher 需要无前缀格式
        code = raw_code.split(".")[-1] if "." in raw_code else raw_code
        if not code or len(code) < 4:
            return None
        fetcher = DataFetcher()
        df = fetcher.get_daily(code, days=500)
        if df is None or len(df) < MIN_DATA_DAYS:
            return None

        name = fetcher.get_name(code)
        df = df.reset_index(drop=True)
        n = len(df)
        price = float(df["close"].iloc[-1])

        df = Analyzer.calc_atr(df)
        df = Analyzer.calc_ma(df, [20, 50, 200])
        supports, resistances = Analyzer.detect_levels(df)
        levels = {"supports": supports, "resistances": resistances, "current": price}

        strategy = Strategy(df, 100000)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()

        atr_val = df["atr"].iloc[-1] if "atr" in df.columns else price * 0.02
        stop_price = round(price - atr_val * 1.5, 2) if atr_val > 0 else round(price * 0.93, 2)
        exit_prices = [r for r in resistances] if resistances else [round(price * 1.08, 2)]

        # Wyckoff
        trend_dir = trend.get("direction", "未知")
        wyckoff_sigs = []
        phase_label = ""
        if len(df) >= 50 and trend_dir != "未知":
            c_arr = df["close"].values.astype(float)
            h_arr = df["high"].values.astype(float)
            l_arr = df["low"].values.astype(float)
            v_arr = df["volume"].values.astype(float)
            o_arr = df["open"].values.astype(float)
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(
                c_arr.tolist(), h_arr.tolist(), l_arr.tolist(), v_arr.tolist(), trend_dir, []
            )
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
                c_arr.tolist(), h_arr.tolist(), l_arr.tolist(), o_arr.tolist(), v_arr.tolist()
            )

        # 大盘切片
        market_slice = None
        if market_df is not None:
            try:
                mask = market_df["date"] <= df["date"].iloc[-1]
                market_slice = market_df[mask].copy()
            except Exception:
                pass

        # 行业
        industry = ""
        try:
            industry = AkshareProvider.get_stock_industry(code)
        except Exception:
            pass

        # 评分 → 提取因子原始值
        scorer = StockScorer(
            df=df, price=price, trend=trend, vol=vol, levels=levels,
            stop_price=stop_price, exit_prices=exit_prices,
            wyckoff_signals=wyckoff_sigs, wyckoff_phase=phase_label,
            symbol=code, industry=industry, default_sector_score=55,
            market_df=market_slice,
        )

        # 获取3个盲区因子的原始数据
        rr = scorer.score_risk_reward()
        wy = scorer.score_wyckoff()
        rs = scorer.score_relative_strength()

        # 前向收益
        fwd = {}
        for d in FORWARD_DAYS:
            if n >= d + 1:
                fwd[d] = (df["close"].iloc[-1] / df["close"].iloc[-(d+1)] - 1) * 100
            else:
                fwd[d] = None

        # 近期历史收益（用于对比信号产生前后的表现）
        hist_ret = {}
        for d in [20, 60]:
            if n >= d + 1:
                hist_ret[d] = (df["close"].iloc[-1] / df["close"].iloc[-(d+1)] - 1) * 100
            else:
                hist_ret[d] = None

        best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")

        return {
            "code": code, "name": name, "price": price,
            "risk_reward_ratio": rr.get("ratio", 0),
            "risk_reward_score": rr.get("score", 50),
            "wyckoff_signal": best_sig[0],
            "wyckoff_score": wy.get("raw_score", 0) if isinstance(wy, dict) else 0,
            "wyckoff_score_val": wy.get("score", 50) if isinstance(wy, dict) else 50,
            "relative_strength_score": rs.get("score", 50),
            "trend_dir": trend_dir,
            "phase": phase_label,
            "fwd_5d": fwd[5], "fwd_10d": fwd[10], "fwd_20d": fwd[20],
            "hist_20d": hist_ret[20], "hist_60d": hist_ret[60],
        }
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)[:120]}"
        if not hasattr(analyze_single, "_debug_cnt"):
            analyze_single._debug_cnt = 0
        if analyze_single._debug_cnt < 3:
            print(f"\n  ERROR [{raw_code}]: {err_msg}")
            analyze_single._debug_cnt += 1
        return None


def run_blindspot_validation(n=200):
    """运行盲区因子验证"""
    t0 = time.time()

    # 获取市场指数
    print("获取大盘数据...")
    market_df = None
    try:
        import requests
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or data.get("data", {}).get("sh000001", {}).get("day", [])
        if klines and len(klines) >= 100:
            rows = []
            for k in klines:
                rows.append({"date": pd.Timestamp(k[0]), "close": float(k[2])})
            market_df = pd.DataFrame(rows).reset_index(drop=True)
    except Exception:
        market_df = None

    # 选股
    codes = get_stock_list(n)
    print(f"候选股: {len(codes)}只 | 大盘数据: {'有' if market_df is not None else '无'}")

    # 逐只分析
    results = []
    errors = 0
    for i, code in enumerate(codes):
        rec = analyze_single(code, market_df)
        if rec:
            results.append(rec)
        else:
            errors += 1

        if (i + 1) % 50 == 0 or i == len(codes) - 1:
            print(f"  进度: {i+1}/{len(codes)} 有效{len(results)} 错误{errors}", end="\r")

    print(f"\n分析完成: {len(results)}只有效 ({time.time()-t0:.0f}s)")

    if len(results) < 30:
        print("样本不足，退出")
        return

    # ============================================================
    # 分析1: risk_reward 有效性
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  因 子：risk_reward (权重30%, 评分系统最大权重)")
    print(f"  逻 辑：支撑位附近买入、阻力位有足够空间→高盈亏比→正期望")
    print(f"  方 法：分4组(RR<1 / 1-1.5 / 1.5-2 / >=2)对比前向收益")
    print(f"{'='*70}")

    rr_groups = {"<1": [], "1-1.5": [], "1.5-2": [], ">=2": []}
    for r in results:
        ratio = r["risk_reward_ratio"]
        if ratio < 1:       rr_groups["<1"].append(r)
        elif ratio < 1.5:   rr_groups["1-1.5"].append(r)
        elif ratio < 2:     rr_groups["1.5-2"].append(r)
        else:               rr_groups[">=2"].append(r)

    print(f"\n  {'分组':<8} {'样本':>6} {'5日后':>8} {'10日后':>8} {'20日后':>8}  {'20日胜率':>8} {'历史20日':>8}")
    print(f"  {'-'*60}")
    for label in ["<1", "1-1.5", "1.5-2", ">=2"]:
        group = rr_groups[label]
        if not group:
            continue
        f5 = np.mean([r["fwd_5d"] for r in group if r["fwd_5d"] is not None])
        f10 = np.mean([r["fwd_10d"] for r in group if r["fwd_10d"] is not None])
        f20 = np.mean([r["fwd_20d"] for r in group if r["fwd_20d"] is not None])
        wr20 = np.mean([r["fwd_20d"] > 0 for r in group if r["fwd_20d"] is not None]) * 100
        h20 = np.mean([r["hist_20d"] for r in group if r["hist_20d"] is not None])
        print(f"  RR {label:<5} {len(group):>6} {f5:>+7.2f}% {f10:>+7.2f}% {f20:>+7.2f}%  {wr20:>7.0f}% {h20:>+7.2f}%")

    # 差异检验：高RR vs 低RR
    high_rr = rr_groups[">=2"]
    low_rr = rr_groups["<1"]
    if high_rr and low_rr:
        h_f20 = np.mean([r["fwd_20d"] for r in high_rr if r["fwd_20d"] is not None])
        l_f20 = np.mean([r["fwd_20d"] for r in low_rr if r["fwd_20d"] is not None])
        diff = h_f20 - l_f20
        print(f"\n  差异检验: 高RR(>=2) vs 低RR(<1): 20日收益差 = {diff:+.2f}%")
        if diff > 2:
            print(f"  >> risk_reward 有效! 高RR组跑赢低RR组 {diff:.1f}%")
        elif diff > 0:
            print(f"  >> risk_reward 弱有效 (方向正确但幅度有限)")
        else:
            print(f"  >> risk_reward 可能无效 (方向不对)")

    # ============================================================
    # 分析2: wyckoff 信号有效性
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  因 子：wyckoff (权重10%)")
    print(f"  逻 辑：Spring/SOS/LPS等信号出现后→上涨概率增大")
    print(f"  方 法：按信号类型分组对比前向收益")
    print(f"{'='*70}")

    # 统计各信号出现频率
    sig_groups = {}
    for r in results:
        sig = r["wyckoff_signal"]
        if sig in ("-", "数据不足", ""):
            continue
        if sig not in sig_groups:
            sig_groups[sig] = []
        sig_groups[sig].append(r)

    if sig_groups:
        print(f"\n  {'信号':<12} {'样本':>6} {'5日后':>8} {'10日后':>8} {'20日后':>8}  {'20日胜率':>8}")
        print(f"  {'-'*60}")
        for sig in ["SOS", "Spring", "LPS", "Compression", "弱Spring", "弱LPS"]:
            if sig not in sig_groups:
                continue
            group = sig_groups[sig]
            f5 = np.mean([r["fwd_5d"] for r in group if r["fwd_5d"] is not None])
            f10 = np.mean([r["fwd_10d"] for r in group if r["fwd_10d"] is not None])
            f20 = np.mean([r["fwd_20d"] for r in group if r["fwd_20d"] is not None])
            wr20 = np.mean([r["fwd_20d"] > 0 for r in group if r["fwd_20d"] is not None]) * 100
            label = SIG_NAMES.get(sig, sig)
            print(f"  {label:<12} {len(group):>6} {f5:>+7.2f}% {f10:>+7.2f}% {f20:>+7.2f}%  {wr20:>7.0f}%")

        # 有信号 vs 无信号
        has_sig = [r for r in results if r["wyckoff_signal"] not in ("-", "数据不足", "")]
        no_sig = [r for r in results if r["wyckoff_signal"] in ("-", "数据不足", "")]
        if has_sig and no_sig:
            hs_f20 = np.mean([r["fwd_20d"] for r in has_sig if r["fwd_20d"] is not None])
            ns_f20 = np.mean([r["fwd_20d"] for r in no_sig if r["fwd_20d"] is not None])
            print(f"\n  有信号({len(has_sig)}只): 20日{hs_f20:+.2f}% | 无信号({len(no_sig)}只): 20日{ns_f20:+.2f}%")
            print(f"  >> 信号组收益差 = {hs_f20-ns_f20:+.2f}%")
    else:
        print(f"\n  本截面无有效威科夫信号")

    # ============================================================
    # 分析3: relative_strength 有效性
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  因 子：relative_strength (权重15%)")
    print(f"  逻 辑：个股跑赢大盘=有主力资金关照→未来继续跑赢")
    print(f"  方 法：分3组(弱/中/强)对比前向收益")
    print(f"{'='*70}")

    rs_groups = {"弱(<40)": [], "中(40-60)": [], "强(>60)": []}
    for r in results:
        rs = r["relative_strength_score"]
        if rs < 40:         rs_groups["弱(<40)"].append(r)
        elif rs <= 60:      rs_groups["中(40-60)"].append(r)
        else:               rs_groups["强(>60)"].append(r)

    print(f"\n  {'分组':<10} {'样本':>6} {'5日后':>8} {'10日后':>8} {'20日后':>8}  {'20日胜率':>8} {'历史20日':>8}")
    print(f"  {'-'*60}")
    for label in ["弱(<40)", "中(40-60)", "强(>60)"]:
        group = rs_groups[label]
        if not group:
            continue
        f5 = np.mean([r["fwd_5d"] for r in group if r["fwd_5d"] is not None])
        f10 = np.mean([r["fwd_10d"] for r in group if r["fwd_10d"] is not None])
        f20 = np.mean([r["fwd_20d"] for r in group if r["fwd_20d"] is not None])
        wr20 = np.mean([r["fwd_20d"] > 0 for r in group if r["fwd_20d"] is not None]) * 100
        h20 = np.mean([r["hist_20d"] for r in group if r["hist_20d"] is not None])
        print(f"  RS {label:<6} {len(group):>6} {f5:>+7.2f}% {f10:>+7.2f}% {f20:>+7.2f}%  {wr20:>7.0f}% {h20:>+7.2f}%")

    strong = rs_groups["强(>60)"]
    weak = rs_groups["弱(<40)"]
    if strong and weak:
        s_f20 = np.mean([r["fwd_20d"] for r in strong if r["fwd_20d"] is not None])
        w_f20 = np.mean([r["fwd_20d"] for r in weak if r["fwd_20d"] is not None])
        print(f"\n  差异检验: 强RS(>60) vs 弱RS(<40): 20日收益差 = {s_f20-w_f20:+.2f}%")

    # ============================================================
    # 4. summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    print(f"  sample: {len(results)}只股票 | 总耗时: {time.time()-t0:.0f}s")
    print(f"  注意: 这是一个截面的快照, 结论有随机性")
    print(f"  要获得可靠结论需多截面滚动验证(类似IC回测)")

    # 保存结果
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data_cache", f"blindspot_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 存摘要而非全量
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_stocks": len(results),
        "risk_reward": {label: len(group) for label, group in rr_groups.items()},
        "wyckoff": {sig: len(group) for sig, group in sig_groups.items()},
        "relative_strength": {label: len(group) for label, group in rs_groups.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  摘要已保存: {out_path}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=STOCK_COUNT)
    args = parser.parse_args()
    run_blindspot_validation(n=args.stocks)
