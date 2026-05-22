"""
因子IC回测 — 逐个因子测预测能力
高分组(前30%) vs 低分组(后30%)的未来5日胜率差 + IC值
"""
import sys, os, time, json, math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from scanner import Scanner
from main import Analyzer, Strategy, load_config
from scoring import StockScorer
from data_providers import get_financial_indicators, get_stock_moneyflow

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"

def get_klines(code):
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get(key, {}).get("qfqday") or data.get("data", {}).get(key, {}).get("day") or []
        if not klines: return None
        rows = [{"date": pd.Timestamp(k[0]), "open": float(k[1]), "close": float(k[2]),
                 "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])} for k in klines]
        return pd.DataFrame(rows)
    except: return None


def get_factors(df, code):
    """计算单只股票的所有因子值"""
    try:
        mask = df["date"] <= pd.Timestamp("2026-05-21")
        sdf = df[mask].copy()
        if len(sdf) < 60: return None

        sdf = Analyzer.calc_atr(sdf)
        sdf = Analyzer.calc_ma(sdf, [20, 50, 200])
        supports, resistances = Analyzer.detect_levels(sdf)
        price = float(sdf["close"].iloc[-1])
        levels = {"supports": supports, "resistances": resistances, "current": price}

        strategy = Strategy(sdf, 100000)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()

        atr_val = sdf["atr"].iloc[-1] if "atr" in sdf.columns else price * 0.02
        stop_price = round(price - atr_val * 3.0, 2)
        atr_target = round(price + atr_val * 5.0, 2)
        exit_prices = [atr_target] + ([r for r in resistances[:2]] if resistances else [])

        wyckoff_sigs = []
        phase_label = ""
        trend_dir = trend["direction"]
        if len(sdf) >= 50 and trend_dir != "未知":
            c = sdf["close"].values.astype(float)
            h = sdf["high"].values.astype(float)
            l = sdf["low"].values.astype(float)
            v = sdf["volume"].values.astype(float)
            from scanner import WyckoffAnalyzer
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, [])
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir)

        fin_metrics = get_financial_indicators(code)
        moneyflow = get_stock_moneyflow(code)

        scorer = StockScorer(
            df=sdf, price=price, trend=trend, vol=vol, levels=levels,
            stop_price=stop_price, exit_prices=exit_prices,
            position={"shares": 0, "avg_price": 0}, symbol=code,
        )
        scoring = scorer.compute()

        # 提取各因子得分
        factors = scoring.get("breakdown", {})
        factor_scores = {}
        for k, v in factors.items():
            if isinstance(v, dict) and "score" in v:
                factor_scores[k] = v["score"]

        # 前向 5 日收益
        idx = len(df[mask])
        fwd_5d = None
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx+5]["close"] - price) / price * 100

        return {
            "code": code,
            "factors": factor_scores,
            "composite": scoring.get("composite_score"),
            "fwd_5d": fwd_5d,
            "trend": trend_dir,
            "sig": wyckoff_sigs[0][0] if wyckoff_sigs else "-",
        }
    except Exception as e:
        return None


print("=" * 70)
print("  因子 IC 回测")
print("=" * 70)
print("  统计期: 截至 2026-05-21，前向 5 日收益")
print()

t0 = time.time()

# 获取候选池
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
candidates = sc.candidates[:200]
print(f"  候选: {len(candidates)}只")

# 获取K线
stock_dfs = {}
for i, stk in enumerate(candidates):
    df = get_klines(stk["code"])
    if df is not None and len(df) > 105:
        stock_dfs[stk["code"]] = df
    if (i+1) % 50 == 0:
        print(f"  K线加载: {i+1}/{len(candidates)}", end="\r")
print(f"  K线有效: {len(stock_dfs)}只")

# 计算因子
all_data = []
for i, (code, df) in enumerate(stock_dfs.items()):
    r = get_factors(df, code)
    if r:
        all_data.append(r)
    if (i+1) % 50 == 0:
        print(f"  因子计算: {i+1}/{len(stock_dfs)}", end="\r")
print(f"  有效数据: {len(all_data)}只\n")

# 收集所有因子名
factor_names = set()
for r in all_data:
    factor_names.update(r["factors"].keys())
factor_names = sorted(factor_names)

# 对每个因子做 IC 测试
results = []
for fn in factor_names:
    vals = [(r["factors"].get(fn), r["fwd_5d"]) for r in all_data
            if fn in r["factors"] and r["fwd_5d"] is not None]
    if len(vals) < 20:
        continue

    scores, fwds = zip(*vals)
    scores = np.array(scores)
    fwds = np.array(fwds)

    # Rank IC (Spearman)
    from scipy.stats import spearmanr
    ic, ic_p = spearmanr(scores, fwds)

    # 高分组 vs 低分组
    threshold_high = np.percentile(scores, 70)
    threshold_low = np.percentile(scores, 30)

    high_mask = scores >= threshold_high
    low_mask = scores <= threshold_low

    high_fwd = fwds[high_mask]
    low_fwd = fwds[low_mask]

    high_win = np.mean(high_fwd > 0) * 100
    low_win = np.mean(low_fwd > 0) * 100
    high_avg = np.mean(high_fwd)
    low_avg = np.mean(low_fwd)

    win_diff = high_win - low_win
    avg_diff = high_avg - low_avg

    results.append({
        "factor": fn,
        "n": len(vals),
        "ic": round(ic, 4),
        "ic_p": round(ic_p, 4),
        "high_n": len(high_fwd),
        "low_n": len(low_fwd),
        "high_win": round(high_win, 1),
        "low_win": round(low_win, 1),
        "win_diff": round(win_diff, 1),
        "high_avg": round(high_avg, 2),
        "low_avg": round(low_avg, 2),
        "avg_diff": round(avg_diff, 2),
    })

# 排序：按 IC 绝对值
results.sort(key=lambda x: -abs(x["ic"]))

print(f"{'='*90}")
print(f"  按 IC 绝对值排序")
print(f"{'='*90}")
print(f"  {'因子':<18} {'数量':>5} {'IC':>7} {'p值':>7} {'高分组胜率':>9} {'低分组胜率':>9} {'胜率差':>7} {'高分组收益':>9} {'收益差':>7}")
print(f"  {'-'*80}")
for r in results:
    sig = "*" if r["ic_p"] < 0.05 else " "
    print(f"  {r['factor']:<18} {r['n']:>5} {r['ic']:>+7.4f}{sig} {r['ic_p']:>7.4f} {r['high_win']:>7.1f}% {r['low_win']:>7.1f}% {r['win_diff']:>+6.1f}% {r['high_avg']:>+7.2f}% {r['avg_diff']:>+6.2f}%")

# 综合评分 IC
comp_vals = [(r["composite"], r["fwd_5d"]) for r in all_data if r["composite"] is not None and r["fwd_5d"] is not None]
if len(comp_vals) > 20:
    cs, cf = zip(*comp_vals)
    ic_c, ic_cp = spearmanr(cs, cf)
    th_high = np.percentile(cs, 70)
    th_low = np.percentile(cs, 30)
    ch = np.mean([cf[i] for i in range(len(cs)) if cs[i] >= th_high])
    cl = np.mean([cf[i] for i in range(len(cs)) if cs[i] <= th_low])
    print(f"\n  {'综合评分':<18} {len(comp_vals):>5} {ic_c:>+7.4f} {ic_cp:>7.4f} {'':>19} {'':>9} {'':>6} {ch:>+7.2f}% {ch-cl:>+6.2f}%")

# 威科夫信号 IC
sig_map = {"SOS": 80, "Spring": 70, "LPS": 65, "EVR": 60, "Compression": 55, "Upthrust": 50, "-": 30}
sig_vals = [(sig_map.get(r["sig"], 30), r["fwd_5d"]) for r in all_data if r["fwd_5d"] is not None]
if len(sig_vals) > 20:
    ss, sf = zip(*sig_vals)
    ic_s, ic_sp = spearmanr(ss, sf)
    print(f"  {'威科夫信号':<18} {len(sig_vals):>5} {ic_s:>+7.4f} {ic_sp:>7.4f}")

print(f"\n  总耗时: {time.time()-t0:.0f}s")
print(f"{'='*70}")
