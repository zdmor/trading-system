"""
30天全市场卖出信号回测 — Upthrust/EVR 表现验证
"""
import sys, os, time, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from main import Analyzer, Strategy
from scanner import WyckoffAnalyzer, Scanner
from scoring import StockScorer

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"

def get_klines(code):
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            return None
        klines = data.get("data", {}).get(key, {}).get("qfqday") or \
                 data.get("data", {}).get(key, {}).get("day") or []
        if not klines:
            return None
        rows = []
        for k in klines:
            rows.append({
                "date": pd.Timestamp(k[0]),
                "open": float(k[1]), "close": float(k[2]),
                "high": float(k[3]), "low": float(k[4]),
                "volume": float(k[5]),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None

def analyze_at_date(df, code, name, cutoff_date):
    try:
        mask = df["date"] <= pd.Timestamp(cutoff_date)
        sdf = df[mask].copy()
        if len(sdf) < 60:
            return None

        sdf = Analyzer.calc_atr(sdf)
        sdf = Analyzer.calc_ma(sdf, [20, 50, 200])
        supports, resistances = Analyzer.detect_levels(sdf)
        price = float(sdf["close"].iloc[-1])
        levels = {"supports": supports, "resistances": resistances, "current": price}

        strategy = Strategy(sdf, 100000)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()

        trend_dir = trend["direction"]
        wyckoff_sigs = []
        phase_label = ""
        if len(sdf) >= 50 and trend_dir != "未知":
            c = sdf["close"].values.astype(float)
            h = sdf["high"].values.astype(float)
            l = sdf["low"].values.astype(float)
            v = sdf["volume"].values.astype(float)
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, []
            )
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir
            )

        best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")
        wyckoff_score = best_sig[1] if len(best_sig) > 1 else 0

        sell_signals = {"Upthrust", "EVR", "弱Upthrust", "弱Spring"}
        return {
            "code": code, "name": name, "date": cutoff_date,
            "signal": best_sig[0], "score": wyckoff_score,
            "trend": trend_dir, "phase": phase_label, "price": price,
            "is_sell": best_sig[0] in sell_signals,
        }
    except Exception:
        return None

def forward_return(df, analyze_date, days=2):
    mask = df["date"] <= pd.Timestamp(analyze_date)
    idx = len(df[mask])
    if idx + days >= len(df):
        return None
    buy_p = float(df.iloc[idx]["close"])
    sell_p = float(df.iloc[idx + days]["close"])
    return round((sell_p - buy_p) / buy_p * 100, 2)


print("=" * 70)
print("  30天卖出信号回测")
print("=" * 70)

# 1. 候选池
print("  获取候选股票池...")
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
candidates = sc.candidates[:80]
print(f"  候选: {len(candidates)}只")

# 2. 交易日范围（近30个交易日 ≈ 6周）
today = datetime.now()
dates = []
for i in range(45):
    d = today - timedelta(days=i)
    if d.weekday() < 5:
        dates.append(d.strftime("%Y-%m-%d"))
dates = sorted(dates[-31:])  # 30个交易日 + 今天
print(f"  回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)")

# 3. 获取K线
print("  批量获取K线数据...")
stock_data = {}
for i, stk in enumerate(candidates):
    df = get_klines(stk["code"])
    if df is not None and len(df) >= 60:
        stock_data[stk["code"]] = {"df": df, "name": stk["name"]}
    if (i + 1) % 20 == 0:
        print(f"  K线缓存: {i+1}/{len(candidates)}", end="\r")
print(f"  K线缓存: {len(candidates)}/{len(candidates)} — 有效: {len(stock_data)}只")

# 4. 逐日回测
all_sells = []
t0 = time.time()

for di, d in enumerate(dates):
    day_sells = []
    for code, info in stock_data.items():
        result = analyze_at_date(info["df"], code, info["name"], d)
        if result and result["is_sell"]:
            fwd2 = forward_return(info["df"], d, 2)
            fwd5 = forward_return(info["df"], d, 5)
            if fwd2 is not None:
                result["fwd_2d"] = fwd2
                result["fwd_5d"] = fwd5
                day_sells.append(result)
                all_sells.append(result)

    elapsed = time.time() - t0
    print(f"  [{di+1}/{len(dates)}] {d} 卖出信号:{len(day_sells)}只"
          f"  ({elapsed:.0f}s)" if len(day_sells) == 0 else
          f"  [{di+1}/{len(dates)}] {d} 卖出信号:{len(day_sells)}只")

# 5. 分析
total = len(all_sells)
if total == 0:
    print("\n  没有卖出信号")
else:
    sells_upthrust = [r for r in all_sells if r["signal"] == "Upthrust"]
    sells_evr = [r for r in all_sells if r["signal"] == "EVR"]
    sells_weak = [r for r in all_sells if r["signal"] in ("弱Upthrust", "弱Spring")]

    # 胜率（卖出信号 → 下跌算赢）
    win_2d = sum(1 for r in all_sells if r["fwd_2d"] < 0)
    win_5d = sum(1 for r in all_sells if r["fwd_5d"] is not None and r["fwd_5d"] < 0)

    avg_2d = np.mean([r["fwd_2d"] for r in all_sells])
    avg_5d = np.mean([r["fwd_5d"] for r in all_sells if r["fwd_5d"] is not None])

    # 按信号类型分组
    print(f"\n{'='*70}")
    print(f"  卖出信号回测结果（{dates[0]} ~ {dates[-1]}）")
    print(f"{'='*70}")
    print(f"  总卖出信号: {total}次")
    print(f"  Upthrust: {len(sells_upthrust)}次")
    print(f"  EVR: {len(sells_evr)}次")
    print(f"  弱信号: {len(sells_weak)}次")
    print(f"\n  {'─'*50}")
    print(f"  整体表现:")
    print(f"  {'指标':<20} {'值':>12}")
    print(f"  {'-'*32}")
    print(f"  平均2日收益: {avg_2d:>+8.2f}%")
    print(f"  平均5日收益: {avg_5d:>+8.2f}%")
    print(f"  2日胜率(跌): {win_2d/total*100:>6.1f}%")
    print(f"  5日胜率(跌): {win_5d/total*100:>6.1f}%")

    # 按信号类型
    for label, sig_list in [("Upthrust", sells_upthrust), ("EVR", sells_evr)]:
        if not sig_list:
            continue
        w2 = sum(1 for r in sig_list if r["fwd_2d"] < 0)
        a2 = np.mean([r["fwd_2d"] for r in sig_list])
        a5 = np.mean([r["fwd_5d"] for r in sig_list if r["fwd_5d"] is not None])
        print(f"\n  {label}:")
        print(f"    次数: {len(sig_list)} | 平均2日: {a2:+.2f}% | 2日胜率: {w2/len(sig_list)*100:.0f}% | 平均5日: {a5:+.2f}%")

    # 最好/最差
    best = max(all_sells, key=lambda x: x["fwd_5d"] if x["fwd_5d"] is not None else -999)
    worst = min(all_sells, key=lambda x: x["fwd_5d"] if x["fwd_5d"] is not None else 999)
    print(f"\n  最好5日: {best['name']}({best['code']}) {best['signal']} 5日{best['fwd_5d']:+.2f}%")
    print(f"  最差5日: {worst['name']}({worst['code']}) {worst['signal']} 5日{worst['fwd_5d']:+.2f}%")

# 6. 保存
print(f"\n  总耗时: {time.time()-t0:.0f}s")
print(f"{'='*70}")
