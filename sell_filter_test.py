"""
卖出信号过滤器测试 — 测不同过滤条件组合的胜率
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
MARKET_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"

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

def get_market_klines():
    """获取上证指数K线"""
    try:
        r = requests.get(MARKET_URL, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or data.get("data", {}).get("sh000001", {}).get("day") or []
        if not klines: return None
        rows = [{"date": pd.Timestamp(k[0]), "close": float(k[2])} for k in klines]
        df = pd.DataFrame(rows)
        df["ma50"] = df["close"].rolling(50).mean()
        df["ma200"] = df["close"].rolling(200).mean()
        return df
    except: return None

def market_trend(mdf, date):
    """判断大盘在指定日期的趋势"""
    mask = mdf["date"] <= pd.Timestamp(date)
    sdf = mdf[mask]
    if len(sdf) < 50: return "未知"
    c = sdf["close"].iloc[-1]
    ma50 = sdf["ma50"].iloc[-1]
    ma200 = sdf["ma200"].iloc[-1] if not pd.isna(sdf["ma200"].iloc[-1]) else 0
    if c > ma50 > ma200: return "多头"
    if c < ma50 < ma200: return "空头"
    return "震荡"

def calc_20d_return(df, date):
    """计算截止日期前20日涨幅"""
    mask = df["date"] <= pd.Timestamp(date)
    sdf = df[mask]
    if len(sdf) < 21: return 0
    c = sdf["close"].values.astype(float)
    return (c[-1] / c[-21] - 1) * 100

def analyze_sell(df, code, name, date, mdf):
    try:
        mask = df["date"] <= pd.Timestamp(date)
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
        trend_dir = trend["direction"]

        wyckoff_sigs = []
        phase_label = ""
        if len(sdf) >= 50 and trend_dir != "未知":
            c = sdf["close"].values.astype(float)
            h = sdf["high"].values.astype(float)
            l = sdf["low"].values.astype(float)
            v = sdf["volume"].values.astype(float)
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, [])
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir)

        best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")
        if best_sig[0] not in ("Upthrust", "EVR"):
            return None

        # 综合评分
        atr_val = sdf["atr"].iloc[-1] if "atr" in sdf.columns else price * 0.02
        stop_price = round(price - atr_val * 1.5, 2) if atr_val > 0 else round(price * 0.93, 2)
        scorer = StockScorer(df=sdf, price=price, trend=trend, vol=vol, levels=levels,
                             stop_price=stop_price, exit_prices=[r for r in resistances] if resistances else [round(price * 1.08, 2)],
                             wyckoff_signals=wyckoff_sigs, wyckoff_phase=phase_label)
        scoring = scorer.compute()
        composite = scoring["composite_score"]

        # 市场趋势
        m_trend = market_trend(mdf, date)
        ret_20d = calc_20d_return(df, date)

        # 判断行情强度（市场+趋势综合）
        is_strong_market = m_trend == "多头"
        stock_in_uptrend = trend_dir == "多头"

        # 前向收益
        mask2 = df["date"] <= pd.Timestamp(date)
        idx = len(df[mask2])
        fwd2 = float(df.iloc[idx+2]["close"]) - float(df.iloc[idx]["close"]) if idx+2 < len(df) else None
        fwd5 = float(df.iloc[idx+5]["close"]) - float(df.iloc[idx]["close"]) if idx+5 < len(df) else None
        fwd2_pct = round((fwd2 / float(df.iloc[idx]["close"])) * 100, 2) if fwd2 is not None else None
        fwd5_pct = round((fwd5 / float(df.iloc[idx]["close"])) * 100, 2) if fwd5 is not None else None

        return {
            "code": code, "name": name, "date": date,
            "signal": best_sig[0], "score": best_sig[1], "composite": composite,
            "phase": phase_label, "trend": trend_dir,
            "market_trend": m_trend, "ret_20d": ret_20d,
            "is_strong_market": is_strong_market, "stock_in_uptrend": stock_in_uptrend,
            "fwd_2d": fwd2_pct, "fwd_5d": fwd5_pct,
        }
    except:
        return None


print("=" * 70)
print("  卖出信号过滤器测试")
print("=" * 70)

# 1. 候选池
print("  获取数据...")
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
candidates = sc.candidates[:80]
market_df = get_market_klines()

today = datetime.now()
dates = []
for i in range(45):
    d = today - timedelta(days=i)
    if d.weekday() < 5:
        dates.append(d.strftime("%Y-%m-%d"))
dates = sorted(dates[-31:])

# 2. 获取K线
stock_data = {}
for i, stk in enumerate(candidates):
    df = get_klines(stk["code"])
    if df is not None and len(df) >= 60:
        stock_data[stk["code"]] = {"df": df, "name": stk["name"]}
    if (i + 1) % 20 == 0:
        print(f"  K线缓存: {i+1}/{len(candidates)}", end="\r")
print(f"  有效: {len(stock_data)}只 | 交易日: {len(dates)}天")

# 3. 回测 + 收集所有信号
t0 = time.time()
all_signals = []
for di, d in enumerate(dates):
    for code, info in stock_data.items():
        r = analyze_sell(info["df"], code, info["name"], d, market_df)
        if r:
            all_signals.append(r)
    if (di + 1) % 5 == 0:
        print(f"  进度: {di+1}/{len(dates)}天 ({time.time()-t0:.0f}s)")

total = len(all_signals)
print(f"\n  总卖出信号: {total}次")

# 4. 测试各种过滤器组合
def test_filter(name, filter_fn, signals):
    filtered = [r for r in signals if filter_fn(r)]
    if not filtered:
        return {"name": name, "count": 0, "win_2d": 0, "avg_2d": 0, "win_5d": 0, "avg_5d": 0}
    w2 = sum(1 for r in filtered if r["fwd_2d"] is not None and r["fwd_2d"] < 0)
    w5 = sum(1 for r in filtered if r["fwd_5d"] is not None and r["fwd_5d"] < 0)
    a2 = np.mean([r["fwd_2d"] for r in filtered if r["fwd_2d"] is not None])
    a5 = np.mean([r["fwd_5d"] for r in filtered if r["fwd_5d"] is not None])
    n2 = sum(1 for r in filtered if r["fwd_2d"] is not None)
    n5 = sum(1 for r in filtered if r["fwd_5d"] is not None)
    return {
        "name": name, "count": len(filtered),
        "win_2d": round(w2/n2*100, 1) if n2 else 0,
        "avg_2d": round(a2, 2), "n_2d": n2,
        "win_5d": round(w5/n5*100, 1) if n5 else 0,
        "avg_5d": round(a5, 2), "n_5d": n5,
    }

filters = [
    ("全部信号（无过滤）", lambda r: True),
    ("大盘空头/震荡时", lambda r: r["market_trend"] != "多头"),
    ("大盘空头时", lambda r: r["market_trend"] == "空头"),
    ("个股趋势为空头时", lambda r: r["trend"] == "空头"),
    ("个股趋势非多头", lambda r: r["trend"] != "多头"),
    ("综合分<50", lambda r: r["composite"] < 50),
    ("综合分<45", lambda r: r["composite"] < 45),
    ("综合分<40", lambda r: r["composite"] < 40),
    ("大盘非多头+综合分<50", lambda r: r["market_trend"] != "多头" and r["composite"] < 50),
    ("大盘非多头+综合分<45", lambda r: r["market_trend"] != "多头" and r["composite"] < 45),
    ("20日涨幅>10%+大盘非多头", lambda r: r["ret_20d"] > 10 and r["market_trend"] != "多头"),
    ("20日涨幅>20%+大盘非多头", lambda r: r["ret_20d"] > 20 and r["market_trend"] != "多头"),
    ("个股空头+综合分<45", lambda r: r["trend"] == "空头" and r["composite"] < 45),
]

results = []
for name, fn in filters:
    r = test_filter(name, fn, all_signals)
    results.append(r)
    print(f"  {r['name']:<28} 次数:{r['count']:>4}  2日胜率:{r['win_2d']:>5.1f}%  2日平均:{r['avg_2d']:>+7.2f}%"
          f"  5日胜率:{r['win_5d']:>5.1f}%  5日平均:{r['avg_5d']:>+7.2f}%")

print(f"\n{'='*70}")
print("  最佳过滤器（按2日胜率排名）:")
print(f"{'='*70}")
results.sort(key=lambda x: -x["win_2d"])
for r in results[:5]:
    print(f"  {r['name']:<28} 次数:{r['count']:>4}  2日胜率:{r['win_2d']:>5.1f}%  2日收益:{r['avg_2d']:>+7.2f}%")

print(f"\n  总耗时: {time.time()-t0:.0f}s")
print(f"{'='*70}")
