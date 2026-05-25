"""
因子IC快速回测 v2 — 纯手工因子，不依赖main模块
"""
import sys, os, time, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

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

t0 = time.time()
# 载入候选股
from scanner import Scanner
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
codes = [c["code"] for c in sc.candidates[:200]]

stock_dfs = {}
for code in codes:
    df = get_klines(code)
    if df is not None and len(df) > 105:
        stock_dfs[code] = df
print(f"  K线有效: {len(stock_dfs)}只")

# 手工因子计算：5个纯K线因子，不依赖任何模块
def calc_factors(df):
    """纯Pandas计算，不依赖任何Analyzerr"""
    try:
        cutoff_date = df["date"].iloc[-11]
        mask = df["date"] <= cutoff_date
        sdf = df[mask].copy()
        if len(sdf) < 60: return None, None

        price = float(sdf["close"].iloc[-1])
        closes = sdf["close"].values.astype(float)
        highs = sdf["high"].values.astype(float)
        lows = sdf["low"].values.astype(float)
        volumes = sdf["volume"].values.astype(float)

        n = len(sdf)

        # ATR
        trs = []
        for i in range(1, n):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        atr = np.mean(trs[-14:]) if len(trs) >= 14 else np.mean(trs)

        # MA
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        # 收益率
        ret_1d = (closes[-1] / closes[-2] - 1) * 100
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        # RSI(14)
        gains, losses = 0, 0
        for i in range(1, min(15, n)):
            d = closes[-i] - closes[-i-1]
            if d > 0: gains += d
            else: losses -= d
        rsi = 50
        if gains + losses > 0:
            rsi = 100 - 100 / (1 + gains / max(losses, 0.001))

        # 量比
        vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_ma20, 1)

        # 量趋势
        vol_5 = np.mean(volumes[-5:]) if n >= 5 else volumes[-1]
        vol_prev5 = np.mean(volumes[-10:-5]) if n >= 10 else 1
        vol_trend = vol_5 / max(vol_prev5, 1)

        # K线形态
        last = sdf.iloc[-1]
        is_up = last["close"] >= last["open"]
        body = abs(last["close"] - last["open"])
        upper = last["high"] - max(last["close"], last["open"])
        lower = min(last["close"], last["open"]) - last["low"]

        # 波动率（20日）
        rets_20d = [(closes[i]/closes[i-1]-1)*100 for i in range(max(0,n-20), n)]
        vol_20 = np.std(rets_20d) if len(rets_20d) > 1 else 2

        # 多空判断
        ma_bull = 1 if ma20 > ma50 > ma200 else 0
        price_above_ma20 = 1 if price > ma20 else 0

        # ＝＝ 因子（0-100归一化） ＝＝
        factors = {}

        # 1. 趋势动量
        mom = 50 + ret_5d * 3 + ret_20d * 1
        factors["动量"] = max(0, min(100, mom))

        # 2. RSI评分（动量方向，高RSI=强势延续）
        if rsi > 70:
            rsi_s = 80
        elif rsi > 60:
            rsi_s = 70
        elif 40 <= rsi <= 60:
            rsi_s = 55
        elif rsi >= 30:
            rsi_s = 40
        else:
            rsi_s = 25
        factors["RSI"] = rsi_s

        # 3. K线形态
        kscore = 50
        if is_up and body > 0:
            kscore = 65
            if upper < body * 0.3: kscore += 10
            if lower > body: kscore += 5
        else:
            kscore = 35
            if lower > body * 2: kscore = 55  # 长下影锤子线
        if body < atr * 0.3: kscore = 40  # 十字星
        factors["K线"] = max(0, min(100, kscore))

        # 4. 量能
        vscore = 50
        if vol_ratio > 2: vscore = 80
        elif vol_ratio > 1.5: vscore = 65
        elif vol_ratio < 0.5: vscore = 30
        if vol_trend > 1.2: vscore += 10
        elif vol_trend < 0.7: vscore -= 10
        factors["量能"] = max(0, min(100, vscore))

        # 5. MA排列
        mscore = 80 if ma_bull else 30
        if price_above_ma20: mscore += 10
        factors["均线"] = max(0, min(100, mscore))

        # 6. 波动率（ATR% 适中波动加分）
        atr_pct = atr / price * 100
        if atr_pct < 1.0:
            vol_score = 30   # 波幅太低，死水
        elif atr_pct < 2.5:
            vol_score = 75   # 适中波动，有空间
        elif atr_pct < 4.5:
            vol_score = 60   # 偏高波动
        else:
            vol_score = 35   # 过高波动，风险大
        factors["波动"] = max(0, min(100, vol_score))

        # 7. 前向5日收益
        idx = len(df[mask])
        fwd_5d = (df.iloc[idx+5]["close"] / price - 1) * 100 if idx + 5 < len(df) else None

        return factors, fwd_5d
    except Exception as e:
        return None, None

t1 = time.time()
all_factors = []
all_fwd = []
for i, (code, df) in enumerate(stock_dfs.items()):
    factors, fwd = calc_factors(df)
    if factors and fwd is not None:
        all_factors.append(factors)
        all_fwd.append(fwd)
    if (i+1) % 50 == 0:
        print(f"  因子计算: {i+1}/{len(stock_dfs)}", end="\r")
print(f"  有效数据: {len(all_factors)}只 ({time.time()-t1:.0f}s)")

if not all_factors:
    print("  没有有效因子数据")
    sys.exit(1)

def spearman_rank(x, y):
    n = len(x)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    rho = 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))
    return rho

# 逐个因子
factor_names = list(all_factors[0].keys())
results = []
for fn in factor_names:
    scores = np.array([f[fn] for f in all_factors])
    fwds = np.array(all_fwd)
    ic = spearman_rank(scores, fwds)

    th_high = np.percentile(scores, 70)
    th_low = np.percentile(scores, 30)
    high_fwd = fwds[scores >= th_high]
    low_fwd = fwds[scores <= th_low]

    results.append({
        "factor": fn, "n": len(scores),
        "ic": round(ic, 4),
        "high_win": round(np.mean(high_fwd > 0) * 100, 1),
        "low_win": round(np.mean(low_fwd > 0) * 100, 1),
        "high_avg": round(np.mean(high_fwd), 2),
        "low_avg": round(np.mean(low_fwd), 2),
    })

# 综合评分IC
comp = np.array([np.mean(list(f.values())) for f in all_factors])
comp_ic = spearman_rank(comp, all_fwd)
th = np.percentile(comp, 70)
ch = np.mean([all_fwd[i] for i in range(len(all_fwd)) if comp[i] >= th])
cl = np.mean([all_fwd[i] for i in range(len(all_fwd)) if comp[i] <= np.percentile(comp, 30)])

results.sort(key=lambda x: -abs(x["ic"]))

print(f"\n{'='*95}")
print(f"  因子IC回测 (n={len(all_factors)})")
print(f"{'='*95}")
print(f"  {'因子':<8} {'IC':>7} {'高胜率':>7} {'低胜率':>7} {'胜率差':>7} {'高收益':>7} {'低收益':>7}")
print(f"  {'-'*55}")
for r in results:
    print(f"  {r['factor']:<8} {r['ic']:>+7.4f} {r['high_win']:>6.1f}% {r['low_win']:>6.1f}% {r['high_win']-r['low_win']:>+6.1f}% {r['high_avg']:>+6.2f}% {r['low_avg']:>+6.2f}%")

print(f"\n  综合平均分: IC={comp_ic:+.4f}  高分收益={ch:+.2f}%  低分收益={cl:+.2f}%")
print(f"\n  总耗时: {time.time()-t0:.0f}s")
