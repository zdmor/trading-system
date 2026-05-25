"""
P1-8: 全因子IC回测（新6因子结构 + 按市态分组）
适配重构后的因子: tech_strength, risk_reward, volume, candlestick, sector, relative_strength

用法: python p1_8_ic_backtest.py --top 200
"""
import sys, os, time, argparse
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"
INDEX_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"


def get_klines(code):
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = (data.get("data", {}).get(key, {}).get("qfqday") or
                  data.get("data", {}).get(key, {}).get("day") or [])
        if not klines:
            return None
        rows = [{"date": pd.Timestamp(k[0]), "open": float(k[1]), "close": float(k[2]),
                 "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])} for k in klines]
        return pd.DataFrame(rows)
    except:
        return None


def spearman_rank(x, y):
    n = len(x)
    if n < 3:
        return 0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    return 1 - 6 * np.sum(d**2) / (n * (n**2 - 1))


def calc_new_factors(df, idx, index_df=None, idx_map=None):
    """计算新6因子结构（与scoring.py保持一致）"""
    try:
        sdf = df.iloc[:idx+1]
        n = len(sdf)
        if n < 60:
            return None, None, None

        price = float(sdf["close"].iloc[-1])
        closes = sdf["close"].values.astype(float)
        highs = sdf["high"].values.astype(float)
        lows = sdf["low"].values.astype(float)
        volumes = sdf["volume"].values.astype(float)

        # MA
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        # ATR
        trs = []
        for i in range(1, n):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        atr = np.mean(trs[-14:]) if trs else 0.01

        # RSI
        gains, losses = 0, 0
        for i in range(1, min(15, n)):
            d = closes[-i] - closes[-i-1]
            gains += max(d, 0); losses += max(-d, 0)
        rsi = 50 if gains+losses==0 else 100-100/(1+gains/max(losses,0.001))

        # 量比
        vol_ratio = volumes[-1] / max(np.mean(volumes[-20:]), 1) if n>=20 else 1
        vol_trend = np.mean(volumes[-5:]) / max(np.mean(volumes[-10:-5]), 1) if n>=10 else 1

        # 支撑阻力
        recent_high = np.max(highs[-60:]) if n>=60 else price*1.1
        recent_low = np.min(lows[-60:]) if n>=60 else price*0.9

        factors = {}

        # ─── 1. tech_strength (威科夫+动量合并) ───
        # Wyckoff 代理
        wyckoff = 50
        if ma50 > ma200 and price > ma50:
            wyckoff += 15
        elif price > ma20:
            wyckoff += 8
        if price > recent_low*1.02 and price < recent_low*1.05 and vol_ratio > 1.3:
            wyckoff += 12
        if price > ma50*1.02 and vol_ratio > 1.8 and closes[-1] > closes[-2]:
            wyckoff += 10
        if vol_ratio < 0.7 and abs(closes[-1]-closes[-2]) < atr*0.5:
            wyckoff += 5
        wyckoff = max(0, min(100, wyckoff))

        # 动量
        momentum = 50
        if price > ma20 > ma50 > ma200: momentum += 25
        elif price > ma50 > ma200: momentum += 15
        elif price > ma20: momentum += 5
        elif price < ma20 < ma50: momentum -= 10
        if rsi > 70: momentum += 15
        elif rsi > 60: momentum += 10
        elif rsi > 50: momentum += 5
        elif rsi < 30: momentum -= 10
        atr_pct = atr/price*100
        if 1.5 <= atr_pct <= 4.0: momentum += 10
        momentum = max(0, min(100, momentum))

        factors["tech_strength"] = int((wyckoff + momentum) / 2)

        # ─── 2. risk_reward ───
        stop_price = recent_low * 0.97
        risk = price - stop_price
        reward = recent_high - price
        rr = reward / max(risk, 0.01)
        if rr >= 3: rr_score = 90
        elif rr >= 2: rr_score = 75
        elif rr >= 1.5: rr_score = 60
        elif rr >= 1: rr_score = 45
        else: rr_score = 25
        factors["risk_reward"] = rr_score

        # ─── 3. volume (量比动量) ───
        vscore = 50
        if vol_ratio > 2.5: vscore = 85
        elif vol_ratio > 2.0: vscore = 75
        elif vol_ratio > 1.5: vscore = 60
        elif vol_ratio > 1.0: vscore = 50
        elif vol_ratio > 0.5: vscore = 35
        else: vscore = 20
        if vol_trend > 1.15: vscore += 10
        elif vol_trend < 0.85: vscore -= 10
        factors["volume"] = max(0, min(100, vscore))

        # ─── 4. candlestick ───
        last = sdf.iloc[-1]
        is_up = last["close"] >= last["open"]
        body = abs(last["close"]-last["open"])
        upper = last["high"]-max(last["close"],last["open"])
        lower = min(last["close"],last["open"])-last["low"]
        kscore = 50
        if is_up and body > atr*0.3:
            kscore = 65
            if lower > body*1.5: kscore += 10
        else:
            kscore = 40
            if lower > body*2: kscore = 55
        factors["candlestick"] = max(0, min(100, kscore))

        # ─── 5. sector (行业排名代理) ───
        factors["sector"] = 50

        # ─── 6. relative_strength ───
        rs = 50
        if index_df is not None and idx_map is not None:
            tp_date = df["date"].iloc[idx]
            idx_i = idx_map.get(str(tp_date.date()))
            if idx_i is not None and idx_i >= 20:
                i_closes = index_df["close"].values.astype(float)
                s_5d = (closes[-1]/closes[-6]-1)*100 if n>=6 else 0
                i_5d = (i_closes[idx_i]/i_closes[idx_i-5]-1)*100 if idx_i>=5 else 0
                s_20d = (closes[-1]/closes[-21]-1)*100 if n>=21 else 0
                i_20d = (i_closes[idx_i]/i_closes[idx_i-20]-1)*100 if idx_i>=20 else 0
                d5, d20 = s_5d-i_5d, s_20d-i_20d
                def ds(d):
                    if d>5: return 15
                    if d>2: return 8
                    if d>0.5: return 3
                    if d>-0.5: return 0
                    if d>-2: return -3
                    if d>-5: return -8
                    return -15
                rs = max(0, min(100, 50 + ds(d5)*0.5 + ds(d20)*0.5))
        factors["relative_strength"] = rs

        # 前向收益
        fwd_5d = (df.iloc[idx+5]["close"] / price - 1) * 100 if idx+5 < len(df) else None

        # 市态判定: 大盘MA排列
        regime = "neutral"
        if index_df is not None and idx_map is not None:
            tp_date = df["date"].iloc[idx]
            idx_i = idx_map.get(str(tp_date.date()))
            if idx_i is not None and idx_i >= 200:
                i_closes = index_df["close"].values.astype(float)[:idx_i+1]
                i_ma20 = np.mean(i_closes[-20:])
                i_ma50 = np.mean(i_closes[-50:])
                i_ma200 = np.mean(i_closes[-200:])
                i_cur = i_closes[-1]
                if i_cur > i_ma50 > i_ma200: regime = "bull"
                elif i_cur < i_ma50 < i_ma200: regime = "bear"

        return factors, fwd_5d, regime

    except Exception:
        return None, None, None


def run_ic_backtest(top_n=200):
    """全因子IC回测 + 按市态分组"""
    print("=" * 90)
    print("  P1-8: 全因子IC回测 (新6因子 + 市态分组)")
    print("=" * 90)

    t0 = time.time()

    # 获取候选股
    from scanner import Scanner
    sc = Scanner()
    sc.fetch_all_stocks()
    sc.filter_candidates()
    codes = [c["code"] for c in sc.candidates[:top_n]]
    print(f"  候选股: {len(codes)}只")

    # K线
    stock_dfs = {}
    for code in codes:
        df = get_klines(code)
        if df is not None and len(df) > 160:
            stock_dfs[code] = df
    print(f"  K线有效: {len(stock_dfs)}只 ({time.time()-t0:.0f}s)")

    # 大盘
    index_df = get_klines("sh000001")
    idx_map = {str(d.date()): i for i, d in enumerate(index_df["date"])} if index_df is not None else None

    # 时间截面
    all_dates = None
    for df in stock_dfs.values():
        dt = set(df["date"].values)
        all_dates = dt if all_dates is None else all_dates & dt
    sorted_dates = sorted(pd.DatetimeIndex(all_dates))
    usable_start = 120
    usable_end = len(sorted_dates) - 5
    time_points = list(range(usable_start, usable_end, 5))

    date_to_idx = {}
    for code, df in stock_dfs.items():
        date_to_idx[code] = {str(d.date()): i for i, d in enumerate(df["date"])}

    factor_keys = ["tech_strength", "risk_reward", "volume", "candlestick", "sector", "relative_strength"]
    factor_labels = ["技术强势", "盈亏比", "量能", "K线形态", "板块强度", "相对强度"]
    all_ics = {k: [] for k in factor_keys}
    regime_ics = {"bull": {k: [] for k in factor_keys},
                  "neutral": {k: [] for k in factor_keys},
                  "bear": {k: [] for k in factor_keys}}
    all_comp_ics = []
    regime_counts = {"bull": 0, "neutral": 0, "bear": 0}

    total_tp = len(time_points)
    for ti, tp_idx in enumerate(time_points):
        tp_date = sorted_dates[tp_idx]
        all_factors = {k: [] for k in factor_keys}
        all_fwds = []
        all_regimes = []

        for code, df in stock_dfs.items():
            idx = date_to_idx[code].get(str(tp_date.date()))
            if idx is None or idx < 60:
                continue
            factors, fwd, regime = calc_new_factors(df, idx, index_df, idx_map)
            if factors and fwd is not None:
                for k in factor_keys:
                    all_factors[k].append(factors[k])
                all_fwds.append(fwd)
                all_regimes.append(regime)

        if len(all_fwds) < 20:
            continue

        # 综合评分（等权）
        comp = np.array([np.mean([all_factors[k][i] for k in factor_keys]) for i in range(len(all_fwds))])
        comp_ic = spearman_rank(comp, np.array(all_fwds))
        all_comp_ics.append(comp_ic)

        # 单因子IC + 按市态分组
        for k in factor_keys:
            scores = np.array(all_factors[k])
            ic = spearman_rank(scores, np.array(all_fwds))
            all_ics[k].append(ic)

            for ri, r in enumerate(all_regimes):
                if ri < len(scores) and ri < len(all_fwds):
                    regime_ics[r][k].append((scores[ri], all_fwds[ri]))

        regime_counts["bull"] += all_regimes.count("bull")
        regime_counts["neutral"] += all_regimes.count("neutral")
        regime_counts["bear"] += all_regimes.count("bear")

        if (ti+1) % 10 == 0:
            print(f"  截面: {ti+1}/{total_tp} [{tp_date.date()}] n={len(all_fwds)}", end="\r")
    print(f"  截面: {total_tp}/{total_tp}  完成 ({time.time()-t0:.0f}s)")

    # ============ 结果汇总 ============
    print(f"\n{'='*90}")
    print(f"  全因子IC回测结果 — {len(all_comp_ics)}个截面")
    print(f"{'='*90}")
    print(f"  {'因子':<14} {'均值IC':>8} {'标准差':>8} {'ICIR':>8} {'胜率':>8} {'贡献':>8}")
    print(f"  {'-'*70}")

    results = []
    for k, lbl in zip(factor_keys, factor_labels):
        ics = np.array(all_ics[k])
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / max(std_ic, 0.001)
        win_rate = np.mean(ics > 0) * 100
        results.append({"factor": lbl, "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "win_rate": win_rate})

    results.sort(key=lambda x: -x["mean_ic"])
    for r in results:
        print(f"  {r['factor']:<14} {r['mean_ic']:>+8.4f} {r['std_ic']:>8.4f} {r['icir']:>+8.2f} {r['win_rate']:>7.1f}% {r['mean_ic']*100:>+7.1f}bp")

    # 综合
    comp_ics = np.array(all_comp_ics)
    print(f"  {'─'*70}")
    print(f"  {'综合(等权)':<14} {np.mean(comp_ics):>+8.4f} {np.std(comp_ics):>8.4f} {np.mean(comp_ics)/max(np.std(comp_ics),0.001):>+8.2f} {np.mean(comp_ics>0)*100:>7.1f}%")

    # ─── IC缓存桥接 ───
    try:
        from factor_weights import update_ic_cache
        ic_data = {}
        for k in factor_keys:
            ics = np.array(all_ics[k])
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            icir = mean_ic / max(std_ic, 0.001)
            win_rate = float(np.mean(ics > 0))
            ic_data[k] = {"ic": mean_ic, "icir": icir, "win_rate": win_rate}
        update_ic_cache(ic_data)
        print(f"  IC缓存已写入 factor_ic_cache.json")
    except Exception as e:
        print(f"  IC缓存写入跳过: {e}")

    # ============ 按市态分组 ============
    print(f"\n{'='*90}")
    print(f"  按市态分组 IC — Bull/Neutral/Bear")
    print(f"{'='*90}")
    for regime, label in [("bull","牛市"), ("neutral","震荡"), ("bear","熊市")]:
        print(f"\n  [{label}]")
        for k, lbl in zip(factor_keys, factor_labels):
            points = regime_ics[regime][k]
            if len(points) < 5:
                continue
            scores_arr = np.array([p[0] for p in points])
            fwds_arr = np.array([p[1] for p in points])
            ic = spearman_rank(scores_arr, fwds_arr)
            print(f"    {lbl:<14} IC={ic:>+7.4f}  n={len(points)}")

    print(f"\n  总耗时: {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=150)
    args = parser.parse_args()
    run_ic_backtest(top_n=args.top)
