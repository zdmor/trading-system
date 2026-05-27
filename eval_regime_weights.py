"""
市态权重调整评估：按截面所在市态分组计算 IC/ICIR
================================================
目的：用数据替代拍脑袋的乘数，确定反转市下各因子应分配多少权重。

流程：
  1. 加载503只流动性股票缓存 + 上证指数
  2. 对132个截面，判断当时的大盘市态（基于历史 MA50/200）
  3. 分组计算各因子在趋势市/反转市/中性市的 IC/ICIR
  4. 按 ICIR 比例输出推荐权重
"""
import sys, os, time, json, requests, numpy as np, pandas as pd
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = r"D:\ClaudeWorkspace\trading_system\data_cache\daily"
WINDOW_START = "2023-06-01"
WINDOW_END = "2026-05-27"
MIN_KLINE_DAYS = 120

FACTOR_NAMES = [
    "动量", "RSI", "K线", "量能", "均线", "波动",
    "Δ动量", "Δ量能", "Δ均线",
    "质_量能", "质_形态", "质_动量", "互证",
]

PYTHON = r"C:\Users\sut-b\AppData\Local\Programs\Python\Python312\python.exe"


def spearman_rank(x, y):
    n = len(x)
    if n < 10:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    s = 1.0 - 6.0 * np.sum(d * d) / (n * (n * n - 1))
    return max(-1.0, min(1.0, s))


def get_market_regime_at_date(target_date, kline_data, short_ma=50, long_ma=200):
    """判断指定日期的历史市态（基于当时的MA50/MA200，无前瞻偏差）"""
    idx = kline_data["date"].searchsorted(target_date)
    if idx < long_ma:
        return "neutral"
    closes = kline_data["close"].values[:idx].astype(float)
    n = len(closes)
    if n < short_ma:
        return "neutral"
    ma_short = np.mean(closes[-short_ma:])
    ma_long = np.mean(closes[-long_ma:]) if n >= long_ma else np.mean(closes[:short_ma])
    if ma_short > ma_long * 1.02:
        return "trending"
    elif ma_short < ma_long * 0.98:
        return "reversal"
    return "neutral"


def load_market_data():
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,800,qfq"
    r = requests.get(url, timeout=15)
    data = r.json()
    klines = (data.get("data", {}).get("sh000001", {}).get("qfqday") or
              data.get("data", {}).get("sh000001", {}).get("day", []))
    rows = [{"date": pd.Timestamp(k[0]), "close": float(k[2])} for k in klines]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def load_stock_cache():
    stock_dfs = {}
    for f in os.listdir(CACHE_DIR):
        if not f.endswith(".csv") or f.startswith("ic_results"):
            continue
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f), parse_dates=["date"])
            df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)]
            code = f.replace(".csv", "").replace("_", ".")
            if len(df) >= MIN_KLINE_DAYS:
                stock_dfs[code] = df.reset_index(drop=True)
        except Exception:
            pass
    return stock_dfs


def calc_factors_vectorized(df, idx):
    """直接从 factor_ic_bulk_v4.py 复制的因子计算（简化版，只算评分系统用的6因子+波动）"""
    try:
        sdf = df.iloc[:idx + 1]
        if len(sdf) < 60:
            return None
        closes = sdf["close"].values.astype(np.float64)
        highs = sdf["high"].values.astype(np.float64)
        lows = sdf["low"].values.astype(np.float64)
        volumes = sdf["volume"].values.astype(np.float64)
        opens = sdf["open"].values.astype(np.float64)
        n = len(sdf)
        price = float(closes[-1])

        tr = np.maximum(highs[1:] - lows[1:],
                        np.maximum(np.abs(highs[1:] - closes[:-1]),
                                   np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        diffs = np.diff(closes[-15:]) if n >= 16 else np.diff(closes)
        gains = np.sum(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 0
        losses = -np.sum(diffs[diffs < 0]) if len(diffs[diffs < 0]) > 0 else 0
        rsi = 100 - 100 / (1 + gains / max(losses, 0.001)) if (gains + losses) > 0 else 50

        vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_ma20, 1)
        vol_5 = np.mean(volumes[-5:]) if n >= 5 else volumes[-1]
        vol_prev5 = np.mean(volumes[-10:-5]) if n >= 10 else 1
        vol_trend = vol_5 / max(vol_prev5, 1)

        last_close, last_open = closes[-1], opens[-1]
        last_high, last_low = highs[-1], lows[-1]
        is_up = last_close >= last_open
        body = abs(last_close - last_open)
        upper = last_high - max(last_close, last_open)
        lower = min(last_close, last_open) - last_low

        if n >= 21:
            rets_pct = np.diff(closes[-21:]) / closes[-21:-1] * 100
            vol_20 = np.std(rets_pct) if len(rets_pct) > 1 else 2
        else:
            vol_20 = 2

        ma_bull = 1 if ma20 > ma50 > ma200 else 0
        price_above_ma20 = 1 if price > ma20 else 0

        factors = {}
        factors["动量"] = max(0, min(100, 50 + ret_5d * 3 + ret_20d * 1))
        if rsi > 70: factors["RSI"] = 80
        elif rsi > 60: factors["RSI"] = 70
        elif 40 <= rsi <= 60: factors["RSI"] = 55
        elif rsi >= 30: factors["RSI"] = 40
        else: factors["RSI"] = 25
        factors["均线"] = max(0, min(100,
            50 + (5 if ma_bull else -5) + (10 if price_above_ma20 else -10) +
            min(10, int((ma20 - ma50) / max(ma50, 1) * 50))))
        factors["量能"] = max(0, min(100,
            50 + min(10, int((vol_trend - 1) * 50)) +
            (10 if vol_ratio > 1.2 and is_up else -10 if vol_ratio > 1.2 else 0)))
        factors["K线"] = max(0, min(100, 50 +
            (10 if body > 0 and (lower >= body * 2 or is_up) else -10 if not is_up else 0)))
        factors["波动"] = max(0, min(100, min(100, vol_20 * 5)))
        factors["互证"] = max(0, min(100, 50 +
            (1 if vol_ratio > 1.2 and is_up else -1 if vol_ratio > 1.2 else 0) * 15 +
            (1 if ma_bull and ret_5d > 0 else -1 if not ma_bull and ret_5d < 0 else 0) * 15))
        factors["质_量能"] = max(0, min(100,
            max(0, vol_trend - 1) * 30 + (1 if vol_ratio > 1.2 else 0) * 20))
        factors["质_形态"] = max(0, min(100,
            (1 if lower >= body * 2 else 0) * 30 + (1 if upper <= body * 0.5 else 0) * 20))
        factors["质_动量"] = max(0, min(100,
            (1 if ma_bull else 0) * 30 + (1 if vol_ratio > 1.2 else 0) * 20 +
            (1 if price_above_ma20 else 0) * 15))
        factors["Δ动量"] = 0  # placeholder
        factors["Δ量能"] = 0
        factors["Δ均线"] = 0

        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None
        return factors, fwd_5d
    except Exception:
        return None


def main():
    t0 = time.time()
    print("=" * 60)
    print("  市态分组 IC/ICIR 评估")
    print("=" * 60)

    # 1. 加载大盘
    print("\n[1/4] 加载大盘数据...")
    market_df = load_market_data()
    print(f"  上证指数: {len(market_df)} 个交易日")

    # 2. 加载股票缓存
    print("\n[2/4] 加载股票缓存...")
    stock_dfs = load_stock_cache()
    print(f"  有效股票: {len(stock_dfs)} 只")

    # 3. 构建时间截面
    print("\n[3/4] 构建时间截面...")
    first_df = next(iter(stock_dfs.values()))
    all_dates = pd.to_datetime(first_df["date"])
    window_start = pd.Timestamp(WINDOW_START)
    window_end = pd.Timestamp(WINDOW_END)
    mask = (all_dates >= window_start) & (all_dates <= window_end)
    dates_in_window = all_dates[mask]
    time_points = list(range(0, len(dates_in_window), 5))[1:]
    dates_sorted = [str(d.date()) for d in dates_in_window.iloc[time_points]]
    print(f"  时间截面: {len(dates_sorted)} 个 (间隔5日)")

    # 股票日期索引
    stock_date_idx = {}
    for code, df in stock_dfs.items():
        stock_date_idx[code] = {str(d.date()): i for i, d in enumerate(df["date"])}

    # 4. 滚动IC + 按市态分组
    print("\n[4/4] 滚动IC分析...")
    regime_ics = {"trending": {}, "reversal": {}, "neutral": {}}
    section_count = {"trending": 0, "reversal": 0, "neutral": 0}
    factor_names = FACTOR_NAMES

    for fn in factor_names:
        for r in regime_ics:
            regime_ics[r][fn] = []

    all_comp_ics = {"trending": [], "reversal": [], "neutral": []}

    for ti, tp_date_str in enumerate(dates_sorted):
        tp_date = pd.Timestamp(tp_date_str)
        regime = get_market_regime_at_date(tp_date, market_df)
        section_count[regime] += 1

        all_factor_dicts = []
        all_fwds = []
        for code, df in stock_dfs.items():
            idx = stock_date_idx[code].get(tp_date_str)
            if idx is None or idx < 60:
                continue
            result = calc_factors_vectorized(df, idx)
            if result and result[0] and result[1] is not None:
                factors, fwd = result
                all_factor_dicts.append(factors)
                all_fwds.append(fwd)

        if len(all_factor_dicts) < 30:
            continue

        fwds_arr = np.array(all_fwds, dtype=np.float64)
        for fn in factor_names:
            if fn in ("Δ动量", "Δ量能", "Δ均线"):
                continue
            scores = np.array([f[fn] for f in all_factor_dicts], dtype=np.float64)
            ic = spearman_rank(scores, fwds_arr)
            regime_ics[regime][fn].append(ic)

        comp = np.array([np.mean(list(f.values())) for f in all_factor_dicts], dtype=np.float64)
        comp_ic = spearman_rank(comp, fwds_arr)
        all_comp_ics[regime].append(comp_ic)

        if (ti + 1) % 20 == 0 or ti == len(dates_sorted) - 1:
            print(f"  截面: {ti+1}/{len(dates_sorted)} [{tp_date_str}] "
                  f"regime={regime} n={len(all_factor_dicts)}")

    # 5. 输出
    print(f"\n{'='*70}")
    print(f"  按市态分组的 IC/ICIR")
    print(f"  截面分布: 趋势={section_count['trending']} "
          f"反转={section_count['reversal']} 中性={section_count['neutral']}")
    print(f"{'='*70}")

    score_key_map = {
        "动量": "tech_strength", "RSI": "tech_strength",
        "K线": "candlestick", "量能": "volume",
        "均线": "tech_strength", "波动": "volatility",
        "互证": "mutual_confirmation",
        "质_量能": "volume_quality", "质_形态": "pattern_quality",
        "质_动量": "momentum_quality",
    }

    for regime in ["trending", "reversal", "neutral"]:
        cnt = section_count[regime]
        if cnt < 5:
            print(f"\n  [{regime}] 截面不足({cnt})，跳过")
            continue
        print(f"\n  [{regime.upper()}]  {cnt} 个截面")
        print(f"  {'因子':<10} {'均值IC':>8} {'ICIR':>6} {'胜率':>4}  评分映射")
        print(f"  {'-'*45}")
        rankings = []
        for fn in factor_names:
            if fn in ("Δ动量", "Δ量能", "Δ均线"):
                continue
            ics = regime_ics[regime][fn]
            if len(ics) < 5:
                continue
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / max(std_ic, 0.001)
            win_rate = np.sum(np.array(ics) > 0) / len(ics)
            sk = score_key_map.get(fn, fn)
            rankings.append((fn, mean_ic, icir, win_rate, sk))

        rankings.sort(key=lambda x: -abs(x[2]))
        for fn, mean_ic, icir, wr, sk in rankings:
            print(f"  {fn:<10} {mean_ic:>+8.4f} {icir:>+6.2f} {wr:>4.0%}  ->{sk}")

    # 6. 反转市推荐权重
    print(f"\n{'='*70}")
    print(f"  反转市推荐权重（基于 ICIR 比例分配）")
    print(f"{'='*70}")

    rev_cnt = section_count["reversal"]
    if rev_cnt >= 5:
        rev_data = {}
        for fn in factor_names:
            if fn in ("Δ动量", "Δ量能", "Δ均线"):
                continue
            ics = regime_ics["reversal"][fn]
            if len(ics) < 5:
                continue
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / max(std_ic, 0.001)
            sk = score_key_map.get(fn, fn)
            rev_data[sk] = rev_data.get(sk, []) + [icir]

        # 合并到评分因子名
        merged = {}
        for sk, icirs in rev_data.items():
            abs_vals = [abs(v) for v in icirs if abs(v) > 0.05]
            merged[sk] = np.mean(abs_vals) if abs_vals else 0

        # 按比例分配权重
        total_abs = sum(merged.values())
        if total_abs > 0:
            print(f"  {'评分因子':<20} {'|ICIR|':>6} {'推荐权重':>8}")
            print(f"  {'-'*36}")
            for sk, abs_icir in sorted(merged.items(), key=lambda x: -x[1]):
                w = abs_icir / total_abs
                print(f"  {sk:<20} {abs_icir:>6.2f} {w:>8.1%}")

    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
