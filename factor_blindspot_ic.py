"""
盲区因子 IC 验证 — risk_reward / relative_strength / wyckoff
===============================================================
这3个因子合计占评分系统52%权重，从未被IC框架验证过。
原因：需要完整的分析管道（支撑阻力/形态识别），无法用简化OHLCV计算。

本工具通过以下方式解决：
  - risk_reward: 用20日高低点作为支撑/阻力的近似
  - relative_strength: 用腾讯指数K线计算相对收益
  - wyckoff: 直接用 WyckoffAnalyzer (scanner.py) 的analyze_all/detect_phase

用法: python factor_blindspot_ic.py --stocks 500
"""
import sys, os, time, json, argparse
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from scanner import WyckoffAnalyzer

CACHE_DIR = r"D:\ClaudeWorkspace\trading_system\data_cache\daily"
FACTOR_NAMES = ["risk_reward", "relative_strength", "wyckoff_signal", "wyckoff_phase"]


def get_market_index():
    """获取上证指数日线"""
    import requests
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,800,qfq"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or \
                 data.get("data", {}).get("sh000001", {}).get("day", [])
        if klines and len(klines) >= 100:
            rows = [{"date": pd.Timestamp(k[0]), "close": float(k[2])} for k in klines]
            return pd.DataFrame(rows).reset_index(drop=True)
    except:
        pass
    return None


def calc_blindspot_factors(df, idx, mkt_df=None):
    """
    在 df 的 idx 位置计算3个盲区因子 + 前向收益。

    Returns:
      (factors_dict, fwd_5d) or (None, None)
    """
    try:
        sdf = df.iloc[:idx + 1]
        if len(sdf) < 60:
            return None, None

        closes = sdf["close"].values.astype(np.float64)
        highs = sdf["high"].values.astype(np.float64)
        lows = sdf["low"].values.astype(np.float64)
        volumes = sdf["volume"].values.astype(np.float64)
        opens = sdf["open"].values.astype(np.float64)
        n = len(sdf)
        price = float(closes[-1])

        # MA
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]
        ma_bull = ma20 > ma50 > ma200

        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        # ATR(14)
        tr = np.maximum(highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        factors = {}
        details = {}

        # ============================================================
        # 因子1: risk_reward (简版)
        # 支撑 = 最近20日最低点(缩量), 阻力 = 最近20日最高点(放量)
        # ============================================================
        lookback = min(20, n)
        recent_high = np.max(highs[-lookback:])
        recent_low = np.min(lows[-lookback:])

        # 阻力 = max(最近20日最高点, 最近阻力位)
        resistance = recent_high
        # 支撑 = 最近20日最低点
        support = recent_low

        if support >= price:
            # 支撑在现价之上 → 无有效支撑
            rr_score = 10
            rr_ratio = 0
        else:
            risk = price - support
            reward = resistance - price
            if risk <= 0:
                rr_score = 10
                rr_ratio = 0
            else:
                rr_ratio = reward / max(risk, 0.001)
                # 与 scoring.py 保持一致的映射
                if rr_ratio >= 3.0:       base = 95
                elif rr_ratio >= 2.0:     base = 80
                elif rr_ratio >= 1.5:     base = 60
                elif rr_ratio >= 1.0:     base = 40
                else:                     base = 20
                # 空头趋势打折
                discount = 0.7 if not ma_bull else 1.0
                rr_score = max(5, min(100, int(base * discount)))

        factors["risk_reward"] = rr_score
        details["rr_ratio"] = round(rr_ratio, 2)

        # ============================================================
        # 因子2: relative_strength (简版)
        # 个股 vs 大盘的多周期相对收益
        # ============================================================
        if mkt_df is not None and len(mkt_df) >= 21:
            # 找到与当前个股日期对齐的指数数据
            current_date = df["date"].iloc[idx]
            mkt_mask = mkt_df["date"] <= current_date
            mkt_slice = mkt_df[mkt_mask]
            if len(mkt_slice) >= 21:
                mkt_closes = mkt_slice["close"].values.astype(np.float64)
                mkt_current = mkt_closes[-1]

                def rel_ret(s_arr, m_arr, n_days):
                    if len(s_arr) < n_days + 1 or len(m_arr) < n_days + 1:
                        return 0
                    s_ret = (s_arr[-1] / s_arr[-(n_days+1)] - 1) * 100
                    m_ret = (m_arr[-1] / m_arr[-(n_days+1)] - 1) * 100
                    return s_ret - m_ret

                r1 = rel_ret(closes, mkt_closes, 1)
                r5 = rel_ret(closes, mkt_closes, 5)
                r20 = rel_ret(closes, mkt_closes, 20)

                def diff_score(diff):
                    if diff > 5: return 15
                    if diff > 2: return 8
                    if diff > 0.5: return 3
                    if diff > -0.5: return 0
                    if diff > -2: return -3
                    if diff > -5: return -8
                    return -15

                # 与 scoring.py 保持一致的加权
                total = diff_score(r1) * 0.5 + diff_score(r5) * 0.3 + diff_score(r20) * 0.2
                rs_score = max(0, min(100, 50 + total))
            else:
                rs_score = 50
        else:
            rs_score = 50

        factors["relative_strength"] = rs_score

        # ============================================================
        # 因子3: wyckoff — 直接用 WyckoffAnalyzer
        # ============================================================
        # 趋势方向
        trend_dir = "多头" if ma_bull else ("空头" if ma20 < ma50 < ma200 else "震荡")

        wyckoff_score = 50
        wyckoff_signal_raw = "-"
        wyckoff_signal_name = "无信号"
        phase_label = ""

        try:
            # detect phase
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(
                closes.tolist(), highs.tolist(), lows.tolist(),
                volumes.tolist(), trend_dir, [])
        except Exception:
            phase_label = ""

        try:
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
                closes.tolist(), highs.tolist(), lows.tolist(),
                opens.tolist(), volumes.tolist())
        except Exception:
            wyckoff_sigs = []

        # 信号 → 分数映射
        if wyckoff_sigs:
            best_sig = wyckoff_sigs[0]
            wyckoff_signal_raw = best_sig[0]
            wyckoff_signal_name = best_sig[0]

            type_map = {"SOS": 90, "Spring": 85, "LPS": 75,
                        "Compression": 65, "Markup": 80,
                        "Upthrust": 30, "EVR": 40}
            is_weak = best_sig[0].startswith("弱")
            clean_sig = best_sig[0].replace("弱", "")
            base = type_map.get(clean_sig, 50)
            if is_weak:
                base -= 15
            # 阶段加成
            if "Markup" in phase_label or "Phase D" in phase_label:
                base += 5
            elif "Phase B" in phase_label or "Phase C" in phase_label:
                base += 3
            wyckoff_score = max(0, min(100, base))
        else:
            # 无信号时，根据阶段给基础分
            if "Phase B" in phase_label:
                wyckoff_score = 55
            elif "Phase D" in phase_label or "Markup" in phase_label:
                wyckoff_score = 60
            elif "派发" in phase_label or "Phase E" in phase_label:
                wyckoff_score = 35
            else:
                wyckoff_score = 50

        factors["wyckoff_signal"] = wyckoff_score

        # wyckoff_phase 作为分类因子：编码为分数
        phase_score = 50
        if "吸筹" in phase_label or "Accumulation" in phase_label:
            phase_score = 75
        elif "Markup" in phase_label or "主升" in phase_label:
            phase_score = 85
        elif "派发" in phase_label or "Distribution" in phase_label:
            phase_score = 25
        elif "下跌" in phase_label or "Markdown" in phase_label:
            phase_score = 30
        elif "震荡" in phase_label or "Reaccumulation" in phase_label:
            phase_score = 65
        factors["wyckoff_phase"] = phase_score

        # 前向5日收益
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None

        return factors, fwd_5d

    except Exception:
        return None, None


# ============================================================
# 加载缓存 + 时间截面
# ============================================================
def load_data(top_n=500):
    """从缓存加载数据并构造时间截面"""
    from scanner import Scanner

    # 1. 候选股
    s = Scanner()
    s.fetch_all_stocks()
    candidates = s.filter_candidates()
    codes = []
    seen = set()
    for c in candidates:
        code = c["code"]
        name = c.get("name", "")
        sym = code.split(".")[-1]
        if sym.startswith(("8", "4", "9")) or "ST" in name or code in seen:
            continue
        seen.add(code)
        codes.append(code)
        if len(codes) >= top_n:
            break
    print(f"  候选股: {len(codes)}只")

    # 2. 加载缓存
    stock_dfs = {}
    csv_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    loaded = 0
    for pf in csv_files:
        try:
            code = pf.replace(".csv", "").replace("_", ".")
            if code not in codes:
                continue
            path = os.path.join(CACHE_DIR, pf)
            df = pd.read_csv(path, parse_dates=["date"])
            df = df[(df["date"] >= "2023-06-01") & (df["date"] <= "2026-05-27")]
            if len(df) >= 120:
                stock_dfs[code] = df.reset_index(drop=True)
                loaded += 1
        except Exception:
            pass
    print(f"  有效: {loaded}只")

    if loaded < 10:
        return None, None, None, None, None

    # 3. 时间轴
    first_df = next(iter(stock_dfs.values()))
    all_ts = first_df["date"].values
    mask = (all_ts >= np.datetime64("2023-06-01")) & (all_ts <= np.datetime64("2026-05-27"))
    sorted_dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in all_ts[mask]]
    usable_start, usable_end = 60, len(sorted_dates) - 5
    time_points = list(range(usable_start, usable_end, 5))

    stock_date_idx = {}
    for code, df in stock_dfs.items():
        stock_date_idx[code] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df["date"])}

    print(f"  时间截面: {len(time_points)}个 ({sorted_dates[usable_start]} -> {sorted_dates[usable_end-1]})")

    return stock_dfs, sorted_dates, stock_date_idx, time_points, codes


# ============================================================
# IC 分析
# ============================================================
def spearman_rho(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    d = rx - ry
    return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))


def ic_trend_analysis(ics):
    arr = np.array(ics, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 10:
        return {"slope": 0, "recent_mean": 0, "early_mean": 0, "change": 0, "alert": "insufficient", "stability": 1.0}
    x = np.arange(n, dtype=np.float64)
    slope = np.polyfit(x, arr, 1)[0] * n
    split = n // 2
    early_mean = float(np.mean(arr[:split]))
    recent_mean = float(np.mean(arr[split:]))
    change = recent_mean - early_mean
    window = min(20, n // 3)
    if n >= window * 2:
        rolling_std = np.array([np.std(arr[max(0, i-window):i+1]) for i in range(n)])
        recent_vol = float(np.mean(rolling_std[-window:]))
        early_vol = float(np.mean(rolling_std[:window]))
        stability = early_vol / max(recent_vol, 0.001)
    else:
        stability = 1.0
    if slope < 0 and stability < 0.8:
        alert = "decay"
    elif slope < 0 or stability < 0.7:
        alert = "watch"
    else:
        alert = "normal"
    return {"slope": round(slope, 4), "recent_mean": round(recent_mean, 4), "early_mean": round(early_mean, 4),
            "change": round(change, 4), "alert": alert, "stability": round(stability, 2)}


def run_blindspot_ic(top_n=500):
    """主流程"""
    t0 = time.time()

    print("=" * 60)
    print("  盲区因子 IC 验证")
    print(f"  因子: {', '.join(FACTOR_NAMES)}")
    print("=" * 60)

    # 加载数据
    print("\n[1/3] 加载数据...")
    stock_dfs, sorted_dates, stock_date_idx, time_points, codes = load_data(top_n)
    if stock_dfs is None:
        print("  数据不足")
        return

    # 获取市场指数
    print("\n[2/3] 获取市场指数...")
    mkt_df = get_market_index()
    print(f"  上证指数: {len(mkt_df) if mkt_df is not None else '无'}天数据")

    # 滚动IC
    print(f"\n[3/3] 滚动IC分析...")
    all_ics = {fn: [] for fn in FACTOR_NAMES}
    total_tp = len(time_points)
    valid_tp = 0

    for ti, tp_idx in enumerate(time_points):
        tp_date = sorted_dates[tp_idx]
        all_factors = []
        all_fwds = []

        for code in codes:
            df = stock_dfs.get(code)
            if df is None:
                continue
            idx = stock_date_idx[code].get(tp_date)
            if idx is None or idx < 60:
                continue
            factors, fwd = calc_blindspot_factors(df, idx, mkt_df)
            if factors and fwd is not None:
                all_factors.append(factors)
                all_fwds.append(fwd)

        if len(all_factors) < 30:
            continue

        valid_tp += 1
        fwds_arr = np.array(all_fwds, dtype=np.float64)

        for fn in FACTOR_NAMES:
            scores = np.array([f[fn] for f in all_factors], dtype=np.float64)
            ic = spearman_rho(scores, fwds_arr)
            if not np.isnan(ic):
                all_ics[fn].append(ic)

        if (ti + 1) % 20 == 0 or ti == total_tp - 1:
            print(f"  截面: {ti+1}/{total_tp} [{tp_date}] n={len(all_factors)}", end="\r")

    print(f"\n  有效截面: {valid_tp}个")

    if valid_tp < 10:
        print("  截面不足，结果不可靠")
        return

    # ============================================================
    # 输出
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  盲区因子 IC 结果 — {valid_tp}个截面")
    print(f"{'='*70}")
    print(f"  {'因子':<18} {'均值IC':>8} {'标准差':>8} {'ICIR':>8} {'胜率':>6} {'趋势':>8}")
    print(f"  {'-'*60}")

    trends = {}
    for fn in FACTOR_NAMES:
        ics = np.array(all_ics[fn])
        if len(ics) < 5:
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / max(std_ic, 0.001)
        win_rate = float(np.mean(ics > 0)) * 100
        ta = ic_trend_analysis(ics)
        trends[fn] = ta
        sym = {"normal": "OK", "watch": "..", "decay": "!!", "insufficient": "--"}.get(ta["alert"], "??")
        print(f"  {fn:<18} {mean_ic:>+8.4f} {std_ic:>8.4f} {icir:>+8.2f} {win_rate:>5.0f}% {sym:>8}")

    # 与主因子对比
    print(f"\n{'='*70}")
    print(f"  与主因子IC对比")
    print(f"{'='*70}")

    # 主因子IC参考值（来自 v4 0906）
    main_ic = {
        "波动": +0.0505, "RSI": -0.0361, "互证": -0.0356, "动量": -0.0337,
        "质_量能": -0.0284, "均线": -0.0249, "质_动量": -0.0247,
        "量能": -0.0239, "K线": -0.0186, "质_形态": -0.0161,
    }

    print(f"  {'因子':<18} {'本因子IC':>8} {'主因子IC参考':>12} {'对比':>8}")
    print(f"  {'-'*48}")

    blind_map = {
        "risk_reward": "无直接对应",
        "relative_strength": "无直接对应",
        "wyckoff_signal": "无直接对应",
        "wyckoff_phase": "无直接对应",
    }

    for fn in FACTOR_NAMES:
        ics = np.array(all_ics[fn])
        mean_ic = float(np.mean(ics)) if len(ics) > 0 else 0
        # 找最接近的主因子做参考
        ref_name = blind_map.get(fn, "")
        print(f"  {fn:<18} {mean_ic:>+8.4f} {ref_name:>12} {'新因子':>8}")

    print(f"\n{'='*70}")
    print(f"  risk_reward 权重30% / relative_strength 权重15% / wyckoff 权重10%")
    print(f"  这3个是盲区因子，IC验证结果决定是否调整权重")
    print(f"{'='*70}")
    print(f"\n  总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=500)
    args = parser.parse_args()
    run_blindspot_ic(top_n=args.stocks)
