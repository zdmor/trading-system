"""
因子单调性检验 + 相关性矩阵
=============================================
补上 IC 框架缺失的两个维度：
  1. 单调性检验：分10组看每档平均收益是否单调递增/递减
  2. 因子相关性矩阵：发现冗余因子（ρ > 0.7 即冗余）

用法: python factor_monotonicity.py

依赖 factor_ic_bulk_v4.py 的缓存目录 (data_cache/daily)
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
from datetime import datetime
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = r"D:\ClaudeWorkspace\trading_system\data_cache\daily"
MIN_STOCKS_PER_SECTION = 30
MIN_SECTIONS = 10

# 与 v4 保持一致的因子列表
FACTOR_NAMES = ["动量", "RSI", "K线", "量能", "均线", "波动",
                "质_量能", "质_形态", "质_动量", "互证"]

# ============================================================
# 因子计算（与 v4 calc_factors_vectorized 保持一致）
# ============================================================
def calc_factors(df, idx):
    """在 df 的 idx 位置计算因子值。返回 (factors_dict, fwd_5d)"""
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

        # ATR(14)
        tr = np.maximum(highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        # MA
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        # RSI(14)
        diffs = np.diff(closes[-15:]) if n >= 16 else np.diff(closes)
        gains = np.sum(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 0
        losses = -np.sum(diffs[diffs < 0]) if len(diffs[diffs < 0]) > 0 else 0
        rsi = 100 - 100 / (1 + gains / max(losses, 0.001)) if (gains + losses) > 0 else 50

        vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_ma20, 1)

        vol_5 = np.mean(volumes[-5:]) if n >= 5 else volumes[-1]
        vol_prev5 = np.mean(volumes[-10:-5]) if n >= 10 else 1
        vol_trend = vol_5 / max(vol_prev5, 1)

        last_close = closes[-1]
        last_open = opens[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        is_up = last_close >= last_open
        body = abs(last_close - last_open)
        upper = last_high - max(last_close, last_open)
        lower = min(last_close, last_open) - last_low

        ma_bull = 1 if ma20 > ma50 > ma200 else 0
        price_above_ma20 = 1 if price > ma20 else 0

        factors = {}

        # 1. 动量
        mom = 50 + ret_5d * 3 + ret_20d * 1
        factors["动量"] = max(0, min(100, mom))

        # 2. RSI
        if rsi > 70:       factors["RSI"] = 80
        elif rsi > 60:     factors["RSI"] = 70
        elif 40 <= rsi <= 60: factors["RSI"] = 55
        elif rsi >= 30:    factors["RSI"] = 40
        else:              factors["RSI"] = 25

        # 3. K线
        kscore = 50
        if is_up and body > 0:
            kscore = 65
            if upper < body * 0.3: kscore += 10
            if lower > body: kscore += 5
        else:
            kscore = 35
            if lower > body * 2: kscore = 55
        if body < atr * 0.3: kscore = 40
        factors["K线"] = max(0, min(100, kscore))

        # 4. 量能
        vscore = 50
        if vol_ratio > 2:          vscore = 80
        elif vol_ratio > 1.5:      vscore = 65
        elif vol_ratio < 0.5:      vscore = 30
        if vol_trend > 1.2:        vscore += 10
        elif vol_trend < 0.7:      vscore -= 10
        factors["量能"] = max(0, min(100, vscore))

        # 5. 均线
        mscore = 80 if ma_bull else 30
        if price_above_ma20: mscore += 10
        factors["均线"] = max(0, min(100, mscore))

        # 6. 波动率
        atr_pct = atr / price * 100
        if atr_pct < 1.0:          factors["波动"] = 30
        elif atr_pct < 2.5:        factors["波动"] = 75
        elif atr_pct < 4.5:        factors["波动"] = 60
        else:                      factors["波动"] = 35

        # 7. 质_量能
        causal_vol = 50
        if vol_ratio > 1.3:
            if ma_bull and price > ma20:         causal_vol = 85
            elif price < ma20 and ret_5d < -3:   causal_vol = 30
            else:                                causal_vol = 65
        elif vol_ratio < 0.6:
            if ma_bull and ret_5d > 0:           causal_vol = 70
            else:                                causal_vol = 40
        factors["质_量能"] = causal_vol

        # 8. 质_形态
        causal_k = 50
        total_range = last_high - last_low
        body_ratio = body / max(total_range, 0.01)
        if is_up and body_ratio > 0.5:
            if vol_ratio > 1.2:    causal_k = 85
            elif vol_ratio < 0.7:  causal_k = 40
        elif not is_up and body_ratio > 0.5:
            if vol_ratio > 1.3:    causal_k = 25
            else:                  causal_k = 45
        else:
            if lower > body * 2 and vol_ratio < 0.8:   causal_k = 75
            elif upper > body * 2 and vol_ratio > 1.3: causal_k = 30
        factors["质_形态"] = causal_k

        # 9. 质_动量
        causal_mom = 50
        if ret_5d > 2 and ret_20d > 3:
            if vol_trend > 1.1:    causal_mom = 85
            else:                  causal_mom = 60
        elif ret_5d < -2 and ret_20d < -3:
            if vol_trend > 1.1:    causal_mom = 25
            else:                  causal_mom = 40
        elif -1 <= ret_5d <= 1:
            if vol_ratio < 0.6 and price > ma20:   causal_mom = 70
            elif vol_ratio > 1.5 and price < ma20: causal_mom = 30
        factors["质_动量"] = causal_mom

        # 10. 互证
        consensus = 0
        if vol_ratio > 1.2 and is_up:        consensus += 1
        elif vol_ratio > 1.2 and not is_up:  consensus -= 1
        if ma_bull and ret_5d > 0:           consensus += 1
        elif not ma_bull and ret_5d < 0:     consensus -= 1
        if rsi > 60 and ret_5d > 0:          consensus += 1
        elif rsi < 40 and ret_5d < 0:        consensus -= 1
        factors["互证"] = max(0, min(100, 50 + consensus * 16))

        # 前向5日收益
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None

        return factors, fwd_5d
    except Exception:
        return None, None


# ============================================================
# 主分析
# ============================================================
def run_monotonicity(top_n=500, shift=5):
    """运行单调性检验 + 相关性分析"""
    t0 = time.time()

    # 1. 获取候选股
    print("加载候选股...")
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
        if len(codes) >= top_n:
            break
    print(f"  候选股: {len(codes)}只")

    # 2. 加载缓存数据
    print("加载K线缓存...")
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

    if loaded < 50:
        print("  数据不足，退出")
        return

    # 3. 构建时间轴
    first_df = next(iter(stock_dfs.values()))
    all_ts = first_df["date"].values
    mask = (all_ts >= np.datetime64("2023-06-01")) & (all_ts <= np.datetime64("2026-05-27"))
    sorted_dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in all_ts[mask]]
    usable_start, usable_end = 60, len(sorted_dates) - 5
    time_points = list(range(usable_start, usable_end, shift))

    stock_date_idx = {}
    for code, df in stock_dfs.items():
        stock_date_idx[code] = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df["date"])}

    print(f"  时间截面: {len(time_points)}个 ({sorted_dates[usable_start]} → {sorted_dates[usable_end-1]})")

    # ============================================================
    # 4. 遍历时间截面，收集因子 + 前向收益
    # ============================================================
    # 结构: {factor: [ {date_str, decile_1_fwd, decile_2_fwd, ...}, ... ]}
    monotonicity_data = {f: [] for f in FACTOR_NAMES}

    # 相关性矩阵: 每个截面存一个因子矩阵 (n_stocks × n_factors)
    corr_factors_list = []  # [{factor: [values]}, ...] 每个截面一个

    total_tp = len(time_points)
    for ti, tp_idx in enumerate(time_points):
        tp_date = sorted_dates[tp_idx]
        stock_factors = []
        stock_fwds = []

        for code, df in stock_dfs.items():
            idx = stock_date_idx[code].get(tp_date)
            if idx is None or idx < 60:
                continue
            fac, fwd = calc_factors(df, idx)
            if fac and fwd is not None:
                stock_factors.append(fac)
                stock_fwds.append(fwd)

        if len(stock_factors) < MIN_STOCKS_PER_SECTION:
            continue

        fwds_arr = np.array(stock_fwds, dtype=np.float64)

        # ---- 单调性: 每因子分10组 ----
        for fn in FACTOR_NAMES:
            scores = np.array([f[fn] for f in stock_factors], dtype=np.float64)
            # 分10组 (0-9, group 0 = 最低分)
            decile_idx = np.digitize(scores, np.percentile(scores, np.arange(10, 100, 10)))  # 0-9
            decile_fwd = {}
            for d in range(10):
                mask_d = decile_idx == d
                if np.sum(mask_d) >= 3:
                    decile_fwd[f"d{d}"] = float(np.mean(fwds_arr[mask_d]))
            decile_fwd["n"] = len(scores)
            monotonicity_data[fn].append(decile_fwd)

        # ---- 相关性: 收集因子向量 ----
        corr_entry = {}
        for fn in FACTOR_NAMES:
            corr_entry[fn] = [f[fn] for f in stock_factors]
        corr_factors_list.append(corr_entry)

        if (ti + 1) % 20 == 0 or ti == total_tp - 1:
            print(f"  截面: {ti+1}/{total_tp} [{tp_date}] n={len(stock_factors)}", end="\r")

    print(f"\n  有效截面: {len(corr_factors_list)}个 ({time.time()-t0:.0f}s)")

    if len(corr_factors_list) < MIN_SECTIONS:
        print(f"  截面不足{MIN_SECTIONS}个，结果不可靠")
        return

    # ============================================================
    # 5. 单调性报告
    # ============================================================
    print("\n" + "=" * 80)
    print("  因子单调性检验 — 各因子分10组后的前向5日收益")
    print("  数据: 所有时间截面的分组均值汇总")
    print("=" * 80)

    print(f"\n  {'因子':<10} {'D1(低)':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'D5':>8} {'D6':>8} {'D7':>8} {'D8':>8} {'D9':>8} {'D10(高)':>8}  {'单调分':>6} {'IC':>7}")
    print(f"  {'-'*100}")

    monotonicity_scores = {}
    for fn in FACTOR_NAMES:
        records = monotonicity_data[fn]
        if len(records) < MIN_SECTIONS:
            continue

        # 汇总各档均值
        decile_means = {}
        decile_counts = {}
        for d in range(10):
            vals = [r[f"d{d}"] for r in records if f"d{d}" in r]
            if vals:
                decile_means[d] = float(np.mean(vals))
                decile_counts[d] = len(vals)

        if len(decile_means) < 6:
            continue

        # 单调性评分: Spearman相关 between decile_rank (0-9) 和 decile_mean
        ranks = sorted(decile_means.keys())
        decile_values = [decile_means[d] for d in ranks]
        if len(decile_values) >= 5:
            # 单调性 = 线性回归斜率（每档收益的递增趋势）
            x = np.arange(len(decile_values))
            slope = np.polyfit(x, decile_values, 1)[0]
            # 归一化到 [-10, 10]
            mono_score = max(-10, min(10, slope * 10))
        else:
            mono_score = 0

        monotonicity_scores[fn] = mono_score

        # 格式化输出
        line = f"  {fn:<10}"
        for d in range(10):
            v = decile_means.get(d)
            if v is not None:
                line += f" {v:>+7.2f}%"
            else:
                line += f" {'':>7}"

        # 单调分指示器
        mono_indicator = ""
        if mono_score > 2:
            mono_indicator = " +"
        elif mono_score < -2:
            mono_indicator = " -"
        elif abs(mono_score) <= 1:
            mono_indicator = " ."

        n_records = len(records)
        line += f" {mono_score:>+5.1f}{mono_indicator}"

        # 来自IC结果的对比（从已知数据手填）
        ic_map = {"波动": "+0.051", "RSI": "-0.036", "互证": "-0.036",
                  "动量": "-0.034", "质_量能": "-0.028", "均线": "-0.025",
                  "质_动量": "-0.025", "量能": "-0.024", "K线": "-0.019",
                  "质_形态": "-0.016"}
        line += f" {ic_map.get(fn, ''):>7}"

        print(line)

    # 单调性结论
    print(f"  {'-'*100}")
    print(f"  单调分>2: 收益随因子值递增（正向单调）")
    print(f"  单调分<-2: 收益随因子值递减（反向单调）")
    print(f"  单调分≈0:  收益与因子值无单调关系（问题因子）")

    # ============================================================
    # 6. 因子相关性矩阵
    # ============================================================
    print("\n" + "=" * 80)
    print("  因子相关性矩阵 — Spearman ρ | 跨截面均值")
    print("  ρ > 0.7 = 高度相关, 其中一个可能是冗余因子")
    print("=" * 80)

    def spearman_rho(x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        n = len(x)
        if n < 10 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        d = rx - ry
        return float(1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1)))

    # 跨截面计算相关性矩阵
    corr_sum = {f: {g: [] for g in FACTOR_NAMES} for f in FACTOR_NAMES}
    for entry in corr_factors_list:
        for fi in FACTOR_NAMES:
            for gj in FACTOR_NAMES:
                if FACTOR_NAMES.index(fi) <= FACTOR_NAMES.index(gj):
                    rho = spearman_rho(entry[fi], entry[gj])
                    corr_sum[fi][gj].append(rho)

    # 打印矩阵（上三角）
    header = f"  {'因子':<10}"
    for fn in FACTOR_NAMES:
        header += f" {fn:>6}"
    print(header)
    print(f"  {'-' * (10 + len(FACTOR_NAMES) * 7)}")

    for fi in FACTOR_NAMES:
        line = f"  {fi:<10}"
        for gj in FACTOR_NAMES:
            vals = [v for v in corr_sum[fi][gj] if not np.isnan(v)]
            if not vals:
                line += f" {'':>6}"
            elif FACTOR_NAMES.index(fi) > FACTOR_NAMES.index(gj):
                line += f" {'':>6}"  # 下三角留空
            else:
                rho_mean = float(np.mean(vals))
                rho_std = float(np.std(vals))
                if fi == gj:
                    line += f" {'—':>6}"
                elif abs(rho_mean) > 0.5:
                    line += f" {rho_mean:>+.2f}!"  # ! 提示高相关
                else:
                    line += f" {rho_mean:>+5.2f}"
        print(line)

    # 找高相关对
    print(f"\n  高相关因子对 (|ρ| > 0.5):")
    found = False
    for fi in FACTOR_NAMES:
        for gj in FACTOR_NAMES:
            if FACTOR_NAMES.index(fi) >= FACTOR_NAMES.index(gj):
                continue
            vals = [v for v in corr_sum[fi][gj] if not np.isnan(v)]
            if vals:
                rho_mean = float(np.mean(vals))
                if abs(rho_mean) > 0.5:
                    found = True
                    print(f"    {fi:<10} <-> {gj:<10}  rho={rho_mean:>+.3f}")
    if not found:
        print(f"    无")

    # ============================================================
    # 7. 综合判断：这个因子是否应该保留？
    # ============================================================
    print(f"\n{'=' * 80}")
    print(f"  因子有效性综合判定")
    print(f"{'=' * 80}")

    # 从 v4 最新结果 (2026-05-27 0906) 提取 IC 数据
    ic_data = {
        "波动": {"ic": 0.0505, "icir": 0.31, "wr": 64.4, "trend": "watch"},
        "RSI":  {"ic": -0.0361, "icir": -0.26, "wr": 35.6, "trend": "normal"},
        "互证": {"ic": -0.0356, "icir": -0.26, "wr": 41.7, "trend": "normal"},
        "动量": {"ic": -0.0337, "icir": -0.20, "wr": 42.4, "trend": "normal"},
        "质_量能": {"ic": -0.0284, "icir": -0.26, "wr": 37.1, "trend": "normal"},
        "均线": {"ic": -0.0249, "icir": -0.21, "wr": 46.2, "trend": "watch"},
        "质_动量": {"ic": -0.0247, "icir": -0.19, "wr": 46.2, "trend": "normal"},
        "量能": {"ic": -0.0239, "icir": -0.25, "wr": 34.1, "trend": "decay"},
        "K线": {"ic": -0.0186, "icir": -0.14, "wr": 39.4, "trend": "watch"},
        "质_形态": {"ic": -0.0161, "icir": -0.15, "wr": 42.4, "trend": "decay"},
    }

    print(f"\n  {'因子':<10} {'IC':>8} {'ICIR':>6} {'胜率':>6} {'趋势':>8} {'单调':>6} {'结论':<10}")
    print(f"  {'-' * 60}")

    for fn in FACTOR_NAMES:
        ic = ic_data.get(fn, {})
        mono = monotonicity_scores.get(fn, 0)
        ic_val = ic.get("ic", 0)
        icir = ic.get("icir", 0)
        wr = ic.get("wr", 0)
        trend = ic.get("trend", "")

        # 综合判定
        signals = []
        if abs(ic_val) < 0.02:
            signals.append("IC弱")
        if abs(icir) < 0.15:
            signals.append("IR低")
        if trend == "decay":
            signals.append("衰减")
        if abs(mono) < 1.5:
            signals.append("无单调")

        if len(signals) >= 3:
            conclusion = "建议移除"
        elif len(signals) >= 2:
            conclusion = "需降权"
        elif trend == "decay":
            conclusion = "关注中"
        elif "IC弱" in signals:
            conclusion = "可用但弱"
        else:
            conclusion = "有效"

        trend_symbol = {"normal": "OK", "watch": "..", "decay": "!!"}.get(trend, "??")
        print(f"  {fn:<10} {ic_val:>+8.4f} {icir:>+6.2f} {wr:>5.0f}% {trend_symbol:>8} {mono:>+5.1f}  {conclusion:<10}")

    print(f"  {'-' * 60}")
    print(f"  结论说明:")
    print(f"    '建议移除' = IC弱 | IR低 | 衰减/无单调, 三项同时成立")
    print(f"    '需降权'   = 至少两项不达标")
    print(f"    '关注中'   = 趋势衰减但有预测力")
    print(f"\n  重要提示:")
    print(f"    IC回测因子是简化版, 与实际评分系统因子有差异")
    print(f"    risk_reward(30%) / wyckoff(10%) / relative_strength(12%) 未被IC覆盖")
    print(f"    需单独验证这三个因子的单调性和与其它因子的相关性")
    print(f"\n  总耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run_monotonicity(top_n=500, shift=5)
