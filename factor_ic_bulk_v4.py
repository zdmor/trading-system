"""
因子IC滚动回测 v4 — 2000只股票 × 3年窗口 × 10因子
================================================================
特性：
  - 数据缓存优先（parquet格式，增量拉取）
  - Tencent K线API批量拉取（并发15线程）
  - 每500只checkpoint保存中间结果
  - 全pandas vectorized因子计算
  - 10因子：6基础 + 3因果质量 + 1互证 (Δ因子已移除，IC确认纯噪音)
  - 输出：IC均值/ICIR/胜率/趋势分析/衰减信号/分段表现
"""
import sys, os, time, json, argparse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 配置
# ============================================================
CACHE_DIR = r"D:\ClaudeWorkspace\trading_system\data_cache\daily"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,800,qfq"
STOCKS_COUNT = 2000
WINDOW_START = "2023-06-01"
WINDOW_END = "2026-05-27"
SHIFT_DAYS = 5           # 每5个交易日取一个截面
MIN_KLINE_DAYS = 120     # 最少需要120天K线数据
THREADS = 15
CHECKPOINT_EVERY = 500

PYTHON = r"C:\Users\sut-b\AppData\Local\Programs\Python\Python312\python.exe"


# ============================================================
# 阶段1: 候选股获取
# ============================================================
def get_candidates(target_count=2000):
    """Use Scanner's filter_candidates() for liquidity-filtered pool.
    Scanner already handles:
      - Trading hours: filter by min_amount (default 5亿 from config)
      - Non-trading hours: return all snapshot (no amount data available)
    We then dedupe by code + strip ST/Beijing exchange/delisting.
    """
    from scanner import Scanner
    s = Scanner()
    s.fetch_all_stocks()
    candidates = s.filter_candidates()  # already amount-filtered (or all during off-hours)
    
    # Deduplicate by code, strip ST / Beijing exchange / delisting
    bj_prefixes = ("8", "4", "9")
    seen = set()
    codes = []
    for c in candidates:
        code = c["code"]
        name = c.get("name", "")
        sym = code.split(".")[-1] if "." in code else code
        if sym.startswith(bj_prefixes) or code.startswith("bj"):
            continue
        if "ST" in name or "*ST" in name:
            continue
        if code in seen:
            continue
        seen.add(code)
        codes.append(code)
        if len(codes) >= target_count:
            break
    
    # Diagnostic: report liquidity profile
    has_realtime = any(v.get("amount", 0) > 0 for v in s.snapshot.values())
    tag = "Trading" if has_realtime else "Off-hours (no realtime amount)"
    amounts = [c.get("amount", 0) for c in candidates if c.get("amount", 0) > 0]
    avg_amt = sum(amounts) / len(amounts) / 1e8 if amounts else 0
    print(f"  {tag}: Scanner filtered {len(candidates)} stocks, "
          f"after dedup/ST/BJ -> {len(codes)}"
          + (f" | avg amount {avg_amt:.1f}亿" if avg_amt > 0 else ""))
    
    return codes[:target_count]


def filter_by_liquidity(candidates, min_daily_amount=5e8, lookback_days=60):
    """用历史日均成交额过滤候选股（腾讯K-line volume 单位=手，需 *100）。
    只保留缓存中存在且近 lookback_days 日均成交额 >= min_daily_amount 的股票。
    任何时候跑结果一致——不依赖实时行情。
    """
    filtered = []
    for code in candidates:
        csv_path = os.path.join(CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            recent = df.tail(lookback_days)
            if len(recent) < 20:
                continue
            # Tencent API volume unit = 手(100股), so turnover = volume * 100 * close
            avg_amount = (recent["volume"] * 100 * recent["close"]).mean()
            if avg_amount >= min_daily_amount:
                filtered.append(code)
        except Exception:
            continue
    print(f"  Liquidity filter (hist avg>={min_daily_amount/1e8:.0f}亿): "
          f"{len(candidates)} -> {len(filtered)} stocks")
    return filtered


# ============================================================
# 阶段2: K线数据缓存（parquet）
# ============================================================
def parse_kline_response(data, code):
    """解析腾讯K线API响应 → DataFrame"""
    key = code.replace(".", "")
    klines = (data.get("data", {}).get(key, {}).get("qfqday") or
              data.get("data", {}).get(key, {}).get("day") or [])
    if not klines:
        return None
    rows = [{
        "date": pd.Timestamp(k[0]),
        "open": float(k[1]),
        "close": float(k[2]),
        "high": float(k[3]),
        "low": float(k[4]),
        "volume": float(k[5])
    } for k in klines]
    return pd.DataFrame(rows)


def fetch_single_kline(code):
    """拉取单只股票K线"""
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return code, None
        df = parse_kline_response(r.json(), code)
        return code, df
    except Exception:
        return code, None


def build_cache(candidates):
    """批量拉取K线数据并缓存"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # 检查已有缓存
    existing = set()
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".csv"):
            existing.add(f.replace(".csv", "").replace("_", "."))
    
    need_fetch = [c for c in candidates if c not in existing]
    print(f"  已有缓存: {len(existing)} 只 | 需拉取: {len(need_fetch)} 只")
    
    if not need_fetch:
        return len(existing)
    
    # 加载checkpoint
    checkpoint_file = os.path.join(CACHE_DIR, "_checkpoint.json")
    completed = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                completed = set(json.load(f))
        except:
            pass
    
    remaining = [c for c in need_fetch if c not in completed]
    
    # 如果已全部完成
    if not remaining:
        print("  Checkpoint显示全部完成，验证缓存...")
        return len(existing) + len(need_fetch)
    
    # 并发拉取
    fetched = 0
    errors = 0
    batch_start = time.time()
    
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(fetch_single_kline, code): code for code in remaining}
        
        for future in as_completed(futures):
            code, df = future.result()
            fetched += 1
            
            if df is not None and len(df) >= MIN_KLINE_DAYS:
                csv_path = os.path.join(CACHE_DIR, f"{code.replace('.', '_')}.csv")
                df.to_csv(csv_path, index=False)
                completed.add(code)
            elif df is not None:
                errors += 1
            else:
                errors += 1
            
            # 进度
            if fetched % 100 == 0:
                elapsed = time.time() - batch_start
                rate = fetched / max(elapsed, 1)
                eta = (len(remaining) - fetched) / max(rate, 0.01)
                print(f"  拉取: {fetched}/{len(remaining)} ({fetched/len(remaining)*100:.0f}%) "
                      f"| {rate:.1f}只/s | 预计剩余 {eta:.0f}s", end="\r")
            
            # Checkpoint每500只
            if fetched % CHECKPOINT_EVERY == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(list(completed), f)
                print(f"\n  [checkpoint] {len(completed)} 只已缓存")
    
    # 最终checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(list(completed), f)
    
    total_valid = len(existing) + len(completed)
    print(f"\n  缓存完成: {total_valid} 只有效 ({fetched}拉取, {errors}无效)")
    return total_valid


# ============================================================
# 阶段3: 因子计算（纯vectorized版本）
# ============================================================
def calc_factors_vectorized(df, idx):
    """
    在 df 的 idx 位置计算10个因子值（使用 idx 及之前的数据）
    返回 (factors_dict, fwd_5d_return)

    因子列表：
      [基础6] 动量 RSI K线 量能 均线 波动
      [因果3] 质_量能 质_形态 质_动量
      [互证1] 互证
      (Δ因子已于v4.1移除 — IC确认全部纯噪音, IC≈0)
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

        # --- ATR(14) ---
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)

        # --- MA ---
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        # --- 收益率 ---
        ret_1d = (closes[-1] / closes[-2] - 1) * 100
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        # --- RSI(14) ---
        diffs = np.diff(closes[-15:]) if n >= 16 else np.diff(closes)
        gains = np.sum(diffs[diffs > 0]) if len(diffs[diffs > 0]) > 0 else 0
        losses = -np.sum(diffs[diffs < 0]) if len(diffs[diffs < 0]) > 0 else 0
        rsi = 100 - 100 / (1 + gains / max(losses, 0.001)) if (gains + losses) > 0 else 50

        # --- 量比 ---
        vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_ma20, 1)

        # --- 量趋势 ---
        vol_5 = np.mean(volumes[-5:]) if n >= 5 else volumes[-1]
        vol_prev5 = np.mean(volumes[-10:-5]) if n >= 10 else 1
        vol_trend = vol_5 / max(vol_prev5, 1)

        # --- K线形态 ---
        last_close = closes[-1]
        last_open = opens[-1]
        last_high = highs[-1]
        last_low = lows[-1]
        is_up = last_close >= last_open
        body = abs(last_close - last_open)
        upper = last_high - max(last_close, last_open)
        lower = min(last_close, last_open) - last_low

        # --- 波动率(20日) ---
        rets_20d_arr = (closes[max(0, n-21):] / closes[max(0, n-21):][0] - 1) * 100 if n >= 21 else np.array([0])
        if len(rets_20d_arr) > 1:
            rets_20d_pct = np.diff(closes[max(0, n-21):]) / closes[max(0, n-21):-1] * 100
            vol_20 = np.std(rets_20d_pct) if len(rets_20d_pct) > 1 else 2
        else:
            vol_20 = 2

        # --- 多空判断 ---
        ma_bull = 1 if ma20 > ma50 > ma200 else 0
        price_above_ma20 = 1 if price > ma20 else 0

        factors = {}

        # ============ 基础因子 ============
        # 1. 动量
        mom = 50 + ret_5d * 3 + ret_20d * 1
        factors["动量"] = max(0, min(100, mom))

        # 2. RSI
        if rsi > 70:
            factors["RSI"] = 80
        elif rsi > 60:
            factors["RSI"] = 70
        elif 40 <= rsi <= 60:
            factors["RSI"] = 55
        elif rsi >= 30:
            factors["RSI"] = 40
        else:
            factors["RSI"] = 25

        # 3. K线形态
        kscore = 50
        if is_up and body > 0:
            kscore = 65
            if upper < body * 0.3:
                kscore += 10
            if lower > body:
                kscore += 5
        else:
            kscore = 35
            if lower > body * 2:
                kscore = 55
        if body < atr * 0.3:
            kscore = 40
        factors["K线"] = max(0, min(100, kscore))

        # 4. 量能
        vscore = 50
        if vol_ratio > 2:
            vscore = 80
        elif vol_ratio > 1.5:
            vscore = 65
        elif vol_ratio < 0.5:
            vscore = 30
        if vol_trend > 1.2:
            vscore += 10
        elif vol_trend < 0.7:
            vscore -= 10
        factors["量能"] = max(0, min(100, vscore))

        # 5. 均线
        mscore = 80 if ma_bull else 30
        if price_above_ma20:
            mscore += 10
        factors["均线"] = max(0, min(100, mscore))

        # 6. 波动率
        atr_pct = atr / price * 100
        if atr_pct < 1.0:
            factors["波动"] = 30
        elif atr_pct < 2.5:
            factors["波动"] = 75
        elif atr_pct < 4.5:
            factors["波动"] = 60
        else:
            factors["波动"] = 35

        # [Δ因子已移除 — IC确认全部为纯噪音]
        # ============ 因果质量因子 ============
        # 量能因果
        causal_vol = 50
        if vol_ratio > 1.3:
            if ma_bull and price > ma20:
                causal_vol = 85
            elif price < ma20 and ret_5d < -3:
                causal_vol = 30
            else:
                causal_vol = 65
        elif vol_ratio < 0.6:
            if ma_bull and ret_5d > 0:
                causal_vol = 70
            else:
                causal_vol = 40
        factors["质_量能"] = causal_vol

        # K线因果
        causal_k = 50
        total_range = last_high - last_low
        body_ratio = body / max(total_range, 0.01)
        if is_up and body_ratio > 0.5:
            if vol_ratio > 1.2:
                causal_k = 85
            elif vol_ratio < 0.7:
                causal_k = 40
        elif not is_up and body_ratio > 0.5:
            if vol_ratio > 1.3:
                causal_k = 25
            else:
                causal_k = 45
        else:
            if lower > body * 2 and vol_ratio < 0.8:
                causal_k = 75
            elif upper > body * 2 and vol_ratio > 1.3:
                causal_k = 30
        factors["质_形态"] = causal_k

        # 动量因果
        causal_mom = 50
        if ret_5d > 2 and ret_20d > 3:
            if vol_trend > 1.1:
                causal_mom = 85
            else:
                causal_mom = 60
        elif ret_5d < -2 and ret_20d < -3:
            if vol_trend > 1.1:
                causal_mom = 25
            else:
                causal_mom = 40
        elif -1 <= ret_5d <= 1:
            if vol_ratio < 0.6 and price > ma20:
                causal_mom = 70
            elif vol_ratio > 1.5 and price < ma20:
                causal_mom = 30
        factors["质_动量"] = causal_mom

        # ============ 互证因子 ============
        consensus = 0
        if vol_ratio > 1.2 and is_up:
            consensus += 1
        elif vol_ratio > 1.2 and not is_up:
            consensus -= 1
        if ma_bull and ret_5d > 0:
            consensus += 1
        elif not ma_bull and ret_5d < 0:
            consensus -= 1
        if rsi > 60 and ret_5d > 0:
            consensus += 1
        elif rsi < 40 and ret_5d < 0:
            consensus -= 1
        factors["互证"] = max(0, min(100, 50 + consensus * 16))

        # ============ 前向5日收益 ============
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None

        return factors, fwd_5d

    except Exception:
        return None, None


# ============================================================
# 阶段4: 滚动IC分析
# ============================================================
def spearman_rank(x, y):
    """Spearman秩相关系数，已处理NaN和常量序列"""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 10:
        return np.nan
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    # 处理并列
    d = rx - ry
    rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return rho


def ic_trend_analysis(ics):
    """
    IC趋势衰减分析
    返回: {slope, recent_mean, early_mean, change, alert, stability, slope_sign}
    """
    arr = np.array(ics, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 10:
        return {"slope": 0, "recent_mean": 0, "early_mean": 0,
                "change": 0, "alert": "insufficient", "stability": 0, "slope_sign": 0}

    # 线性回归斜率
    x = np.arange(n, dtype=np.float64)
    slope = np.polyfit(x, arr, 1)[0] * n

    # 近期 vs 早期
    split = n // 2
    early_mean = float(np.mean(arr[:split]))
    recent_mean = float(np.mean(arr[split:]))
    change = recent_mean - early_mean

    # IC稳定性
    window = min(20, n // 3)
    if n >= window * 2:
        rolling_std = np.array([np.std(arr[max(0, i-window):i+1]) for i in range(n)])
        recent_vol = float(np.mean(rolling_std[-window:]))
        early_vol = float(np.mean(rolling_std[:window]))
        stability = early_vol / max(recent_vol, 0.001)
    else:
        stability = 1.0

    # 衰减判定: slope<0 + stability<0.8 → "!!"
    if slope < 0 and stability < 0.8:
        alert = "decay"
    elif slope < 0 or stability < 0.7:
        alert = "watch"
    else:
        alert = "normal"

    return {
        "slope": round(slope, 4),
        "recent_mean": round(recent_mean, 4),
        "early_mean": round(early_mean, 4),
        "change": round(change, 4),
        "alert": alert,
        "stability": round(stability, 2),
        "slope_sign": -1 if slope < 0 else 1,
    }


def run_rolling_ic_analysis(stock_dfs, dates_sorted, stock_date_idx, time_points):
    """
    执行滚动IC分析（纯vectorized批量计算）
    
    返回:
      all_ics: {factor_name: [ic_values_per_timepoint]}
      all_comp_ics: [composite_ic_values]
      timepoint_details: [(date, n_stocks, factor_ics)]
    """
    factor_names = ["动量", "RSI", "K线", "量能", "均线", "波动",
                    "质_量能", "质_形态", "质_动量", "互证"]
    all_ics = {fn: [] for fn in factor_names}
    all_comp_ics = []
    timepoint_details = []

    total_tp = len(time_points)
    t0 = time.time()

    for ti, tp_idx in enumerate(time_points):
        tp_date_str = dates_sorted[tp_idx]  # 已经是字符串 "YYYY-MM-DD"
        all_factor_dicts = []
        all_fwds = []

        # 批量：对所有股票在这一时间点计算因子
        for code, df in stock_dfs.items():
            idx = stock_date_idx[code].get(tp_date_str)
            if idx is None or idx < 60:
                continue
            factors, fwd = calc_factors_vectorized(df, idx)
            if factors and fwd is not None:
                all_factor_dicts.append(factors)
                all_fwds.append(fwd)

        if len(all_factor_dicts) < 30:
            continue

        fwds_arr = np.array(all_fwds, dtype=np.float64)
        tp_ics = {}

        for fn in factor_names:
            scores = np.array([f[fn] for f in all_factor_dicts], dtype=np.float64)
            ic = spearman_rank(scores, fwds_arr)
            all_ics[fn].append(ic)
            tp_ics[fn] = ic

        # 综合评分IC
        comp = np.array([np.mean(list(f.values())) for f in all_factor_dicts], dtype=np.float64)
        comp_ic = spearman_rank(comp, fwds_arr)
        all_comp_ics.append(comp_ic)

        timepoint_details.append((tp_date_str, len(all_factor_dicts), tp_ics, comp_ic))

        if (ti + 1) % 10 == 0 or ti == total_tp - 1:
            elapsed = time.time() - t0
            rate = (ti + 1) / max(elapsed, 1)
            eta = (total_tp - ti - 1) / max(rate, 0.01)
            print(f"  截面: {ti+1}/{total_tp} [{tp_date_str}] n={len(all_factor_dicts)} "
                  f"| {rate*60:.1f}截面/min | 剩余{eta:.0f}s", end="\r")
    
    print(f"\n  滚动IC完成: {len(all_comp_ics)}个有效截面 ({time.time()-t0:.0f}s)")
    return all_ics, all_comp_ics, timepoint_details


def print_results(all_ics, all_comp_ics, timepoint_details, prev_rank=None):
    """打印完整分析结果"""
    factor_names = list(all_ics.keys())
    n_tp = len(all_comp_ics)

    print(f"\n{'='*110}")
    print(f"  因子IC滚动回测 v4 — {len(all_comp_ics)}个截面 × {len(factor_names)}因子")
    print(f"  数据窗口: {WINDOW_START} → {WINDOW_END}")
    print(f"{'='*110}")
    print(f"  {'因子':<10} {'均值IC':>8} {'标准差':>8} {'ICIR':>8} {'胜率':>7} {'正IC%':>6} {'负IC%':>6}  {'分布图'}")

    # ============ 全周期排名 ============
    results = []
    for fn in factor_names:
        ics = np.array([v for v in all_ics[fn] if not np.isnan(v)])
        if len(ics) < 10:
            continue
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / max(std_ic, 0.001)
        win_rate = np.mean(ics > 0) * 100
        pos_pct = np.mean(ics > 0.02) * 100
        neg_pct = np.mean(ics < -0.02) * 100

        # 迷你直方图
        bins = np.linspace(-0.3, 0.3, 9)
        hist, _ = np.histogram(ics, bins=bins)
        max_h = max(hist) if max(hist) > 0 else 1
        hist_str = "".join(["#" * max(1, int(h / max_h * 8)) for h in hist])

        results.append({
            "factor": fn, "mean_ic": mean_ic, "std_ic": std_ic,
            "icir": icir, "win_rate": win_rate,
            "pos_pct": pos_pct, "neg_pct": neg_pct, "hist": hist_str
        })

    results.sort(key=lambda x: -abs(x["mean_ic"]))

    print(f"  {'-'*80}")
    for i, r in enumerate(results):
        rank_icon = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f" {i+1} ")
        # 与上次对比
        prev_badge = ""
        if prev_rank and r["factor"] in prev_rank:
            prev_pos = prev_rank[r["factor"]]
            delta = prev_pos - (i + 1)
            if delta > 0:
                prev_badge = f" ↑{delta}"
            elif delta < 0:
                prev_badge = f" ↓{abs(delta)}"
            else:
                prev_badge = " —"
        
        print(f"  {rank_icon} {r['factor']:<10} {r['mean_ic']:>+8.4f} {r['std_ic']:>8.4f} "
              f"{r['icir']:>+8.2f} {r['win_rate']:>6.1f}% {r['pos_pct']:>5.0f}% {r['neg_pct']:>5.0f}%  "
              f"{r['hist']:<12}{prev_badge}")

    # 综合评分
    comp_arr = np.array([v for v in all_comp_ics if not np.isnan(v)])
    comp_mean = np.mean(comp_arr)
    comp_std = np.std(comp_arr)
    comp_ir = comp_mean / max(comp_std, 0.001)
    comp_win = np.mean(comp_arr > 0) * 100
    print(f"  {'-'*80}")
    print(f"  {'综合平均':<10} {comp_mean:>+8.4f} {comp_std:>8.4f} {comp_ir:>+8.2f} {comp_win:>6.1f}%")

    # ============ 分段表现（近1/3、中1/3、早1/3）============
    print(f"\n{'='*110}")
    print(f"  分段表现 — 三阶段独立IC")
    print(f"{'='*110}")

    for label, sl in [("早期1/3", slice(0, n_tp // 3)),
                      ("中期1/3", slice(n_tp // 3, 2 * n_tp // 3)),
                      ("近期1/3", slice(2 * n_tp // 3, n_tp))]:
        header = f"\n  [{label}] 截面{sl.start+1}-{min(sl.stop, n_tp)}"
        print(header)
        print(f"  {'因子':<10} {'IC均值':>8} {'标准差':>8} {'ICIR':>8} {'胜率':>7}  {'趋势'}")

        seg_results = []
        for r in results:
            fn = r["factor"]
            ics = np.array([v for v in all_ics[fn] if not np.isnan(v)])
            seg = ics[sl]
            if len(seg) < 3:
                continue
            m = np.mean(seg)
            s = np.std(seg)
            ir = m / max(s, 0.001)
            wr = np.mean(seg > 0) * 100
            seg_results.append((fn, m, s, ir, wr))

        seg_results.sort(key=lambda x: -abs(x[1]))
        for fn, m, s, ir, wr in seg_results:
            trend_arrow = "↗" if m > 0.01 else ("↘" if m < -0.01 else "→")
            print(f"    {fn:<10} {m:>+8.4f} {s:>8.4f} {ir:>+8.2f} {wr:>6.1f}%  {trend_arrow}")

        seg_comp = comp_arr[sl]
        if len(seg_comp) >= 3:
            cm = np.mean(seg_comp)
            cs = np.std(seg_comp)
            cir = cm / max(cs, 0.001)
            cw = np.mean(seg_comp > 0) * 100
            print(f"    {'综合平均':<10} {cm:>+8.4f} {cs:>8.4f} {cir:>+8.2f} {cw:>6.1f}%")

    # ============ IC健康度 & 衰减信号 ============
    print(f"\n{'='*110}")
    print(f"  IC健康度 — 衰减信号检测 (slope<0 + stability<0.8 → '!!')")
    print(f"{'='*110}")
    print(f"  {'因子':<10} {'近期IC':>8} {'早期IC':>8} {'IC变化':>8} {'slope':>8} {'稳定':>6}  {'信号':>6}  {'判定'}")
    print(f"  {'-'*80}")

    trends = {}
    decay_list = []
    watch_list = []
    for r in results:
        fn = r["factor"]
        ics = np.array([v for v in all_ics[fn] if not np.isnan(v)])
        ta = ic_trend_analysis(ics)
        trends[fn] = ta

        symbol = {"normal": "OK", "watch": "..", "decay": "!!", "insufficient": "--"}.get(ta["alert"], "??")
        
        if ta["alert"] == "decay":
            decay_list.append(fn)
        elif ta["alert"] == "watch":
            watch_list.append(fn)

        print(f"  {fn:<10} {ta['recent_mean']:>+8.4f} {ta['early_mean']:>+8.4f} "
              f"{ta['change']:>+8.4f} {ta['slope']:>+8.4f} {ta['stability']:>5.1f}  "
              f"{symbol:>6}")

    comp_ta = ic_trend_analysis(comp_arr)
    print(f"  {'-'*80}")
    print(f"  {'综合平均':<10} {comp_ta['recent_mean']:>+8.4f} {comp_ta['early_mean']:>+8.4f} "
          f"{comp_ta['change']:>+8.4f} {comp_ta['slope']:>+8.4f} {comp_ta['stability']:>5.1f}")

    if decay_list:
        print(f"\n  {'⚠'*3} 衰减风险因子: {', '.join(decay_list)}")
        print(f"      → slope<0 且 stability<0.8, 建议下调权重或停用")
    if watch_list:
        print(f"  {'·'*3} 需关注: {', '.join(watch_list)}")

    # ============ 与上次结果对比 ============
    if prev_rank:
        print(f"\n{'='*110}")
        print(f"  与上次结果对比")
        print(f"{'='*110}")
        print(f"  {'因子':<10} {'本次IC':>8} {'本次排名':>8} {'上次排名':>8} {'变化':>8}  {'稳定性'}")

        # 上次互证排名第一(+0.059)
        stable_factors = []
        unstable_factors = []
        for r in results:
            fn = r["factor"]
            curr_rank = results.index(r) + 1
            if fn in prev_rank:
                prev_pos = prev_rank[fn]
                delta = prev_pos - curr_rank
                is_stable = abs(delta) <= 2
                if is_stable:
                    stable_factors.append(fn)
                else:
                    unstable_factors.append(fn)
                stability_tag = "稳定" if is_stable else f"{'↑' if delta>0 else '↓'}{abs(delta)}位"
                print(f"  {fn:<10} {r['mean_ic']:>+8.4f} {curr_rank:>5}/{len(results):<5} "
                      f"{prev_pos:>5} {delta:>+5}  {stability_tag}")
            else:
                print(f"  {fn:<10} {r['mean_ic']:>+8.4f} {curr_rank:>5}/{len(results):<5} "
                      f"  {'NEW':>5} {'—':>5}  新因子")

        if stable_factors:
            print(f"\n  稳定因子(排名变化≤2): {', '.join(stable_factors)}")
        if unstable_factors:
            print(f"  不稳定因子(排名变化>2): {', '.join(unstable_factors)}")

    return results, trends


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", type=int, default=STOCKS_COUNT)
    parser.add_argument("--shift", type=int, default=SHIFT_DAYS)
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--prev-results", type=str, default=None,
                        help="上次结果JSON文件路径（用于对比）")
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 80)
    print("  因子IC滚动回测 v4")
    print(f"  {args.stocks}只股票 | {WINDOW_START}→{WINDOW_END} | 截面间隔{args.shift}日")
    print("=" * 80)

    # ---- 阶段1: 候选股 ----
    print("\n[1/4] 获取候选股...")
    candidates = get_candidates(args.stocks)
    if not candidates:
        print("  无候选股，退出")
        return

    # ---- 阶段2: 数据缓存 ----
    if not args.skip_cache:
        print(f"\n[2/4] 构建K线数据缓存...")
        valid_count = build_cache(candidates)
        if valid_count < 30:
            print(f"  缓存数据不足({valid_count}只)，无法分析")
            return
    else:
        valid_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")])
        print(f"\n[2/4] 跳过缓存，使用已有 {valid_count} 只")

    # ---- 阶段2.5: 流动性过滤（基于历史成交额，不依赖实时行情）----
    print(f"\n[2.5/4] 流动性过滤...")
    candidates = filter_by_liquidity(candidates)
    if len(candidates) < 30:
        print(f"  流动性过滤后不足30只({len(candidates)})，退出")
        return

    # ---- 阶段3: 加载缓存 + 构造截面 ----
    print(f"\n[3/4] 加载缓存并构造时间截面...")
    t_load = time.time()

    # 加载所有parquet
    stock_dfs = {}
    candidate_codes = set(candidates)  # only load filtered candidates
    csv_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    loaded = 0
    for pf in csv_files:
        try:
            code = pf.replace(".csv", "").replace("_", ".")
            if code not in candidate_codes:
                continue
            path = os.path.join(CACHE_DIR, pf)
            df = pd.read_csv(path, parse_dates=["date"])
            # 过滤日期范围
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)]
            if len(df) >= MIN_KLINE_DAYS:
                stock_dfs[code] = df.reset_index(drop=True)
                loaded += 1
        except Exception:
            pass
    
    print(f"  加载有效: {loaded}/{len(candidates)} 只候选股有缓存数据")

    if loaded < 30:
        print("  有效数据不足，退出")
        return

    # Build timeline from FIXED window, each timepoint picks stocks with data
    import numpy as np
    window_start_ts = np.datetime64(WINDOW_START)
    window_end_ts = np.datetime64(WINDOW_END)
    first_df = next(iter(stock_dfs.values()))
    all_ts = first_df["date"].values
    mask = (all_ts >= window_start_ts) & (all_ts <= window_end_ts)
    timeline_ts = all_ts[mask]
    sorted_dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in timeline_ts]
    print(f"  时间窗口: {sorted_dates[0]} -> {sorted_dates[-1]} ({len(sorted_dates)}天)")
    
    if len(sorted_dates) < 25:
        print(f"  截面过少({len(sorted_dates)}天)，无法分析")
        return
    usable_start = 60  # need 60 days history for factor calc
    usable_end = len(sorted_dates) - 5

    # stock_date_idx: {code: {date_str: df_row_index}} (用字符串键)
    stock_date_idx = {}
    for code, df in stock_dfs.items():
        idx_map = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(df["date"])}
        stock_date_idx[code] = idx_map

    time_points = list(range(usable_start, usable_end, args.shift))
    print(f"  日期范围: {sorted_dates[usable_start]} → {sorted_dates[usable_end-1]}")
    print(f"  时间截面: {len(time_points)}个 (间隔{args.shift}日)")
    print(f"  加载耗时: {time.time()-t_load:.0f}s")

    # ---- 阶段4: 滚动IC分析 ----
    print(f"\n[4/4] 滚动IC分析...")
    all_ics, all_comp_ics, timepoint_details = run_rolling_ic_analysis(
        stock_dfs, sorted_dates, stock_date_idx, time_points
    )

    # ---- 加载上次结果对比 ----
    prev_rank = None
    if args.prev_results and os.path.exists(args.prev_results):
        try:
            with open(args.prev_results, encoding='utf-8') as f:
                prev_data = json.load(f)
            # 从上次结果提取排名
            prev_rank = {}
            for item in prev_data.get("rankings", prev_data.get("results", [])):
                if isinstance(item, dict):
                    name = item.get("factor", item.get("name", ""))
                    rank = item.get("rank", prev_data.get("rankings", {}).get(name, None))
                    if name and rank:
                        prev_rank[name] = int(rank)
            print(f"\n  加载上次排名: {len(prev_rank)}个因子")
        except Exception as e:
            print(f"\n  加载上次结果失败: {e}")

    # ---- 输出结果 ----
    results, trends = print_results(all_ics, all_comp_ics, timepoint_details, prev_rank)

    # ---- 保存结果 ----
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "stocks_count": loaded,
            "window": f"{WINDOW_START}→{WINDOW_END}",
            "timepoints": len(time_points),
            "shift_days": args.shift,
        },
        "rankings": [
            {
                "factor": r["factor"],
                "rank": i + 1,
                "mean_ic": r["mean_ic"],
                "std_ic": r["std_ic"],
                "icir": r["icir"],
                "win_rate": r["win_rate"],
                "trend": trends.get(r["factor"], {}),
            }
            for i, r in enumerate(results)
        ],
        "composite": {
            "mean_ic": round(float(np.mean([v for v in all_comp_ics if not np.isnan(v)])), 4),
            "std_ic": round(float(np.std([v for v in all_comp_ics if not np.isnan(v)])), 4),
        },
        "decay_signals": {
            "decay": [fn for fn, t in trends.items() if t["alert"] == "decay"],
            "watch": [fn for fn, t in trends.items() if t["alert"] == "watch"],
        },
    }
    
    result_path = os.path.join(CACHE_DIR, f"ic_results_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n  结果已保存: {result_path}")
    print(f"  总耗时: {time.time()-t_total:.0f}s ({loaded}只股票 × {len(time_points)}截面)")
    print("=" * 80)


if __name__ == "__main__":
    main()