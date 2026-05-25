"""
因子IC滚动回测 v3 — 多窗口 + 稳定性指标
计算系统实际使用的7个手工K线因子的IC稳定性
用法: python factor_ic_rolling.py --top 200
"""
import sys, os, time, json, argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,800,qfq"


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


def calc_factors_at_idx(df, idx):
    """
    在 df 的 idx 位置计算因子值（使用 idx 及之前的数据）
    返回 (factors_dict, fwd_5d_return)
    """
    try:
        sdf = df.iloc[:idx + 1].copy()
        if len(sdf) < 60:
            return None, None

        price = float(sdf["close"].iloc[-1])
        closes = sdf["close"].values.astype(float)
        highs = sdf["high"].values.astype(float)
        lows = sdf["low"].values.astype(float)
        volumes = sdf["volume"].values.astype(float)
        n = len(sdf)

        # ATR
        trs = []
        for i in range(1, n):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = np.mean(trs[-14:]) if len(trs) >= 14 else np.mean(trs)

        # MA
        ma20 = np.mean(closes[-20:]) if n >= 20 else closes[-1]
        ma50 = np.mean(closes[-50:]) if n >= 50 else closes[-1]
        ma200 = np.mean(closes[-200:]) if n >= 200 else closes[-1]

        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0
        ret_20d = (closes[-1] / closes[-21] - 1) * 100 if n >= 21 else 0

        # RSI(14)
        gains, losses = 0, 0
        for i in range(1, min(15, n)):
            d = closes[-i] - closes[-i - 1]
            if d > 0:
                gains += d
            else:
                losses -= d
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

        # 波动率
        rets_20d = [(closes[i] / closes[i - 1] - 1) * 100 for i in range(max(0, n - 20), n)]
        vol_20 = np.std(rets_20d) if len(rets_20d) > 1 else 2

        ma_bull = 1 if ma20 > ma50 > ma200 else 0
        price_above_ma20 = 1 if price > ma20 else 0

        # ============ 因子（0-100归一化）============
        factors = {}

        # 1. 动量
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
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None

        # 前向5日收益
        if idx + 5 < len(df):
            fwd_5d = (df.iloc[idx + 5]["close"] / price - 1) * 100
        else:
            fwd_5d = None

        return factors, fwd_5d
    except Exception as e:
        return None, None


def spearman_rank(x, y):
    n = len(x)
    if n < 3:
        return 0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    rho = 1 - 6 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return rho


def run_rolling_ic(top_n=200, windows=None, shift=5):
    """
    滚动IC回测主函数
    """
    if windows is None:
        windows = [20, 60, 120]
    print(f"滚动IC回测: top{top_n}只 | 窗口{windows}日 | 每{shift}日shift")
    print("=" * 100)

    # 1. 获取候选股
    t0 = time.time()
    from scanner import Scanner
    sc = Scanner()
    sc.fetch_all_stocks()
    sc.filter_candidates()
    codes = [c["code"] for c in sc.candidates[:top_n]]
    print(f"  候选股: {len(codes)}只 ({time.time()-t0:.0f}s)")

    # 2. 获取K线
    t1 = time.time()
    stock_dfs = {}
    for code in codes:
        df = get_klines(code)
        if df is not None and len(df) > 160:
            stock_dfs[code] = df
    print(f"  K线有效: {len(stock_dfs)}只 ({time.time()-t1:.0f}s)")
    if not stock_dfs:
        print("  无有效K线数据")
        return

    # 3. 确定时间轴：取所有股票共有日期范围，构造时间点序列
    all_dates = None
    for df in stock_dfs.values():
        dt = set(df["date"].values)
        if all_dates is None:
            all_dates = dt
        else:
            all_dates = all_dates & dt

    sorted_dates = sorted(pd.DatetimeIndex(all_dates))
    # 跳过最前面的120天（需要前向预测空间）
    usable_start = 120
    # 也需要为每个时间点留5日前向预测空间
    usable_end = len(sorted_dates) - 5

    # 构建时间索引映射：stock_df中的date → idx
    date_to_idx = {}
    for code, df in stock_dfs.items():
        idx_map = {str(d.date()): i for i, d in enumerate(df["date"])}
        date_to_idx[code] = idx_map

    # 4. 遍历每个时间点（每shift日取一个截面）
    time_points = list(range(usable_start, usable_end, shift))

    factor_names = ["动量", "RSI", "K线", "量能", "均线", "波动"]
    all_ics = {fn: [] for fn in factor_names}
    all_comp_ics = []

    total_tp = len(time_points)
    for ti, tp_idx in enumerate(time_points):
        tp_date = sorted_dates[tp_idx]
        all_factors = []
        all_fwds = []

        for code, df in stock_dfs.items():
            idx = date_to_idx[code].get(str(tp_date.date()))
            if idx is None or idx < 60:
                continue
            factors, fwd = calc_factors_at_idx(df, idx)
            if factors and fwd is not None:
                all_factors.append(factors)
                all_fwds.append(fwd)

        if len(all_factors) < 30:
            continue

        # 按窗口筛选：使用不同窗口的股票数量
        # 这里所有股票已经过至少60天数据检查，直接计算各窗口IC
        # 窗口IC在这里是指用全部股票的截面计算IC

        # 单个因子IC
        for fn in factor_names:
            scores = np.array([f[fn] for f in all_factors])
            ic = spearman_rank(scores, np.array(all_fwds))
            all_ics[fn].append(ic)

        # 综合评分IC（等权平均）
        comp = np.array([np.mean(list(f.values())) for f in all_factors])
        comp_ic = spearman_rank(comp, np.array(all_fwds))
        all_comp_ics.append(comp_ic)

        if (ti + 1) % 10 == 0:
            print(f"  时间截面: {ti+1}/{total_tp}  [{tp_date.date()}]  n={len(all_factors)}", end="\r")
    print(f"  时间截面: {total_tp}/{total_tp}  完成 ({time.time()-t1:.0f}s)")

    # 5. 汇总结果
    print(f"\n{'=' * 100}")
    print(f"  滚动IC回测结果 — {total_tp}个时间截面, shift={shift}日")
    print(f"{'=' * 100}")
    print(f"  {'因子':<8} {'均值IC':>8} {'标准差':>8} {'ICIR':>8} {'IC胜率':>8} {'正IC':>6} {'负IC':>6}  {'IC分布'}")
    print(f"  {'-' * 90}")

    # 按 |均值IC| 排序
    results = []
    for fn in factor_names:
        ics = np.array(all_ics[fn])
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / max(std_ic, 0.001)
        win_rate = np.mean(ics > 0) * 100
        pos_pct = np.mean(ics > 0.02) * 100
        neg_pct = np.mean(ics < -0.02) * 100

        # 直方图（10个字符）
        bins = [-0.3, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3]
        labels_hist = [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3]
        hist = np.histogram(ics, bins=bins)[0]
        max_h = max(hist) if max(hist) > 0 else 1
        hist_str = "".join(["#" * max(1, int(h / max_h * 10)) for h in hist])

        results.append({
            "factor": fn,
            "mean_ic": mean_ic,
            "std_ic": std_ic,
            "icir": icir,
            "win_rate": win_rate,
            "pos_pct": pos_pct,
            "neg_pct": neg_pct,
            "hist": hist_str,
        })

    results.sort(key=lambda x: -abs(x["mean_ic"]))

    for r in results:
        print(f"  {r['factor']:<8} {r['mean_ic']:>+8.4f} {r['std_ic']:>8.4f} {r['icir']:>+8.2f} {r['win_rate']:>7.1f}% {r['pos_pct']:>5.0f}% {r['neg_pct']:>5.0f}%  {r['hist']:<20}")

    # 综合评分
    comp_ics = np.array(all_comp_ics)
    comp_mean = np.mean(comp_ics)
    comp_std = np.std(comp_ics)
    comp_ir = comp_mean / max(comp_std, 0.001)
    comp_win = np.mean(comp_ics > 0) * 100
    print(f"  {'-' * 90}")
    print(f"  {'综合平均':<8} {comp_mean:>+8.4f} {comp_std:>8.4f} {comp_ir:>+8.2f} {comp_win:>7.1f}%")

    # 6. 按窗口分组（前20/中20/后20截面）
    n_total = len(comp_ics)
    if n_total >= 15:
        third = n_total // 3
        for label, sl in [("早期1/3", slice(0, third)),
                          ("中期1/3", slice(third, 2 * third)),
                          ("近期1/3", slice(2 * third, n_total))]:
            print(f"\n  [{label}]")
            for r in results:
                ics = np.array(all_ics[r["factor"]])
                seg = ics[sl]
                if len(seg) > 0:
                    mean_ic = np.mean(seg)
                    std_ic = np.std(seg)
                    icir = mean_ic / max(std_ic, 0.001)
                    win_rate = np.mean(seg > 0) * 100
                    print(f"    {r['factor']:<8} IC={mean_ic:>+7.4f} ±{std_ic:.4f}  IR={icir:>+5.1f}  胜率{win_rate:.0f}%")

            seg_comp = comp_ics[sl]
            if len(seg_comp) > 0:
                cm = np.mean(seg_comp)
                cs = np.std(seg_comp)
                cir = cm / max(cs, 0.001)
                cw = np.mean(seg_comp > 0) * 100
                print(f"    {'综合平均':<8} IC={cm:>+7.4f} ±{cs:.4f}  IR={cir:>+5.1f}  胜率{cw:.0f}%")

    # 6/7. RSI反转验证 + 核心组合
    print(f"\n  组合方案模拟:")
    print(f"  {'-' * 60}")
    ics = np.array(all_ics["RSI"])
    r_mean = np.mean(ics)
    r_std = np.std(ics)
    r_ir = r_mean / max(r_std, 0.001)
    r_win = np.mean(ics > 0) * 100
    print(f"  RSI已修正: IC={r_mean:>+7.4f} ±{r_std:.4f}  IR={r_ir:>+5.1f}  胜率{r_win:.0f}%")

    # 等权组合：动量+均线+RSI
    core = ["动量", "均线", "RSI"]
    comb_ics = [np.mean([all_ics[f][ti] for f in core]) for ti in range(len(all_ics["动量"]))]
    comb_arr = np.array(comb_ics)
    print(f"  {'+'.join(core):<8} IC={np.mean(comb_arr):>+7.4f} ±{np.std(comb_arr):.4f}  IR={np.mean(comb_arr)/max(np.std(comb_arr),0.001):>+5.1f}  胜率{np.mean(comb_arr>0)*100:.0f}%")

    print(f"\n  总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=200, help="候选股数量")
    parser.add_argument("--shift", type=int, default=5, help="截面间隔（日）")
    args = parser.parse_args()
    run_rolling_ic(top_n=args.top, shift=args.shift)
