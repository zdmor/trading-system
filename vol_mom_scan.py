"""
量比动量全市场扫描 + 分布分析
数据源: 新浪(实时行情) + 腾讯财经(K线) — 与 scanner.py 一致
输出: CSV + stdout 分析摘要
"""
import sys, csv, time, os, json
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from volume_momentum import VolumeMomentum

SINA_HQ_URL = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"


def fetch_spot_sina(top_n=200):
    """从新浪获取成交额前N只股票"""
    all_data = []
    for node in ("sh_a", "sz_a"):
        page = 1
        while True:
            params = {"page": page, "num": 100, "sort": "amount",
                       "asc": "0", "node": node, "_s_r_a": "init"}
            try:
                data = requests.get(SINA_HQ_URL, params=params, timeout=20).json()
            except Exception:
                break
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            all_data.extend(data)
            if len(data) < 100:
                break
            page += 1
            if page > 20:
                break

    # 按成交额排序取 top_n
    records = []
    for d in all_data:
        try:
            amount = float(d.get("amount", 0))
            if amount <= 0:
                continue
            records.append({
                "code": str(d.get("symbol", "")).strip(),
                "name": str(d.get("name", "")).strip(),
                "amount": round(amount / 1e8, 2),
            })
        except (ValueError, TypeError):
            continue

    records.sort(key=lambda x: -x["amount"])
    return records[:top_n]


def fetch_kline_tencent(code, days=60):
    """从腾讯财经获取K线"""
    # code 格式转换: sh600001 / sz000001
    if code.startswith("6"):
        key = f"sh{code}"
    elif code.startswith("0") or code.startswith("3"):
        key = f"sz{code}"
    else:
        key = code

    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            return None, None

        klines = (data.get("data", {}).get(key, {}).get("qfqday") or
                  data.get("data", {}).get(key, {}).get("day") or [])

        if not klines:
            return None, None

        # kline format: ["2026-05-23", 5.10, 5.20, 5.05, 5.18, 12345678]
        closes, volumes = [], []
        for k in klines[-days:]:
            closes.append(float(k[2]))
            volumes.append(float(k[5]))

        return closes, volumes
    except Exception:
        return None, None


def build_vol_history(volumes, lookback=120):
    """从成交量序列构建量比历史序列（用于百分位归一化）"""
    if len(volumes) < 22:
        return None
    hist_ratios = []
    n = len(volumes)
    start = max(0, n - lookback - 1)
    for i in range(start, n - 1):
        window_start = max(0, i - 20)
        avg = np.mean(volumes[window_start:i]) if i > window_start else volumes[i]
        hist_ratios.append(volumes[i] / avg if avg > 0 else 1.0)
    return hist_ratios[-lookback:] if len(hist_ratios) > lookback else hist_ratios


def run_scan(top_n=200, regime="neutral", kline_days=200):
    """扫描成交额前N只个股"""
    print(f"[1/3] 获取全市场行情 (新浪)...")
    stocks = fetch_spot_sina(top_n)
    if not stocks:
        print("  新浪行情返回空")
        return []
    print(f"  取到 {len(stocks)} 只 (成交额排序)")

    print(f"[2/3] 逐只获取K线 (腾讯, {kline_days}天) + 量比动量分析...")
    results = []
    t0 = time.time()
    errors = 0

    for i, st in enumerate(stocks):
        code = st["code"]
        try:
            closes, volumes = fetch_kline_tencent(code, days=kline_days)
            if not closes or len(closes) < 20:
                errors += 1
                continue

            vol_history = build_vol_history(volumes)
            vm = VolumeMomentum(volumes, regime=regime, closes=closes,
                               vol_history=vol_history)
            r = vm.analyze()

            results.append({
                "code": code,
                "name": st["name"],
                "amount": st["amount"],
                "vol_ratio": r["vol_ratio"],
                "vol_ratio_ma": r["vol_ratio_ma"],
                "slope": r["slope"],
                "composite_score": r["composite_score"],
                "direction": r["direction"],
                "signal": r["signal"],
                "suggestion": r["suggestion"],
                "checks_passed": r["checks_passed"],
                "percentile": r.get("percentile"),
            })
        except Exception:
            errors += 1

        if (i + 1) % 20 == 0:
            print(f"   进度: {i+1}/{len(stocks)} ({time.time()-t0:.0f}s, {errors} err)")

    print(f"  完成: {len(results)} 只有效 ({time.time()-t0:.0f}s, {errors} err)")
    return results


def print_analysis(results):
    """分布分析"""
    if not results:
        print("  无数据")
        return

    scores = [r["composite_score"] for r in results]
    slopes = [r["slope"] for r in results]

    print("\n" + "=" * 65)
    print("量比动量扫描分析报告")
    print("=" * 65)

    # 1. 综合分分布
    bands = [
        (5.0, float("inf"), "强信号"),
        (3.0, 5.0, "中等信号"),
        (1.5, 3.0, "一般信号"),
        (0.5, 1.5, "偏弱"),
        (float("-inf"), 0.5, "弱势信号"),
    ]
    print(f"\n1. 综合分分布 (n={len(results)})")
    print(f"   {'区间':<20} {'数量':<6} {'占比':<8}")
    print(f"   " + "-" * 36)
    for lo, hi, label in bands:
        count = sum(1 for s in scores if lo <= s < hi)
        pct = count / len(results) * 100
        print(f"   {label:<20} {count:<6} {pct:>6.1f}%")

    # 2. 方向分布
    print(f"\n2. 量价方向分布")
    directions = {}
    for r in results:
        d = r["direction"]
        directions[d] = directions.get(d, 0) + 1
    for d, cnt in sorted(directions.items(), key=lambda x: -x[1]):
        print(f"   {d}: {cnt} ({cnt/len(results)*100:.1f}%)")

    # 3. 信号分布
    print(f"\n3. 信号等级分布")
    signals = {}
    for r in results:
        s = r["signal"]
        signals[s] = signals.get(s, 0) + 1
    for s, cnt in sorted(signals.items(), key=lambda x: -x[1]):
        print(f"   {s}: {cnt} ({cnt/len(results)*100:.1f}%)")

    # 3b. 百分位分布
    pcts = [r.get("percentile") for r in results if r.get("percentile") is not None]
    if pcts:
        pct_bands = [(90, "强(>=P90)"), (75, "中(>=P75)"), (50, "一般(>=P50)"), (25, "弱(>=P25)"), (0, "极弱(<P25)")]
        print(f"\n3b. 百分位信号分布 (n={len(pcts)})")
        for lo, label in pct_bands:
            count = sum(1 for p in pcts if p >= lo)
            print(f"   {label}: {count} ({count/len(pcts)*100:.1f}%)")

    # 4. 描述统计
    for name, arr in [("综合分", scores), ("斜率", slopes)]:
        arr_s = sorted(arr)
        p25 = arr_s[len(arr_s)//4]
        p50 = arr_s[len(arr_s)//2]
        p75 = arr_s[3*len(arr_s)//4]
        print(f"\n4. {name}统计")
        print(f"   均值={np.mean(arr):.4f}  中位={np.median(arr):.4f}")
        print(f"   P25={p25:.4f}  P75={p75:.4f}  P90={arr_s[9*len(arr_s)//10]:.4f}  P95={arr_s[19*len(arr_s)//20]:.4f}")
        print(f"   最大={max(arr):.4f}  最小={min(arr):.4f}")

    # 5. Top 10
    print(f"\n5. Top 10 (综合分)")
    sorted_r = sorted(results, key=lambda r: -r["composite_score"])
    hdr = f"   {'#':<3} {'代码':<9} {'名称':<9} {'综合分':<8} {'斜率':<8} {'量比':<7} {'均线':<7} {'方向':<10} {'信号':<10}"
    print(hdr)
    print(f"   " + "-" * 76)
    for i, r in enumerate(sorted_r[:10]):
        print(f"   {i+1:<3} {r['code']:<9} {r['name']:<9} {r['composite_score']:<8.2f} "
              f"{r['slope']:<8.4f} {r['vol_ratio']:<7.2f} {r['vol_ratio_ma']:<7.2f} "
              f"{r['direction']:<10} {r['signal']:<10}")

    # 6. 阈值调整建议
    print(f"\n6. 阈值调整建议 (基于分布)")
    p75 = sorted(scores)[3*len(scores)//4]
    p90 = sorted(scores)[9*len(scores)//10]
    p95 = sorted(scores)[19*len(scores)//20]
    above_strong = sum(1 for s in scores if s >= 5.0)
    above_medium = sum(1 for s in scores if s >= 3.0)
    above_normal = sum(1 for s in scores if s >= 1.5)

    print(f"   当前偏弱上界: 1.5  | 中等上界: 3.0  | 强信号下界: 5.0")
    print(f"   P75={p75:.3f}  P90={p90:.3f}  P95={p95:.3f}")
    print(f"   强信号(>=5.0): {above_strong} ({above_strong/len(scores)*100:.1f}%)")
    print(f"   中等+(>=3.0): {above_medium} ({above_medium/len(scores)*100:.1f}%)")
    print(f"   一般+(>=1.5): {above_normal} ({above_normal/len(scores)*100:.1f}%)")

    # 建议方案
    target_above_normal = 0.25  # 希望25%的票进入一般+区间
    target_above_medium = 0.08  # 希望8%进入中等+
    target_above_strong = 0.02  # 希望2%进入强信号

    scores_sorted = sorted(scores)
    suggested_normal = scores_sorted[int((1 - target_above_normal) * len(scores_sorted))]
    suggested_medium = scores_sorted[int((1 - target_above_medium) * len(scores_sorted))]
    suggested_strong = scores_sorted[int((1 - target_above_strong) * len(scores_sorted))]

    print(f"\n   建议阈值 (目标: 一般25%/中等8%/强2%):")
    print(f"   一般上界: {suggested_normal:.2f} (当前1.5)")
    print(f"   中等上界: {suggested_medium:.2f} (当前3.0)")
    print(f"   强信号下界: {suggested_strong:.2f} (当前5.0)")

    # 7. 斜率分析
    print(f"\n7. 斜率分布")
    pos = sum(1 for s in slopes if s > 0.01)
    neg = sum(1 for s in slopes if s < -0.01)
    flat = len(slopes) - pos - neg
    print(f"   正斜率(>0.01): {pos} ({pos/len(slopes)*100:.1f}%)")
    print(f"   负斜率(<-0.01): {neg} ({neg/len(slopes)*100:.1f}%)")
    print(f"   平斜率: {flat} ({flat/len(slopes)*100:.1f}%)")

    # 8. 过滤通过率
    passed = sum(1 for r in results if r["checks_passed"])
    print(f"\n8. 三重过滤通过率: {passed}/{len(results)} ({passed/len(results)*100:.1f}%)")

    return scores


def save_csv(results, path):
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--regime", default="neutral", choices=["bull", "neutral", "bear"])
    args = parser.parse_args()

    print(f"量比动量扫描 | top={args.top} regime={args.regime}")
    t0 = time.time()
    results = run_scan(top_n=args.top, regime=args.regime)
    print_analysis(results)
    save_csv(results, f"vol_mom_scan_{args.regime}_{args.top}.csv")
    print(f"\n总耗时: {time.time()-t0:.0f}s")
