"""
Spring 质量评分回测
从已知的 Spring 信号出发，计算质量评分 vs 后续表现，验证"高可信度 Spring 胜率 > 低可信度 Spring"。
"""
import json, os, sys
import numpy as np
import requests
from datetime import datetime, timedelta

HIST_FILE = os.path.join(os.path.dirname(__file__), "picks_history.json")
REC_FILE = os.path.join(os.path.dirname(__file__), "recommendation_history.json")
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"

# ========== 数据获取 ==========

def fetch_kline(code):
    """获取个股日线"""
    prefix = "sh" if code.startswith("6") else "sz"
    url = TENCENT_KLINE_URL.format(code=f"{prefix}{code}")
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        key = f"{prefix}{code}"
        klines = data.get("data", {}).get(key, {}).get("qfqday") or \
                 data.get("data", {}).get(key, {}).get("day") or []
        closes, highs, lows, volumes = [], [], [], []
        for k in klines:
            try:
                closes.append(float(k[2]))
                highs.append(float(k[3]))
                lows.append(float(k[4]))
                volumes.append(float(k[5]) * 100)
            except (ValueError, IndexError):
                pass
        return closes, highs, lows, volumes
    except Exception as e:
        print(f"  拉取失败 {code}: {e}")
        return [], [], [], []


def fetch_market_index():
    """获取上证指数日线"""
    url = TENCENT_KLINE_URL.format(code="sh000001")
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or \
                 data.get("data", {}).get("sh000001", {}).get("day") or []
        return [float(k[2]) for k in klines if k[2]]
    except Exception:
        return []


# ========== Spring 质量评分（纯 numpy 版本，兼容无 spring_quality 模块场景） ==========

def compute_spring_confidence(closes_arr, highs_arr, lows_arr, volumes_arr, trend, market_closes):
    """完全复制 spring_quality.compute_confidence 的逻辑，避免 import 问题"""
    components = {}
    reasons = []

    hi20 = float(np.max(highs_arr[-20:]))
    lo20 = float(np.min(lows_arr[-20:]))
    r20 = hi20 - lo20
    position = (closes_arr[-1] - lo20) / r20 if r20 > 0 else 0.5
    trend_pos = 25 if position < 0.2 else (20 if position < 0.35 else (12 if position < 0.55 else (0 if position < 0.75 else -10)))
    components["trend_pos"] = trend_pos

    if len(closes_arr) >= 60:
        prior = (closes_arr[-1] / np.mean(closes_arr[-60:-40]) - 1) * 100
    elif len(closes_arr) >= 30:
        prior = (closes_arr[-1] / closes_arr[-30] - 1) * 100
    else:
        prior = 0
    prior_score = 20 if prior < 5 else (15 if prior < 15 else (5 if prior < 30 else -10))
    components["prior_advance"] = prior_score

    if len(volumes_arr) >= 25:
        avg_vol = np.mean(volumes_arr[-25:-5])
    elif len(volumes_arr) >= 10:
        avg_vol = np.mean(volumes_arr[:5])
    else:
        avg_vol = 1
    sv = volumes_arr[-1] if len(volumes_arr) > 0 else 0
    vr = sv / avg_vol if avg_vol > 0 else 1.0
    vol_score = 25 if vr < 0.7 else (18 if vr < 1.0 else (5 if vr < 1.3 else -5))
    components["volume_conf"] = vol_score

    support = float(np.min(lows_arr[-12:-3]))
    touches = sum(1 for l in lows_arr[-20:] if support > 0 and abs(l - support) / support < 0.01)
    sup_score = 15 if touches <= 2 else (8 if touches <= 4 else 0)
    components["support_dist"] = sup_score

    def atr(h, l, c, period=14):
        if len(h) < period + 1:
            return 0
        trs = []
        for i in range(-period, 0):
            trs.append(max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])))
        return float(np.mean(trs)) if trs else 0

    atr_val = atr(highs_arr, lows_arr, closes_arr, 14)
    if len(closes_arr) >= 30:
        atr_hist = atr(np.array(highs_arr[-30:]), np.array(lows_arr[-30:]), np.array(closes_arr[-30:]), 14)
        ap = atr_val / atr_hist if atr_hist > 0 else 0.5
    else:
        ap = 0.5
    vol_env = 15 if ap < 0.8 else (8 if ap < 1.2 else -5)
    components["volatility"] = vol_env

    mkt_score = 0
    if market_closes is not None and len(market_closes) >= 20:
        m_ma20 = float(np.mean(market_closes[-20:]))
        m_ma60 = float(np.mean(market_closes[-60:])) if len(market_closes) >= 60 else m_ma20
        m_cur = float(market_closes[-1])
        mkt = 20 if m_cur > m_ma20 > m_ma60 else (10 if m_cur > m_ma20 else (0 if m_cur > m_ma60 else -15))
        m_cv = float(np.std(market_closes[-20:])) / float(np.mean(market_closes[-20:]))
        if m_cv < 0.02:
            mkt += 10
        elif m_cv > 0.05:
            mkt -= 5
        mkt_score = mkt
    components["market"] = mkt_score

    raw = trend_pos + prior_score + vol_score + sup_score + vol_env + mkt_score
    return {"confidence": max(0, min(100, raw)), "components": components}


# ========== 回测核心 ==========

def backtest_spring_signals():
    """回测已记录的 Spring 信号"""
    # 1) 加载已知信号
    known = []  # [(code, name, signal, score, date, price)]

    # 从 picks_history 加载
    if os.path.exists(HIST_FILE):
        with open(HIST_FILE, encoding="utf-8") as f:
            picks = json.load(f)
        for p in picks:
            if "Spring" in p.get("signal", ""):
                known.append((p["code"], p["name"], p["signal"], p["score"], p["date"], p["price"]))

    # 从 recommendation_history 加载（top 列表中的 Spring）
    if os.path.exists(REC_FILE):
        with open(REC_FILE, encoding="utf-8") as f:
            recs = json.load(f)
        for date_str, day in recs.items():
            for t in day.get("top", []):
                if "Spring" in t.get("signal", ""):
                    tup = (t["code"], t["name"], t["signal"], t["score"], date_str, t["price"])
                    # 去重
                    if not any(x[0] == t["code"] and x[4] == date_str for x in known):
                        known.append(tup)
            # 也找 recommendations 和 pool 中的 Spring
            for section in ("recommendations", "pool", "research"):
                for t in day.get(section, []):
                    if "Spring" in t.get("signal", ""):
                        tup = (t["code"], t.get("name", ""), t["signal"], t.get("score", 0), date_str, t.get("price", 0))
                        if not any(x[0] == t["code"] and x[4] == date_str for x in known):
                            known.append(tup)

    if not known:
        print("无 Spring 信号历史记录")
        return

    print(f"找到 {len(known)} 个 Spring 信号记录，正在回测...\n")

    # 2) 获取大盘数据
    print("获取上证指数数据...")
    market_closes = fetch_market_index()
    print(f"  大盘数据: {len(market_closes)} 条\n")

    # 3) 逐个回测
    results = []
    for i, (code, name, signal, score, dt_str, price) in enumerate(known):
        print(f"  [{i+1}/{len(known)}] {name}({code}) {signal} {dt_str}...", end=" ")

        # 获取完整K线
        closes, highs, lows, volumes = fetch_kline(code)
        if len(closes) < 30:
            print(f"数据不足({len(closes)})")
            continue

        arr_c = np.array(closes)
        arr_h = np.array(highs)
        arr_l = np.array(lows)
        arr_v = np.array(volumes)

        # 确定 Spring 日期在数组中的索引
        # 用价格匹配：找 closes 中最接近推荐价的日期
        target_price = float(price)
        idx = -1
        min_diff = float("inf")
        for j in range(len(closes) - 1, max(len(closes) - 30, 0), -1):
            diff = abs(closes[j] - target_price)
            if diff < min_diff:
                min_diff = diff
                idx = j

        if idx < 10:
            print(f"无法定位 Spring 位置")
            continue

        # 4) 计算 Spring 质量评分（使用 Spring 当天的数据）
        # 用 idx 作为"当前"，截取到 idx+1 为止的数据
        spring_closes = arr_c[:idx+1]
        spring_highs = arr_h[:idx+1]
        spring_lows = arr_l[:idx+1]
        spring_volumes = arr_v[:idx+1]

        sq = compute_spring_confidence(
            spring_closes, spring_highs, spring_lows, spring_volumes,
            "", market_closes  # 大盘数据用同一份（时间范围可能不太精确但可接受）
        )
        conf = sq["confidence"]

        # 5) 计算后续 N 日收益
        forward = {}
        for n_days, label in [(1, "1日"), (5, "5日"), (10, "10日"), (20, "20日")]:
            end_idx = idx + n_days
            if end_idx < len(closes):
                ret = (closes[end_idx] / closes[idx] - 1) * 100
                forward[label] = round(ret, 2)
            else:
                forward[label] = None

        results.append({
            "code": code, "name": name, "date": dt_str,
            "price": price, "confidence": conf,
            "detect_score": score, "signal": signal,
            "fwd": forward
        })
        def fmt(val):
            return f"{val:+.2f}%" if val is not None else "N/A"
        print(f"可信度{conf:.0f} 1日{fmt(forward['1日'])} 5日{fmt(forward['5日'])} 10日{fmt(forward['10日'])}")

    # 6) 汇总报告
    if not results:
        print("\n无有效回测结果")
        return

    print("\n" + "=" * 70)
    print("  Spring 质量评分回测报告")
    print("=" * 70)
    print(f"  {code:<8}")
    print()

    # 按可信度排序
    results.sort(key=lambda x: x["confidence"])

    print(f"  {'#':<3} {'名称':<8} {'信号':<10} {'可信度':<8} {'1日':<8} {'5日':<8} {'10日':<8} {'20日':<8}")
    print("  " + "-" * 65)
    for i, r in enumerate(results, 1):
        def fmt(v):
            if v is None: return "—"
            return f"{v:+.2f}%"
        print(f"  {i:<3} {r['name']:<8} {r['signal']:<10} {r['confidence']:<8} "
              f"{fmt(r['fwd']['1日']):<8} {fmt(r['fwd']['5日']):<8} "
              f"{fmt(r['fwd']['10日']):<8} {fmt(r['fwd']['20日']):<8}")

    # 7) 高可信度 vs 低可信度 分组统计
    high = [r for r in results if r["confidence"] >= 50]
    low = [r for r in results if r["confidence"] < 50]

    def avg_fwd(group, label):
        if not group:
            return 0, 0
        wins = 0
        total = 0
        for r in group:
            v = r["fwd"].get(label)
            if v is not None:
                total += v
                if v > 0:
                    wins += 1
        avg = total / len(group) if group else 0
        wr = wins / len(group) * 100
        return avg, wr

    print()
    print(f"  {'分组':<12} {'数量':<6} {'平均1日':<10} {'1日胜率':<10} {'平均5日':<10} {'5日胜率':<10} {'平均10日':<10} {'10日胜率':<10}")
    print("  " + "-" * 80)
    for gname, group in [("高可信度(>=50)", high), ("低可信度(<50)", low), ("全部", results)]:
        if not group:
            continue
        a1, w1 = avg_fwd(group, "1日")
        a5, w5 = avg_fwd(group, "5日")
        a10, w10 = avg_fwd(group, "10日")
        print(f"  {gname:<12} {len(group):<6} {a1:+.2f}%{'':8} {w1:.0f}%{'':6} {a5:+.2f}%{'':8} {w5:.0f}%{'':6} {a10:+.2f}%{'':8} {w10:.0f}%{'':6}")

    print()
    print("  说明:")
    print("  - 可信度越高，Spring 应该越可靠")
    print("  - 预期: 高可信度组的平均收益和胜率显著高于低可信度组")
    print("  - 样本量不足时结果仅供参考")
    print()


if __name__ == "__main__":
    backtest_spring_signals()
