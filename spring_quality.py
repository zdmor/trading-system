"""
Spring 信号质量评分
评估 Spring 信号的可信度（0-100），基于个股位置 + 量价配合 + 市场环境。
"""

import numpy as np


def compute_confidence(closes, highs, lows, volumes, trend, market_closes=None):
    """
    计算 Spring 信号的可信度评分。

    六维评分：
      1. trend_pos (0-25): 价格在20日区间中的位置
      2. prior_advance (0-20): Spring 前涨幅（底部 vs 高位）
      3. volume_conf (0-25): 成交量缩量配合（卖压枯竭度）
      4. support_dist (0-15): 支撑位测试频次
      4.5 poisson_freq (-10~+15): 泊松频率，Spring 稀有度检验
      5. volatility (0-15): ATR 波动率环境
      6. market (-15~+30): 大盘趋势环境（bonus，可选）

    返回:
      confidence: 0-100 综合可信度
      bonus: 建议加到 detect_spring 评分的额外分
      reasons: 文本明细
    """
    components = {}
    reasons = []

    # ------ 1. 趋势位置百分位 (0-25) ------
    hi20 = float(np.max(highs[-20:]))
    lo20 = float(np.min(lows[-20:]))
    r20 = hi20 - lo20
    position = (closes[-1] - lo20) / r20 if r20 > 0 else 0.5

    if position < 0.2:
        trend_pos = 25
        reasons.append(f"低位Spring[{(position*100):.0f}%区间]")
    elif position < 0.35:
        trend_pos = 20
        reasons.append(f"偏低[{(position*100):.0f}%]")
    elif position < 0.55:
        trend_pos = 12
        reasons.append(f"中位[{(position*100):.0f}%]")
    elif position < 0.75:
        trend_pos = 0
        reasons.append(f"偏高[{(position*100):.0f}%]")
    else:
        trend_pos = -10
        reasons.append(f"高位[{(position*100):.0f}%]")
    components["trend_pos"] = trend_pos

    # ------ 2. 前期涨幅 (0-20) ------
    if len(closes) >= 60:
        prior = (closes[-1] / np.mean(closes[-60:-40]) - 1) * 100
    elif len(closes) >= 30:
        prior = (closes[-1] / closes[-30] - 1) * 100
    else:
        prior = 0

    if prior < 5:
        prior_score = 20
        reasons.append(f"底部({prior:.1f}%)")
    elif prior < 15:
        prior_score = 15
        reasons.append(f"温和({prior:.1f}%)")
    elif prior < 30:
        prior_score = 5
        reasons.append(f"已涨{prior:.1f}%")
    else:
        prior_score = -10
        reasons.append(f"高位{prior:.1f}%")
    components["prior_advance"] = prior_score

    # ------ 3. 量价配合 (0-15, 回测区分力近零, 降权处理) ------
    # 单日量比区分力为零，改用3日量能趋势(确认期成交量是否放大)
    if len(volumes) >= 25:
        avg_vol = float(np.mean(volumes[-25:-5]))
    elif len(volumes) >= 10:
        avg_vol = float(np.mean(volumes[:5]))
    else:
        avg_vol = 1

    sv = float(volumes[-1]) if len(volumes) > 0 else 0
    vr = sv / avg_vol if avg_vol > 0 else 1.0

    # 3日量比均值(确认期)
    if len(volumes) >= 3:
        v3 = float(np.mean(volumes[-3:])) / avg_vol if avg_vol > 0 else 1.0
    else:
        v3 = vr

    vol_score = 0
    if vr > 1.5 and v3 > 1.2:
        vol_score = 15; reasons.append(f"放量确认(v{vr:.1f}/v3{v3:.1f})")
    elif vr < 1.0 and v3 < 1.0:
        vol_score = 10; reasons.append(f"缩量支撑(v{vr:.1f}/v3{v3:.1f})")
    elif vr < 1.3:
        vol_score = 5; reasons.append(f"量平(v{vr:.1f})")
    else:
        vol_score = -5; reasons.append(f"放量回落(v{vr:.1f})")
    components["volume_conf"] = vol_score

    # ------ 4. 支撑质量 (0-15) ------
    support = float(np.min(lows[-12:-3]))
    touches = sum(1 for l in lows[-20:] if support > 0 and abs(l - support) / support < 0.01)
    if touches <= 2:
        sup_score = 15
        reasons.append(f"支撑{touches}次")
    elif touches <= 4:
        sup_score = 8
        reasons.append(f"支撑{touches}次")
    else:
        sup_score = 0
        reasons.append(f"过测{touches}次")
    components["support_dist"] = sup_score

    # ------ 4.5 泊松频率 (-5 ~ +15, 方向已根据回测反转) ------
    # 回测 10319 个信号验证：高频Spring 胜率(38.8%) > 低频(34.3%)，原假设"稀有=好"在A股不成立
    # 频繁测试支撑是吸筹特征，不是出货
    poisson_events = 0
    if len(lows) >= 65:
        for i in range(-60, -3):
            if i + 1 >= len(lows):
                break
            sp = float(np.min(lows[max(i-10, -len(lows)):i]))
            if sp <= 0:
                continue
            low_i = float(lows[i])
            close_i = float(closes[i]) if abs(i) <= len(closes) else 0
            if low_i < sp * 0.997 and close_i > sp * 0.99:
                poisson_events += 1
    poisson_events = max(0, poisson_events - 1)

    if poisson_events >= 3:
        poisson_score = 15
        reasons.append(f"高频支撑测试({poisson_events}次/60日)")
    elif poisson_events == 2:
        poisson_score = 8
        reasons.append(f"中频({poisson_events}次)")
    elif poisson_events == 1:
        poisson_score = 0
        reasons.append(f"低频({poisson_events}次)")
    else:
        poisson_score = -5
        reasons.append(f"首次({poisson_events}次)")
    components["poisson_freq"] = poisson_score

    # ------ 5. 波动率环境 (0-15) ------
    atr = _atr(highs, lows, closes, 14)
    if len(closes) >= 30:
        atr_hist = _atr(highs[-30:], lows[-30:], closes[-30:], 14)
        ap = atr / atr_hist if atr_hist > 0 else 0.5
    else:
        ap = 0.5

    if ap < 0.8:
        vol_env = 15
        reasons.append("低波")
    elif ap < 1.2:
        vol_env = 8
        reasons.append("常波")
    else:
        vol_env = -5
        reasons.append("高波")
    components["volatility"] = vol_env

    # ------ 6. 大盘环境 (-15 ~ +30) ------
    mkt_score = 0
    if market_closes is not None and len(market_closes) >= 20:
        m_ma20 = float(np.mean(market_closes[-20:]))
        m_ma60 = float(np.mean(market_closes[-60:])) if len(market_closes) >= 60 else m_ma20
        m_cur = float(market_closes[-1])

        if m_cur > m_ma20 > m_ma60:
            mkt = 20; reasons.append("大盘多")
        elif m_cur > m_ma20:
            mkt = 10; reasons.append("大盘中性")
        elif m_cur > m_ma60:
            mkt = 0; reasons.append("大盘偏弱")
        else:
            mkt = -15; reasons.append("大盘空")

        m_std = float(np.std(market_closes[-20:]))
        m_mean = float(np.mean(market_closes[-20:]))
        m_cv = m_std / m_mean if m_mean > 0 else 0
        if m_cv < 0.02:
            mkt += 10; reasons.append("大盘稳")
        elif m_cv > 0.05:
            mkt -= 5; reasons.append("大盘波动大")

        mkt_score = mkt
    components["market"] = mkt_score

    # ------ 综合 ------
    raw = trend_pos + prior_score + vol_score + sup_score + poisson_score + vol_env + mkt_score
    confidence = max(0, min(100, raw))

    bonus = _to_bonus(confidence)

    return {
        "confidence": max(0, min(100, confidence)),
        "bonus": bonus,
        "components": components,
        "reasons": "; ".join(reasons),
    }


def _atr(highs, lows, closes, period=14):
    """ATR(period)"""
    if len(highs) < period + 1:
        return 0
    trs = []
    for i in range(-period, 0):
        h, l, pc = float(highs[i]), float(lows[i]), float(closes[i - 1])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(trs)) if trs else 0


def _to_bonus(confidence):
    """可信度 → 建议加到 detect_spring 评分的额外分"""
    if confidence >= 80: return 20
    if confidence >= 65: return 10
    if confidence >= 50: return 5
    if confidence >= 35: return 0
    if confidence >= 20: return -5
    return -15
