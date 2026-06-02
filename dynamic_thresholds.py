# -*- coding: utf-8 -*-
"""
动态阈值模块

根据市场状态自适应调整买卖阈值，替代固定 buy=50/sell=35。

核心原则：
  - 强趋势 → 降低买入阈值（更容易进场）
  - 高波动 → 提高买入阈值（需要更确认的信号）
  - 大跌后 → 提高确认要求
  - 所有指标从 OHLCV 原地计算，不调 Tushare API

用法:
  from dynamic_thresholds import get_dynamic_thresholds
  thresh = get_dynamic_thresholds(closes, highs, lows, volumes)
  buy_thr = thresh["buy_threshold"]
  sell_thr = thresh["sell_threshold"]
"""
import numpy as np
import math


def _ema(x, span):
    """指数移动平均"""
    if len(x) == 0:
        return np.array([])
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(x, dtype=float)
    result[0] = x[0]
    for i in range(1, len(x)):
        result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]
    return result


def _atr(highs, lows, closes, period=14):
    """计算 ATR"""
    n = len(closes)
    if n < 2:
        return np.array([0.0] * n)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # 用EMA平滑TR
    alpha = 2.0 / (period + 1)
    atr_vals = np.zeros(n)
    atr_vals[0] = np.mean(tr[:min(n, period)])
    for i in range(1, n):
        atr_vals[i] = alpha * tr[i] + (1 - alpha) * atr_vals[i - 1]
    return atr_vals


def _trend_strength(closes, ma50_arr):
    """
    趋势强度: (close - MA50) / MA50 归一化到 [-100, 100]
    正值=多头偏强，负值=空头偏弱
    """
    n = len(closes)
    strength = np.zeros(n)
    for i in range(n):
        if ma50_arr[i] > 0:
            raw = (closes[i] - ma50_arr[i]) / ma50_arr[i] * 100  # percent
            # 归一化: tanh 压缩极端值
            strength[i] = raw
        else:
            strength[i] = 0
    # clamp to [-100, 100]
    strength = np.clip(strength, -100, 100)
    return strength


def _volatility_percentile(atr_arr, closes, lookback=360):
    """
    波动率百分位: ATR(14) / close, 与历史 lookback 天比
    返回 [0, 100] 百分比
    """
    n = len(closes)
    vol_ratio = np.zeros(n)
    for i in range(n):
        if closes[i] > 0 and atr_arr[i] > 0:
            vol_ratio[i] = atr_arr[i] / closes[i]
    
    percentile = np.zeros(n)
    for i in range(n):
        start = max(0, i - lookback + 1)
        window = vol_ratio[start:i + 1]
        if len(window) >= 10:
            percentile[i] = (np.sum(window < vol_ratio[i]) / len(window)) * 100
        else:
            percentile[i] = 50  # 数据不足默认中值
    
    return percentile


def _ma_slope(ma20_arr, window=10):
    """
    MA斜率: (MA20[t] - MA20[t-10]) / MA20[t-10] 归一化
    """
    n = len(ma20_arr)
    slope = np.zeros(n)
    for i in range(window, n):
        if ma20_arr[i - window] > 0:
            raw = (ma20_arr[i] - ma20_arr[i - window]) / ma20_arr[i - window] * 100
            slope[i] = raw
    return slope


def _recent_max_drawdown(closes, window=60):
    """近 window 天从高点回落幅度 (%)"""
    n = len(closes)
    dd = np.zeros(n)
    for i in range(window, n):
        seg = closes[i - window:i + 1]
        peak = np.max(seg)
        current = closes[i]
        if peak > 0:
            dd[i] = (peak - current) / peak * 100
    return dd


def _smooth(values, span=5):
    """EMA 平滑"""
    if len(values) < span:
        return values
    return _ema(values, span)


def get_dynamic_thresholds(closes, highs, lows, volumes):
    """
    自适应计算买卖阈值

    Args:
        closes:   np.array, 收盘价 (至少 400 天)
        highs:    np.array, 最高价
        lows:     np.array, 最低价
        volumes:  np.array, 成交量

    Returns:
        dict: {
            "buy_threshold": float,     # [35, 65]
            "sell_threshold": float,    # [20, 50]
            "trend_strength": float,    # [-100, 100]
            "volatility_percentile": float,  # [0, 100]
            "recent_drawdown": float,   # %
            "ma_slope": float,          # %
            "detail": str,              # 说明
        }
    """
    n = len(closes)
    if n < 50:
        return {
            "buy_threshold": 50, "sell_threshold": 35,
            "trend_strength": 0, "volatility_percentile": 50,
            "recent_drawdown": 0, "ma_slope": 0,
            "detail": f"数据不足({n}天<50), 用默认阈值"
        }

    # 计算 MA
    def _sma(x, period):
        result = np.zeros_like(x)
        for i in range(len(x)):
            s = max(0, i - period + 1)
            result[i] = np.mean(x[s:i + 1])
        return result

    ma20 = _sma(closes, 20)
    ma50 = _sma(closes, 50)
    
    # 1. 趋势强度
    ts_raw = _trend_strength(closes, ma50)
    ts_latest = ts_raw[-1]
    
    # 2. 波动率百分位
    atr_arr = _atr(highs, lows, closes, 14)
    vol_pct = _volatility_percentile(atr_arr, closes, 360)
    vol_pct_latest = vol_pct[-1]
    
    # 3. MA斜率
    slope_arr = _ma_slope(ma20, 10)
    slope_latest = slope_arr[-1]
    
    # 4. 近期最大回撤
    dd_arr = _recent_max_drawdown(closes, 60)
    dd_latest = dd_arr[-1]

    # ====== 计算阈值 ======
    base_buy = 50.0
    base_sell = 35.0

    # 趋势调整: 趋势每强 1%, 买阈降 0.3
    trend_adj = -ts_latest * 0.3

    # 波动调整: 百分位每高 10, 买阈升 1
    vol_adj = (vol_pct_latest - 50) * 0.1

    # MA斜率调整: 上升趋势→略降, 下降趋势→略升
    slope_adj = -np.clip(slope_latest, -10, 10) * 0.2

    # 大跌后需要更高确认
    dd_adj = 0.0
    if dd_latest > 20:
        dd_adj = 5.0
    elif dd_latest > 10:
        dd_adj = 3.0
    elif dd_latest > 5:
        dd_adj = 1.0

    # 综合
    buy_thr = base_buy + trend_adj + vol_adj + slope_adj + dd_adj
    sell_thr = base_sell + trend_adj * 0.5 + vol_adj + slope_adj * 0.5 + dd_adj * 0.5

    # clamp
    buy_thr = np.clip(buy_thr, 35, 65)
    sell_thr = np.clip(sell_thr, 20, 50)

    # 确保 buy > sell + 10
    if buy_thr - sell_thr < 10:
        mid = (buy_thr + sell_thr) / 2
        buy_thr = mid + 5
        sell_thr = mid - 5
        buy_thr = np.clip(buy_thr, 35, 65)
        sell_thr = np.clip(sell_thr, 20, 50)

    # 说明文字
    parts = []
    if abs(ts_latest) > 5:
        direction = "降低" if ts_latest > 5 else "升高"
        parts.append(f"强{'多' if ts_latest > 0 else '空'}趋势(TS={ts_latest:.0f}%)→{direction}阈值")
    if vol_pct_latest > 70:
        parts.append(f"高波动(P{vol_pct_latest:.0f})")
    elif vol_pct_latest < 30:
        parts.append(f"低波动(P{vol_pct_latest:.0f})")
    if dd_latest > 10:
        parts.append(f"近期回撤{dd_latest:.0f}%")
    detail = "; ".join(parts) if parts else "市场中性"

    return {
        "buy_threshold": round(float(buy_thr), 1),
        "sell_threshold": round(float(sell_thr), 1),
        "trend_strength": round(float(ts_latest), 1),
        "volatility_percentile": round(float(vol_pct_latest), 1),
        "recent_drawdown": round(float(dd_latest), 1),
        "ma_slope": round(float(slope_latest), 2),
        "detail": detail,
    }


def get_threshold_sequence(closes, highs, lows, volumes,
                           start_idx=250, step=5,
                           smooth_span=5):
    """
    对整个回测窗口生成阈值序列（含平滑）

    Args:
        closes/highs/lows/volumes: 完整 K 线数据
        start_idx: 回测起始位置
        step: 扫描间隔
        smooth_span: EMA 平滑窗口

    Returns:
        list of dict: 每个扫描点的阈值
    """
    n = len(closes)
    raw_sequences = []
    
    for idx in range(start_idx, n, step):
        # 从当前往前取最多 400 天
        s = max(0, idx - 400)
        cs = closes[s:idx + 1]
        hs = highs[s:idx + 1]
        ls = lows[s:idx + 1]
        vs = volumes[s:idx + 1]
        
        thresh = get_dynamic_thresholds(cs, hs, ls, vs)
        thresh["idx"] = idx
        raw_sequences.append(thresh)
    
    if not raw_sequences:
        return []
    
    # EMA 平滑阈值（避免跳变）
    if len(raw_sequences) >= smooth_span:
        buy_vals = np.array([r["buy_threshold"] for r in raw_sequences])
        sell_vals = np.array([r["sell_threshold"] for r in raw_sequences])
        
        buy_smoothed = _ema(buy_vals, smooth_span)
        sell_smoothed = _ema(sell_vals, smooth_span)
        
        for i, r in enumerate(raw_sequences):
            r["buy_threshold"] = round(float(buy_smoothed[i]), 1)
            r["sell_threshold"] = round(float(sell_smoothed[i]), 1)
    
    return raw_sequences


# ====== 自检 ======
if __name__ == "__main__":
    print("dynamic_thresholds 自检:")
    
    # 生成模拟数据测试
    np.random.seed(42)
    n = 500
    closes = np.cumsum(np.random.randn(n) * 0.5) + 100
    highs = closes + np.abs(np.random.randn(n) * 1)
    lows = closes - np.abs(np.random.randn(n) * 1)
    volumes = np.random.randint(1000000, 10000000, n).astype(float)
    
    # 单点测试
    thresh = get_dynamic_thresholds(closes, highs, lows, volumes)
    print(f"  单点: buy={thresh['buy_threshold']}, sell={thresh['sell_threshold']}")
    print(f"  TS={thresh['trend_strength']:.1f}% VolP={thresh['volatility_percentile']:.0f}")
    print(f"  DD={thresh['recent_drawdown']:.1f}% Slope={thresh['ma_slope']:.2f}")
    print(f"  说明: {thresh['detail']}")
    
    # 序列测试
    seq = get_threshold_sequence(closes, highs, lows, volumes, start_idx=250, step=5)
    print(f"\n  序列: {len(seq)} 个点")
    if seq:
        buy_range = (min(r["buy_threshold"] for r in seq),
                      max(r["buy_threshold"] for r in seq))
        sell_range = (min(r["sell_threshold"] for r in seq),
                       max(r["sell_threshold"] for r in seq))
        print(f"  买阈范围: [{buy_range[0]:.1f}, {buy_range[1]:.1f}]")
        print(f"  卖阈范围: [{sell_range[0]:.1f}, {sell_range[1]:.1f}]")
        print(f"  首5点: {[r['buy_threshold'] for r in seq[:5]]}")
    print("  自检完成 - OK")
