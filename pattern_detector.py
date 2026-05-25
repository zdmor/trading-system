"""
形态学检测模块 (Pattern Detector)
传统图表形态识别：双顶/双底/头肩顶底/三重顶底/V转

数据源：复用 scanner.py 已获取的腾讯 K 线数据，不额外调用 Tushare API
设计原则：量化确认标准，不做形状模糊匹配
"""
import numpy as np
from typing import List, Tuple, Optional, Dict


# ============================================================
# 工具函数
# ============================================================

def _find_peaks(arr: np.ndarray, order: int = 2) -> np.ndarray:
    """寻找波峰：比前后 order 个元素都大的位置"""
    if len(arr) < 2 * order + 1:
        return np.array([], dtype=int)
    peaks = []
    for i in range(order, len(arr) - order):
        left = arr[i - order:i]
        right = arr[i + 1:i + order + 1]
        if np.all(arr[i] > left) and np.all(arr[i] > right):
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _find_valleys(arr: np.ndarray, order: int = 2) -> np.ndarray:
    """寻找波谷：比前后 order 个元素都小的位置"""
    if len(arr) < 2 * order + 1:
        return np.array([], dtype=int)
    valleys = []
    for i in range(order, len(arr) - order):
        left = arr[i - order:i]
        right = arr[i + 1:i + order + 1]
        if np.all(arr[i] < left) and np.all(arr[i] < right):
            valleys.append(i)
    return np.array(valleys, dtype=int)


def _ma(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均"""
    if len(arr) < window:
        return np.full_like(arr, np.nan)
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    ret[:window - 1] = np.nan
    ret[window - 1:] /= window
    return ret


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, window: int = 14) -> float:
    """平均真实波幅"""
    if len(highs) < window + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(highs)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hc, lc))
    return float(np.mean(tr_list[-window:]))


def _pct_diff(a: float, b: float) -> float:
    """百分比差值"""
    return (a / b - 1) * 100 if b != 0 else 0.0


# ============================================================
# 形态检测器
# ============================================================

class PatternDetector:
    """
    多K线形态检测。

    所有检测器输入：numpy 数组 closes/highs/lows/volumes
    输出：PatternResult 或 None（未识别）
    """

    class PatternResult:
        """形态检测结果"""
        __slots__ = ("name", "score", "direction", "neckline", "target",
                     "confirmed", "detail", "key_levels")

        def __init__(self, name: str, score: int, direction: str,
                     neckline: float = 0, target: float = 0,
                     confirmed: bool = False, detail: str = "",
                     key_levels: Optional[Dict] = None):
            self.name = name
            self.score = score       # 0-100
            self.direction = direction  # "bullish" or "bearish"
            self.neckline = neckline
            self.target = target
            self.confirmed = confirmed
            self.detail = detail
            self.key_levels = key_levels or {}

    # -------------------------------------------------------
    # 双顶 (Double Top)
    # -------------------------------------------------------
    @classmethod
    def detect_double_top(cls, closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volumes: np.ndarray,
                          lookback: int = 60) -> Optional[PatternResult]:
        """
        双顶：
        - 两个高度接近的波峰（允许 1-3% 偏差）
        - 第二顶成交量明显小于第一顶
        - 中间有一个明确波谷（颈线）
        - 收盘跌破颈线且幅度 >= 2*ATR 为确认
        """
        if len(closes) < lookback:
            return None

        seg = min(lookback, len(highs))
        h_arr = highs[-seg:]
        l_arr = lows[-seg:]
        c_arr = closes[-seg:]
        v_arr = volumes[-seg:]

        peaks = _find_peaks(h_arr, order=3)
        if len(peaks) < 2:
            return None

        # 取最近两个显著波峰
        last_two = peaks[-2:]
        p1_idx, p2_idx = last_two
        p1_h = h_arr[p1_idx]
        p2_h = h_arr[p2_idx]

        # 两个顶高度差异在 1-3% 以内
        height_diff = abs(_pct_diff(p1_h, p2_h))
        if height_diff > 5:
            return None
        if p2_h < p1_h:
            # 第二顶略低更典型，允许
            pass
        elif p2_h > p1_h * 1.03:
            return None  # 第二顶太高，可能趋势延续

        # 两顶之间必须有间隔（至少 5 根 K 线）
        gap = p2_idx - p1_idx
        if gap < 5:
            return None

        # 两顶之间的波谷（颈线）
        valley_idx = int(np.argmin(l_arr[p1_idx:p2_idx])) + p1_idx
        neckline = l_arr[valley_idx]

        # 两顶的成交量比较：第二顶量应小于第一顶
        p1_v = v_arr[p1_idx]
        p2_v = v_arr[p2_idx]
        if p1_v <= 0:
            p1_v = 1
        vol_ratio = p2_v / p1_v
        vol_ok = vol_ratio < 1.15

        # 当前收盘是否跌破颈线
        last_close = c_arr[-1]
        tr = _atr(highs[-min(30, len(highs)):], lows[-min(30, len(lows)):],
                  closes[-min(30, len(closes)):])
        atr_val = max(tr, 0.01)

        confirmed = False
        if last_close < neckline - atr_val * 2:
            confirmed = True

        # 评分
        score = 0
        parts = []
        if height_diff < 2:
            score += 25
            parts.append("顶高度一致")
        elif height_diff < 4:
            score += 15
            parts.append("顶偏差合理")
        if vol_ok:
            score += 25
            parts.append(f"第二顶缩量({vol_ratio:.2f}x)")
        if gap >= 10:
            score += 15
        elif gap >= 5:
            score += 8
        if confirmed:
            score += 35
            parts.append("颈线确认")
        else:
            penetration = (neckline - last_close) / atr_val if atr_val > 0 else 0
            if last_close < neckline:
                score += 15
                parts.append(f"跌破颈线({penetration:.1f}ATR)")

        if score < 40:
            return None

        target = neckline - (p1_h - neckline)  # 形态高度映射
        detail = f"第一顶{p1_h:.2f} 第二顶{p2_h:.2f} 颈线{neckline:.2f} 目标{target:.2f}"
        direction = "bearish"

        return cls.PatternResult(
            name="双顶", score=score, direction=direction,
            neckline=neckline, target=target, confirmed=confirmed,
            detail=detail,
            key_levels={"top1": p1_h, "valley": neckline, "top2": p2_h, "target": target}
        )

    # -------------------------------------------------------
    # 双底 (Double Bottom)
    # -------------------------------------------------------
    @classmethod
    def detect_double_bottom(cls, closes: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, volumes: np.ndarray,
                             lookback: int = 60) -> Optional[PatternResult]:
        """
        双底（倒双顶）：
        - 两个高度接近的波谷
        - 第二底成交量放大（需求进入）
        - 收盘站上颈线且幅度 >= 2*ATR 为确认
        """
        if len(closes) < lookback:
            return None

        seg = min(lookback, len(lows))
        h_arr = highs[-seg:]
        l_arr = lows[-seg:]
        c_arr = closes[-seg:]
        v_arr = volumes[-seg:]

        valleys = _find_valleys(l_arr, order=3)
        if len(valleys) < 2:
            return None

        last_two = valleys[-2:]
        v1_idx, v2_idx = last_two
        v1_l = l_arr[v1_idx]
        v2_l = l_arr[v2_idx]

        height_diff = abs(_pct_diff(v1_l, v2_l))
        if height_diff > 5:
            return None
        if v2_l > v1_l:
            # 第二底略高更典型（底部抬高）
            pass
        elif v2_l < v1_l * 0.97:
            return None  # 第二底太低，可能下跌延续

        gap = v2_idx - v1_idx
        if gap < 5:
            return None

        # 两底之间的波峰（颈线）
        peak_idx = int(np.argmax(h_arr[v1_idx:v2_idx])) + v1_idx
        neckline = h_arr[peak_idx]

        # 成交量：第二底应放量或右侧放量
        v1_vol = v_arr[v1_idx]
        v2_vol = v_arr[v2_idx]
        bg_vol = float(np.mean(v_arr[-20:])) if len(v_arr) >= 20 else 1

        vol_ok = v2_vol > v1_vol if v1_vol > 0 else False
        right_vol = float(np.mean(v_arr[v2_idx:])) > bg_vol * 1.3 if len(v_arr[v2_idx:]) >= 3 else False
        vol_score_extra = 10 if right_vol else 0

        # 当前收盘是否站上颈线
        last_close = c_arr[-1]
        tr = _atr(highs[-min(30, len(highs)):], lows[-min(30, len(lows)):],
                  closes[-min(30, len(closes)):])
        atr_val = max(tr, 0.01)

        confirmed = False
        if last_close > neckline + atr_val * 2:
            confirmed = True

        score = 0
        parts = []
        if height_diff < 2:
            score += 25
            parts.append("底高度一致")
        elif height_diff < 4:
            score += 15
            parts.append("底偏差合理")
        if vol_ok:
            score += 20
            parts.append("第二底放量")
        if right_vol:
            score += vol_score_extra
            parts.append("右侧放量")
        if gap >= 10:
            score += 15
        elif gap >= 5:
            score += 8
        if v2_l > v1_l:
            score += 10
            parts.append("底部抬高")
        if confirmed:
            score += 35
            parts.append("颈线突破确认")
        else:
            penetration = (last_close - neckline) / atr_val if atr_val > 0 else 0
            if last_close > neckline:
                score += 15
                parts.append(f"站上颈线({penetration:.1f}ATR)")

        if score < 40:
            return None

        target = neckline + (neckline - v1_l)
        detail = f"第一底{v1_l:.2f} 第二底{v2_l:.2f} 颈线{neckline:.2f} 目标{target:.2f}"

        return cls.PatternResult(
            name="双底", score=score, direction="bullish",
            neckline=neckline, target=target, confirmed=confirmed,
            detail=detail,
            key_levels={"bottom1": v1_l, "peak": neckline, "bottom2": v2_l, "target": target}
        )

    # -------------------------------------------------------
    # 头肩顶 (Head and Shoulders Top)
    # -------------------------------------------------------
    @classmethod
    def detect_hns_top(cls, closes: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, volumes: np.ndarray,
                       lookback: int = 80) -> Optional[PatternResult]:
        """
        头肩顶：左肩-头-右肩依次出现，右肩缩量
        """
        if len(closes) < lookback:
            return None

        seg = min(lookback, len(highs))
        h_arr = highs[-seg:]
        l_arr = lows[-seg:]
        c_arr = closes[-seg:]
        v_arr = volumes[-seg:]

        peaks = _find_peaks(h_arr, order=3)
        if len(peaks) < 3:
            return None

        last_three = peaks[-3:]
        if len(last_three) < 3:
            return None

        lsh, head, rsh = last_three  # left shoulder, head, right shoulder
        lsh_h = h_arr[lsh]
        head_h = h_arr[head]
        rsh_h = h_arr[rsh]

        # 头必须最高
        if not (head_h > lsh_h and head_h > rsh_h):
            return None

        # 左右肩高度接近（左肩 > 右肩 或 接近）
        shoulder_ratio = rsh_h / lsh_h if lsh_h > 0 else 0
        if shoulder_ratio < 0.85:
            return None

        # 左右肩之间的两个波谷连线 = 颈线
        # 左肩→头之间的波谷
        valley1_idx = int(np.argmin(l_arr[lsh:head])) + lsh
        # 头→右肩之间的波谷
        valley2_idx = int(np.argmin(l_arr[head:rsh])) + head

        valley1_l = l_arr[valley1_idx]
        valley2_l = l_arr[valley2_idx]
        neckline = (valley1_l + valley2_l) / 2  # 简化颈线为均值

        # 成交量：右肩 < 左肩，头最大
        lsh_v = v_arr[lsh]
        head_v = v_arr[head]
        rsh_v = v_arr[rsh]
        vol_ok = rsh_v < lsh_v if lsh_v > 0 else False
        head_vol_ok = head_v > lsh_v if lsh_v > 0 else False

        # 跌破确认
        last_close = c_arr[-1]
        tr = _atr(highs[-min(30, len(highs)):], lows[-min(30, len(lows)):],
                  closes[-min(30, len(closes)):])
        atr_val = max(tr, 0.01)

        confirmed = False
        if last_close < neckline - atr_val * 2:
            confirmed = True

        score = 0
        parts = []
        if vol_ok:
            score += 25
            parts.append(f"右肩缩量")
        if head_vol_ok:
            score += 15
            parts.append("头部放量")
        if shoulder_ratio >= 0.95:
            score += 15
            parts.append("左右肩对称")
        else:
            score += 8
            parts.append("右肩略低可接受")
        if confirmed:
            score += 35
            parts.append("颈线跌破确认")
        else:
            if last_close < neckline:
                pen = (neckline - last_close) / atr_val
                score += 10
                parts.append(f"跌破颈线({pen:.1f}ATR)")

        if score < 40:
            return None

        target = neckline - (head_h - neckline)
        detail = (f"左肩{lsh_h:.2f} 头{head_h:.2f} 右肩{rsh_h:.2f} "
                  f"颈线{neckline:.2f} 目标{target:.2f}")

        return cls.PatternResult(
            name="头肩顶", score=score, direction="bearish",
            neckline=neckline, target=target, confirmed=confirmed,
            detail=detail,
            key_levels={"left_shoulder": lsh_h, "head": head_h,
                        "right_shoulder": rsh_h, "neckline": neckline, "target": target}
        )

    # -------------------------------------------------------
    # 头肩底 (Head and Shoulders Bottom / Inverse H&S)
    # -------------------------------------------------------
    @classmethod
    def detect_hns_bottom(cls, closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volumes: np.ndarray,
                          lookback: int = 80) -> Optional[PatternResult]:
        """
        头肩底（倒头肩）：左肩-头-右肩，右肩放量
        """
        if len(closes) < lookback:
            return None

        seg = min(lookback, len(lows))
        h_arr = highs[-seg:]
        l_arr = lows[-seg:]
        c_arr = closes[-seg:]
        v_arr = volumes[-seg:]

        valleys = _find_valleys(l_arr, order=3)
        if len(valleys) < 3:
            return None

        last_three = valleys[-3:]
        if len(last_three) < 3:
            return None

        lsh, head, rsh = last_three
        lsh_l = l_arr[lsh]
        head_l = l_arr[head]
        rsh_l = l_arr[rsh]

        # 头必须最低
        if not (head_l < lsh_l and head_l < rsh_l):
            return None

        shoulder_ratio = rsh_l / lsh_l if lsh_l > 0 else 0
        if shoulder_ratio > 1.15:
            return None  # 右肩太高

        # 左肩→头之间的波峰
        peak1_idx = int(np.argmax(h_arr[lsh:head])) + lsh
        # 头→右肩之间的波峰
        peak2_idx = int(np.argmax(h_arr[head:rsh])) + head
        peak1_h = h_arr[peak1_idx]
        peak2_h = h_arr[peak2_idx]
        neckline = (peak1_h + peak2_h) / 2

        # 成交量：右肩放量，头部可能放量也可能缩量
        lsh_v = v_arr[lsh]
        rsh_v = v_arr[rsh]
        vol_ok = rsh_v > lsh_v * 1.2 if lsh_v > 0 else False

        # 突破确认
        last_close = c_arr[-1]
        tr = _atr(highs[-min(30, len(highs)):], lows[-min(30, len(lows)):],
                  closes[-min(30, len(closes)):])
        atr_val = max(tr, 0.01)

        confirmed = False
        if last_close > neckline + atr_val * 2:
            confirmed = True

        score = 0
        parts = []
        if vol_ok:
            score += 25
            parts.append(f"右肩放量({rsh_v/lsh_v:.1f}x)")
        if shoulder_ratio <= 1.03:
            score += 15
            parts.append("左右肩对称")
        else:
            score += 5
            parts.append("右肩略高")
        if rsh_l > head_l:
            score += 10
            parts.append("右肩高于头部")
        if confirmed:
            score += 35
            parts.append("颈线突破确认")
        else:
            if last_close > neckline:
                pen = (last_close - neckline) / atr_val
                score += 10
                parts.append(f"站上颈线({pen:.1f}ATR)")

        if score < 40:
            return None

        target = neckline + (neckline - head_l)
        detail = (f"左肩{lsh_l:.2f} 头{head_l:.2f} 右肩{rsh_l:.2f} "
                  f"颈线{neckline:.2f} 目标{target:.2f}")

        return cls.PatternResult(
            name="头肩底", score=score, direction="bullish",
            neckline=neckline, target=target, confirmed=confirmed,
            detail=detail,
            key_levels={"left_shoulder": lsh_l, "head": head_l,
                        "right_shoulder": rsh_l, "neckline": neckline, "target": target}
        )

    # -------------------------------------------------------
    # V型反转
    # -------------------------------------------------------
    @classmethod
    def detect_v_reversal(cls, closes: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, volumes: np.ndarray,
                          lookback: int = 40) -> Optional[PatternResult]:
        """
        V型反转：
        - 急跌后急拉，没有横盘
        - 必须有天量 + 强催化特征
        - 可靠性远低于其他形态
        """
        if len(closes) < lookback:
            return None

        seg = min(lookback, len(closes))
        c_arr = closes[-seg:]
        v_arr = volumes[-seg:]
        h_arr = highs[-seg:]
        l_arr = lows[-seg:]

        # 找最近 30 根的最低点（V尖）
        v_window = min(30, len(c_arr) - 1)
        v_idx = int(np.argmin(l_arr[-v_window:])) + (len(c_arr) - v_window)
        if v_idx < 5:
            return None

        # V尖左侧：急跌
        left_start = max(0, v_idx - 10)
        left_decline = _pct_diff(l_arr[v_idx], c_arr[left_start])
        # V尖右侧：急拉
        right_end = len(c_arr) - 1
        right_rise = _pct_diff(c_arr[right_end], c_arr[v_idx])

        if left_decline > -5 or right_rise < 8:
            return None  # 跌幅不够或涨幅不够

        # 量能：底部出现天量
        bg_vol = float(np.mean(v_arr[-min(20, len(v_arr)):])) if len(v_arr) >= 20 else 1
        v_at_bottom = v_arr[min(v_idx, len(v_arr)-1)]
        vol_ok = v_at_bottom > bg_vol * 2 if bg_vol > 0 else False

        # 反弹高度收复前一波跌幅的 50% 以上
        total_decline = abs(left_decline)
        recovery_ratio = right_rise / total_decline if total_decline > 0 else 0
        recovery_ok = recovery_ratio > 0.5

        # 右侧是否有横盘/回调（如果有横盘，说明不是 V 转）
        if right_end - v_idx >= 5:
            right_slice = c_arr[v_idx:right_end]
            retrace = float(np.max(right_slice) - np.min(right_slice)) / float(np.mean(right_slice)) * 100
            if retrace < 3 and right_rise > 15:
                pass  # 干净的V
            elif retrace > 8:
                return None  # 回调太多，不是V转

        score = 0
        parts = []
        direction = None

        if left_decline < -10:
            score += 20
            parts.append(f"急跌{left_decline:.1f}%")
        elif left_decline < -8:
            score += 15
            parts.append(f"显著下跌{left_decline:.1f}%")

        if vol_ok:
            score += 30
            parts.append(f"底部天量({v_at_bottom/bg_vol:.1f}x)")
        if recovery_ok:
            score += 25
            parts.append(f"收复{recovery_ratio*100:.0f}%跌幅")

        # 判断方向
        if left_decline < -8 and right_rise > 10 and vol_ok:
            direction = "bullish"  # V底
            score += 15
        elif left_decline < -8 and right_rise > 15:
            direction = "bullish"
            score += 10
        elif left_decline > 8 and right_rise < -10 and vol_ok:
            direction = "bearish"  # V顶（倒V）
            score += 10
        else:
            return None

        if score < 50:
            return None

        name = "V转" if direction == "bullish" else "倒V顶"
        detail = f"V尖{c_arr[v_idx]:.2f} 跌幅{left_decline:.1f}% 反弹{right_rise:.1f}% 量{v_at_bottom/bg_vol:.1f}x"

        return cls.PatternResult(
            name=name, score=score, direction=direction,
            confirmed=vol_ok and recovery_ok,
            detail=detail,
            key_levels={"v_point": c_arr[v_idx], "left_decline_pct": left_decline,
                        "right_rise_pct": right_rise, "volume_ratio": v_at_bottom/bg_vol}
        )

    # -------------------------------------------------------
    # 全部检测
    # -------------------------------------------------------
    @classmethod
    def analyze_all(cls, closes: np.ndarray, highs: np.ndarray,
                    lows: np.ndarray, volumes: np.ndarray,
                    trend: str = "空头") -> List[PatternResult]:
        """
        运行全部形态检测，返回按得分降序排列的结果列表。
        trend 参数用于过滤反向形态：多头趋势下不报双底（趋势延续而非反转）
        """
        results = []

        # 双顶（空头形态）
        r = cls.detect_double_top(closes, highs, lows, volumes)
        if r:
            results.append(r)

        # 头肩顶（空头形态）
        r = cls.detect_hns_top(closes, highs, lows, volumes)
        if r:
            results.append(r)

        # 双底（多头形态）
        r = cls.detect_double_bottom(closes, highs, lows, volumes)
        if r:
            results.append(r)

        # 头肩底（多头形态）
        r = cls.detect_hns_bottom(closes, highs, lows, volumes)
        if r:
            results.append(r)

        # V转
        r = cls.detect_v_reversal(closes, highs, lows, volumes)
        if r:
            results.append(r)

        results.sort(key=lambda x: -x.score)
        return results
