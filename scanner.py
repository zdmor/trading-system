"""
A股全市场选股扫描器 v2.1
功能: 板块分类 + 威科夫形态检测（Spring/SOS/LPS/Upthrust）

流程: 新浪行情中心 -> 流动性过滤 -> 行业映射 -> 威科夫+趋势分析 -> 板块分组报告

数据源: 腾讯财经HTTP API（k线）、新浪（实时行情）、baostock（行业分类）
"""

import requests
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 威科夫形态分析器
# ============================================================

class WyckoffAnalyzer:
    """检测多种威科夫交易形态"""

    @staticmethod
    def _avg(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 1

    @classmethod
    def analyze_all(cls, closes, highs, lows, volumes, trend="空头"):
        """
        返回该股票所有威科夫信号，按得分降序排列。
        return: (signals list, extra dict with accum_stage/markup)
        """
        if len(closes) < 30:
            return [("数据不足", 0, "")], {}

        signals = []

        # ---- 1. Spring 弹簧 ----
        sig = cls.detect_spring(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 2. SOS 强势信号 ----
        sig = cls.detect_sos(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 3. LPS 最后支撑点 ----
        sig = cls.detect_lps(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 4. Upthrust 上冲回落 ----
        sig = cls.detect_upthrust(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 5. EVR 努力无结果 ----
        sig = cls.detect_evr(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 6. Compression 压缩蓄势 ----
        sig = cls.detect_compression(closes, highs, lows, volumes, trend)
        if sig:
            signals.append(sig)

        # ---- 额外信息 ----
        extra = {}
        sig = cls.detect_accum_stage(closes, highs, lows, volumes, trend)
        if sig:
            extra["accum_stage"] = sig

        sig = cls.detect_markup(closes, highs, lows, volumes, trend)
        if sig:
            extra["markup"] = sig

        # 按得分降序
        signals.sort(key=lambda x: -x[1])
        return signals, extra

    # -------------------------------------------------------
    # Spring 弹簧
    # -------------------------------------------------------
    @classmethod
    def detect_spring(cls, closes, highs, lows, volumes, trend):
        """
        弹簧：价格跌破近期支撑，快速收回，伴随放量
        只在多头/盘整趋势中有效
        """
        if trend == "空头" or trend == "错误":
            return None

        support_data = lows[-12:-3]
        if len(support_data) < 3:
            return None
        support = float(min(support_data))

        cur = float(closes[-1])
        if cur > support * 1.15 or cur < support * 0.95:
            return None

        bg_vol = cls._avg(volumes[-15:-3])

        best = None
        for i in range(len(lows[-3:])):
            low = float(lows[-3:][i])
            close = float(closes[-3:][i])
            vol = float(volumes[-3:][i])

            if low >= support * 0.997 or close <= support:
                continue

            depth = (support - low) / support * 100
            vratio = vol / bg_vol
            bounce = (close - low) / low * 100

            score = 0
            parts = []
            if depth >= 4: score += 25; parts.append(f"深探{depth:.1f}%")
            elif depth >= 2: score += 20; parts.append(f"中探{depth:.1f}%")
            else: score += 12; parts.append(f"浅探{depth:.1f}%")

            if vratio >= 2.5: score += 25; parts.append("巨量")
            elif vratio >= 1.8: score += 20; parts.append("放量")
            elif vratio >= 1.2: score += 12; parts.append("微放量")
            else: score += 5; parts.append("量平")

            if bounce >= 5: score += 25; parts.append(f"强弹{bounce:.1f}%")
            elif bounce >= 3: score += 18; parts.append(f"中弹{bounce:.1f}%")
            elif bounce >= 1.5: score += 10; parts.append(f"弱弹{bounce:.1f}%")

            if i + 1 < len(closes[-3:]):
                nc = float(closes[-3:][i + 1])
                if nc > close: score += 15; parts.append("确认")

            if score > (best[1] if best else 0):
                sig = "Spring" if score >= 55 else "弱Spring"
                best = (sig, score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # SOS Sign of Strength 强势信号
    # -------------------------------------------------------
    @classmethod
    def detect_sos(cls, closes, highs, lows, volumes, trend):
        """
        强势信号：大阳线 + 放量 + 高位收盘
        表明需求强劲，供应被吸收
        """
        if trend != "多头":
            return None
        if len(closes) < 50:
            return None

        # ---- 位阶保护：SOS 不应出现在远离 MA200 的高位（Buying Climax） ----
        if len(closes) >= 200:
            ma50 = float(np.mean(closes[-50:]))
            ma200 = float(np.mean(closes))
            if ma200 > 0:
                bias_200 = (ma50 - ma200) / ma200 * 100
                # 超过 25% 说明已在高位，此时的放量可能是派发而非 SOS
                if bias_200 > 25:
                    return None

        # 只用最近一根K线检测
        today_o = float(closes[-2])  # 昨日收盘 = 今日开盘（简化）
        today_c = float(closes[-1])
        today_h = float(highs[-1])
        today_l = float(lows[-1])
        today_v = float(volumes[-1])

        # 必须是阳线
        if today_c <= today_o:
            return None

        bg_vol = cls._avg(volumes[-20:-1])
        bg_range = cls._avg([highs[i] - lows[i] for i in range(-20, -1)])

        # 阳线实体
        body = today_c - today_o
        total_range = today_h - today_l
        if total_range < 0.01:
            return None
        pos_in_range = (today_c - today_l) / total_range  # 收盘在K线中的位置(0~1)

        vratio = today_v / bg_vol
        range_ratio = total_range / bg_range if bg_range > 0 else 0

        score = 0
        parts = []

        # 涨幅
        pct = (today_c - today_o) / today_o * 100
        if pct >= 5: score += 25; parts.append(f"大涨{pct:.1f}%")
        elif pct >= 3: score += 20; parts.append(f"中涨{pct:.1f}%")
        elif pct >= 1.5: score += 12; parts.append(f"小涨{pct:.1f}%")
        else: return None  # 涨幅太小不算SOS

        # 量
        if vratio >= 2.0: score += 25; parts.append("放量")
        elif vratio >= 1.5: score += 18; parts.append("增量")
        elif vratio >= 1.2: score += 10; parts.append("微增")
        else: score += 5; parts.append("量平")

        # 高位收盘
        if pos_in_range >= 0.8: score += 25; parts.append("收高位")
        elif pos_in_range >= 0.6: score += 15; parts.append("收中上")
        else: score += 5; parts.append("收中位")

        # 振幅
        if range_ratio >= 1.5: score += 15; parts.append("宽幅")
        elif range_ratio >= 1.0: score += 8; parts.append("正常幅")

        # 价格在MA50之上
        if len(closes) >= 50 and today_c > np.mean(closes[-50:]):
            score += 10; parts.append("趋势上")

        if score >= 50:
            return ("SOS", score, " | ".join(parts))
        return None

    # -------------------------------------------------------
    # LPS Last Point of Support 最后支撑点
    # -------------------------------------------------------
    @classmethod
    def detect_lps(cls, closes, highs, lows, volumes, trend):
        """
        最后支撑点：放量上攻后，缩量回调至支撑附近
        威科夫最佳入场点
        """
        if trend != "多头":
            return None
        if len(closes) < 30:
            return None

        # 近期支撑（类似Spring的支撑计算）
        support = float(min(lows[-12:-3]))

        cur = float(closes[-1])
        # 价格必须在支撑附近（支撑上方0~5%）
        if not (support <= cur <= support * 1.05):
            return None

        # 最近3天是否有过放量上涨（LPS的前提是之前有SOS或上涨）
        recent_volumes = volumes[-10:-3]
        bg_vol = cls._avg(volumes[-25:-10])
        had_upsurge = any(
            float(volumes[-10:-3][i]) > bg_vol * 1.3 and
            float(closes[-10:-3][i]) > float(closes[-11:-4][i])
            for i in range(min(5, len(volumes[-10:-3])))
        )

        if not had_upsurge:
            return None

        # 当前成交量应该萎缩
        today_v = float(volumes[-1])
        vratio = today_v / bg_vol if bg_vol > 0 else 1

        score = 30  # 基础分
        parts = []

        # 缩量
        if vratio <= 0.5: score += 30; parts.append("地量")
        elif vratio <= 0.7: score += 22; parts.append("缩量")
        elif vratio <= 0.9: score += 12; parts.append("微缩")
        else: parts.append("量平")

        # K线形态：小实体更好
        body = abs(float(closes[-1]) - float(closes[-2]))
        avg_body = cls._avg([abs(closes[i] - closes[i-1]) for i in range(-10, -1)])
        if body < avg_body * 0.6: score += 15; parts.append("小实体")
        elif body < avg_body * 0.9: score += 8; parts.append("中实体")

        # 支撑效力：之前该支撑是否被Spring确认过
        support_tested = any(float(lows[i]) < support and float(closes[i]) > support
                             for i in range(-15, 0))
        if support_tested: score += 15; parts.append("支撑验证")

        # 价格波动收窄
        recent_range = float(max(highs[-5:])) - float(min(lows[-5:]))
        older_range = float(max(highs[-10:-5])) - float(min(lows[-10:-5]))
        if older_range > 0 and recent_range < older_range * 0.7:
            score += 10; parts.append("波幅收窄")

        detail = " | ".join(parts) if parts else ""
        sig = "LPS" if score >= 60 else "弱LPS"
        return (sig, score, detail) if score >= 40 else None

    # -------------------------------------------------------
    # Upthrust 上冲回落（UT/UTAD）
    # -------------------------------------------------------
    @classmethod
    def detect_upthrust(cls, closes, highs, lows, volumes, trend):
        """
        上冲回落：价格突破阻力后迅速收回，放量
        威科夫做空/派发信号
        """
        # Upthrust 在多头趋势末期也有效，不限制趋势
        if len(closes) < 20:
            return None

        # 阻力：过去12天的最高点（排除最近3天）
        resistance = float(max(highs[-12:-3]))
        if resistance <= 0:
            return None

        cur = float(closes[-1])
        # 价格必须在阻力附近（阻力下方3%到上方3%）
        if not (resistance * 0.97 <= cur <= resistance * 1.03):
            return None

        bg_vol = cls._avg(volumes[-15:-3])

        best = None
        for i in range(len(highs[-3:])):
            high = float(highs[-3:][i])
            close = float(closes[-3:][i])
            low = float(lows[-3:][i])
            vol = float(volumes[-3:][i])

            # 必须突破过阻力
            if high < resistance:
                continue

            # 收盘必须回到阻力之下或附近（上冲失败）
            if close > resistance * 1.01:
                continue

            vratio = vol / bg_vol
            total_range = high - low
            pos_in_range = (close - low) / total_range if total_range > 0 else 0.5

            score = 30  # 基础分
            parts = []

            # 突破幅度
            thrust = (high - resistance) / resistance * 100
            if thrust >= 3: score += 20; parts.append(f"深探{thrust:.1f}%")
            elif thrust >= 1.5: score += 15; parts.append(f"中探{thrust:.1f}%")
            else: score += 8; parts.append(f"浅探{thrust:.1f}%")

            # 量
            if vratio >= 2.0: score += 25; parts.append("巨量")
            elif vratio >= 1.5: score += 18; parts.append("放量")
            elif vratio >= 1.2: score += 10; parts.append("微放量")
            else: score += 5; parts.append("量平")

            # 低位收盘（上冲回落的特征）
            if pos_in_range <= 0.3: score += 20; parts.append("收低位")
            elif pos_in_range <= 0.5: score += 10; parts.append("收中低")

            # 上影线
            upper_wick = high - max(close, float(closes[-3:][i-1]) if i > 0 else close)
            if upper_wick > total_range * 0.4: score += 15; parts.append("长上影")

            # 次日确认
            if i + 1 < len(closes[-3:]):
                nc = float(closes[-3:][i + 1])
                if nc < close: score += 10; parts.append("次日跌")

            if score > (best[1] if best else 0):
                sig = "Upthrust" if score >= 55 else "弱Upthrust"
                best = (sig, score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # EVR Effort vs Result 努力无结果
    # -------------------------------------------------------
    @classmethod
    def detect_evr(cls, closes, highs, lows, volumes, trend):
        """
        努力无结果：放量但价格窄幅波动
        表明需求正在吸收供应（吸筹特征），或高位派发
        只在多头/盘整趋势中有效
        """
        if trend == "空头" or trend == "错误":
            return None
        if len(closes) < 30:
            return None

        # 位阶保护：远离 MA200 的高位放量滞涨按派发处理
        if len(closes) >= 200:
            ma200 = float(np.mean(closes))
            if ma200 > 0:
                cur = float(closes[-1])
                if (cur - ma200) / ma200 > 0.3:
                    return None

        bg_vol = cls._avg(volumes[-20:-1])
        best = None

        # 检测最近3天是否出现"放量但价格不动"
        for i in range(-3, 0):
            vol = float(volumes[i])
            vratio = vol / bg_vol if bg_vol > 0 else 0
            if vratio < 1.5:
                continue  # 量不够大不算努力

            # 当日涨跌幅
            pct = (float(closes[i]) - float(closes[i-1])) / float(closes[i-1]) * 100
            if abs(pct) > 2.5:
                continue  # 价格变动太大，说明有结果

            score = 50
            parts = []

            if vratio >= 2.5: score += 20; parts.append(f"巨量{vratio:.1f}倍")
            elif vratio >= 2.0: score += 15; parts.append(f"放量{vratio:.1f}倍")
            else: score += 8; parts.append(f"增量{vratio:.1f}倍")

            if pct > 0: parts.append("抗跌")
            else: parts.append("滞涨")

            # K线振幅小，确认窄幅
            k_range = float(highs[i]) - float(lows[i])
            avg_range = cls._avg([highs[j] - lows[j] for j in range(-20, -1)])
            if avg_range > 0 and k_range < avg_range * 0.7:
                score += 15; parts.append("窄幅")

            if i == -1:
                # 当日确认收盘不弱于前日
                if float(closes[-1]) >= float(closes[-2]) * 0.99:
                    score += 10; parts.append("确认")

            if score > (best[1] if best else 0):
                best = ("EVR", score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # Compression 压缩蓄势
    # -------------------------------------------------------
    @classmethod
    def detect_compression(cls, closes, highs, lows, volumes, trend):
        """
        压缩蓄势：ATR 收窄 + 缩量，变盘前夜
        价格在收敛区间内整理，波动率持续下降
        """
        if trend == "错误":
            return None
        if len(closes) < 40:
            return None

        # 计算每日真实波幅
        tr = []
        for i in range(1, len(closes)):
            h = float(highs[i])
            l_ = float(lows[i])
            pc = float(closes[i-1])
            tr.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
        if len(tr) < 25:
            return None

        # 近期 ATR vs 历史 ATR（均以百分比表示）
        recent_tr = tr[-10:]
        hist_tr = tr[-30:-10]
        recent_avg = np.mean(recent_tr) / float(closes[-1]) * 100 if len(recent_tr) > 0 else 0
        hist_avg = np.mean(hist_tr) / float(closes[-1]) * 100 if len(hist_tr) > 0 else 0

        if hist_avg <= 0:
            return None

        compress_ratio = recent_avg / hist_avg
        if compress_ratio > 0.75:
            return None  # 波幅不够收敛

        # 量能萎缩
        bg_vol = cls._avg(volumes[-30:-10])
        recent_vol = cls._avg(volumes[-10:])
        vol_ratio = recent_vol / bg_vol if bg_vol > 0 else 1
        if vol_ratio > 0.8:
            return None

        # 不允许连续两天 ATR 扩大（突破收敛）
        violations = sum(1 for j in range(1, len(recent_tr)) if recent_tr[j] > recent_tr[j-1])
        if violations > 2:
            return None

        score = 50
        parts = []

        if compress_ratio <= 0.4: score += 20; parts.append(f"极度压缩({compress_ratio:.0%})")
        elif compress_ratio <= 0.6: score += 15; parts.append(f"明显压缩({compress_ratio:.0%})")
        else: score += 8; parts.append(f"轻度压缩({compress_ratio:.0%})")

        if vol_ratio <= 0.4: score += 15; parts.append("地量")
        elif vol_ratio <= 0.6: score += 10; parts.append("缩量")
        else: parts.append("微缩")

        # 价格在均线附近（蓄势待发）
        cur = float(closes[-1])
        if len(closes) >= 50:
            ma50 = float(np.mean(closes[-50:]))
            if ma50 > 0 and abs(cur - ma50) / ma50 < 0.03:
                score += 15; parts.append("均线附近")

        return ("Compression", score, " | ".join(parts)) if score >= 60 else None


    # -------------------------------------------------------
    # Markup 阶段确认（MA50上穿MA200）
    # -------------------------------------------------------
    @classmethod
    def detect_markup(cls, closes, highs, lows, volumes, trend):
        """
        Markup 主升段：MA50 上穿 MA200 且保持在上方 N 日
        确认进入上升趋势主升段
        """
        if trend != "多头":
            return None
        if len(closes) < 210:
            return None

        ma50 = np.array([np.mean(closes[i-50:i]) for i in range(50, len(closes))])
        ma200 = np.array([np.mean(closes[i-200:i]) for i in range(200, len(closes))])
        if len(ma50) < 5:
            return None

        score = 0
        parts = []

        # 检查 MA50 是否 > MA200（趋势成立）
        if ma50[-1] <= ma200[-1]:
            return None

        score += 20; parts.append("MA50>MA200")

        # 过去 20 日内是否存在 MA50 上穿 MA200
        crossover_detected = False
        for j in range(min(20, len(ma50)-1), 0, -1):
            if ma50[-j-1] <= ma200[-j-1] and ma50[-j] > ma200[-j]:
                crossover_detected = True
                days_since = j
                parts.append(f"上穿{days_since}天前")
                break

        if not crossover_detected:
            # 已经穿越很久了，但确认趋势持续
            if ma50[-1] > ma200[-1] * 1.05:
                score += 15; parts.append("趋势持续")
        else:
            if days_since <= 5: score += 25; parts.append("金叉初现")
            elif days_since <= 15: score += 18; parts.append("金叉确认")
            else: score += 10; parts.append("趋势稳固")

        # MA50 角度（趋势强度）
        ma50_vals = [np.mean(closes[i-50:i]) for i in range(-10, 0)]
        if len(ma50_vals) >= 2:
            angle = (ma50_vals[-1] - ma50_vals[0]) / ma50_vals[0] * 100
            if angle >= 5: score += 20; parts.append("陡峭上升")
            elif angle >= 2: score += 12; parts.append("平稳上升")
            else: score += 5; parts.append("走平")

        # 价格在 MA50 上方
        cur = float(closes[-1])
        if cur > ma50[-1]:
            score += 15; parts.append("价格在MA50上")

        # 量能配合
        bg_vol = cls._avg(volumes[-50:-10])
        recent_vol = cls._avg(volumes[-10:])
        if bg_vol > 0 and recent_vol > bg_vol * 1.2:
            score += 10; parts.append("量能配合")

        return ("Markup", score, " | ".join(parts)) if score >= 50 else None


    # -------------------------------------------------------
    # Accumulation ABC 子阶段
    # -------------------------------------------------------
    @classmethod
    def detect_accum_stage(cls, closes, highs, lows, volumes, trend):
        """
        吸筹期子阶段细分：
        - Accum_A: 下跌停止，量能萎缩，低位盘整
        - Accum_B: 底部区间反复测试，低点逐步抬高
        - Accum_C: 最后缩量回踩不破A低
        """
        if trend in ("空头", "错误"):
            return None
        if len(closes) < 60:
            return None

        cur = float(closes[-1])
        low_60d = float(min(lows[-60:]))

        # 条件：价格必须在年内低位附近（年内低点 +35% 以内）
        low_250d = float(min(lows[-250:])) if len(lows) >= 250 else low_60d
        if low_250d > 0 and cur > low_250d * 1.35:
            return None

        accum_base_low = low_250d

        # 均线胶着：MA50 和 MA200 差距小
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else 0
        ma200 = float(np.mean(closes)) if len(closes) >= 200 else 0
        if ma200 > 0:
            ma_gap = abs(ma50 - ma200) / ma200 * 100
            if ma_gap > 8:
                return None  # 均线分离太远，不在吸筹期

        # 量能萎缩确认
        vol_recent = cls._avg(volumes[-20:])
        vol_hist = cls._avg(volumes[-120:-20])
        if vol_hist > 0 and vol_recent / vol_hist > 0.7:
            return None  # 量能还没干枯

        score = 40
        parts = []

        # ---- 判定 ABC ----
        # B 阶段特征：最近 30 日内多次测试底部
        zone_lows = lows[-30:]
        test_count = sum(1 for l in zone_lows if abs(l - accum_base_low) / accum_base_low <= 0.05)

        if test_count >= 3:
            stage = "Accum_B"
            score += 30; parts.append(f"多次探底({test_count}次)")
        else:
            # C 阶段特征：最近小幅下跌但不破底，量能再度萎缩
            recent_low = float(min(lows[-20:]))
            c_ok = accum_base_low > 0 and recent_low >= accum_base_low * 0.97
            if c_ok:
                vol_dry = cls._avg(volumes[-10:])
                vol_hist_c = cls._avg(volumes[-60:-20])
                if vol_hist_c > 0 and vol_dry / vol_hist_c < 0.6:
                    stage = "Accum_C"
                    score += 25; parts.append("缩量回踩不破底")
                else:
                    stage = "Accum_A"
                    score += 15; parts.append("止跌缩量")
            else:
                stage = "Accum_A"
                score += 15; parts.append("止跌缩量")

        if ma_gap < 4:
            score += 10; parts.append("均线粘合")
        if vol_recent / vol_hist < 0.4:
            score += 10; parts.append("地量")

        return (stage, score, " | ".join(parts))


    # -------------------------------------------------------
    # Phase A-E 威科夫价格周期阶段识别
    # -------------------------------------------------------
    @classmethod
    def detect_phase(cls, closes, highs, lows, volumes, trend, events, extra=None):
        """
        将检测到的威科夫事件归类到价格周期阶段（Phase A-E）。
        利用 Accum ABC / Markup 做更细粒度判断。
        return: (phase_label, description, confidence)
        """
        if extra is None:
            extra = {}
        if len(closes) < 50:
            return ("数据不足", "", 0)

        cur = float(closes[-1])
        ma50 = float(np.mean(closes[-50:]))
        ma200 = float(np.mean(closes)) if len(closes) >= 200 else 0

        # 近期震荡区间（过去60天）
        range_high = float(max(highs[-60:]))
        range_low  = float(min(lows[-60:]))
        range_size = (range_high - range_low) / range_low * 100 if range_low > 0 else 0

        above_range = cur > range_high * 1.01
        below_range = cur < range_low * 0.99
        in_range = not above_range and not below_range

        # 成交量趋势
        avg_vol_50 = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else 1
        avg_vol_10 = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else 1
        if avg_vol_10 < avg_vol_50 * 0.7:
            vol_state = "缩量"
        elif avg_vol_10 > avg_vol_50 * 1.3:
            vol_state = "放量"
        else:
            vol_state = "量平"

        # 提取事件类型
        et = set(e[0] for e in events if e[1] > 0)
        has_spring   = any("Spring" in e for e in et)
        has_sos      = "SOS" in et
        has_lps      = "LPS" in et
        has_upthrust = "Upthrust" in et
        has_evr      = "EVR" in et
        has_compression = "Compression" in et
        extra_accum = extra.get("accum_stage")
        extra_markup = extra.get("markup")

        # ----- 根据趋势 + 事件 + 位置 判断阶段 -----
        if trend == "多头":
            # Markup 优先：主升段
            if extra_markup:
                return ("Phase D→E — Markup主升段",
                        f"MA50上穿MA200确认，趋势强度{extra_markup[1]}，{vol_state}，多头主导", 88)

            if has_sos:
                if above_range:
                    return ("Phase D→E — 突破确认/上升趋势",
                            f"价格突破区间{range_low:.1f}-{range_high:.1f}，SOS信号确认，多头主导", 88)
                return ("Phase D — SOS强势信号确认",
                        f"SOS出现，价格在区间{range_low:.1f}-{range_high:.1f}内整理，{vol_state}，等待突破", 82)
            if above_range:
                return ("Phase E — 上升趋势中",
                        f"均线多头排列({ma50/ma200-1:+.1%})，沿趋势运行", 82)
            if in_range and has_lps:
                return ("Phase D — LPS最后支撑点",
                        f"缩量回调至支撑附近，最佳入场区域", 85)
            if in_range and has_spring:
                return ("Phase C — Spring弹簧确认",
                        f"支撑位附近探底回升，底部确认", 85)
            if in_range and has_compression:
                return ("Phase B→C — 压缩蓄势",
                        f"波动率收窄+缩量，变盘前夜，{vol_state}", 80)

            # 利用 Accum ABC 子阶段
            if in_range and extra_accum:
                stage_name = {"Accum_A": "A(止跌缩量)", "Accum_B": "B(探底测试)", "Accum_C": "C(最后回踩)"}
                sub = stage_name.get(extra_accum[0], extra_accum[0])
                return (f"Phase A→B — 吸筹{sub}",
                        f"价格在{range_low:.1f}-{range_high:.1f}整理，{vol_state}，{extra_accum[2]}", 72)

            if in_range:
                if range_size < 20:
                    return ("Phase B — 吸筹区间震荡",
                            f"价格在{range_low:.1f}-{range_high:.1f}整理，{vol_state}，蓄力待发", 72)
                return ("Phase A→B — 吸筹筑底期",
                        f"均线走平，{vol_state}，关注区间方向选择", 65)

        elif trend == "空头":
            if below_range:
                return ("Phase E — 下降趋势中",
                        "空头主导，不宜做多，等待底部结构形成", 80)
            if in_range and has_upthrust:
                return ("Phase C — Upthrust上冲回落",
                        "突破阻力后收回，派发特征，警惕进一步下跌", 85)
            if in_range:
                return ("Phase B — 派发区间震荡",
                        f"价格反弹受阻，{vol_state}，注意二次探底风险", 68)
            return ("Phase A — 派发初期",
                    "高位滞涨，供应开始出现，注意趋势转变", 60)

        # 盘整
        return ("Phase B — 区间整理",
                f"均线交织，价格在{range_low:.1f}-{range_high:.1f}区间波动", 55)


# ============================================================
# Scanner
# ============================================================

class Scanner:

    SINA_HQ_URL = "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"

    INDUSTRY_CACHE = os.path.join(os.path.dirname(__file__), "industry_cache.json")

    def __init__(self):
        self.all_stocks = []
        self.snapshot = {}
        self.industry_map = {}
        self.candidates = []
        self.results = []

    # ==================== 行业分类 ====================

    def build_industry_map(self, force=False):
        if not force and os.path.exists(self.INDUSTRY_CACHE):
            try:
                with open(self.INDUSTRY_CACHE, encoding="utf-8") as f:
                    self.industry_map = json.load(f)
                return
            except Exception: pass

        import baostock as bs
        bs.login()
        try:
            rs = bs.query_stock_industry()
            while rs.next():
                row = rs.get_row_data()
                if len(row) >= 4 and row[3]:
                    self.industry_map[row[1]] = row[3]
        finally:
            bs.logout()

        try:
            with open(self.INDUSTRY_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.industry_map, f, ensure_ascii=False)
        except Exception: pass

    def get_industry(self, code):
        return self.industry_map.get(code, "其他")

    # ==================== 获取全市场数据 ====================

    @staticmethod
    def _fetch_page(page, num=100):
        params = {"page": page, "num": num, "sort": "symbol",
                  "asc": "1", "node": "hs_a", "_s_r_a": "init"}
        r = requests.get(Scanner.SINA_HQ_URL, params=params, timeout=20)
        return r.json()

    def fetch_all_stocks(self, max_pages=80):
        all_data = []
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = {pool.submit(self._fetch_page, p): p for p in range(1, max_pages + 1)}
            for f in as_completed(futures):
                try:
                    data = f.result()
                    if data:
                        all_data.extend(data)
                except Exception: pass

        stocks, snapshot = [], {}
        for item in all_data:
            code = item.get("code", "")
            name = item.get("name", "")
            symbol = item.get("symbol", "")

            if symbol.startswith("bj") or code.startswith("9"):
                continue

            bs_code = f"{symbol[:2]}.{symbol[2:]}" if len(symbol) >= 4 else f"sz.{code}"
            try:
                price = float(item.get("trade", 0))
                amount = float(item.get("amount", 0))
                volume = float(item.get("volume", 0))
                turnover = float(item.get("turnoverratio", 0))
                high = float(item.get("high", 0))
                low = float(item.get("low", 0))
                change_pct = float(item.get("changepercent", 0))
            except (ValueError, TypeError):
                continue

            if price <= 0 or amount <= 0:
                continue

            stocks.append({"code": bs_code, "name": name})
            snapshot[bs_code] = {
                "code": bs_code, "name": name, "price": price,
                "amount": amount, "volume": volume, "turnover": turnover,
                "high": high, "low": low, "open": float(item.get("open", price)),
                "pre_close": float(item.get("yestodayclose", price / (1 + change_pct/100) if abs(change_pct) > 0 else price)),
                "change_pct": change_pct,
            }

        self.all_stocks = stocks
        self.snapshot = snapshot
        return stocks, snapshot

    # ==================== 过滤 ====================

    def filter_candidates(self, min_amount=5e8):
        candidates = [d for d in self.snapshot.values() if d["amount"] >= min_amount]
        candidates.sort(key=lambda x: x["amount"], reverse=True)
        self.candidates = candidates
        return candidates

    # ==================== 个股分析 ====================

    def _analyze_stock(self, code):
        """完整分析：趋势 + 威科夫形态 + 阶段识别"""
        result = {"trend": "错误", "strength": 0,
                  "wyckoff_sig": "-", "wyckoff_score": 0, "wyckoff_detail": "",
                  "phase": "数据不足", "phase_detail": "",
                  "industry": self.get_industry(code)}

        try:
            key = code.replace(".", "")
            url = self.TENCENT_KLINE_URL.format(code=key)
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("code") != 0:
                return result

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

            # 趋势判断
            if len(closes) >= 200:
                ma50 = np.mean(closes[-50:])
                ma200 = np.mean(closes)
                s = (ma50 / ma200 - 1) * 100
                result["trend"] = "多头" if ma50 > ma200 else "空头"
                result["strength"] = round(s, 1)
            elif len(closes) >= 50:
                result["trend"] = "数据不足"
            else:
                result["trend"] = "新股"

            # 威科夫形态分析
            signals = []
            extra_info = {}
            if len(closes) >= 30:
                signals, extra_info = WyckoffAnalyzer.analyze_all(
                    closes, highs, lows, volumes, trend=result["trend"]
                )
                if signals and signals[0][1] > 0:
                    result["wyckoff_sig"] = signals[0][0]
                    result["wyckoff_score"] = signals[0][1]
                    result["wyckoff_detail"] = signals[0][2]

            # 威科夫阶段识别
            if len(closes) >= 50:
                phase, phase_detail, _ = WyckoffAnalyzer.detect_phase(
                    closes, highs, lows, volumes, result["trend"], signals, extra_info
                )
                result["phase"] = phase
                result["phase_detail"] = phase_detail

        except Exception:
            pass

        return result

    def run_analysis(self, max_stocks=80):
        results = []
        total = min(len(self.candidates), max_stocks)

        for i, stock in enumerate(self.candidates[:max_stocks]):
            analysis = self._analyze_stock(stock["code"])
            results.append({**stock, **analysis})
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{total}")
            time.sleep(0.02)

        # 排序：威科夫信号得分 > 多头 > 成交额
        def sort_key(x):
            ws = -x["wyckoff_score"]  # 高分在前
            tr = 0 if x["trend"] == "多头" else 1
            return (ws, tr, -x["amount"])

        results.sort(key=sort_key)
        self.results = results
        return results

    # ==================== 报告 ====================

    def print_report(self, top_n=15):
        bullish = sum(1 for r in self.results if r["trend"] == "多头")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 信号统计
        sig_counts = {}
        for r in self.results:
            t = r["wyckoff_sig"]
            if t not in ("-", "无信号", "数据不足"):
                sig_counts[t] = sig_counts.get(t, 0) + 1

        print("\n" + "=" * 80)
        print(f"  A股扫描报告  {now}")
        print("=" * 80)
        print(f"  覆盖: {len(self.all_stocks)} 只  流动性通过: {len(self.candidates)} 只")
        print(f"  多头: {bullish}只  ", end="")
        for k, v in sig_counts.items():
            print(f" {k}:{v}只 ", end="")
        print("\n")

        if not self.results:
            print("  无符合条件标的")
            return

        # === 板块分布 ===
        sectors = {}
        for r in self.results:
            ind = r.get("industry", "其他")
            if ind not in sectors:
                sectors[ind] = {"count": 0, "sig_count": 0, "names": []}
            sectors[ind]["count"] += 1
            if r["wyckoff_sig"] not in ("-", "无信号", "数据不足"):
                sectors[ind]["sig_count"] += 1
            if len(sectors[ind]["names"]) < 3:
                sectors[ind]["names"].append(r["name"])

        sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["count"])
        print(f"  【板块分布】")
        for ind, info in sorted_sec[:12]:
            tag = f" 🌱{info['sig_count']}信号" if info['sig_count'] > 0 else ""
            top = ", ".join(info["names"])
            print(f"  {ind:<18} {info['count']:>2}只{tag:<12} | {top}")
        print()

        # === Top N ===
        print(f"  [Top {top_n}]")
        print(f"  {'#':<3} {'代码':<8} {'名称':<7} {'板块':<12} {'趋势':<10} {'威科夫':<18} {'价格':<8} {'成交额':<8}")
        print(f"  " + "-" * 96)

        for i, r in enumerate(self.results[:top_n]):
            sym = r["code"].split(".")[1]
            name = r["name"][:6]
            ind = r.get("industry", "其他")[:10]
            trend = f"{r['trend']}({r['strength']:+.1f}%)" if r["trend"] in ("多头", "空头") else r["trend"]

            sig = r["wyckoff_sig"]
            sc = r["wyckoff_score"]
            sig_str = f"{sig}({sc})" if sig not in ("-", "无信号") else sig

            price = f"{r['price']:.2f}"
            amt = f"{r['amount']/1e8:.1f}亿"
            print(f"  {i+1:<3} {sym:<8} {name:<7} {ind:<12} {trend:<10} {sig_str:<18} {price:<8} {amt:<8}")

        print()

        # === 信号精选 ===
        valid = [r for r in self.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
        if valid:
            print(f"  【威科夫信号精选 Top {min(10, len(valid))}】")
            print(f"  {'信号':<14} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'细节':<32}")
            print(f"  " + "-" * 88)
            for r in valid[:10]:
                sym = r["code"].split(".")[1]
                name = r["name"][:6]
                ind = r.get("industry", "其他")[:10]
                sig = r["wyckoff_sig"]
                sc = r["wyckoff_score"]
                detail = r.get("wyckoff_detail", "")[:30]
                print(f"  {sig:<14} {sym:<8} {name:<7} {ind:<12} {sc:<5} {detail:<32}")
            print()

    # ==================== 主流程 ====================

    def run(self, min_amount=5e8, max_analysis=80, top_n=15, quick=False):
        print("A股扫描器 v2.1（威科夫形态检测）")
        print("=" * 62)
        print(f"成交额门槛: {min_amount/1e8:.0f}亿\n")

        print("[1/3] 获取全市场数据 (新浪)...", end=" ")
        t0 = time.time()
        self.fetch_all_stocks()
        print(f"{len(self.snapshot)} 只有效 ({time.time()-t0:.1f}s)")

        print("[2/3] 流动性过滤...", end=" ")
        t0 = time.time()
        self.filter_candidates(min_amount)
        print(f"{len(self.candidates)} 只 ({time.time()-t0:.1f}s)")

        print("[2.5/3] 行业分类...", end=" ")
        t0 = time.time()
        self.build_industry_map()
        print(f"{len(self.industry_map)} 只映射 ({time.time()-t0:.1f}s)")

        if quick:
            self.results = []
            for s in self.candidates[:max_analysis]:
                self.results.append({
                    **s, "trend": "-", "strength": 0,
                    "wyckoff_sig": "-", "wyckoff_score": 0, "wyckoff_detail": "",
                    "industry": self.get_industry(s["code"]),
                })
        else:
            print("[3/3] 威科夫+趋势分析...")
            t0 = time.time()
            self.run_analysis(max_analysis)
            print(f"  完成 ({time.time()-t0:.1f}s)")

        self.print_report(top_n)


def run_scanner(min_amount=5e8, quick=False):
    Scanner().run(min_amount=min_amount, quick=quick)


if __name__ == "__main__":
    min_amt = 5e8
    quick = False
    for arg in sys.argv[1:]:
        if arg.replace(".", "").isdigit():
            min_amt = float(arg) * 1e8
        elif arg == "--quick":
            quick = True
    run_scanner(min_amt, quick)
