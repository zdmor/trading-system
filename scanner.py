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
import yaml
import warnings
warnings.filterwarnings("ignore")

from data_providers import AkshareProvider
from data_providers import get_financial_indicators, get_stock_quality, get_sector_heat, get_industry_map as dp_get_industry_map
from data_providers.tushare_provider import TushareProvider

try:
    from research_watchlist import ResearchTracker
    _RESEARCH_TOOL = True
except ImportError:
    _RESEARCH_TOOL = False

try:
    from recommendation_tracker import save_scan_result, print_review
    _REC_TRACKER = True
except ImportError:
    _REC_TRACKER = False

from pattern_detector import PatternDetector
from notifier import send_card, make_div, make_hr, make_note

try:
    from lhb_analyzer import enrich_stock_list, analyze_stock
    _LHB_TOOL = True
except ImportError:
    _LHB_TOOL = False

try:
    from sentiment_indicator import get_sentiment, get_daily_sentiment, get_weekly_sentiment, get_monthly_sentiment
    _SENTIMENT_TOOL = True
except ImportError:
    _SENTIMENT_TOOL = False

try:
    from buzz_monitor import scan_overheat, get_buzz_growth, get_stock_buzz
    _BUZZ_TOOL = True
except ImportError:
    _BUZZ_TOOL = False


# ============================================================
# 配置加载
# ============================================================

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

def _load_yaml(name):
    path = os.path.join(_CONFIG_DIR, name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

# 全局缓存，避免重复读文件
_PATTERNS_CFG = None
_SCANNER_CFG = None

def get_patterns_config():
    global _PATTERNS_CFG
    if _PATTERNS_CFG is None:
        _PATTERNS_CFG = _load_yaml("patterns.yaml")
    return _PATTERNS_CFG

def get_scanner_config():
    global _SCANNER_CFG
    if _SCANNER_CFG is None:
        _SCANNER_CFG = _load_yaml("scanner.yaml")
    return _SCANNER_CFG


# ============================================================
# 威科夫形态分析器
# ============================================================

class WyckoffAnalyzer:
    """检测多种威科夫交易形态"""

    _cfg_cache = {}

    @classmethod
    def _cfg(cls, section):
        """读取 patterns.yaml 中指定段落的配置"""
        if section not in cls._cfg_cache:
            cls._cfg_cache[section] = get_patterns_config().get(section, {})
        return cls._cfg_cache[section]

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
        """弹簧：价格跌破近期支撑，快速收回，伴随放量"""
        cfg = cls._cfg("spring")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头", "错误"]):
            return None
        sw = cfg.get("support_window", [-12, -3])
        pr = cfg.get("price_range", [0.95, 1.15])
        vbw = cfg.get("volume_baseline_window", [-15, -3])
        dw = cfg.get("detect_window", [-3, None])
        thr = cfg.get("thresholds", {})

        support = float(min(lows[sw[0]:sw[1]])) if len(lows) >= abs(sw[0]) else 1
        cur = float(closes[-1])
        if cur > support * pr[1] or cur < support * pr[0]:
            return None

        bg_vol = cls._avg(volumes[vbw[0]:vbw[1]])
        lookback = lows[dw[0]:dw[1]]
        closes_lb = closes[dw[0]:dw[1]]

        scoring = cfg.get("scoring", {})
        depth_cfg = scoring.get("depth", [])
        vol_cfg = scoring.get("volume_ratio", [])
        bounce_cfg = scoring.get("bounce", [])
        nd_confirm = scoring.get("next_day_confirm", 15)

        # ---- start_zone_check: 跌势起点位置（全局，循环外计算一次）-----
        zone_bonus = 0
        zone_label = ""
        if len(closes) >= 11 and len(highs) >= 10:
            start_price = float(closes[-11])   # 10日前收盘价 = 跌势起点
            recent_10d_high = float(max(highs[-10:]))
            zone_range = recent_10d_high - support
            if zone_range > 0:
                start_pct = (start_price - support) / zone_range * 100
                if start_pct >= 30:
                    zone_bonus = 5
                    zone_label = f"起区{start_pct:.0f}%"
                elif start_pct < 20:
                    zone_bonus = -10
                    zone_label = f"近撑{start_pct:.0f}%"

        best = None
        for i in range(len(lookback)):
            low = float(lookback[i])
            close = float(closes_lb[i])
            vol = float(volumes[-3:][i]) if len(volumes) >= 3 else 1

            if low >= support * 0.997 or close <= support:
                continue

            depth = (support - low) / support * 100
            vratio = vol / bg_vol if bg_vol > 0 else 1
            bounce = (close - low) / low * 100

            score = 0
            parts = []
            for d in depth_cfg:
                if depth >= d[0]: score += d[1]; parts.append(d[2].format(depth=depth)); break
            for v in vol_cfg:
                if vratio >= v[0]: score += v[1]; parts.append(v[2]); break
            for b in bounce_cfg:
                if bounce >= b[0]: score += b[1]; parts.append(b[2].format(bounce=bounce)); break
            if i + 1 < len(closes_lb) and float(closes_lb[i+1]) > close:
                score += nd_confirm; parts.append("确认")

            # 叠加 start_zone 评分
            if zone_bonus != 0:
                score += zone_bonus
                parts.append(zone_label)

            if score > (best[1] if best else 0):
                sig = "Spring" if score >= thr.get("strong_signal", 55) else thr.get("weak_label", "弱Spring")
                best = (sig, score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # SOS Sign of Strength 强势信号
    # -------------------------------------------------------
    @classmethod
    def detect_sos(cls, closes, highs, lows, volumes, trend):
        """强势信号：大阳线 + 放量 + 高位收盘"""
        cfg = cls._cfg("sos")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头"]):
            return None
        if len(closes) < cfg.get("min_bars", 50):
            return None

        # 位阶保护
        mp = cfg.get("ma200_bias_protection", {})
        if mp.get("enabled", True) and len(closes) >= 200:
            ma50 = float(np.mean(closes[-50:]))
            ma200 = float(np.mean(closes))
            if ma200 > 0 and (ma50 - ma200) / ma200 * 100 > mp.get("max_bias_pct", 25):
                return None

        vbw = cfg.get("volume_baseline_window", [-20, -1])
        rbw = cfg.get("range_baseline_window", [-20, -1])
        scoring = cfg.get("scoring", {})
        pc_cfg = scoring.get("price_pct", [])
        vr_cfg = scoring.get("volume_ratio", [])
        cp_cfg = scoring.get("close_position", [])
        rr_cfg = scoring.get("range_ratio", [])
        min_score = cfg.get("thresholds", {}).get("min_score", 50)

        today_o = float(closes[-2])
        today_c = float(closes[-1])
        today_h = float(highs[-1])
        today_l = float(lows[-1])
        today_v = float(volumes[-1])

        if today_c <= today_o:
            return None

        bg_vol = cls._avg(volumes[vbw[0]:vbw[1]]) if vbw[1] else cls._avg(volumes[vbw[0]:])
        bg_range = cls._avg([highs[i] - lows[i] for i in range(rbw[0], rbw[1])])
        total_range = today_h - today_l
        if total_range < 0.01:
            return None

        pos_in_range = (today_c - today_l) / total_range
        vratio = today_v / bg_vol if bg_vol > 0 else 1
        range_ratio = total_range / bg_range if bg_range > 0 else 0
        pct = (today_c - today_o) / today_o * 100

        score = 0
        parts = []
        matched = False
        for d in pc_cfg:
            if pct >= d[0]: score += d[1]; parts.append(d[2].format(pct=pct)); matched = True; break
        if not matched:
            return None
        for v in vr_cfg:
            if vratio >= v[0]: score += v[1]; parts.append(v[2]); break
        for c in cp_cfg:
            if pos_in_range >= c[0]: score += c[1]; parts.append(c[2]); break
        for r in rr_cfg:
            if range_ratio >= r[0]: score += r[1]; parts.append(r[2]); break
        if len(closes) >= 50 and today_c > np.mean(closes[-50:]):
            score += scoring.get("above_ma50", 10); parts.append("趋势上")

        return ("SOS", score, " | ".join(parts)) if score >= min_score else None

    # -------------------------------------------------------
    # LPS Last Point of Support 最后支撑点
    # -------------------------------------------------------
    @classmethod
    def detect_lps(cls, closes, highs, lows, volumes, trend):
        """最后支撑点：放量上攻后，缩量回调至支撑附近"""
        cfg = cls._cfg("lps")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头"]):
            return None
        if len(closes) < cfg.get("min_bars", 30):
            return None

        sw = cfg.get("support_window", [-12, -3])
        pr = cfg.get("price_range", [1.0, 1.05])
        surge_win = cfg.get("surge_window", [-10, -3])
        surge_bw = cfg.get("surge_baseline_window", [-25, -10])
        surge_vr = cfg.get("surge_volume_ratio", 1.3)
        surge_range = cfg.get("surge_confirm_range", 5)
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})

        support = float(min(lows[sw[0]:sw[1]]))
        cur = float(closes[-1])
        if not (support * pr[0] <= cur <= support * pr[1]):
            return None

        # 前置放量上涨检测
        bg_vol = cls._avg(volumes[surge_bw[0]:surge_bw[1]])
        had_upsurge = any(
            float(volumes[surge_win[0]:surge_win[1]][i]) > bg_vol * surge_vr
            and float(closes[surge_win[0]:surge_win[1]][i]) > float(closes[surge_win[0]-1:surge_win[1]-1][i])
            for i in range(min(surge_range, len(volumes[surge_win[0]:surge_win[1]])))
        ) if bg_vol > 0 else False
        if not had_upsurge:
            return None

        today_v = float(volumes[-1])
        vratio = today_v / bg_vol if bg_vol > 0 else 1
        vr_cfg = scoring.get("volume_ratio", [])

        score = thr.get("base_score", 30)
        parts = []
        for v in vr_cfg:
            if vratio <= v[0]:
                score += v[1]
                parts.append(v[2])
                break
        if not parts:
            parts.append("量平")

        body = abs(float(closes[-1]) - float(closes[-2]))
        avg_body = cls._avg([abs(closes[i] - closes[i-1]) for i in range(-10, -1)])
        br_cfg = scoring.get("body_ratio", [])
        for b in br_cfg:
            if body < avg_body * b[0]:
                score += b[1]; parts.append(b[2]); break

        st_win = scoring.get("support_tested_window", [-15, 0])
        if any(float(lows[i]) < support and float(closes[i]) > support for i in range(st_win[0], st_win[1] or 0)):
            score += scoring.get("support_tested_score", 15); parts.append("支撑验证")

        rn = scoring.get("range_narrowing", {})
        if rn:
            rnw = rn.get("window", [-10, -5, -5, 0])
            recent_r = float(max(highs[rnw[2]:rnw[3]])) - float(min(lows[rnw[2]:rnw[3]]))
            older_r = float(max(highs[rnw[0]:rnw[1]])) - float(min(lows[rnw[0]:rnw[1]]))
            if older_r > 0 and recent_r < older_r * rn.get("ratio_threshold", 0.7):
                score += rn.get("score", 10); parts.append("波幅收窄")

        detail = " | ".join(parts) if parts else ""
        sig = "LPS" if score >= thr.get("strong_signal", 60) else thr.get("weak_label", "弱LPS")
        return (sig, score, detail) if score >= thr.get("min_score", 40) else None

    # -------------------------------------------------------
    # Upthrust 上冲回落（UT/UTAD）
    # -------------------------------------------------------
    @classmethod
    def detect_upthrust(cls, closes, highs, lows, volumes, trend):
        """上冲回落：价格突破阻力后迅速收回，放量"""
        cfg = cls._cfg("upthrust")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头"]):
            return None
        if len(closes) < cfg.get("min_bars", 20):
            return None

        rw = cfg.get("resistance_window", [-12, -3])
        pr = cfg.get("price_range", [0.97, 1.03])
        vbw = cfg.get("volume_baseline_window", [-15, -3])
        dw = cfg.get("detect_window", [-3, None])
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})

        resistance = float(max(highs[rw[0]:rw[1]]))
        if resistance <= 0:
            return None

        cur = float(closes[-1])
        if not (resistance * pr[0] <= cur <= resistance * pr[1]):
            return None

        bg_vol = cls._avg(volumes[vbw[0]:vbw[1]])
        lookback_h = highs[dw[0]:dw[1]]
        lookback_c = closes[dw[0]:dw[1]]
        lookback_l = lows[dw[0]:dw[1]]

        thrust_cfg = scoring.get("thrust", [])
        vr_cfg = scoring.get("volume_ratio", [])
        cp_cfg = scoring.get("close_position", [])
        uw_ratio = scoring.get("upper_wick_ratio", 0.4)
        uw_score = scoring.get("upper_wick_score", 15)
        nd_score = scoring.get("next_day_confirm", 10)
        base = scoring.get("base_score", 30)
        strong = thr.get("strong_signal", 55)
        weak = thr.get("weak_label", "弱Upthrust")

        best = None
        for i in range(len(lookback_h)):
            high = float(lookback_h[i])
            close = float(lookback_c[i])
            low = float(lookback_l[i])
            vol = float(volumes[dw[0]:dw[1]][i]) if len(volumes) >= abs(dw[0]) else 1

            if high < resistance:
                continue
            if close > resistance * 1.01:
                continue

            vratio = vol / bg_vol if bg_vol > 0 else 1
            total_range = high - low
            pos_in_range = (close - low) / total_range if total_range > 0 else 0.5
            thrust = (high - resistance) / resistance * 100

            score = base
            parts = []
            for d in thrust_cfg:
                if thrust >= d[0]: score += d[1]; parts.append(d[2].format(thrust=thrust)); break
            for v in vr_cfg:
                if vratio >= v[0]: score += v[1]; parts.append(v[2]); break
            for c in cp_cfg:
                if pos_in_range <= c[0]: score += c[1]; parts.append(c[2]); break
            upper_wick = high - max(close, float(lookback_c[i-1]) if i > 0 else close)
            if upper_wick > total_range * uw_ratio:
                score += uw_score; parts.append("长上影")
            if i + 1 < len(lookback_c) and float(lookback_c[i+1]) < close:
                score += nd_score; parts.append("次日跌")

            if score > (best[1] if best else 0):
                sig = "Upthrust" if score >= strong else weak
                best = (sig, score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # EVR Effort vs Result 努力无结果
    # -------------------------------------------------------
    @classmethod
    def detect_evr(cls, closes, highs, lows, volumes, trend):
        """努力无结果：放量但价格窄幅波动"""
        cfg = cls._cfg("evr")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头", "错误"]):
            return None
        if len(closes) < cfg.get("min_bars", 30):
            return None

        mp = cfg.get("ma200_bias_protection", {})
        if mp.get("enabled", True) and len(closes) >= 200:
            ma200 = float(np.mean(closes))
            if ma200 > 0 and (float(closes[-1]) - ma200) / ma200 > mp.get("max_bias_pct", 30) / 100:
                return None

        vbw = cfg.get("volume_baseline_window", [-20, -1])
        dw = cfg.get("detect_window", [-3, 0])
        min_vr = cfg.get("min_volume_ratio", 1.5)
        max_pc = cfg.get("max_price_change", 2.5)
        scoring = cfg.get("scoring", {})
        vr_cfg = scoring.get("volume_ratio", [])
        rn = scoring.get("range_narrowing", {})

        bg_vol = cls._avg(volumes[vbw[0]:vbw[1]])
        best = None

        for idx in range(dw[0], dw[1] or 0):
            vol = float(volumes[idx])
            vratio = vol / bg_vol if bg_vol > 0 else 0
            if vratio < min_vr:
                continue
            pct = (float(closes[idx]) - float(closes[idx-1])) / float(closes[idx-1]) * 100
            if abs(pct) > max_pc:
                continue

            score = scoring.get("base_score", 50)
            parts = []
            for v in vr_cfg:
                if vratio >= v[0]: score += v[1]; parts.append(v[2].format(vratio=vratio)); break
            parts.append("抗跌" if pct > 0 else "滞涨")

            if rn:
                rnw = rn.get("window", [-20, -1])
                k_range = float(highs[idx]) - float(lows[idx])
                avg_r = cls._avg([highs[j] - lows[j] for j in range(rnw[0], rnw[1])])
                if avg_r > 0 and k_range < avg_r * rn.get("ratio_threshold", 0.7):
                    score += rn.get("score", 15); parts.append("窄幅")
            if idx == -1 and float(closes[-1]) >= float(closes[-2]) * scoring.get("confirm_threshold", 0.99):
                score += scoring.get("confirm_score", 10); parts.append("确认")

            if score > (best[1] if best else 0):
                best = ("EVR", score, " | ".join(parts))

        return best

    # -------------------------------------------------------
    # Compression 压缩蓄势
    # -------------------------------------------------------
    @classmethod
    def detect_compression(cls, closes, highs, lows, volumes, trend):
        """压缩蓄势：ATR 收窄 + 缩量，变盘前夜"""
        cfg = cls._cfg("compression")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["错误"]):
            return None
        if len(closes) < cfg.get("min_bars", 40):
            return None

        tr = []
        for i in range(1, len(closes)):
            h, l_, pc = float(highs[i]), float(lows[i]), float(closes[i-1])
            tr.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
        if len(tr) < cfg.get("min_tr_bars", 25):
            return None

        rtw = cfg.get("recent_tr_window", 10)
        htw = cfg.get("hist_tr_window", [30, 10])
        vbw = cfg.get("volume_baseline_window", [-30, -10])
        max_cr = cfg.get("max_compress_ratio", 0.75)
        max_vr = cfg.get("max_vol_ratio", 0.8)
        max_viol = cfg.get("max_violations", 2)
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})

        recent_tr = tr[-rtw:]
        hist_tr = tr[-htw[0]:-htw[1]] if len(tr) >= htw[0] else tr[:htw[0]-htw[1]]
        recent_avg = np.mean(recent_tr) / float(closes[-1]) * 100 if recent_tr else 0
        hist_avg = np.mean(hist_tr) / float(closes[-1]) * 100 if hist_tr else 0
        if hist_avg <= 0:
            return None

        compress_ratio = recent_avg / hist_avg
        if compress_ratio > max_cr:
            return None

        bg_vol = cls._avg(volumes[vbw[0]:vbw[1]])
        recent_vol_avg = cls._avg(volumes[-rtw:])
        vol_ratio = recent_vol_avg / bg_vol if bg_vol > 0 else 1
        if vol_ratio > max_vr:
            return None

        violations = sum(1 for j in range(1, len(recent_tr)) if recent_tr[j] > recent_tr[j-1])
        if violations > max_viol:
            return None

        cr_cfg = scoring.get("compress_ratio", [])
        vr_cfg = scoring.get("volume_ratio", [])
        ma50_prox = scoring.get("ma50_proximity", {})
        score = thr.get("base_score", 50)
        parts = []

        for c in cr_cfg:
            if compress_ratio <= c[0]: score += c[1]; parts.append(c[2].format(ratio=compress_ratio)); break
        for v in vr_cfg:
            if vol_ratio <= v[0]: score += v[1]; parts.append(v[2]); break
        if not parts: parts.append("微缩")

        if ma50_prox.get("enabled") and len(closes) >= 50:
            ma50 = float(np.mean(closes[-50:]))
            max_dist = ma50_prox.get("max_distance_pct", 3.0) / 100
            if ma50 > 0 and abs(float(closes[-1]) - ma50) / ma50 < max_dist:
                score += ma50_prox.get("score", 15); parts.append("均线附近")

        return ("Compression", score, " | ".join(parts)) if score >= thr.get("min_score", 60) else None


    # -------------------------------------------------------
    # Markup 阶段确认（MA50上穿MA200）
    # -------------------------------------------------------
    @classmethod
    def detect_markup(cls, closes, highs, lows, volumes, trend):
        """Markup 主升段：MA50 上穿 MA200 且保持在上方 N 日"""
        cfg = cls._cfg("markup")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头"]):
            return None
        if len(closes) < cfg.get("min_bars", 210):
            return None
        if len(closes) < 200:
            return None

        ma50_arr = np.array([np.mean(closes[i-50:i]) for i in range(50, len(closes))])
        ma200_arr = np.array([np.mean(closes[i-200:i]) for i in range(200, len(closes))])
        if len(ma50_arr) < cfg.get("min_ma_bars", 5):
            return None
        if ma50_arr[-1] <= ma200_arr[-1]:
            return None

        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})
        score = scoring.get("base_score", 20)
        parts = ["MA50>MA200"]

        cross_win = cfg.get("crossover_window", 20)
        cd_cfg = scoring.get("crossover_days", [])
        crossover_detected = False
        for j in range(min(cross_win, len(ma50_arr)-1), 0, -1):
            if ma50_arr[-j-1] <= ma200_arr[-j-1] and ma50_arr[-j] > ma200_arr[-j]:
                crossover_detected = True
                days_since = j
                parts.append(f"上穿{days_since}天前")
                for cd in cd_cfg:
                    if days_since <= cd[0]: score += cd[1]; parts.append(cd[2]); break
                break
        if not crossover_detected:
            ts = scoring.get("trend_sustained", {})
            if ma50_arr[-1] > ma200_arr[-1] * (1 + ts.get("ma_gap_pct", 5) / 100):
                score += ts.get("score", 15); parts.append("趋势持续")

        angle_cfg = scoring.get("ma50_angle", [])
        ma50_vals = [np.mean(closes[i-50:i]) for i in range(-10, 0)]
        if len(ma50_vals) >= 2:
            angle = (ma50_vals[-1] - ma50_vals[0]) / ma50_vals[0] * 100
            for a in angle_cfg:
                if angle >= a[0]: score += a[1]; parts.append(a[2]); break

        if float(closes[-1]) > ma50_arr[-1]:
            score += scoring.get("above_ma50", 15); parts.append("价格在MA50上")

        bg_vol = cls._avg(volumes[-50:-10])
        recent_vol = cls._avg(volumes[-10:])
        if bg_vol > 0 and recent_vol > bg_vol * scoring.get("volume_ratio", 1.2):
            score += scoring.get("volume_score", 10); parts.append("量能配合")

        return ("Markup", score, " | ".join(parts)) if score >= thr.get("min_score", 50) else None

    # -------------------------------------------------------
    # 缠论第三类买点 — 执行层信号
    # -------------------------------------------------------
    @classmethod
    def detect_chan_third_buy(cls, closes, highs, lows, volumes, trend):
        """
        缠论第三类买点检测：中枢形成 → 突破上沿 → 回踩不进入

        三买本质：
        趋势已经确立（中枢形成 = 多空力量平衡已被打破），
        第一次回踩不重新进入中枢 = 确认趋势延续的最佳入场点。
        """
        cfg = cls._cfg("chan_third_buy")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头", "错误"]):
            return None
        if len(closes) < cfg.get("min_bars", 60):
            return None

        n = len(closes)

        # ---- 1. 顶底分型识别 ----
        fw = cfg.get("fractal_window", 2)
        tops = []    # [(index, high)]
        bottoms = []  # [(index, low)]

        for i in range(fw, n - fw):
            # 顶分型：中间高 > 两侧各fw根
            if all(highs[i] > highs[i - j - 1] for j in range(fw)) and \
               all(highs[i] >= highs[i + j + 1] for j in range(fw)):
                tops.append((i, highs[i]))
            # 底分型：中间低 < 两侧各fw根
            if all(lows[i] < lows[i - j - 1] for j in range(fw)) and \
               all(lows[i] <= lows[i + j + 1] for j in range(fw)):
                bottoms.append((i, lows[i]))

        if len(tops) < 3 or len(bottoms) < 3:
            return None

        # ---- 2. 构建交替笔序列 ----
        pts = []
        ti, bi = 0, 0

        # 谁先出现谁开头
        if tops[0][0] < bottoms[0][0]:
            pts.append(("top", tops[0][0], tops[0][1]))
            ti = 1
        else:
            pts.append(("bottom", bottoms[0][0], bottoms[0][1]))
            bi = 1

        while ti < len(tops) and bi < len(bottoms):
            last = pts[-1]
            if last[0] == "top":
                if bottoms[bi][0] > last[1]:
                    pts.append(("bottom", bottoms[bi][0], bottoms[bi][1]))
                bi += 1
            else:
                if tops[ti][0] > last[1]:
                    pts.append(("top", tops[ti][0], tops[ti][1]))
                ti += 1

        # 需要至少6个点（3笔）才能构成中枢
        if len(pts) < 6:
            return None

        # ---- 3. 寻找最近的中枢（3笔重叠区域） ----
        max_ll = cfg.get("max_breakout_lookback", 60)
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})
        max_rt_dist = cfg.get("max_retest_distance_pct", 5.0)

        # 从最近的潜在中枢开始搜索
        for seg_start in range(len(pts) - 4, max(0, len(pts) - 10), -2):
            if seg_start < 0 or seg_start + 4 > len(pts):
                continue

            p0, p1, p2, p3 = pts[seg_start:seg_start + 4]

            # 三笔的价格区间
            s1_h = max(highs[p0[1]:p1[1] + 1])
            s1_l = min(lows[p0[1]:p1[1] + 1])
            s2_h = max(highs[p1[1]:p2[1] + 1])
            s2_l = min(lows[p1[1]:p2[1] + 1])
            s3_h = max(highs[p2[1]:p3[1] + 1])
            s3_l = min(lows[p2[1]:p3[1] + 1])

            # 中枢重叠区间：三笔的max低点 < min高点（有交集）
            pivot_top = min(s1_h, s2_h, s3_h)
            pivot_bot = max(s1_l, s2_l, s3_l)

            if pivot_bot >= pivot_top:
                continue  # 无重叠，不是中枢

            pivot_end_idx = p3[1]  # 中枢完成位置

            # ---- 4. 突破检测 ----
            breakout_idx = None
            search_end = min(n, pivot_end_idx + max_ll)
            for i in range(pivot_end_idx, search_end):
                if highs[i] > pivot_top:
                    breakout_idx = i
                    break

            if breakout_idx is None:
                continue

            # ---- 5. 回踩检测 ----
            retest_low = None
            retest_idx = None
            max_allowed = pivot_top * (1 + max_rt_dist / 100)
            min_allowed = pivot_top  # 不能回到中枢内

            for i in range(breakout_idx, n):
                l = lows[i]
                if l > min_allowed:
                    # 回踩到接近中枢上沿但没进去
                    if l <= max_allowed:
                        retest_low = l
                        retest_idx = i
                        break
                else:
                    # 重新进入中枢，信号失效
                    break

            if retest_low is None:
                continue

            # ---- 6. 评分 ----
            rt_dist = (retest_low - pivot_top) / pivot_top * 100
            score = scoring.get("base_score", 60)
            parts = []

            # 回踩质量
            for r in scoring.get("retest_categories", []):
                if rt_dist <= r[0]:
                    score += r[1]
                    if len(r) > 2 and r[2]:
                        parts.append(r[2])
                    break

            # 多头趋势确认
            if trend == "多头":
                score += scoring.get("bull_trend_bonus", 10)
                parts.append("多头趋势")

            # 突破放量
            if breakout_idx < len(volumes):
                bk_vol = volumes[breakout_idx]
                bg = closes[-20:-1] if n >= 20 else closes[:-1]
                avg_vol = cls._avg(volumes[-min(20, len(volumes)-1):-1]) if n >= 5 else cls._avg(volumes)
                if avg_vol > 0 and bk_vol > avg_vol * scoring.get("breakout_volume_ratio", 1.5):
                    score += scoring.get("breakout_volume_score", 10)
                    parts.append("放量突破")

            # 回踩后回升确认
            if closes[-1] > pivot_top and retest_idx < n - 1:
                if closes[-1] > closes[retest_idx]:
                    score += scoring.get("recovery_score", 10)
                    parts.append("回升确认")

            parts.append(f"三买{rt_dist:.1f}%")
            sig = "三买" if score >= thr.get("strong_signal", 70) else thr.get("weak_label", "弱三买")
            return (sig, score, " | ".join(parts))

        return None

    # -------------------------------------------------------
    # Accumulation ABC 子阶段
    # -------------------------------------------------------
    @classmethod
    def detect_accum_stage(cls, closes, highs, lows, volumes, trend):
        """吸筹期子阶段细分：A 止跌 / B 探底 / C 回踩"""
        cfg = cls._cfg("accumulation")
        if not cfg.get("enabled", True) or trend in cfg.get("exclude_trends", ["空头", "错误"]):
            return None
        if len(closes) < cfg.get("min_bars", 60):
            return None

        cur = float(closes[-1])
        low_60d = float(min(lows[-60:]))
        low_250d = float(min(lows[-250:])) if len(lows) >= 250 else low_60d
        max_price_pct = cfg.get("max_price_from_low_pct", 35)
        if low_250d > 0 and cur > low_250d * (1 + max_price_pct / 100):
            return None

        accum_base_low = low_250d
        max_ma_gap = cfg.get("max_ma_gap_pct", 8)
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else 0
        ma200 = float(np.mean(closes)) if len(closes) >= 200 else 0
        if ma200 > 0:
            ma_gap = abs(ma50 - ma200) / ma200 * 100
            if ma_gap > max_ma_gap:
                return None

        vol_r = cfg.get("vol_window", [20, 120])
        vol_recent = cls._avg(volumes[-vol_r[0]:])
        vol_hist = cls._avg(volumes[-vol_r[1]:-vol_r[0]])
        if vol_hist > 0 and vol_recent / vol_hist > cfg.get("max_vol_ratio", 0.7):
            return None

        scoring = cfg.get("scoring", {})
        score = scoring.get("base_score", 40)
        parts = []

        b_cfg = scoring.get("b_stage", {})
        tw = b_cfg.get("test_window", 30)
        tt = b_cfg.get("test_threshold_pct", 5) / 100
        tc = b_cfg.get("test_count_threshold", 3)
        zone_lows = lows[-tw:]
        test_count = sum(1 for l in zone_lows if accum_base_low > 0 and abs(l - accum_base_low) / accum_base_low <= tt)

        if test_count >= tc:
            stage = b_cfg.get("label", "Accum_B")
            score += b_cfg.get("score", 30)
            parts.append(b_cfg.get("desc", "多次探底({count}次)").format(count=test_count))
        else:
            c_cfg = scoring.get("c_stage", {})
            recent_low = float(min(lows[-c_cfg.get("window", 20):]))
            c_ok = accum_base_low > 0 and recent_low >= accum_base_low * c_cfg.get("min_from_base_pct", 97) / 100
            if c_ok:
                vw = c_cfg.get("vol_window", [60, 20])
                vol_dry = cls._avg(volumes[-vw[1]:])
                vol_hist_c = cls._avg(volumes[-vw[0]:-vw[1]])
                if vol_hist_c > 0 and vol_dry / vol_hist_c < c_cfg.get("max_vol_ratio", 0.6):
                    stage = c_cfg.get("label", "Accum_C")
                    score += c_cfg.get("score", 25); parts.append(c_cfg.get("desc", "缩量回踩不破底"))
                else:
                    a_cfg = scoring.get("a_stage", {})
                    stage = a_cfg.get("label", "Accum_A")
                    score += a_cfg.get("score", 15); parts.append(a_cfg.get("desc", "止跌缩量"))
            else:
                a_cfg = scoring.get("a_stage", {})
                stage = a_cfg.get("label", "Accum_A")
                score += a_cfg.get("score", 15); parts.append(a_cfg.get("desc", "止跌缩量"))

        if ma200 > 0 and ma_gap < cfg.get("tight_ma_gap_pct", 4):
            score += scoring.get("ma_tight_score", 10); parts.append("均线粘合")
        if vol_hist > 0 and vol_recent / vol_hist < cfg.get("dry_vol_ratio", 0.4):
            score += scoring.get("dry_vol_score", 10); parts.append("地量")

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
        cf = cls._cfg("phase")
        min_bars = cf.get("min_bars", 50)
        if extra is None:
            extra = {}
        if len(closes) < min_bars:
            return ("数据不足", "", 0)

        cur = float(closes[-1])
        ma50 = float(np.mean(closes[-50:]))
        ma200 = float(np.mean(closes)) if len(closes) >= 200 else 0
        rw = cf.get("range_window", 60)
        bt = cf.get("breakout_threshold", 1.0) / 100

        range_high = float(max(highs[-rw:]))
        range_low = float(min(lows[-rw:]))
        above_range = cur > range_high * (1 + bt)
        below_range = cur < range_low * (1 - bt)
        in_range = not above_range and not below_range

        vs = cf.get("vol_state", {})
        avg_vol_50 = float(np.mean(volumes[-50:])) if len(volumes) >= 50 else 1
        avg_vol_10 = float(np.mean(volumes[-10:])) if len(volumes) >= 10 else 1
        if avg_vol_10 < avg_vol_50 * vs.get("shrink_ratio", 0.7):
            vol_state = "缩量"
        elif avg_vol_10 > avg_vol_50 * vs.get("expand_ratio", 1.3):
            vol_state = "放量"
        else:
            vol_state = "量平"

        et = set(e[0] for e in events if e[1] > 0)
        has_sos      = "SOS" in et
        has_lps      = "LPS" in et
        has_spring   = any("Spring" in e for e in et)
        has_upthrust = "Upthrust" in et
        has_evr      = "EVR" in et
        has_compression = "Compression" in et
        extra_accum = extra.get("accum_stage")
        extra_markup = extra.get("markup")
        conf = cf.get("confidence", {})
        pb = cf.get("phase_b", {})
        range_size = (range_high - range_low) / range_low * 100 if range_low > 0 else 0

        if trend == "多头":
            if extra_markup:
                return (f"Phase D→E — Markup主升段",
                        f"MA50上穿MA200确认，趋势强度{extra_markup[1]}，{vol_state}，多头主导",
                        conf.get("markup", 88))
            if has_sos:
                label = "Phase D→E — 突破确认/上升趋势" if above_range else "Phase D — SOS强势信号确认"
                detail = (f"价格突破区间{range_low:.1f}-{range_high:.1f}，SOS信号确认，多头主导"
                          if above_range else
                          f"SOS出现，价格在区间{range_low:.1f}-{range_high:.1f}内整理，{vol_state}，等待突破")
                return (label, detail, conf.get("sos_breakout" if above_range else "sos_in_range", 85))
            if above_range:
                return ("Phase E — 上升趋势中",
                        f"均线多头排列({ma50/ma200-1:+.1%})，沿趋势运行" if ma200 > 0 else "多头趋势运行",
                        conf.get("trend_up", 82))
            if in_range and has_lps:
                return ("Phase D — LPS最后支撑点", "缩量回调至支撑附近，最佳入场区域", conf.get("lps", 85))
            if in_range and has_spring:
                return ("Phase C — Spring弹簧确认", "支撑位附近探底回升，底部确认", conf.get("spring", 85))
            if in_range and has_compression:
                return ("Phase B→C — 压缩蓄势",
                        f"波动率收窄+缩量，变盘前夜，{vol_state}", conf.get("compression", 80))
            if in_range and extra_accum:
                sn = {"Accum_A": "A(止跌缩量)", "Accum_B": "B(探底测试)", "Accum_C": "C(最后回踩)"}
                return (f"Phase A→B — 吸筹{sn.get(extra_accum[0], extra_accum[0])}",
                        f"价格在{range_low:.1f}-{range_high:.1f}整理，{vol_state}，{extra_accum[2]}",
                        conf.get("accum", 72))
            if in_range:
                if range_size < pb.get("max_range_size_pct", 20):
                    return ("Phase B — 吸筹区间震荡",
                            f"价格在{range_low:.1f}-{range_high:.1f}整理，{vol_state}，蓄力待发",
                            conf.get("phase_b", 72))
                return ("Phase A→B — 吸筹筑底期",
                        f"均线走平，{vol_state}，关注区间方向选择", conf.get("phase_b_build", 65))

        elif trend == "空头":
            if below_range:
                return ("Phase E — 下降趋势中", "空头主导，不宜做多，等待底部结构形成", conf.get("downtrend", 80))
            if in_range and has_upthrust:
                return ("Phase C — Upthrust上冲回落",
                        "突破阻力后收回，派发特征，警惕进一步下跌", conf.get("upthrust_in_range", 85))
            if in_range:
                return ("Phase B — 派发区间震荡",
                        f"价格反弹受阻，{vol_state}，注意二次探底风险", conf.get("distribute", 68))
            return ("Phase A — 派发初期",
                    "高位滞涨，供应开始出现，注意趋势转变", conf.get("distribute_early", 60))

        return ("Phase B — 区间整理",
                f"均线交织，价格在{range_low:.1f}-{range_high:.1f}区间波动", conf.get("range_trade", 55))


# ============================================================
# Scanner
# ============================================================

class Scanner:

    _sc = get_scanner_config()

    SINA_HQ_URL = _sc.get("api", {}).get("sina_hq",
        "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData")
    TENCENT_KLINE_URL = _sc.get("api", {}).get("tencent_kline",
        "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq")

    INDUSTRY_CACHE = os.path.join(os.path.dirname(__file__),
        _sc.get("industry", {}).get("cache_file", "industry_cache.json"))

    def __init__(self):
        self.all_stocks = []
        self.snapshot = {}
        self.industry_map = {}
        self.candidates = []
        self.results = []
        self._sc = get_scanner_config()

    # ==================== 行业分类 ====================

    def build_industry_map(self, force=False, use_akshare=True):
        if not force and os.path.exists(self.INDUSTRY_CACHE):
            try:
                with open(self.INDUSTRY_CACHE, encoding="utf-8") as f:
                    self.industry_map = json.load(f)
                if self.industry_map:
                    return
            except Exception: pass

        if use_akshare:
            try:
                print("  (AKShare 行业分类)...", end=" ")
                t0 = time.time()
                self.industry_map = dp_get_industry_map()
                if self.industry_map:
                    print(f"{len(self.industry_map)} 只 ({time.time()-t0:.1f}s)")
                    self._save_industry_cache()
                    return
            except Exception as e:
                print(f"AKShare失败({e}), 回退baostock...")

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

        self._save_industry_cache()

    def _save_industry_cache(self):
        try:
            with open(self.INDUSTRY_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.industry_map, f, ensure_ascii=False)
        except Exception: pass

    def get_industry(self, code):
        return self.industry_map.get(code, "其他")

    # ==================== 获取全市场数据 ====================

    @staticmethod
    def _fetch_page(page, num=100, timeout=20, node="hs_a"):
        params = {"page": page, "num": num, "sort": "symbol",
                  "asc": "1", "node": node, "_s_r_a": "init"}
        r = requests.get(Scanner.SINA_HQ_URL, params=params, timeout=timeout)
        return r.json()

    def fetch_all_stocks(self, max_pages=None):
        fc = self._sc.get("fetch", {})
        max_pages = max_pages or fc.get("max_pages", 80)
        page_size = fc.get("page_size", 100)
        workers = fc.get("thread_pool", 15)
        timeout = fc.get("request_timeout", 20)
        all_data = []
        # 同时获取沪市(sh_a)和深市(sz_a)，hs_a已改为仅北交所
        for node in ("sh_a", "sz_a"):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pages_per_exchange = max(30, max_pages // 2)
                futures = {}
                for p in range(1, pages_per_exchange + 1):
                    futures[pool.submit(self._fetch_page, p, page_size, timeout, node)] = p
                for f in as_completed(futures):
                    try:
                        data = f.result()
                        if data:
                            all_data.extend(data)
                    except Exception:
                        pass

        stocks, snapshot = [], {}
        if all_data:
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

        # ── 兜底：Sina API 限流时用 AKShare 股票代码列表 ──
        if not snapshot:
            try:
                import akshare as ak
                df = ak.stock_info_a_code_name()
                for _, row in df.iterrows():
                    code = str(row["code"])
                    name = str(row["name"])
                    if code.startswith("9"):
                        continue
                    market = "sh" if code.startswith("6") else "sz"
                    bs_code = f"{market}.{code}"
                    stocks.append({"code": bs_code, "name": name})
                    snapshot[bs_code] = {
                        "code": bs_code, "name": name, "price": 0,
                        "amount": 0, "volume": 0, "turnover": 0,
                        "high": 0, "low": 0, "open": 0, "pre_close": 0, "change_pct": 0,
                    }
            except Exception:
                pass

        self.all_stocks = stocks
        self.snapshot = snapshot
        return stocks, snapshot

    # ==================== 过滤 ====================

    def filter_candidates(self, min_amount=None):
        min_amount = min_amount if min_amount is not None else self._sc.get("filter", {}).get("min_amount", 500000000)
        # 有实时数据时按成交额过滤
        has_realtime = any(v["amount"] > 0 for v in self.snapshot.values())
        if has_realtime:
            candidates = [d for d in self.snapshot.values() if d["amount"] >= min_amount]
            candidates.sort(key=lambda x: x["amount"], reverse=True)
        else:
            # 兜底：无实时数据时全量返回（限流模式）
            candidates = list(self.snapshot.values())
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

            # 形态学检测（双顶/双底/头肩顶底/V转）
            result["pattern_name"] = "-"
            result["pattern_score"] = 0
            result["pattern_detail"] = ""
            if len(closes) >= 40:
                patterns = PatternDetector.analyze_all(
                    np.array(closes), np.array(highs),
                    np.array(lows), np.array(volumes),
                    trend=result["trend"]
                )
                if patterns:
                    result["pattern_name"] = patterns[0].name
                    result["pattern_score"] = patterns[0].score
                    det = patterns[0].detail
                    if patterns[0].confirmed:
                        det += " [确认]"
                    elif "颈线" in det or "站上" in det or "跌破" in det:
                        det += " [未确认]"
                    result["pattern_detail"] = det
                    result["_all_patterns"] = [(p.name, p.score, p.direction,
                                                p.confirmed, p.target) for p in patterns]

            # 缠论三买执行信号
            result["chan_third_buy"] = None
            if len(closes) >= 60:
                try:
                    chan_sig = WyckoffAnalyzer.detect_chan_third_buy(
                        closes, highs, lows, volumes, trend=result["trend"]
                    )
                    if chan_sig:
                        result["chan_third_buy"] = {
                            "signal": chan_sig[0],
                            "score": chan_sig[1],
                            "detail": chan_sig[2]
                        }
                except Exception:
                    pass

        except Exception:
            pass

        return result

    def enrich_with_financials(self, stocks_with_sigs, max_items=None):
        """对威科夫信号股票补充基本面（PE/PB/ROE/市值）+ 筹码分布"""
        analysis_cfg = self._sc.get("analysis", {})
        max_items = max_items or analysis_cfg.get("financial_batch_size", 10)
        fin_interval = analysis_cfg.get("financial_interval", 0.3)
        enriched = []
        for s in stocks_with_sigs[:max_items]:
            try:
                code = s["code"].split(".")[1] if "." in s["code"] else s["code"]
                fin = get_financial_indicators(code)
                s["pe"] = fin.get("pe")
                s["pb"] = fin.get("pb")
                s["roe"] = fin.get("roe")
                s["total_mv"] = fin.get("total_mv")
                # 筹码分布
                ts_code = code + ".SZ" if not code.startswith("6") else code + ".SH"
                chips = TushareProvider.get_chip_distribution(ts_code)
                s["chip_support"] = chips.get("chip_support")
                s["chip_resistance"] = chips.get("chip_resistance")
                s["chip_current"] = chips.get("current_price")
            except Exception:
                s["pe"] = s["pb"] = s["roe"] = s["total_mv"] = None
                s["chip_support"] = s["chip_resistance"] = None
            enriched.append(s)
            time.sleep(fin_interval)
        return enriched

    def enrich_with_quality(self, stocks_with_sigs, max_items=None):
        """
        个股质地检查（第三层过滤）: ROE/营收/质押/商誉/ST
        对评分前N的候选股运行，标记 quality_passed
        深度研究确认的个股可跳过部分检查
        """
        ac = self._sc.get("analysis", {})
        rt = self._sc.get("research_tracker", {})
        max_items = max_items or ac.get("quality_batch_size", 30)
        checked = 0
        passed = 0
        for s in stocks_with_sigs[:max_items]:
            try:
                code = s["code"].split(".")[1] if "." in s["code"] else s["code"]
                q = get_stock_quality(code)

                # 深度研究个股: 跳过部分检查
                if _RESEARCH_TOOL and rt.get("enabled", True):
                    bypass = ResearchTracker.get_quality_bypass(code)
                    if bypass.get("roe") and q.get("checks", {}).get("roe_qualified") is False:
                        q["checks"]["roe_qualified"] = True
                    if bypass.get("revenue") and q.get("checks", {}).get("revenue_growing") is False:
                        q["checks"]["revenue_growing"] = True
                    q["all_passed"] = all(q.get("checks", {}).values())

                s["quality"] = q.get("checks", {})
                s["quality_passed"] = q.get("all_passed", False)
                # 复制几个关键指标方便展示
                for k in ("roe_3y_ok", "revenue_yoy", "pledge_ratio", "debt_to_assets", "roe"):
                    if k in q:
                        s[k] = q[k]
                checked += 1
                if s["quality_passed"]:
                    passed += 1
            except Exception:
                s["quality"] = {}
                s["quality_passed"] = True  # 数据异常不拦截
            time.sleep(0.3)  # 防 Tushare 限频
        if checked:
            sys.stdout.write(f"\r  质地检查: {checked}只, 通过{passed}只, 排除{checked-passed}只\n")
            sys.stdout.flush()
        return stocks_with_sigs

    def run_analysis(self, max_stocks=None):
        analysis_cfg = self._sc.get("analysis", {})
        max_stocks = max_stocks or analysis_cfg.get("max_stocks", 80)
        stock_interval = analysis_cfg.get("stock_interval", 0.02)
        results = []
        total = min(len(self.candidates), max_stocks)

        for i, stock in enumerate(self.candidates[:max_stocks]):
            analysis = self._analyze_stock(stock["code"])
            results.append({**stock, **analysis})
            if (i + 1) % 10 == 0 and total > 10:
                sys.stdout.write(f"\r  分析进度: {i+1}/{total}")
                sys.stdout.flush()
            time.sleep(stock_interval)

        if total > 10:
            sys.stdout.write(f"\r  分析进度: {total}/{total}\n")
            sys.stdout.flush()

        # 排序：三买 > 威科夫信号得分 > 多头 > 成交额
        def sort_key(x):
            c3 = 0
            cb = x.get("chan_third_buy")
            if cb and cb["signal"] == "三买":
                c3 = -100  # 三买优先展示
            elif cb and cb["signal"] == "弱三买":
                c3 = -50
            ws = -x["wyckoff_score"]
            tr = 0 if x["trend"] == "多头" else 1
            return (c3, ws, tr, -x["amount"])

        results.sort(key=sort_key)
        self.results = results
        return results

    # ==================== 持仓/股票池分析 ====================

    def analyze_watchlist(self):
        """分析自选股池 + 持仓个股状态"""
        wl = self._sc.get("watchlist", {})
        pool = wl.get("pool", [])
        holdings = wl.get("holdings", [])
        watchlist_results = {"pool": [], "holdings": []}

        # 分析关注池
        for item in pool:
            code = item["code"]
            bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
            snap = self.snapshot.get(bs_code, {})
            name = snap.get("name", code)
            analysis = self._analyze_stock(bs_code)
            analysis["code"] = code
            analysis["name"] = name
            analysis["price"] = snap.get("price", 0)
            watchlist_results["pool"].append(analysis)
            time.sleep(0.02)

        # 分析持仓
        for item in holdings:
            code = item["code"]
            bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
            snap = self.snapshot.get(bs_code, {})
            name = snap.get("name", code)
            analysis = self._analyze_stock(bs_code)
            analysis["code"] = code
            analysis["name"] = name
            analysis["price"] = snap.get("price", 0)
            analysis["cost"] = item.get("cost", 0)
            analysis["shares"] = item.get("shares", 0)
            if analysis["cost"] > 0 and analysis["price"] > 0:
                analysis["pnl_pct"] = round((analysis["price"] / analysis["cost"] - 1) * 100, 1)
            else:
                analysis["pnl_pct"] = None
            watchlist_results["holdings"].append(analysis)
            time.sleep(0.02)

        self.watchlist_results = watchlist_results
        return watchlist_results

    # ==================== 报告 ====================

    def _calc_breadth(self):
        """从全市场快照计算涨跌比、涨跌停"""
        up = down = limit_up = limit_down = 0
        for v in self.snapshot.values():
            chg = v.get("change_pct", 0)
            if chg > 0:
                up += 1
            elif chg < 0:
                down += 1
            if chg >= 9.8:
                limit_up += 1
            elif chg <= -9.8:
                limit_down += 1
        total = up + down
        ad_ratio = round(up / down, 2) if down > 0 else 99
        ul_ratio = round(limit_up / limit_down, 2) if limit_down > 0 else limit_up
        return {
            "up": up, "down": down, "total": total,
            "ad_ratio": ad_ratio, "limit_up": limit_up,
            "limit_down": limit_down, "ul_ratio": ul_ratio,
        }

    def _get_concept_board_performance(self):
        """获取概念板块当日涨幅Top10（用于市场情绪参考）"""
        try:
            boards = AkshareProvider.get_concept_boards()
            if boards:
                return [(b.get("板块名称", ""), b.get("涨跌幅", 0)) for b in boards[:10]]
        except Exception:
            pass
        return []

    def print_report(self, top_n=15):
        bullish = sum(1 for r in self.results if r["trend"] == "多头")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 信号统计
        sig_counts = {}
        c3_count = 0
        for r in self.results:
            t = r["wyckoff_sig"]
            if t not in ("-", "无信号", "数据不足"):
                sig_counts[t] = sig_counts.get(t, 0) + 1
            cb = r.get("chan_third_buy")
            if cb and cb["signal"] == "三买":
                c3_count += 1

        # 推荐回顾
        if _REC_TRACKER:
            print_review()

        # 龙虎榜数据（缓存一次，全报告复用）
        lhb_map = {}
        lhb_holdings = {}
        if _LHB_TOOL:
            try:
                codes_ts = []
                for r in self.results:
                    c = r["code"].split(".")[1]
                    suffix = ".SZ" if not c.startswith("6") else ".SH"
                    codes_ts.append(c + suffix)
                lhb_map = enrich_stock_list(codes_ts)
                if hasattr(self, "watchlist_results"):
                    for h in self.watchlist_results.get("holdings", []):
                        code = h["code"]
                        ts_code = code + (".SZ" if not code.startswith("6") else ".SH")
                        h_info = analyze_stock(ts_code)
                        if h_info:
                            lhb_holdings[code] = h_info
            except Exception:
                pass

        print("\n" + "=" * 80)
        print(f"  A股扫描报告  {now}")
        print("=" * 80)
        print(f"  覆盖: {len(self.all_stocks)} 只  流动性通过: {len(self.candidates)} 只")
        print(f"  多头: {bullish}只  ", end="")
        for k, v in sig_counts.items():
            print(f" {k}:{v}只 ", end="")
        if c3_count:
            print(f" 三买:{c3_count}只 ", end="")

        # 质地通过率
        quality_checked = [r for r in self.results if r.get("quality_passed") is not None]
        if quality_checked:
            passed = sum(1 for r in quality_checked if r["quality_passed"])
            pct = passed / len(quality_checked) * 100
            print(f"\n  质地检查: {len(quality_checked)}只 通过{passed}只 ({pct:.0f}%)", end="")
        print("\n")

        if not self.results:
            print("  无符合条件标的")
            return

        # === 市场情绪指标（三层） ===
        if _SENTIMENT_TOOL:
            try:
                d = get_daily_sentiment()
                w = get_weekly_sentiment()
                m = get_monthly_sentiment()
                c = get_sentiment()

                def items_line(items):
                    return " | ".join(f"{it['name']}{it['value']}[{it['level']}]" for it in items)

                mom_icon = {"升温": "↗", "降温": "↘", "持平": "→"}.get(w["momentum"], "")
                print(f"  ══ 市场情绪 ══  综合:{c['score']}[{c['level']}]")
                print(f"  日频: {d['score']}[{d['level']}]  {items_line(d['items'])}")
                print(f"  周频: {w['score']}[{w['level']}]  {mom_icon}Δ{w['delta']:+d}  {items_line(w['items'])}")
                print(f"  月频: {m['score']}[{m['level']}]  {items_line(m['items'])}")
                print()
            except Exception:
                pass

        # === 大盘健康状况 ===
        try:
            from market import MarketAnalyzer
            indices = MarketAnalyzer.fetch_indices()
        except Exception:
            indices = {}

        breadth = self._calc_breadth()

        print(f"  【大盘健康状况】")
        # 指数
        idx_line = ""
        if indices:
            for key in ["sh000001", "sz399001", "sz399006", "sh000688"]:
                idx = indices.get(key)
                if idx:
                    idx_line += f"{idx['name']}{idx['change_pct']:+.2f}% "
        print(f"  指数: {idx_line}")

        # 涨跌统计
        if breadth:
            ad = breadth.get("ad_ratio", 0)
            up = breadth.get("up", 0)
            down = breadth.get("down", 0)
            lu = breadth.get("limit_up", 0)
            ld = breadth.get("limit_down", 0)
            # 判断涨跌状态
            if ad >= 2:
                ad_label = "普涨格局"
            elif ad >= 1.2:
                ad_label = "涨多跌少"
            elif ad >= 0.8:
                ad_label = "偏弱"
            else:
                ad_label = "极弱"
            if lu > ld * 3 and lu > 30:
                emo_label = "做多情绪强"
            elif lu > ld:
                emo_label = "正常"
            else:
                emo_label = "做空占优"

            print(f"  涨跌: {up}涨/{down}跌  (涨跌比{ad}, {ad_label})")
            print(f"  涨停: {lu}只  跌停: {ld}只  ({emo_label})")

        # PE分位
        try:
            pe_info, cap = MarketAnalyzer.get_pe_percentile()
            if pe_info:
                print(f"  估值: PE {pe_info['pe']} 近5年{pe_info['percentile']}%分位 ({pe_info['level']})")
        except Exception:
            pass

        # 综合健康判断
        health_p = 0
        health_w = 0
        if indices:
            avg_chg = np.mean([i["change_pct"] for i in indices.values()])
            if avg_chg >= 0.5:
                health_p += 1
            elif avg_chg <= -0.5:
                health_w += 1
        if breadth:
            if breadth["ad_ratio"] >= 1.5:
                health_p += 1
            elif breadth["ad_ratio"] < 0.8:
                health_w += 1
            if breadth["ul_ratio"] >= 3:
                health_p += 1
            elif breadth["ul_ratio"] < 1:
                health_w += 1
        score = health_p - health_w
        if score >= 2:
            hl = "健康 — 可积极参与"
        elif score >= 1:
            hl = "较好 — 精选个股操作"
        elif score >= 0:
            hl = "一般 — 注意仓位管理"
        elif score >= -1:
            hl = "偏弱 — 建议降低仓位"
        else:
            hl = "较差 — 防守为主"
        print(f"  综合判断: {hl}")
        print()

        # === 热门板块轮动 ===
        sectors = {}
        for r in self.results:
            ind = r.get("industry", "其他")
            if ind not in sectors:
                sectors[ind] = {"count": 0, "sig_count": 0, "sig_buy": 0, "names": []}
            sectors[ind]["count"] += 1
            sig = r["wyckoff_sig"]
            if sig not in ("-", "无信号", "数据不足"):
                sectors[ind]["sig_count"] += 1
                if sig in ("Spring", "SOS", "LPS", "弱Spring"):
                    sectors[ind]["sig_buy"] += 1
            if len(sectors[ind]["names"]) < 3:
                sectors[ind]["names"].append(r["name"])

        sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["count"])
        print(f"  【热门板块轮动】")
        print(f"  {'板块':<26} {'个股':<5} {'信号':<6} {'买入':<6} {'热度':<8} {'代表':<24}")
        print(f"  " + "-" * 80)
        for ind, info in sorted_sec[:10]:
            sig_tag = f"  {info['sig_count']} " if info['sig_count'] > 0 else "  - "
            buy_tag = f"  {info['sig_buy']} " if info['sig_buy'] > 0 else "  - "
            # 热度
            heat_str = "-"
            try:
                heat = get_sector_heat(ind)
                if heat and heat.get("composite"):
                    heat_str = f"{heat['composite']}"
            except Exception:
                pass
            top = ", ".join(info["names"])
            print(f"  {ind:<26} {info['count']:<5} {sig_tag:<6} {buy_tag:<6} {heat_str:<8} {top:<24}")

        # 轮动判断
        top_sector = sorted_sec[0][0] if sorted_sec else ""
        top_count = sorted_sec[0][1]["count"] if sorted_sec else 0
        total_analyzed = len(self.results)
        concentration = top_count / total_analyzed * 100 if total_analyzed > 0 else 0
        # 统计买入信号比例
        total_buy = sum(1 for r in self.results if r["wyckoff_sig"] in ("Spring", "SOS", "LPS", "弱Spring") and r.get("quality_passed") == True)
        total_sell = sum(1 for r in self.results if r["wyckoff_sig"] in ("Upthrust",) and r.get("quality_passed") == True)
        # 板块数量
        sec_count = len(sorted_sec)

        if concentration > 50:
            rotation_note = f"高度集中在{top_sector}，占比{concentration:.0f}%，缺乏接力板块"
        elif concentration > 30:
            rotation_note = f"集中在{top_sector}，占比{concentration:.0f}%，少量分散"
        else:
            rotation_note = f"分布相对分散({sec_count}个板块)，轮动正常"
        if total_sell > total_buy * 2:
            rotation_note += "，卖出信号偏多，注意回调风险"
        print(f"  轮动: {rotation_note}")
        print()

        # === 因子拥挤度分析 ===
        n_res = len(self.results)
        if n_res >= 30:
            # 趋势拥挤
            bull_count = sum(1 for r in self.results if r["trend"] == "多头")
            bear_count = sum(1 for r in self.results if r["trend"] == "空头")
            bull_pct = bull_count / n_res * 100
            bear_pct = bear_count / n_res * 100

            # 威科夫信号拥挤
            buy_sigs = {"Spring", "SOS", "LPS", "弱Spring", "Compression"}
            sell_sigs = {"Upthrust", "EVR"}
            sig_high = sum(1 for r in self.results if r.get("wyckoff_score", 0) >= 70)
            sig_low = sum(1 for r in self.results if r.get("wyckoff_score", 0) <= 30)
            sig_high_pct = sig_high / n_res * 100
            sig_low_pct = sig_low / n_res * 100

            # 成交量因子拥挤（strength 含量价信息）
            vol_high = sum(1 for r in self.results if r.get("strength", 0) >= 65)
            vol_low = sum(1 for r in self.results if r.get("strength", 0) <= 35)
            vol_high_pct = vol_high / n_res * 100

            print(f"  【因子拥挤度 — 高分占比越高=因子越拥挤(可能衰减)】")
            print(f"  {'因子/信号':<14} {'高分%':>8} {'拥挤':>6}")
            print(f"  " + "-" * 30)
            def crowd_label(pct):
                if pct > 50: return "!!过挤"
                if pct > 35: return "拥挤.."
                if pct > 20: return "偏高"
                return "正常"
            print(f"  {'多头趋势':<14} {bull_pct:>7.0f}%  {crowd_label(bull_pct):>6}")
            print(f"  {'空头趋势':<14} {bear_pct:>7.0f}%  {crowd_label(bear_pct):>6}")
            print(f"  {'威科夫高分(≥70)':<14} {sig_high_pct:>7.0f}%  {crowd_label(sig_high_pct):>6}")
            print(f"  {'威科夫低分(≤30)':<14} {sig_low_pct:>7.0f}%  {crowd_label(sig_low_pct):>6}")
            print(f"  {'量价强势(≥65)':<14} {vol_high_pct:>7.0f}%  {crowd_label(vol_high_pct):>6}")
            print()

        # === 股票池状态 ===
        if hasattr(self, "watchlist_results"):
            pool = self.watchlist_results.get("pool", [])
            if pool:
                print(f"  【股票池状态】")
                print(f"  {'代码':<8} {'名称':<10} {'趋势':<10} {'威科夫':<18} {'阶段':<14} {'价格':<8}")
                print(f"  " + "-" * 72)
                for s in pool:
                    sym = s["code"]
                    nn = s["name"][:8]
                    tr = f"{s['trend']}({s['strength']:+.1f}%)" if s["trend"] in ("多头", "空头") else s["trend"]
                    sg = f"{s['wyckoff_sig']}({s['wyckoff_score']})" if s["wyckoff_sig"] not in ("-", "无信号") else "-"
                    ph = s.get("phase", "")[:12]
                    pr = f"{s['price']:.2f}" if s["price"] else "-"
                    print(f"  {sym:<8} {nn:<10} {tr:<10} {sg:<18} {ph:<14} {pr:<8}")
                print()

        # === 持仓状态和建议 ===
        if hasattr(self, "watchlist_results"):
            holdings = self.watchlist_results.get("holdings", [])
            if holdings:
                print(f"  【持仓状态和建议】")
                print(f"  {'代码':<8} {'名称':<8} {'趋势':<8} {'威科夫':<16} {'价格':<8} {'成本':<8} {'浮盈':<8} {'龙虎榜':<10} {'建议':<12}")
                print(f"  " + "-" * 92)
                for s in holdings:
                    sym = s["code"]
                    nn = s["name"][:6]
                    tr = f"{s['trend']}({s['strength']:+.1f}%)" if s["trend"] in ("多头", "空头") else s["trend"]
                    sg = f"{s['wyckoff_sig']}({s['wyckoff_score']})" if s["wyckoff_sig"] not in ("-", "无信号") else "-"
                    pr = f"{s['price']:.2f}" if s["price"] else "-"
                    cost = f"{s['cost']:.2f}" if s.get("cost") else "-"
                    pnl = f"{s['pnl_pct']:+.1f}%" if s.get("pnl_pct") is not None else "-"
                    # 龙虎榜状态
                    lhb_str = "-"
                    if sym in lhb_holdings:
                        hi = lhb_holdings[sym]
                        net = hi.get('top_list', {}).get('net_amount', 0)
                        lhb_str = f"净{net:+.2f}亿" if net else "上榜"
                    # 建议规则
                    sug = "持有"
                    if s["trend"] == "空头":
                        sug = "减仓/止损"
                    elif s["wyckoff_sig"] in ("Upthrust",) and s.get("pnl_pct", 0) is not None and s["pnl_pct"] > 5:
                        sug = "减仓"
                    elif s["wyckoff_sig"] in ("Spring", "SOS", "LPS"):
                        sug = "加仓/持有"
                    elif s.get("pnl_pct") is not None and s["pnl_pct"] < -15:
                        sug = "警惕止损"
                    print(f"  {sym:<8} {nn:<8} {tr:<8} {sg:<16} {pr:<8} {cost:<8} {pnl:<8} {lhb_str:<10} {sug:<12}")
                print()

        # === 市值风云研究股票池 ===
        if _RESEARCH_TOOL:
            try:
                import json as _json
                rw_path = os.path.join(os.path.dirname(__file__), "research_watchlist.json")
                if os.path.exists(rw_path):
                    with open(rw_path, encoding="utf-8") as _f:
                        rw_data = _json.load(_f)
                    active_research = [s for s in rw_data.get("stocks", []) if s.get("active", True)]
                    if active_research:
                        print(f"  【市值风云研究股票池】")
                        print(f"  {'代码':<8} {'名称':<8} {'扫描趋势':<10} {'信号':<16} {'现价':<8} {'确信度':<6}")
                        print(f"  " + "-" * 62)
                        for rs in active_research:
                            code = rs["code"]
                            name = rs.get("name", "")[:6]
                            conviction = rs.get("conviction", "medium")[:4]
                            bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
                            # 在results中查找
                            found_r = next((r for r in self.results if r["code"] == bs_code), None)
                            if found_r:
                                trend = f"{found_r['trend']}({found_r['strength']:+.1f}%)" if found_r['trend'] in ("多头","空头") else found_r["trend"]
                                sig_s = f"{found_r['wyckoff_sig']}({found_r['wyckoff_score']})" if found_r["wyckoff_sig"] not in ("-","无信号") else "-"
                                price = f"{found_r['price']:.2f}"
                            else:
                                trend = "未入分析"
                                sig_s = "-"
                                price = "-"
                            print(f"  {code:<8} {name:<8} {trend:<10} {sig_s:<16} {price:<8} {conviction:<6}")
                        print()
            except Exception:
                pass

        # === Top N ===
        print(f"  [Top {top_n}]")
        print(f"  {'#':<3} {'代码':<8} {'名称':<7} {'板块':<12} {'趋势':<10} {'威科夫':<18} {'形态':<16} {'三买':<14} {'价格':<8} {'成交额':<8} {'质地':<6}")
        print(f"  " + "-" * 140)

        for i, r in enumerate(self.results[:top_n]):
            sym = r["code"].split(".")[1]
            name = r["name"][:6]

            # 深度研究个股标记
            rt_cfg = self._sc.get("research_tracker", {})
            if _RESEARCH_TOOL and rt_cfg.get("enabled", True):
                try:
                    if ResearchTracker.is_confirmed(sym):
                        name += rt_cfg.get("display_marker", "R")
                except Exception:
                    pass

            ind = r.get("industry", "其他")[:10]
            trend = f"{r['trend']}({r['strength']:+.1f}%)" if r["trend"] in ("多头", "空头") else r["trend"]

            sig = r["wyckoff_sig"]
            sc = r["wyckoff_score"]
            sig_str = f"{sig}({sc})" if sig not in ("-", "无信号") else sig

            # 形态信号
            pn = r.get("pattern_name", "-")
            ps = r.get("pattern_score", 0)
            p_str = f"{pn}({ps})" if pn != "-" else "-"

            # 三买信号
            cb = r.get("chan_third_buy")
            cb_str = "-"
            if cb:
                cb_str = f"{cb['signal']}({cb['score']})" if cb.get("detail") else cb["signal"]

            price = f"{r['price']:.2f}"
            amt = f"{r['amount']/1e8:.1f}亿"
            qf = "通过" if r.get("quality_passed") else ("未检" if r.get("quality_passed") is None else "未过")
            print(f"  {i+1:<3} {sym:<8} {name:<7} {ind:<12} {trend:<10} {sig_str:<18} {p_str:<16} {cb_str:<14} {price:<8} {amt:<8} {qf:<6}")

        print()

        # === 缠论三买执行信号 ===
        c3_stocks = [r for r in self.results if r.get("chan_third_buy") and r["chan_third_buy"]["signal"] == "三买"]
        if c3_stocks:
            print(f"  【缠论三买执行信号 — 中枢突破回踩确认】")
            print(f"  {'信号':<14} {'代码':<8} {'名称':<7} {'板块':<12} {'趋势':<10} {'威科夫':<18} {'细节':<36}")
            print(f"  " + "-" * 110)
            for r in c3_stocks[:10]:
                cb = r["chan_third_buy"]
                sym = r["code"].split(".")[1]
                name = r["name"][:6]
                ind = r.get("industry", "其他")[:10]
                tr = f"{r['trend']}({r['strength']:+.1f}%)" if r["trend"] in ("多头", "空头") else r["trend"]
                sg = r["wyckoff_sig"]
                sc = r["wyckoff_score"]
                sg_str = f"{sg}({sc})" if sg not in ("-", "无信号") else sg
                detail = cb.get("detail", "")[:36]
                print(f"  {cb['signal']}({cb['score']})  {sym:<8} {name:<7} {ind:<12} {tr:<10} {sg_str:<18} {detail:<36}")
            print()

        # === 龙虎榜异动 ===
        if _LHB_TOOL and lhb_map:
            print(f"  【龙虎榜异动】")
            print(f"  {'代码':<8} {'名称':<7} {'净买入(亿)':<12} {'净率':<8} {'上榜原因':<26}")
            print(f"  " + "-" * 65)
            sorted_lhb = sorted(lhb_map.items(), key=lambda x: -abs(x[1].get('net_amount', 0)))
            for ts_code, info in sorted_lhb:
                name = info.get('name', '')[:6]
                net = info.get('net_amount', 0)
                net_str = f"{net:+.2f}" if net else "-"
                rate = f"{info['net_rate']:.1f}%" if info.get('net_rate') else "-"
                reason = (info.get('reason', '') or '')[:24]
                print(f"  {ts_code:<8} {name:<7} {net_str:<12} {rate:<8} {reason:<26}")
            print()

        # === 推荐选股结果 ===
        valid = [r for r in self.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
        if valid:
            quality_ok = [r for r in valid if r.get("quality_passed") == True]
            if quality_ok:
                enriched = self.enrich_with_financials(quality_ok, 10)
                # 分成买入信号和卖出信号
                buy_sigs = [r for r in enriched if r["wyckoff_sig"] in ("Spring", "SOS", "LPS", "弱Spring")]
                sell_sigs = [r for r in enriched if r["wyckoff_sig"] in ("Upthrust", "弱Upthrust", "EVR")]

                if buy_sigs:
                    print(f"  【买入关注 — Spring/SOS/LPS】质地通过 {len(buy_sigs)}只")
                    print(f"  {'信号':<12} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'PE':<7} {'ROE':<6} {'细节':<24}")
                    print(f"  " + "-" * 88)
                    for r in buy_sigs:
                        sym = r["code"].split(".")[1]
                        name = r["name"][:6]
                        ind = r.get("industry", "其他")[:10]
                        sig = r["wyckoff_sig"]
                        sc = r["wyckoff_score"]
                        pe = f"{r['pe']:.1f}" if r.get("pe") else "-"
                        roe = f"{r['roe']:.1f}%" if r.get("roe") else "-"
                        detail = r.get("wyckoff_detail", "")[:24]
                        print(f"  {sig:<12} {sym:<8} {name:<7} {ind:<12} {sc:<5} {pe:<7} {roe:<6} {detail:<24}")
                    print()

                if sell_sigs:
                    print(f"  【警示列表 — Upthrust】质地通过 {len(sell_sigs)}只（非买入点）")
                    print(f"  {'信号':<12} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'PE':<7} {'ROE':<6} {'细节':<24}")
                    print(f"  " + "-" * 88)
                    for r in sell_sigs:
                        sym = r["code"].split(".")[1]
                        name = r["name"][:6]
                        ind = r.get("industry", "其他")[:10]
                        sig = r["wyckoff_sig"]
                        sc = r["wyckoff_score"]
                        pe = f"{r['pe']:.1f}" if r.get("pe") else "-"
                        roe = f"{r['roe']:.1f}%" if r.get("roe") else "-"
                        detail = r.get("wyckoff_detail", "")[:24]
                        print(f"  {sig:<12} {sym:<8} {name:<7} {ind:<12} {sc:<5} {pe:<7} {roe:<6} {detail:<24}")
                    print()

                # 无明确信号的质地通过票
                neutral = [r for r in enriched if r not in buy_sigs and r not in sell_sigs]
                if neutral:
                    print(f"  【其他质地通过】（无威科夫信号或信号不明）:")
                    print("  ", ", ".join(f"{r['name']}({r['code'].split('.')[1]})" for r in neutral[:5]))
                    print()
            else:
                print(f"  【威科夫信号 {len(valid)}只，但均未通过个股质地检查】")
                for r in valid[:5]:
                    q = r.get("quality", {})
                    fails = [k for k, v in q.items() if v == False]
                    print(f"  {r['name']:<6} {r['code']:<12} 排除: {','.join(fails)}")
                print()

        # === 今日小结 ===
        buys = [r for r in self.results if r["wyckoff_sig"] in ("Spring","SOS","LPS","弱Spring") and r.get("quality_passed") == True]
        sells = [r for r in self.results if r["wyckoff_sig"] == "Upthrust" and r.get("quality_passed") == True]
        total_sig = len([r for r in self.results if r["wyckoff_sig"] not in ("-","无信号","数据不足")])
        # 形态信号统计
        pattern_count = len([r for r in self.results if r.get("pattern_name", "-") != "-"])
        pattern_confirmed = len([r for r in self.results if r.get("_all_patterns") and any(p[3] for p in r["_all_patterns"])])
        print(f"  【今日小结】")
        notes = []
        if breadth:
            if breadth.get("ad_ratio", 0) >= 2:
                notes.append(f"普涨格局，但")
            elif breadth.get("ad_ratio", 0) >= 1.2:
                notes.append(f"涨多跌少，但")
        if total_sell > total_buy * 2:
            notes.append(f"卖出信号(Upthrust {len(sells)}只)远超买入信号({len(buys)}只)，不宜追高")
        elif buys:
            notes.append(f"买入信号{len(buys)}只，可关注")
        else:
            notes.append(f"当前无质地通过的买入信号")

        if pattern_count:
            notes.append(f"形态信号{pattern_count}只(确认{pattern_confirmed})")

        if c3_count:
            notes.append(f"三买{c3_count}只")

        if concentration > 50:
            notes.append(f"板块高度集中在{top_sector}，注意轮动风险")

        # 建议
        action_suggest = ""
        if buys and total_sell <= total_buy * 2:
            action_suggest = "轻仓试探买入信号标的，设好止损"
        elif total_sell > total_buy * 2:
            action_suggest = "减仓为主，不开新仓"
        else:
            action_suggest = "等待Spring/SOS出现再动手"
        notes.append(f"建议: {action_suggest}")
        print(f"  {'| '.join(notes)}")
        print()

        # === 概念板块热度 ===
        try:
            concepts = self._get_concept_board_performance()
            if concepts:
                print(f"  【概念板块涨幅 Top10】")
                print(f"  {'板块':<20} {'涨跌幅':<8}")
                print(f"  " + "-" * 30)
                for name, pct in concepts:
                    arrow_f = "+" if pct > 0 else ""
                    print(f"  {name:<20} {arrow_f}{pct:+.2f}%")
                print()
        except Exception:
            pass

        # === 热度增长率（概念2） ===
        if _BUZZ_TOOL:
            try:
                overheat_list = scan_overheat(20)
                if overheat_list:
                    print(f"  【热度监控】")
                    print(f"  {'名称':<10} {'排名':<6} {'综合分':<8} {'涨幅':<8} {'状态':<10}")
                    print(f"  " + "-" * 45)
                    for s in overheat_list[:8]:
                        arrow_f = "+" if s['pct_chg'] > 0 else ""
                        print(f"  {s['name']:<10} #{s['rank']:<4} {s['score']:<8.0f} {arrow_f}{s['pct_chg']:+.2f}%{'':<5} {s['alert']:<10}")
                    print()
            except Exception:
                pass

        # === 3σ 情绪过热（概念3） ===
        if _SENTIMENT_TOOL:
            try:
                s = get_sentiment()
                oh = s.get("overheat", [])
                if oh:
                    print(f"  【3σ 情绪异常】")
                    for o in oh:
                        print(f"  {o['name']}: {o['status']} — {o['detail']}")
                    print()
            except Exception:
                pass

    # ==================== 飞书推送 ====================

    def _push_feishu(self, d, w, m):
        """扫描完成后推送飞书卡片"""
        cfg = get_scanner_config()
        webhook_url = cfg.get("notify", {}).get("webhook_url", "")
        if not webhook_url:
            return

        c_score = d["score"] * 0.4 + w["score"] * 0.3 + m["score"] * 0.3
        combined_level = "黄"
        if c_score >= 70:
            combined_level = "红"
        elif c_score < 40:
            combined_level = "绿"
        header_temp = "red" if combined_level == "红" else ("yellow" if combined_level == "黄" else "blue")
        header_title = f"扫描报告 | 综合{round(c_score)}[{combined_level}]"

        elements = []

        # 三层情绪
        mom_icon = {"升温": "↗", "降温": "↘", "持平": "→"}.get(w.get("momentum", ""), "")
        d_line = " | ".join(f"{i['name']}{i['value']}{i['level']}" for i in d["items"])
        w_line = f"周频{w['score']}分 {mom_icon}Δ{w['delta']:+d}" if w.get("delta") is not None else f"周频{w['score']}分"
        m_line = " | ".join(f"{i['name']}{i['value']}{i['level']}" for i in m["items"])
        elements.append(make_div(f"**日频** {d['score']}分  {d_line}"))
        elements.append(make_note(f"{w_line}  |  **月频** {m['score']}分  {m_line}"))
        elements.append(make_hr())

        # 大盘状况
        from market import MarketAnalyzer
        try:
            indices = MarketAnalyzer.fetch_indices()
            idx_line = " | ".join(f"{n[:6]}{i['change_pct']:+.2f}%" for n, i in indices.items())
            if idx_line:
                elements.append(make_div(f"**大盘**  {idx_line}"))
        except Exception:
            pass

        try:
            brd = MarketAnalyzer.calc_breadth()
            if brd:
                ad = brd.get("ad_ratio", 0)
                label = "普涨" if ad >= 2 else ("偏强" if ad >= 1.2 else "偏弱")
                elements.append(make_note(f"涨跌{brd.get('up',0)}/{brd.get('down',0)}  涨停{brd.get('limit_up',0)}  跌停{brd.get('limit_down',0)}  |  {label}"))
        except Exception:
            pass
        elements.append(make_hr())

        # 信号精选
        valid = [r for r in self.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
        signals = [r for r in valid if r.get("quality_passed") == True]
        if signals:
            elements.append(make_div(f"**信号精选** ({len(signals)}只质地通过)"))
            for r in signals[:_MAX_PUSH_N]:
                sym = r["code"].split(".")[1] if "." in r["code"] else r["code"]
                sig = r["wyckoff_sig"]
                sc = r["wyckoff_score"]
                elements.append(make_note(f"{sym} {r['name'][:6]}  {sig}({sc})  {r.get('industry','')[:8]}"))
            if len(signals) > _MAX_PUSH_N:
                elements.append(make_note(f"... 共{len(signals)}只信号"))
            elements.append(make_hr())

        # 板块集中度
        sectors = {}
        for r in self.results:
            ind = r.get("industry", "其他")
            sectors[ind] = sectors.get(ind, 0) + 1
        sorted_sec = sorted(sectors.items(), key=lambda x: -x[1])
        if sorted_sec:
            top_sec = sorted_sec[0]
            conc = top_sec[1] / len(self.results) * 100
            sec_str = f"**板块** {top_sec[0]} {top_sec[1]}只({conc:.0f}%)"
            if len(sorted_sec) > 1:
                sec_str += f"  次热{sorted_sec[1][0]}{sorted_sec[1][1]}只"
            elements.append(make_div(sec_str))

        # 3σ 情绪异常
        oh = d.get("overheat", [])
        if oh:
            oh_str = "  ".join(f"{o['name']}⚠{o['status']}" for o in oh[:3])
            if oh_str:
                elements.append(make_note(f"⚠ 3σ异常 {oh_str}"))

        # 热度预警（概念2）
        if _BUZZ_TOOL:
            try:
                buzz_alerts = scan_overheat(20)
                if buzz_alerts:
                    alert_str = "  ".join(f"{a['name']}{a['alert']}" for a in buzz_alerts[:3])
                    elements.append(make_note(f"🔥 {alert_str}"))
            except Exception:
                pass

        if hasattr(self, "watchlist_results"):
            holdings = self.watchlist_results.get("holdings", [])
            if holdings:
                h_items = []
                for s in holdings:
                    pnl = f"{s['pnl_pct']:+.1f}%" if s.get("pnl_pct") is not None else "-"
                    sug = ""
                    if s["trend"] == "空头":
                        sug = "止损"
                    elif s["wyckoff_sig"] in ("Upthrust",) and s.get("pnl_pct", 0) is not None and s["pnl_pct"] > 5:
                        sug = "减仓"
                    h_items.append(f"{s['name'][:6]} {pnl}" + (f"({sug})" if sug else ""))
                if h_items:
                    elements.append(make_note("**持仓** " + " | ".join(h_items)))

        # 小结
        buys = len([r for r in signals if r["wyckoff_sig"] in ("Spring", "SOS", "LPS", "弱Spring")])
        sells = len([r for r in signals if r["wyckoff_sig"] == "Upthrust"])
        parts = []
        if buys:
            parts.append(f"买入{buys}只")
        if sells:
            parts.append(f"卖出{sells}只")
        if buys > sells:
            parts.append("偏多")
        elif sells > buys:
            parts.append("偏空")
        if parts:
            elements.append(make_note(" | ".join(parts)))

        send_card(webhook_url, header_title, elements)

    # ==================== 主流程 ====================

    def run(self, min_amount=None, max_analysis=None, top_n=None, quick=False):
        ac = self._sc.get("analysis", {})
        rc = self._sc.get("report", {})
        fc = self._sc.get("filter", {})
        min_amount = min_amount or fc.get("min_amount", 500000000)
        max_analysis = max_analysis or ac.get("max_stocks", 80)
        top_n = top_n or rc.get("top_n_default", 15)

        print("A股扫描器 v2.2（威科夫形态检测，YAML配置）")
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

            # [3.5/3] 个股质地检查（第三层，仅标记不过滤）
            print("[3.5/3] 个股质地检查 (Tushare)...")
            self.enrich_with_quality(self.results)

            # [4/4] 持仓/股票池分析
            print("[4/4] 持仓+股票池分析...")
            self.analyze_watchlist()

        self.print_report(top_n)

        # 推送飞书卡片
        if _SENTIMENT_TOOL:
            try:
                from sentiment_indicator import get_daily_sentiment, get_weekly_sentiment, get_monthly_sentiment
                d = get_daily_sentiment()
                w = get_weekly_sentiment()
                m = get_monthly_sentiment()
                self._push_feishu(d, w, m)
            except Exception:
                pass

        # 保存今日扫描结果到推荐历史
        if _REC_TRACKER and hasattr(self, "results") and self.results:
            try:
                from market import MarketAnalyzer
                indices = MarketAnalyzer.fetch_indices()
            except Exception:
                indices = {}
            # 提取推荐（质地通过+有信号）
            valid = [r for r in self.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
            recommendations = [r for r in valid if r.get("quality_passed") == True]
            pool = self.watchlist_results.get("pool", []) if hasattr(self, "watchlist_results") else []
            holdings = self.watchlist_results.get("holdings", []) if hasattr(self, "watchlist_results") else []
            save_scan_result(self.results, pool, holdings, recommendations, indices, top_n)


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
