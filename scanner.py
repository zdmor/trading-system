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

try:
    from spring_quality import compute_confidence
    _SPRING_QUALITY_OK = True
except ImportError:
    _SPRING_QUALITY_OK = False
    _REC_TRACKER = False

from pattern_detector import PatternDetector
from notifier import send_card, make_div, make_hr, make_note
from update_state import update as update_state_md

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

# 高级模块（静默加载，不阻塞主流程）
try:
    from bayesian_confidence import BayesianConfidence
    _BAYES_OK = True
except Exception:
    _BAYES_OK = False

try:
    from kelly_position import load_factor_stats, kelly_fraction, compute_kelly_position
    _KELLY_OK = True
except Exception:
    _KELLY_OK = False

try:
    from factor_weights import get_weights, BASE_WEIGHTS
    _WEIGHTS_OK = True
except Exception:
    _WEIGHTS_OK = False

# 因子7系统（计算完整多因子评分）
try:
    from factor_7_validate import compute_factors_for_stock
    _FACTOR7_OK = True
except Exception:
    _FACTOR7_OK = False


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

    # ─── 纯价量驱动的威科夫阶段分类器 ───

    @staticmethod
    def _w_atr(highs, lows, closes, period=14):
        """单数组 ATR，不依赖 self.df"""
        n = len(highs)
        if n < period + 1:
            return 0.0
        trs = []
        for i in range(-period, 0):
            h = float(highs[i])
            l = float(lows[i])
            pc = float(closes[i - 1])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(trs)) if trs else 0.0

    @classmethod
    def _phase_features(cls, closes, highs, lows, opens, volumes):
        """计算分类器所需的全部特征，返回 dict"""
        n = len(closes)
        if n < 60:
            return None
        c, h, l, o, v = (
            np.array(closes, dtype=float),
            np.array(highs, dtype=float),
            np.array(lows, dtype=float),
            np.array(opens, dtype=float),
            np.array(volumes, dtype=float),
        )
        # 0. 涨跌停标记 — 用于支撑/阻力测试过滤（_phase_features 无 code 参数，统一用 10%）
        limit_up = np.zeros(n, dtype=bool)
        limit_down = np.zeros(n, dtype=bool)
        limit_pct = 10  # 统一用主板阈值，保守地多过滤几个极端日
        for i in range(max(1, n - 60), n):
            prev = c[i - 1]
            if prev == 0:
                continue
            pct = (c[i] - prev) / prev * 100
            body_range = abs(c[i] - o[i])
            total_range = h[i] - l[i]
            body_ratio = body_range / total_range if total_range > 0 else 1.0
            if pct >= limit_pct * 0.95 and body_ratio >= 0.5:
                limit_up[i] = True
            elif pct <= -limit_pct * 0.95 and body_ratio >= 0.5:
                limit_down[i] = True
        limit_up_10d = int(np.sum(limit_up[-10:]))
        limit_down_10d = int(np.sum(limit_down[-10:]))

        # 1. 60日区间位置
        r_hi = float(np.max(h[-60:]))
        r_lo = float(np.min(l[-60:]))
        pos = (c[-1] - r_lo) / (r_hi - r_lo) * 100.0 if r_hi > r_lo else 50.0

        # 2. 量能趋势
        v10 = float(np.mean(v[-10:]))
        v20 = float(np.mean(v[-20:]))
        v50 = float(np.mean(v[-50:])) if n >= 50 else v10
        vol_ratio = v10 / v50 if v50 > 0 else 1.0

        # 3. 支撑测试 (60日) — 排除跌停日（非真实测试）
        lo60 = float(np.min(l[-60:]))
        supp_mask = l[-60:] <= lo60 * 1.03
        supp_mask &= ~limit_down[-60:]
        support_tests = int(np.sum(supp_mask))

        # 4. 阻力测试 — 排除涨停日
        hi60 = float(np.max(h[-60:]))
        resis_mask = h[-60:] >= hi60 * 0.97
        resis_mask &= ~limit_up[-60:]
        resistance_tests = int(np.sum(resis_mask))

        # 5. 高潮柱 (宽振幅+巨量)
        avg_r30 = float(np.mean(np.abs(h[-30:] - l[-30:])))
        avg_v20_feat = float(np.mean(v[-20:])) if n >= 20 else 1
        climax_idx = -1
        for i in range(-1, -30, -1):  # 从最新往最旧找，找到最近的一根
            rng = float(h[i] - l[i])
            if rng > avg_r30 * 1.5 and v[i] > avg_v20_feat * 2.0:
                climax_idx = i
                break

        # 6. 下影线 — 排除一字板（非真实供需）
        lw_ratios = []
        lw_count = 0
        for i in range(-20, 0):
            if limit_up[i] or limit_down[i]:  # 一字板无真实影线
                lw_ratios.append(0.5)  # 中性值，不影响均值
                continue
            body_bot = min(c[i], o[i])
            total = float(h[i] - l[i])
            r = (body_bot - float(l[i])) / total if total > 0 else 0
            lw_ratios.append(r)
            if r > 0.35:
                lw_count += 1
        avg_lw = float(np.mean(lw_ratios)) if lw_ratios else 0

        # 7. 上影线 — 排除一字板
        uw_ratios = []
        uw_count = 0
        for i in range(-20, 0):
            if limit_up[i] or limit_down[i]:  # 一字板无真实影线
                uw_ratios.append(0.5)  # 中性值
                continue
            body_top = max(c[i], o[i])
            total = float(h[i] - l[i])
            r = (float(h[i]) - body_top) / total if total > 0 else 0
            uw_ratios.append(r)
            if r > 0.4:
                uw_count += 1
        avg_uw = float(np.mean(uw_ratios)) if uw_ratios else 0

        # 8. MA50 / MA200
        ma50 = float(np.mean(c[-50:]))
        ma200 = float(np.mean(c[-200:])) if n >= 200 else ma50
        ma_gap = (ma50 / ma200 - 1) * 100.0 if ma200 > 0 else 0
        above_ma50 = c[-1] > ma50

        # 9. ATR 趋势
        atr_recent = cls._w_atr(h[-20:], l[-20:], c[-20:], 14) if n >= 20 else 0
        atr_hist = cls._w_atr(h[-60:-20], l[-60:-20], c[-60:-20], 14) if n >= 60 else 0
        atr_trend = atr_recent / atr_hist if atr_hist > 0 else 1.0

        # 10. 缩量反弹 (climax 后量萎缩)
        vol_dried_up = False
        if climax_idx < -4:  # climax 至少在4天前
            end = None if climax_idx + 6 >= 0 else max(climax_idx + 6, -1)
            post_vol = np.mean(v[climax_idx + 1: end]) if end is not None else np.mean(v[climax_idx + 1:])
            climax_vol = float(v[climax_idx])
            if climax_vol > 0 and post_vol < climax_vol * 0.6:
                vol_dried_up = True

        # 11. 波幅收窄
        atr10_recent = cls._w_atr(h[-10:], l[-10:], c[-10:], 10) if n >= 10 else 0
        atr20_recent = cls._w_atr(h[-20:], l[-20:], c[-20:], 14) if n >= 20 else 0
        atr30_hist = cls._w_atr(h[-30:], l[-30:], c[-30:], 14) if n >= 30 else 0
        range_contract = (atr10_recent / atr20_recent <= 1.2) if atr20_recent > 0 else True
        range_contract_tight = (atr10_recent / atr30_hist <= 0.85) if atr30_hist > 0 else False

        # 12. 最近low是否未创新低
        lo30 = float(np.min(l[-30:]))
        lo10 = float(np.min(l[-10:]))
        not_new_low = (lo10 >= lo30 * 0.995)

        # 13. highs 是否依次降低
        h_first = float(np.max(h[-10:-7])) if n >= 10 else 0
        h_last  = float(np.max(h[-3:]))
        highs_lowering = h_last < h_first * 0.97

        return {
            "pos": pos, "vol_ratio": vol_ratio, "v10": v10, "v50": v50, "v20": v20,
            "support_tests": support_tests, "resistance_tests": resistance_tests,
            "climax_idx": climax_idx, "vol_dried_up": vol_dried_up,
            "lw_count": lw_count, "lw_10_count": int(np.sum([1 for i in range(-10, 0)
                if not (limit_up[i] or limit_down[i])  # 排除一字板
                and (min(c[i], o[i]) - l[i]) / max(h[i] - l[i], 0.001) > 0.3])),
            "uw_count": uw_count, "avg_lw": avg_lw, "avg_uw": avg_uw,
            "ma50": ma50, "ma200": ma200, "ma_gap": ma_gap, "above_ma50": above_ma50,
            "atr_trend": atr_trend, "range_contract": range_contract,
            "range_contract_tight": range_contract_tight,
            "not_new_low": not_new_low, "highs_lowering": highs_lowering,
            "r_hi": r_hi, "r_lo": r_lo,
            "limit_up_10d": limit_up_10d, "limit_down_10d": limit_down_10d,
        }

    @classmethod
    def classify_phase(cls, closes, highs, lows, opens, volumes,
                       events=None):
        """
        纯价量驱动的威科夫阶段分类器。
        不依赖 MA 交叉，仅用价格行为、量能、影线判断。

        Args:
            closes/highs/lows/opens/volumes: 日线数组（numpy 或 list）
            events: [(signal_name, score), ...] 如 [("Spring", 65), ("SOS", 70)]

        Returns:
            (phase_label, confidence, description)

        判定规则: 每个阶段有强制性必须满足的条件(hard gates)。
        不满足任何必要条件→跳过该阶段。conf < 40 → 降级为 Range。
        """
        if events is None:
            events = []
        event_names = {e[0] if isinstance(e, (tuple, list)) else str(e)
                       for e in events}

        f = cls._phase_features(closes, highs, lows, opens, volumes)
        if f is None:
            return ("数据不足", 0, "K线<60根")

        pos = f["pos"]
        vol_ratio = f["vol_ratio"]
        supp = f["support_tests"]
        resis = f["resistance_tests"]
        has_climax = f["climax_idx"] != -1  # -1 表示没找到，其他值表示有
        dried = f["vol_dried_up"]
        lw_c = f["lw_count"]
        lw_10 = f["lw_10_count"]
        uw_c = f["uw_count"]
        ma_gap = f["ma_gap"]
        above = f["above_ma50"]
        atr = f["atr_trend"]
        contr = f["range_contract"]
        contr_tight = f["range_contract_tight"]
        no_new_lo = f["not_new_low"]
        hi_down = f["highs_lowering"]
        climax_at = f["climax_idx"]

        def _conf(core_ok, core_total, bonus_ok, bonus_total):
            raw = int((core_ok / core_total) * 60 + (bonus_ok / max(1, bonus_total)) * 40)
            return max(10, min(100, raw))

        desc = f"pos={pos:.0f}% vol={vol_ratio:.2f} supp={supp}次 lw={lw_c} uw={uw_c} " \
               f"ma_gap={ma_gap:.0f}% atr_tr={atr:.2f}"

        # ═══════════════════════════════════════════
        # 1. Markdown — 下降趋势
        # ═══════════════════════════════════════════
        md_core = [
            (pos <= 20, pos <= 20),
            (ma_gap is not None and (ma_gap < 5 or pos < 8), f"MA弱/死叉(gap={ma_gap:.0f}%)"),
            (not above, not above),
        ]
        md_has_accum = (supp >= 3 and lw_c >= 4 and vol_ratio <= 1.1 and atr <= 1.2)
        if all(v for v, _ in md_core) and not md_has_accum:
            md_bonus = [hi_down, not any("Spring" in n for n in event_names), atr >= 1.0]
            conf = _conf(len(md_core), len(md_core), sum(md_bonus), len(md_bonus))
            if conf >= 40:
                return ("Markdown", conf, desc + " 下降趋势")

        # ═══════════════════════════════════════════
        # 2. Markup — 主升段
        # ═══════════════════════════════════════════
        mu_core = [
            (pos >= 70, pos >= 70),
            (above, above),
            (ma_gap is not None and ma_gap > 0, ma_gap is not None and ma_gap > 0),
        ]
        if all(v for v, _ in mu_core):
            mu_bonus = [
                vol_ratio >= 0.9,
                any("SOS" in n for n in event_names),
                uw_c <= 5,
                atr >= 1.0,
            ]
            conf = _conf(len(mu_core), len(mu_core), sum(mu_bonus), len(mu_bonus))
            if conf >= 40:
                return ("Markup", conf, desc + " 主升段")

        # ═══════════════════════════════════════════
        # 3. Accum_A — 止跌 (Selling Climax + 自动反弹)
        # ═══════════════════════════════════════════
        aa_core = [
            (pos <= 35, pos <= 35),
            (has_climax, has_climax),
        ]
        if all(v for v, _ in aa_core):
            aa_bonus = [dried, lw_10 >= 3, vol_ratio <= 0.9]
            conf = _conf(len(aa_core), len(aa_core), sum(aa_bonus), len(aa_bonus))
            if conf >= 40:
                return ("Accum_A", conf, desc + " 止跌(SC+反弹)")

        # ═══════════════════════════════════════════
        # 4. Accum_B — 吸筹区间
        # ═══════════════════════════════════════════
        ab_core = [
            (pos <= 45, pos <= 45),
            (vol_ratio <= 1.2, vol_ratio <= 1.2),
            (supp >= 3, supp >= 3),
            (lw_c >= 3, lw_c >= 3),
            (not hi_down or not above, True),  # 不处于下降加速
        ]
        if all(v for v, _ in ab_core):
            ab_bonus = [
                lw_c >= 5,
                contr,
                any("Spring" in n for n in event_names),
                abs(ma_gap) < 15 if ma_gap is not None else True,
            ]
            conf = _conf(len(ab_core), len(ab_core), sum(ab_bonus), len(ab_bonus))
            if conf >= 40:
                return ("Accum_B", conf, desc + " 吸筹区间")

        # ═══════════════════════════════════════════
        # 5. Accum_C — 最后回踩（缩量测试）
        # ═══════════════════════════════════════════
        ac_core = [
            (pos <= 35, pos <= 35),
            (vol_ratio <= 0.75, vol_ratio <= 0.75),
            (no_new_lo, no_new_lo),
        ]
        if all(v for v, _ in ac_core):
            ac_bonus = [
                contr_tight,
                lw_c >= 5,
                abs(ma_gap) < 8 if ma_gap is not None else True,
            ]
            conf = _conf(len(ac_core), len(ac_core), sum(ac_bonus), len(ac_bonus))
            if conf >= 40:
                return ("Accum_C", conf, desc + " 最后回踩(缩量测试)")

        # ═══════════════════════════════════════════
        # 6. Distribute_A — 派发初期（Buying Climax）
        # ═══════════════════════════════════════════
        hi_stifled = False
        if climax_at != -1:  # 有高潮柱
            hi_after = float(np.max(highs[climax_at:])) if abs(climax_at) <= len(highs) else 0
            hi_stifled = hi_after <= f["r_hi"] * 0.99
        da_core = [
            (pos >= 65, pos >= 65),
            (has_climax, has_climax),
        ]
        if all(v for v, _ in da_core):
            da_bonus = [
                hi_stifled,
                uw_c >= 5,
                vol_ratio > 1.0,
                any("Upthrust" in n for n in event_names),
            ]
            conf = _conf(len(da_core), len(da_core), sum(da_bonus), len(da_bonus))
            if conf >= 40:
                return ("Distribute_A", conf, desc + " 派发初期(BC)")

        # ═══════════════════════════════════════════
        # 7. Distribute_B — 派发区间
        # ═══════════════════════════════════════════
        db_core = [
            (pos >= 50, pos >= 50),
            (uw_c >= 5, uw_c >= 5),
        ]
        if all(v for v, _ in db_core):
            db_bonus = [
                vol_ratio >= 1.1,
                resis >= 3,
                ma_gap is not None and ma_gap <= 5,
                any("Upthrust" in n for n in event_names),
                atr > 1.2,
            ]
            conf = _conf(len(db_core), len(db_core), sum(db_bonus), len(db_bonus))
            if conf >= 40:
                return ("Distribute_B", conf, desc + " 派发区间")

        # ═══════════════════════════════════════════
        # 8. Range — 中性区间（含方向提示）
        # ═══════════════════════════════════════════
        hint = ""
        if pos <= 45 and supp >= 2 and vol_ratio <= 1.1:
            hint = " 偏吸筹"
        elif pos >= 65 and uw_c >= 4:
            hint = " 偏派发"
        elif above and (ma_gap is not None and ma_gap > 0):
            hint = " 偏多"
        elif not above:
            hint = " 偏空"
        return ("Range", max(20, int(50 - abs(pos - 50) / 2)), desc + hint)

    @classmethod
    def analyze_all(cls, closes, highs, lows, opens, volumes):
        """
        返回该股票所有威科夫信号（全量检测，不做阶段过滤）。
        将 opens 传入以备 classify_phase 使用。
        return: (signals list, extra dict with accum_stage/markup)
        """
        if len(closes) < 30:
            return [("数据不足", 0, "")], {}

        signals = []

        # ---- Spring 弹簧（两套参数集取高分）----
        sig = cls.detect_spring(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- SOS 强势信号 ----
        sig = cls.detect_sos(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- LPS 最后支撑点 ----
        sig = cls.detect_lps(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- Upthrust 上冲回落 ----
        sig = cls.detect_upthrust(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- EVR 努力无结果 ----
        sig = cls.detect_evr(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- Compression 压缩蓄势 ----
        sig = cls.detect_compression(closes, highs, lows, volumes)
        if sig:
            signals.append(sig)

        # ---- 额外信息 ----
        extra = {}
        sig = cls.detect_accum_stage(closes, highs, lows, volumes)
        if sig:
            extra["accum_stage"] = sig

        sig = cls.detect_markup(closes, highs, lows, volumes)
        if sig:
            extra["markup"] = sig

        # 按得分降序
        signals.sort(key=lambda x: -x[1])
        return signals, extra

    # -------------------------------------------------------
    # Spring 弹簧（v3 — 第一性原理重写）
    # -------------------------------------------------------
    @staticmethod
    def _find_support_zone(closes, highs, lows, window=30, min_touches=2, percentile=20, zone_pct=2.0):
        """
        从交易区间中寻找支撑区域（不是最低点，是价格区间底部）。

        Wyckoff 第一性原理：支撑是需求战胜供给的**区域**，不是单条线。
        区间需要多次测试才确认有效。

        返回:
            (support_level, touch_count, range_width_pct, is_valid)
        """
        if len(lows) < window:
            return (None, 0, 0, False)

        recent_lows = lows[-window:]
        recent_highs = highs[-window:]

        # 支撑 = 20% 分位数（剔除毛刺/恐慌低点）
        support = float(np.percentile(recent_lows, percentile))
        if support <= 0:
            return (None, 0, 0, False)

        # 测试次数：low 在支撑附近 ±zone_pct% 内的次数
        touches = sum(1 for l in recent_lows
                      if abs(l - support) / support * 100 < zone_pct)

        # 区间上沿 = 80% 分位数（剔除极端高点）
        range_top = float(np.percentile(recent_highs, 80))
        range_width = (range_top - support) / support * 100 if support > 0 else 0

        # 有效条件：有多次测试 + 区间有足够宽度（否则是横盘/死水）
        is_valid = touches >= min_touches and range_width >= 2.0

        return (support, touches, range_width, is_valid)

    @classmethod
    def _detect_spring_core(cls, closes, highs, lows, volumes,
                            support, touches, range_width,
                            config_section="spring"):
        """
        Spring 核心检测逻辑，参数化可同时用于多头/空头。

        关键修复：
        1. 不双重计算（深度×弹跳）。改用"收盘在区间中的位置"评分
        2. 浅探 = 高分（卖压枯竭），深探 = 低分/扣分
        3. 量能用 Spring 后趋势判断，不用单日量比
        4. 支撑质量基于测试次数，不是 single min
        """
        cfg = cls._cfg(config_section)
        dw = cfg.get("detect_window", [-10, None])
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})
        max_pen = scoring.get("max_penetration", 10.0)
        recovery_window = cfg.get("recovery_window", 3)

        lookback = lows[dw[0]:dw[1]]
        closes_lb = closes[dw[0]:dw[1]]

        best = None
        for i in range(len(lookback)):
            low = float(lookback[i])
            close = float(closes_lb[i])

            penetration = (support - low) / support * 100
            if penetration <= 0:
                continue

            # --- 同日 Spring ---
            if close > support:
                if penetration > max_pen:
                    continue
                deep_pen = penetration
                recovery_close = close
                delay = 0
                multi_day = False

            # --- 多日 Spring ---
            else:
                recovery_day = -1
                for j in range(i + 1, min(i + 1 + recovery_window, len(lookback))):
                    if float(closes_lb[j]) > support:
                        recovery_day = j
                        break
                if recovery_day < 0:
                    continue
                deep_pen = max(
                    (support - float(lookback[k])) / support * 100
                    for k in range(i, recovery_day + 1)
                )
                if deep_pen > max_pen:
                    continue
                recovery_close = float(closes_lb[recovery_day])
                delay = recovery_day - i
                multi_day = True
            if penetration > max_pen:     # 穿透太深，不是 Spring
                continue

            parts = []
            score = 0

            # 1. 收盘在区间中的位置 (0-30)
            if range_width > 0:
                close_in_range = (recovery_close - support) / (support * range_width / 100) * 100
                if close_in_range >= 80:
                    score += 30; parts.append(f"收高位({close_in_range:.0f}%)")
                elif close_in_range >= 50:
                    score += 22; parts.append(f"收中位({close_in_range:.0f}%)")
                elif close_in_range >= 20:
                    score += 12; parts.append(f"收低位({close_in_range:.0f}%)")
                else:
                    score += 0; parts.append("贴支撑")

            # 2. 支撑质量 (0-25)
            if touches >= 5:
                score += 22; parts.append(f"强支撑({touches}次)")
            elif touches >= 3:
                score += 15; parts.append(f"支撑({touches}次)")
            elif touches >= 2:
                score += 5; parts.append(f"弱支撑({touches}次)")

            # 3. 穿透深度 (0-15)
            if deep_pen < 1.0:
                score += 15; parts.append(f"浅探{deep_pen:.2f}%")
            elif deep_pen < 3.0:
                score += 10; parts.append(f"中探{deep_pen:.2f}%")
            elif deep_pen < 5.0:
                score += 5; parts.append(f"深探{deep_pen:.2f}%")
            else:
                score += 0; parts.append(f"猛跌{deep_pen:.2f}%")

            # 多日弹簧扣分（迟收回 = 弱信号）
            if multi_day:
                delay_penalty = delay * 5
                score -= delay_penalty
                parts.append(f"迟{delay}天-{delay_penalty}")

            # 4. 量能趋势 (0-20)：Spring 后量能递减 = 卖压枯竭
            effective_idx = recovery_day if multi_day else i
            spring_idx = -(len(lookback) - effective_idx)
            # 取 Spring 后所有可用 K 线（上限 3 根），避免负→正索引空切片
            post_vols = []
            if spring_idx < -1:  # 至少还有 1 根后验
                post_vols = volumes[spring_idx + 1:]  # 到末尾
                if len(post_vols) > 3:
                    post_vols = post_vols[:3]

            vol_declining = (
                len(post_vols) >= 2
                and all(post_vols[j] <= post_vols[j-1] for j in range(1, len(post_vols)))
            )
            # Spring 当日量比
            vol_i = float(volumes[spring_idx]) if spring_idx < 0 and abs(spring_idx) <= len(volumes) else 0
            bg_vol = float(np.mean(volumes[-25:-5])) if len(volumes) >= 25 else 1
            vratio = vol_i / bg_vol if bg_vol > 0 else 1

            if vol_declining and vratio < 1.5:
                score += 20; parts.append("缩量确认")
            elif vol_declining:
                score += 15; parts.append(f"放量+缩量(v{vratio:.1f}x)")
            elif vratio > 2.0:
                score += 8; parts.append(f"巨量(v{vratio:.1f}x)")
            elif vratio > 1.2:
                score += 5; parts.append(f"放量(v{vratio:.1f}x)")
            else:
                score += 3; parts.append(f"量平")

            # 5. 次日确认 (0-15)
            nd_idx = recovery_day if multi_day else i
            nd_close = recovery_close if multi_day else close
            if nd_idx + 1 < len(closes_lb) and float(closes_lb[nd_idx + 1]) > nd_close:
                score += 15; parts.append("确认")

            if score > (best[1] if best else 0):
                sig = "Spring" if score >= thr.get("strong_signal", 60) else thr.get("weak_label", "弱Spring")
                best = (sig, score, " | ".join(parts))

        return best

    @classmethod
    def detect_spring(cls, closes, highs, lows, volumes, trend=""):
        """
        Spring 检测。内部运行两套参数集取高分：
        - 标准：30d / 20%ile / 2次支撑 / 10%穿透
        - 严格：45d / 15%ile / 3次支撑 / 8%穿透 + 评分×0.8
        后者在 Markdown 阶段有更多假信号，故用更严条件过滤。
        """
        cfg = cls._cfg("spring")
        if not cfg.get("enabled", True):
            return None

        # 第1套：标准参数
        sw = cfg.get("support_window", 30)
        result1 = None
        support, touches, range_width, valid = cls._find_support_zone(
            closes, highs, lows, window=sw,
            min_touches=cfg.get("min_touches", 2),
        )
        if valid:
            result1 = cls._detect_spring_core(
                closes, highs, lows, volumes,
                support, touches, range_width,
                config_section="spring",
            )

        # 第2套：严格参数（长窗口/多测试/低分位 + 折扣）
        strict_sw = cfg.get("strict_support_window", 45)
        result2 = None
        support2, touches2, rw2, valid2 = cls._find_support_zone(
            closes, highs, lows, window=strict_sw,
            min_touches=cfg.get("strict_min_touches", 3),
            percentile=cfg.get("strict_percentile", 15),
        )
        if valid2:
            r2 = cls._detect_spring_core(
                closes, highs, lows, volumes,
                support2, touches2, rw2,
                config_section="spring",
            )
            if r2:
                sig, score, detail = r2
                score = int(score * 0.8)
                sig = "Spring" if score >= cfg.get("thresholds", {}).get("strong_signal", 60) else "弱Spring"
                result2 = (sig, score, detail + " [严格]")

        if not result1 and not result2:
            return None
        if not result1:
            return result2
        if not result2:
            return result1
        return result1 if result1[1] >= result2[1] else result2

    # -------------------------------------------------------
    # SOS Sign of Strength 强势信号
    # -------------------------------------------------------
    @classmethod
    def detect_sos(cls, closes, highs, lows, volumes, trend=""):
        """强势信号：大阳线 + 放量 + 高位收盘"""
        cfg = cls._cfg("sos")
        if not cfg.get("enabled", True):
            return None
        if len(closes) < cfg.get("min_bars", 50):
            return None

        # 位阶保护
        mp = cfg.get("ma200_bias_protection", {})
        if mp.get("enabled", True) and len(closes) >= 200:
            ma50 = float(np.mean(closes[-50:]))
            ma200 = float(np.mean(closes[-200:]))
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
    def detect_lps(cls, closes, highs, lows, volumes, trend=""):
        """最后支撑点：放量上攻后，缩量回调至支撑附近"""
        cfg = cls._cfg("lps")
        if not cfg.get("enabled", True):
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

        support_level, touches, _, valid = cls._find_support_zone(
            closes, highs, lows, window=abs(sw[0] - sw[1]),
            min_touches=1, zone_pct=2.0,
        )
        if not valid:
            return None
        cur = float(closes[-1])
        if not (support_level * 0.99 <= cur <= support_level * pr[1]):
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
        if any(float(lows[i]) < support_level and float(closes[i]) > support_level for i in range(st_win[0], st_win[1] or 0)):
            score += scoring.get("support_tested_score", 15); parts.append("支撑验证")

        rn = scoring.get("range_narrowing", {})
        if rn:
            rnw = rn.get("window", [-10, -5, -5, 0])
            recent = highs[rnw[2]:rnw[3]]
            older = highs[rnw[0]:rnw[1]]
            if len(recent) == 0 or len(older) == 0:
                return None
            recent_r = float(max(recent)) - float(min(lows[rnw[2]:rnw[3]]))
            older_r = float(max(older)) - float(min(lows[rnw[0]:rnw[1]]))
            if older_r > 0 and recent_r < older_r * rn.get("ratio_threshold", 0.7):
                score += rn.get("score", 10); parts.append("波幅收窄")

        detail = " | ".join(parts) if parts else ""
        sig = "LPS" if score >= thr.get("strong_signal", 60) else thr.get("weak_label", "弱LPS")
        return (sig, score, detail) if score >= thr.get("min_score", 40) else None

    # -------------------------------------------------------
    # Upthrust 上冲回落（UT/UTAD）
    # -------------------------------------------------------
    @classmethod
    def detect_upthrust(cls, closes, highs, lows, volumes, trend=""):
        """上冲回落：价格突破阻力后迅速收回，放量"""
        cfg = cls._cfg("upthrust")
        if not cfg.get("enabled", True):
            return None
        if len(closes) < cfg.get("min_bars", 20):
            return None

        rw = cfg.get("resistance_window", [-12, -3])
        pr = cfg.get("price_range", [0.97, 1.03])
        vbw = cfg.get("volume_baseline_window", [-15, -3])
        dw = cfg.get("detect_window", [-3, None])
        scoring = cfg.get("scoring", {})
        thr = cfg.get("thresholds", {})

        # 阻力区域（80%分位数），剔除毛刺最高点
        resistance = float(np.percentile(highs[rw[0]:rw[1]], 80))
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
    def detect_evr(cls, closes, highs, lows, volumes, trend=""):
        """努力无结果：放量但价格窄幅波动"""
        cfg = cls._cfg("evr")
        if not cfg.get("enabled", True):
            return None
        if len(closes) < cfg.get("min_bars", 30):
            return None

        mp = cfg.get("ma200_bias_protection", {})
        if mp.get("enabled", True) and len(closes) >= 200:
            ma200 = float(np.mean(closes[-200:]))
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
    def detect_compression(cls, closes, highs, lows, volumes, trend=""):
        """压缩蓄势：ATR 收窄 + 缩量，变盘前夜"""
        cfg = cls._cfg("compression")
        if not cfg.get("enabled", True):
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
    def detect_markup(cls, closes, highs, lows, volumes, trend=""):
        """Markup 主升段：MA50 上穿 MA200 且保持在上方 N 日"""
        cfg = cls._cfg("markup")
        if not cfg.get("enabled", True):
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
    def detect_chan_third_buy(cls, closes, highs, lows, volumes, phase=""):
        """
        缠论第三类买点检测：中枢形成 → 突破上沿 → 回踩不进入

        三买本质：
        趋势已经确立（中枢形成 = 多空力量平衡已被打破），
        第一次回踩不重新进入中枢 = 确认趋势延续的最佳入场点。
        """
        cfg = cls._cfg("chan_third_buy")
        if not cfg.get("enabled", True):
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

            # 上升阶段确认（Markup / Accum_B / Accum_C）
            if phase in ("Markup", "Accum_B", "Accum_C"):
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
    def detect_accum_stage(cls, closes, highs, lows, volumes, trend=""):
        """吸筹期子阶段细分：A 止跌 / B 探底 / C 回踩"""
        cfg = cls._cfg("accumulation")
        if not cfg.get("enabled", True):
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
        ma200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else 0
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
    def detect_phase(cls, closes, highs, lows, volumes, trend=None, events=None, extra=None):
        """
        委托 classify_phase() — 纯价量驱动的威科夫阶段识别。
        保留旧签名兼容外部调用（backtest/scoring 等 10+ 处）。

        return: (phase_label, description, confidence)  — 旧版格式
        """
        opens = (extra or {}).get("opens") if isinstance(extra, dict) else None
        if opens is None:
            opens = closes  # 近似：opens≈closes（影线计算受影响但阶段分类正常）
        phase, conf, desc = cls.classify_phase(closes, highs, lows, opens, volumes, events or [])
        return (phase, desc, conf)


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
    def _fetch_page(page, num=100, timeout=10, node="hs_a"):
        params = {"page": page, "num": num, "sort": "symbol",
                  "asc": "1", "node": node, "_s_r_a": "init"}
        try:
            r = requests.get(Scanner.SINA_HQ_URL, params=params, timeout=timeout)
            return r.json()
        except Exception:
            return None

    def fetch_all_stocks(self, max_pages=None):
        fc = self._sc.get("fetch", {})
        max_pages = max_pages or fc.get("max_pages", 80)
        page_size = fc.get("page_size", 100)
        workers = fc.get("thread_pool", 15)
        timeout = fc.get("request_timeout", 10)
        all_data = []
        # 全局超时：Sina API 超过 45 秒无有效数据就走 AKShare 兜底
        SINA_DEADLINE = 45
        fetch_start = time.time()
        # 同时获取沪市(sh_a)和深市(sz_a)，hs_a已改为仅北交所
        for node in ("sh_a", "sz_a"):
            if time.time() - fetch_start > SINA_DEADLINE:
                break
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pages_per_exchange = max(30, max_pages // 2)
                futures = {}
                for p in range(1, pages_per_exchange + 1):
                    futures[pool.submit(self._fetch_page, p, page_size, timeout, node)] = p
                try:
                    remaining = max(5, SINA_DEADLINE - (time.time() - fetch_start))
                    for f in as_completed(futures, timeout=remaining):
                        try:
                            data = f.result()
                            if data:
                                all_data.extend(data)
                        except Exception:
                            pass
                        if time.time() - fetch_start > SINA_DEADLINE:
                            break
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

            closes, highs, lows, volumes, opens = [], [], [], [], []
            for k in klines:
                try:
                    opens.append(float(k[1]))
                    closes.append(float(k[2]))
                    highs.append(float(k[3]))
                    lows.append(float(k[4]))
                    volumes.append(float(k[5]) * 100)
                except (ValueError, IndexError):
                    pass

            # 趋势判断（保留向后兼容）
            if len(closes) >= 200:
                ma50 = np.mean(closes[-50:])
                ma200 = np.mean(closes[-200:])
                s = (ma50 / ma200 - 1) * 100
                result["trend"] = "多头" if ma50 > ma200 else "空头"
                result["strength"] = round(s, 1)
                # 三重滤网: 月线(MA200≈10月线) / 周线(MA50≈10周线) / 日线(MA20)
                price = closes[-1]
                ma20 = np.mean(closes[-20:])
                result["tf_monthly"] = "多" if price > ma200 else "空"
                result["tf_weekly"] = "多" if price > ma50 else "空"
                result["tf_daily"] = "多" if price > ma20 else "空"
                result["tf_score"] = int(price > ma200) + int(price > ma50) + int(price > ma20)
            elif len(closes) >= 50:
                result["trend"] = "数据不足"
            else:
                result["trend"] = "新股"

            # ─── 三步管道：全量信号 → classify_phase → 阶段过滤 ───
            # 第1步：analyze_all 返回所有信号（不过滤）
            signals = []
            extra_info = {}
            if len(closes) >= 30:
                signals, extra_info = WyckoffAnalyzer.analyze_all(
                    closes, highs, lows, opens, volumes
                )

            # 第2步：classify_phase 确定阶段（用 signals 做 bonus）
            if len(closes) >= 60:
                phase_label, phase_conf, phase_desc = WyckoffAnalyzer.classify_phase(
                    closes, highs, lows, opens, volumes, signals
                )
                result["phase"] = phase_label
                result["phase_conf"] = phase_conf
                result["phase_detail"] = phase_desc
            else:
                phase_label = "数据不足"

            # 第3步：按阶段过滤 signals
            PHASE_SIGNAL_MAP = {
                "Accum_A": ["Accum_Stage"],
                "Accum_B": ["Spring", "Compression", "Accum_Stage"],
                "Accum_C": ["Spring", "SOS", "LPS", "Accum_Stage"],
                "Markup": ["SOS", "LPS", "Markup"],
                "Distribute_A": ["EVR", "Upthrust"],
                "Distribute_B": ["Upthrust", "EVR"],
                "Markdown": ["Spring"],
                "Range": ["Compression"],
            }
            allowed = PHASE_SIGNAL_MAP.get(phase_label, [])
            signals = [s for s in signals if s[0] in allowed]
            if not signals:
                signals = [("-", 0, "")]

            if signals and signals[0][1] > 0:
                result["wyckoff_sig"] = signals[0][0]
                result["wyckoff_score"] = signals[0][1]
                result["wyckoff_detail"] = signals[0][2]

            # 形态学检测（双顶/双底/头肩顶底/V转）
            result["pattern_name"] = "-"
            result["pattern_score"] = 0
            result["pattern_detail"] = ""
            if len(closes) >= 40:
                patterns = PatternDetector.analyze_all(
                    np.array(closes), np.array(highs),
                    np.array(lows), np.array(volumes)
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
                        closes, highs, lows, volumes, phase=result.get("phase", "")
                    )
                    if chan_sig:
                        result["chan_third_buy"] = {
                            "signal": chan_sig[0],
                            "score": chan_sig[1],
                            "detail": chan_sig[2]
                        }
                except Exception:
                    pass

            # 多因子评分（factor_7_validate 补充缺失因子）
            if len(closes) >= 30 and _FACTOR7_OK:
                try:
                    data_dict = {
                        "close": np.array(closes),
                        "high": np.array(highs),
                        "low": np.array(lows),
                        "vol": np.array(volumes),
                        "open": np.array(opens),
                    }
                    mkt = getattr(self, "_market_close_arr", None)
                    f7 = compute_factors_for_stock(data_dict, len(closes) - 1, mkt)
                    if f7:
                        result["_factor_scores"] = f7
                    # 混沌元信息
                    try:
                        from factor_7_validate import compute_chaos_meta_for_stock
                        result["_chaos_meta"] = compute_chaos_meta_for_stock(data_dict, len(closes) - 1)
                    except Exception:
                        pass
                except Exception:
                    pass

        except Exception as e:
            import traceback
            print(f"[WARN] _analyze_stock({code}): {e}")
            traceback.print_exc()

        # 保存OHLCV数组用于Spring质量评分
        if _SPRING_QUALITY_OK and len(closes) >= 30:
            result["_ohlcv"] = {
                "closes": closes, "highs": highs, "lows": lows, "volumes": volumes
            }

        return result

    def _enrich_spring_quality(self, results):
        """对Spring信号个股计算质量评分（后处理，需市场数据已加载）"""
        if not _SPRING_QUALITY_OK:
            return results
        mkt = getattr(self, "_market_close_arr", None)
        if mkt is None or len(mkt) < 20:
            return results
        enriched = 0
        for r in results:
            sig = r.get("wyckoff_sig", "")
            if "Spring" not in sig:
                continue
            ohlcv = r.get("_ohlcv")
            if not ohlcv:
                continue
            try:
                quality = compute_confidence(
                    ohlcv["closes"], ohlcv["highs"], ohlcv["lows"],
                    ohlcv["volumes"], r.get("trend", ""), mkt
                )
                r["spring_quality"] = quality
                enriched += 1
            except Exception:
                pass
        if enriched:
            sys.stdout.write(f"\r  Spring质量评分: {enriched}只\n")
            sys.stdout.flush()
        return results

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
        sector_cache = {}  # 行业评分本地缓存

        for i, stock in enumerate(self.candidates[:max_stocks]):
            analysis = self._analyze_stock(stock["code"])
            # 补充板块热度评分（用于 Kelly 仓位计算）
            ind = analysis.get("industry", "")
            if ind and ind != "其他":
                if ind not in sector_cache:
                    try:
                        from sector_heat import analyze as sector_analyze
                        sr = sector_analyze(ind)
                        sector_cache[ind] = sr.get("composite", 50)
                    except Exception:
                        sector_cache[ind] = 50
                analysis["sector_score"] = sector_cache.get(ind, 50)
            else:
                analysis["sector_score"] = 50
            results.append({**stock, **analysis})
            if (i + 1) % 10 == 0 and total > 10:
                sys.stdout.write(f"\r  分析进度: {i+1}/{total}")
                sys.stdout.flush()
            time.sleep(stock_interval)

        if total > 10:
            sys.stdout.write(f"\r  分析进度: {total}/{total}\n")
            sys.stdout.flush()

        # 市场数据获取（Spring quality + 多因子评分都需要，提前到 _enrich_spring_quality 之前）
        if not getattr(self, "_market_close_arr", None):
            try:
                index_key = "sh000001"
                url = self.TENCENT_KLINE_URL.format(code=index_key)
                r = requests.get(url, timeout=15)
                data = r.json()
                klines = data.get("data", {}).get(index_key, {}).get("qfqday") or []
                index_closes = [float(k[2]) for k in klines if k[2]]
                if len(index_closes) > 20:
                    self._market_close_arr = np.array(index_closes)
            except Exception:
                pass

        # Spring 信号质量评分（需市场数据，先于 _post_process_scores 执行）
        results = self._enrich_spring_quality(results)

        # 将 Spring quality bonus 加到 wyckoff_score（走正常加权路径）
        # 回测结论：>=80 高胜率 bonus +20，<40 confidence 不加bonus（低质量Spring自然低排名）
        for r in results:
            sq = r.get("spring_quality", {})
            bonus = sq.get("bonus", 0) if sq else 0
            if bonus:
                r["wyckoff_score"] = min(100, r.get("wyckoff_score", 50) + bonus)

        # 基本面数据提前获取（仅限高分股票，最多 15 只，~4.5s 开销）
        top_by_wyckoff = sorted(results, key=lambda x: -x.get("wyckoff_score", 0))
        top_candidates = [r for r in top_by_wyckoff if r.get("wyckoff_sig", "-") not in ("-", "无信号", "数据不足", "无数据")][:15]
        if top_candidates:
            enriched = self.enrich_with_financials(top_candidates, max_items=15)
            enriched_codes = {r["code"] for r in enriched}
            for r in results:
                if r["code"] in enriched_codes:
                    fin_data = next((e for e in enriched if e["code"] == r["code"]), {})
                    for k in ("pe", "pb", "roe", "total_mv", "chip_support", "chip_resistance"):
                        if k in fin_data:
                            r[k] = fin_data[k]

        # 多因子系统评分后处理（此时 wyckoff_score 已含 Spring quality bonus）
        results = self._post_process_scores(results)

        # 权重快照
        tf_coeffs = {3: 1.15, 2: 1.00, 1: 0.85, 0: 0.60}
        src = getattr(self, "_weight_source", "base")
        snapshot = f"权重来源: {src} | TF系数: tf3={tf_coeffs.get(3):.2f} tf2={tf_coeffs.get(2):.2f} tf1={tf_coeffs.get(1):.2f} tf0={tf_coeffs.get(0):.2f} | {datetime.now().strftime('%m-%d %H:%M')}"
        self._weight_snapshot = snapshot

        # 保存检查点（防止后续步骤崩溃丢数据）
        try:
            self._save_checkpoint()
        except Exception:
            pass

        # 排序：三买 > 系统分 > 威科夫信号得分 > 阶段 > 成交额
        def sort_key(x):
            c3 = 0
            cb = x.get("chan_third_buy")
            if cb and cb["signal"] == "三买":
                c3 = -100
            elif cb and cb["signal"] == "弱三买":
                c3 = -50
            ss = -x.get("system_score", x["wyckoff_score"])
            ws = -x["wyckoff_score"]
            phase_rank = {"Markup": 0, "Accum_B": 1, "Accum_C": 1, "Accum_A": 2,
                          "Range": 3, "Dist_A": 4, "Dist_B": 4,
                          "Distribute_A": 4, "Distribute_B": 4, "Markdown": 5}
            pr = phase_rank.get(x.get("phase", ""), 6)
            return (c3, ss, ws, pr, -x["amount"])

        results.sort(key=sort_key)

        # 行业中性化：同行业最多取 top_n × 15% 只，避免板块扎堆
        top_n = max_stocks or len(results)
        max_per_industry = max(1, int(top_n * 0.15))
        ind_count = {}
        filtered = []
        for r in results:
            ind = r.get("industry", "其他")
            cnt = ind_count.get(ind, 0)
            if cnt >= max_per_industry:
                continue
            ind_count[ind] = cnt + 1
            filtered.append(r)
        results = filtered

        self.results = results
        return results

    # ==================== 检查点保存 ====================

    def _save_checkpoint(self):
        """将当前 results 保存到临时检查点文件，防止后续步骤崩溃丢数据"""
        if not hasattr(self, "results") or not self.results:
            return
        cp = os.path.join(os.path.dirname(__file__), "results_checkpoint.json")
        minimal = []
        for r in self.results:
            minimal.append({
                "code": r.get("code", ""),
                "name": r.get("name", ""),
                "wyckoff_sig": r.get("wyckoff_sig", ""),
                "wyckoff_score": r.get("wyckoff_score", 0),
                "trend": r.get("trend", ""),
                "price": r.get("price", 0),
                "industry": r.get("industry", ""),
            })
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(minimal, f, ensure_ascii=False, indent=2)

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

    # ─── 高级模块状态摘要 ───

    def _advanced_modules_summary(self) -> str:
        """收集 Bayesian / Kelly / RMT / 因子权重 等模块的状态"""
        parts = []

        # Bayesian 置信度
        if _BAYES_OK:
            try:
                bc = BayesianConfidence()
                rpt = bc.report()
                overall = rpt.get("overall", {}).get("mean", 0.5)
                total_trades = rpt.get("overall", {}).get("total", 0)
                parts.append(f"贝叶斯:{overall:.0%}({total_trades}次)")
            except Exception:
                parts.append("贝叶斯:--")

        # Kelly 仓位参考
        if _KELLY_OK:
            try:
                stats = load_factor_stats()
                kelly_pcts = []
                for k, v in stats.items():
                    wr = v.get("win_rate", 0.5)
                    aw = v.get("avg_win", 3.0)
                    al = v.get("avg_loss", 2.0)
                    f = kelly_fraction(wr, aw, al)
                    kelly_pcts.append(f)
                avg_kelly = np.mean(kelly_pcts) if kelly_pcts else 0
                parts.append(f"Kelly:{avg_kelly:.0%}")
            except Exception:
                parts.append("Kelly:--")

        # RMT + 因子权重偏离
        if _WEIGHTS_OK:
            try:
                base = BASE_WEIGHTS
                eff = get_weights()
                changed = []
                for k in base:
                    if k in eff:
                        diff = eff[k] - base[k]
                        if abs(diff) >= 0.01:
                            changed.append(f"{k}{eff[k]:.0%}")
                parts.append(f"权重调整:{' '.join(changed) if changed else '无'}")
            except Exception:
                parts.append("权重调整:--")

        if not parts:
            return ""

        return "  [系统状态] " + " | ".join(parts)

    def _chaos_summary(self, results) -> str:
        """提取混沌状态摘要"""
        chaos_ok = sum(1 for r in results if r.get("_chaos_meta"))
        if not chaos_ok:
            return ""
        # 取 top5 的混沌特征均值
        metas = [r["_chaos_meta"] for r in results[:5] if r.get("_chaos_meta")]
        if not metas:
            return ""
        avg_snr = np.mean([m.get("snr_db", 0) for m in metas])
        avg_svd = np.mean([m.get("svd_entropy", 0.5) for m in metas])
        avg_npe = np.mean([m.get("npe_ratio", 1.0) for m in metas])
        avg_conf = np.mean([m.get("structure_confidence", 0) for m in metas])
        return "  [混沌状态] 个股可预测:{:.0%} 信噪比:{:.0f}dB 熵:{:.2f} NPE:{:.2f}".format(
            avg_conf, avg_snr, avg_svd, avg_npe)

    # ─── 因子名称到权重名称的映射 ───

    _F2W_MAP = {
        "risk_reward": "risk_reward",
        "volume": "volume",
        "candlestick": "candlestick",
        "trend_momentum": "tech_strength",
        "relative_strength": "relative_strength",
        "volatility": "volatility",
        "orbit_compression": "orbit_compression",
        "lyapunov": "lyapunov",
        "hurst": "hurst",
        "fractal_dim": "fractal_dim",
        "attractor_shape": "attractor_shape",
    }

    # ─── 系统评分后处理（RMT权重 + 贝叶斯） ───

    def _post_process_scores(self, results):
        """
        对每只结果应用 RMT 因子权重 + 贝叶斯置信度，生成 system_score。
        同时写入 Kelly 建议仓位和策略建议。
        """
        # 1) 获取 RMT 权重和贝叶斯因子权重调整（市场数据已在 _run_scan 中获取）
        weights = None
        bayes_adj = {}

        if _WEIGHTS_OK:
            try:
                weights = get_weights()
                self._weight_source = "RMT"
            except Exception:
                weights = None
                self._weight_source = "base"
        if _BAYES_OK:
            try:
                bc = BayesianConfidence()
                bayes_adj = bc.factor_weight_adjustment()
            except Exception:
                pass

        # 3) 遍历结果计算 system_score
        for r in results:
            fs = r.get("_factor_scores", {})
            wyckoff = r.get("wyckoff_score", 50)

            if fs and weights:
                weighted_sum = 0.0
                total_w = 0.0
                for factor_name, weight_name in self._F2W_MAP.items():
                    w = weights.get(weight_name, 0)
                    if w > 0:
                        w *= bayes_adj.get(weight_name, 1.0)  # 贝叶斯因子级调整
                        f_val = fs.get(factor_name, 50)
                        weighted_sum += f_val * w
                        total_w += w
                # 将 wyckoff 分作为额外加权项（用权重的均值作为其权重）
                wyckoff_w = np.mean(list(weights.values())) if weights else 0.15
                weighted_sum += wyckoff * wyckoff_w
                total_w += wyckoff_w

                factor_score = weighted_sum / total_w if total_w > 0 else wyckoff
                factor_score = max(0, min(100, factor_score))
                r["system_score"] = round(factor_score, 1)
            else:
                # 回退到原有的 wyckoff_score
                r["system_score"] = float(wyckoff)

            # 5) 三重滤网调整: 基于回测数据 (2026-06-01, t=3.34显著)
            # tf=3:+0.70%/周 → ×1.15, tf=2:+0.14% → ×1.00
            # tf=1:+0.18% → ×0.85, tf=0:-0.02% → ×0.60
            tf = r.get("tf_score", 0)
            tf_coeffs = {3: 1.15, 2: 1.00, 1: 0.85, 0: 0.60}
            coeff = tf_coeffs.get(tf, 1.0)
            r["system_score"] = min(100, r["system_score"] * coeff) if tf == 3 else r["system_score"] * coeff
            r["system_score"] = round(r["system_score"], 1)

            # 5b) 基本面微调（低 PE 加分，高 ROE 加分）
            pe = r.get("pe")
            roe = r.get("roe")
            adj = 0
            if pe is not None and pe > 0:
                if pe < 15:
                    adj += 3
                elif pe > 40:
                    adj -= 3
            if roe is not None:
                if roe > 15:
                    adj += 3
            if adj:
                r["system_score"] = min(100, max(0, r["system_score"] + adj))
                r["system_score"] = round(r["system_score"], 1)

            # 6) Kelly 仓位建议（完整版: compute_kelly_position）
            r["kelly_pct"] = 0.0
            if _KELLY_OK:
                try:
                    fs = r.get("_factor_scores", {})
                    # 构建 factors 列表 (breakdown_lines 格式)
                    kelly_factors = []
                    for factor_name, weight_name in self._F2W_MAP.items():
                        w = weights.get(weight_name, 0) if weights else 0
                        if w > 0:
                            fv = fs.get(factor_name, {})
                            f_score = fv.get("score", fv) if isinstance(fv, dict) else (fv if isinstance(fv, (int, float)) else 50)
                            item = {"key": weight_name, "weight": w, "score": f_score}
                            if factor_name == "risk_reward":
                                fv = fs.get(factor_name, 50)
                                f_score = fv if isinstance(fv, (int, float)) else fv.get("score", 50)
                                # 反向映射 score → ratio: scoring.py 映射表
                                rr_score_to_ratio = {95: 3.5, 80: 2.5, 65: 2.0, 50: 1.5, 35: 1.0, 20: 0.5, 10: 0.3}
                                rr_ratio = rr_score_to_ratio.get(round(f_score / 10) * 10, 1.5)
                                item = {"key": weight_name, "weight": w, "score": f_score, "ratio": rr_ratio}
                            kelly_factors.append(item)
                    # TF 调整也应用到 Kelly 因子分，与 system_score 保持一致
                    if coeff != 1.0:
                        for f_item in kelly_factors:
                            f_item["score"] = min(100, f_item["score"] * coeff)
                    kelly_result = compute_kelly_position(
                        r["system_score"], kelly_factors,
                        market_score=getattr(self, "_market_score", 50),
                        sector_score=r.get("sector_score", 50)
                    )
                    r["kelly_pct"] = round(kelly_result["position_factor"], 2)
                except Exception:
                    r["kelly_pct"] = 0.0

        return results

    # ==================== 报告输出 ====================

    def print_report(self, top_n=15):
        import socket as _sckt
        _old_to = _sckt.getdefaulttimeout()
        _sckt.setdefaulttimeout(8)

        # === monkey-patch: requests 全局 timeout (防Tushare/AKShare挂死) ===
        import requests as _req
        if not getattr(_req.Session, '_patched_timeout', False):
            _orig_request = _req.Session.request
            def _patched_request(self, method, url, **kwargs):
                kwargs.setdefault('timeout', (5, 8))
                return _orig_request(self, method, url, **kwargs)
            _req.Session.request = _patched_request
            _req.Session._patched_timeout = True
        # ================================================================

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
        if hasattr(self, "_weight_snapshot"):
            print(f"  [{self._weight_snapshot}]")
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

        # 大盘健康状况（提前计算用于判断）
        try:
            from market import MarketAnalyzer
            indices = MarketAnalyzer.fetch_indices()
        except Exception:
            indices = {}
        breadth = self._calc_breadth()

        # 综合健康判断（显示在最顶部）
        health_p = 0
        health_w = 0
        if indices:
            avg_chg = np.mean([i["change_pct"] for i in indices.values()])
            if avg_chg >= 0.5: health_p += 1
            elif avg_chg <= -0.5: health_w += 1
        if breadth:
            if breadth["ad_ratio"] >= 1.5: health_p += 1
            elif breadth["ad_ratio"] < 0.8: health_w += 1
            if breadth["ul_ratio"] >= 3: health_p += 1
            elif breadth["ul_ratio"] < 1: health_w += 1
        hl_score = health_p - health_w
        if hl_score >= 2: hl = "健康 — 可积极参与"
        elif hl_score >= 1: hl = "较好 — 精选个股操作"
        elif hl_score >= 0: hl = "一般 — 注意仓位管理"
        elif hl_score >= -1: hl = "偏弱 — 建议降低仓位"
        else: hl = "较差 — 防守为主"
        breadth_data = breadth or {}
        # 综合判断
        print(f"\n  {'='*60}")
        print(f"  >>> 大盘综合判断: {hl}")

        # 指数 + 涨跌（一行）
        idx_line = ""
        if indices:
            for key in ["sh000001", "sz399001", "sz399006", "sh000688"]:
                idx = indices.get(key)
                if idx: idx_line += f"{idx['name']}{idx['change_pct']:+.2f}% "
        bd = breadth_data
        if bd:
            ad = bd.get("ad_ratio", 0)
            up = bd.get("up", 0); down = bd.get("down", 0)
            lu = bd.get("limit_up", 0); ld = bd.get("limit_down", 0)
            ad_label = "普涨" if ad >= 2 else ("涨多跌少" if ad >= 1.2 else ("偏弱" if ad >= 0.8 else "极弱"))
            idx_line += f"| 涨跌:{up}/{down}({ad_label}) 涨停:{lu} 跌停:{ld}"
        try:
            pe_info, cap = MarketAnalyzer.get_pe_percentile()
            if pe_info: idx_line += f" | PE{pe_info['pe']} {pe_info['percentile']}%分位"
        except Exception:
            pass
        print(f"  指数: {idx_line}")
        print(f"  {'='*60}\n")

        # 高级模块状态
        sys_status = self._advanced_modules_summary()
        if sys_status:
            print(sys_status)

        # 混沌状态摘要
        chaos_line = self._chaos_summary(self.results)
        if chaos_line:
            print(chaos_line)

        if not self.results:
            print("  无符合条件标的")
            return

        # === 市场情绪（压缩到一行） ===
        if _SENTIMENT_TOOL:
            try:
                d = get_daily_sentiment()
                w = get_weekly_sentiment()
                m = get_monthly_sentiment()
                c = get_sentiment()

                def items_line(items):
                    return " | ".join(f"{it['name']}{it['value']}[{it['level']}]" for it in items)

                mom_icon = {"升温": "↗", "降温": "↘", "持平": "→"}.get(w["momentum"], "")
                print(f"  情绪: 综合{c['score']}[{c['level']}] | 日{d['score']}[{d['level']}] {items_line(d['items'])} | 周{w['score']}[{w['level']}] {mom_icon}Δ{w['delta']:+d}")
            except Exception:
                pass

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
                import functools
                @functools.lru_cache(maxsize=32)
                def _cached_heat(name):
                    return get_sector_heat(name)
                heat = _cached_heat(ind)
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
                    sug = "▶ 持有"
                    if s["trend"] == "空头":
                        sug = "▼ 警惕止损"
                    elif s["wyckoff_sig"] in ("Upthrust",) and s.get("pnl_pct", 0) is not None and s["pnl_pct"] > 5:
                        sug = "▼ 减仓"
                    elif s["wyckoff_sig"] in ("Spring", "SOS", "LPS"):
                        sug = "▲ 加仓/持有"
                    elif s.get("pnl_pct") is not None and s["pnl_pct"] < -15:
                        sug = "!! 止损"
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
        print(f"  {'#':<3} {'代码':<8} {'名称':<7} {'板块':<12} {'趋势':<10} {'威科夫':<18} {'系统分':<6} {'盈亏比':<6} {'三滤':<4} {'量信':<4} {'价格':<8} {'成交额':<8} {'Kelly':<6} {'质地':<6}")
        print(f"  " + "-" * 136)

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

            # 系统分（多因子加权 + RMT + 贝叶斯 + 三重滤网）
            ss = r.get("system_score", sc)
            ss_str = f"{ss:.0f}" if isinstance(ss, (int, float)) else "-"
            # 盈亏比因子
            fs = r.get("_factor_scores", {})
            rr_val = fs.get("risk_reward", fs.get("盈亏比", None)) if isinstance(fs, dict) else None
            rr_str = f"{rr_val:.0f}" if isinstance(rr_val, (int, float)) else "-"

            # 三重滤网状态
            m = r.get("tf_monthly", "-")
            w = r.get("tf_weekly", "-")
            d = r.get("tf_daily", "-")
            tf_str = f"{m}{w}{d}"

            # Kelly 仓位建议
            kp = r.get("kelly_pct", 0)
            kelly_str = f"{kp:.0%}" if isinstance(kp, (int, float)) else "-"

            # Spring 质量置信度
            sq = r.get("spring_quality", {})
            sq_conf = sq.get("confidence", 0) if sq else 0
            sq_str = f"{sq_conf:.0f}" if "Spring" in r.get("wyckoff_sig", "") and sq_conf > 0 else "-"

            price = f"{r['price']:.2f}"
            amt = f"{r['amount']/1e8:.1f}亿"
            qf = "通过" if r.get("quality_passed") else ("未检" if r.get("quality_passed") is None else "未过")
            marker = ">>" if i < 3 else "  "
            print(f"  {marker}{i+1:<3} {sym:<8} {name:<7} {ind:<12} {trend:<10} {sig_str:<18} {ss_str:<6} {rr_str:<6} {tf_str:<4} {sq_str:<4} {price:<8} {amt:<8} {kelly_str:<6} {qf:<6}")

        # 三重滤网统计
        tf_counts = {}
        for r in self.results:
            m = r.get("tf_monthly", "?")
            w = r.get("tf_weekly", "?")
            d = r.get("tf_daily", "?")
            tag = f"{m}{w}{d}"
            tf_counts[tag] = tf_counts.get(tag, 0) + 1
        print(f"  三重滤网分布: " +
              " | ".join(f"{k}={v}只" for k, v in sorted(tf_counts.items(),
                       key=lambda x: (sum(1 for c in x[0] if c == '多'), -x[1]), reverse=True)))
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
                    print(f"  {'信号':<12} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'可信':<5} {'PE':<7} {'ROE':<6} {'细节':<24}")
                    print(f"  " + "-" * 95)
                    for r in buy_sigs:
                        sym = r["code"].split(".")[1]
                        name = r["name"][:6]
                        ind = r.get("industry", "其他")[:10]
                        sig = r["wyckoff_sig"]
                        sc = r["wyckoff_score"]
                        sq = r.get("spring_quality", {})
                        sq_str = f"{sq['confidence']:.0f}" if sq.get("confidence") else "-"
                        pe = f"{r['pe']:.1f}" if r.get("pe") else "-"
                        roe = f"{r['roe']:.1f}%" if r.get("roe") else "-"
                        detail = r.get("wyckoff_detail", "")[:24]
                        print(f"  {sig:<12} {sym:<8} {name:<7} {ind:<12} {sc:<5} {sq_str:<5} {pe:<7} {roe:<6} {detail:<24}")
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

        _sckt.setdefaulttimeout(_old_to)

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
            # 加载大盘情绪（涨停家数/连板高度/炸板率，补充大盘因子输入）
            try:
                from market_sentiment import get_market_sentiment
                ms = get_market_sentiment()
                self._market_score = ms["score"]
                self._market_sentiment_detail = ms
                print(f"  大盘情绪: {ms['score']}分 [{ms['label']}] 涨停{ms['zting_count']}家 连板{ms['high_board']}板 炸板{ms['zha_ban_rate']:.0%}")
            except Exception:
                pass

            print("[3/3] 威科夫+趋势分析...")
            t0 = time.time()
            self.run_analysis(max_analysis)
            print(f"  完成 ({time.time()-t0:.1f}s)")

            # [3.5/3] 个股质地检查（第三层，仅标记不过滤）
            print("[3.5/3] 个股质地检查 (Tushare)...")
            self.enrich_with_quality(self.results)

        # 无论 quick/full 都分析股票池+持仓
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

            # 清理检查点
            try:
                cp = os.path.join(os.path.dirname(__file__), "results_checkpoint.json")
                if os.path.exists(cp):
                    os.remove(cp)
            except Exception:
                pass

        # 上报选股结果到 system_state
        try:
            from system_state import report_scanner
            report_scanner(self.results[:30] if hasattr(self, "results") else [])
        except Exception:
            pass

        # 更新 _system_state.md (markdown 状态文件)
        try:
            top_n = self._sc.get("report", {}).get("top_n_default", 15)
            top_picks = []
            for i, r in enumerate(self.results[:top_n]):
                code = r["code"].split(".")[1]
                amount = r.get("amount", 0)
                amount_str = f"{amount/1e8:.1f}亿" if amount else "0"
                top_picks.append((
                    i + 1, code, r["name"], r.get("phase", r.get("trend", "-")), r["wyckoff_sig"],
                    r["wyckoff_score"], r.get("system_score", r["wyckoff_score"]),
                    amount_str, r.get("industry", ""),
                ))
            update_state_md(
                scan_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                top_picks=top_picks,
            )
        except Exception:
            pass


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
