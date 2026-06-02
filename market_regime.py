# -*- coding: utf-8 -*-
"""
市场状态分类器

基于 Tushare 公开数据识别 A 股当前市场阶段:
  底部 / 上升 / 加速 / 顶部 / 下跌 / 震荡

数据源: 上证指数日线、全市场成交额、两融余额、跌停家数、涨跌家数比
所有计算滚动窗口内完成，不依赖外部付费数据。

用法:
  from market_regime import get_regime
  r = get_regime(date="20260530")  # 或 get_regime() 取最新
"""
import sys, os, json, time, math
import numpy as np
from datetime import datetime, timedelta

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── 缓存路径 ──
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "output_v2")
_CACHE_FILE = os.path.join(_CACHE_DIR, "_regime_cache.json")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _get_pro():
    import tushare as ts
    return ts.pro_api()


def _load_cache():
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache):
    with open(_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── 数据拉取(带缓存) ──

def _fetch_index_daily(pro, end_date, lookback=120):
    """拉取上证指数日线数据"""
    cache = _load_cache()
    cache_key = f"index_daily_{end_date}"
    if cache_key in cache:
        return cache[cache_key]

    # 计算起始日期
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=lookback + 20)
    start_str = start_dt.strftime("%Y%m%d")

    dfs = []
    # 分批拉取，每次60条
    batch_start = start_str
    while batch_start < end_date:
        batch_end = (datetime.strptime(batch_start, "%Y%m%d") + timedelta(days=60)).strftime("%Y%m%d")
        batch_end = min(batch_end, end_date)
        try:
            df = pro.index_daily(ts_code='000001.SH', start_date=batch_start, end_date=batch_end)
            if df is not None and len(df) > 0:
                dfs.append(df)
            time.sleep(0.2)
        except Exception:
            pass
        batch_start = (datetime.strptime(batch_end, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    if not dfs:
        return None
    import pandas as pd
    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("trade_date").reset_index(drop=True)
    data = result.to_dict(orient="records")
    for r in data:
        r["trade_date"] = str(r["trade_date"])

    cache[cache_key] = data
    _save_cache(cache)
    return data





def _src_to_date_map(records, date_field="trade_date"):
    """将 records 转为 {date_str: record} 映射"""
    return {str(r[date_field]): r for r in records}


def _safe_float(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


# ── 指标计算 ──

def _sma(values, period):
    """简单移动平均"""
    if len(values) < period:
        return np.mean(values) if len(values) > 0 else 0
    return np.mean(values[-period:])


def _calc_all_indicators(index_daily, limit_down_count, advance_ratio,
                            limit_up_count=None, total_stocks=None):
    """
    输入:
      index_daily: [{trade_date, close, pct_chg, vol, ...}]
      limit_down_count: float 跌停家数
      advance_ratio: float 涨家/(涨+跌)
    输出: 5项指标 + 辅助字段
    """
    n_idx = len(index_daily)
    if n_idx < 20:
        return None

    closes = np.array([_safe_float(r, "close") for r in index_daily])
    vols = np.array([_safe_float(r, "vol") for r in index_daily])
    highs = np.array([_safe_float(r, "high") for r in index_daily])
    lows = np.array([_safe_float(r, "low") for r in index_daily])

    # 上证趋势: close / MA20 - 1 (%)
    ma20_close = _sma(closes, 20)
    ma50_close = _sma(closes, 50)
    ma200_close = _sma(closes, 200) if len(closes) >= 200 else None
    idx_trend = (closes[-1] / ma20_close - 1) * 100 if ma20_close > 0 else 0

    # MA 位置关系
    ma50_above_200 = (ma50_close > ma200_close) if ma200_close else None
    ma50_slope = ((closes[-1] / closes[-min(50, len(closes))]) - 1) * 100 if len(closes) >= 50 else 0

    # 量能比: volume / MA(volume, 20)
    ma20_vol = _sma(vols, 20)
    vol_ratio = vols[-1] / ma20_vol if ma20_vol > 0 else 1.0

    # 波动率指数: (high-low)/close 20日标准差
    if n_idx >= 20:
        hilo_ratios = np.array([(highs[i] - lows[i]) / max(closes[i], 0.01)
                                for i in range(max(0, n_idx - 20), n_idx)])
        vol_idx = float(np.std(hilo_ratios) * 100)
    else:
        vol_idx = 0

    # 量能萎缩检测: 最近10日量能 vs 前20日
    if len(vols) >= 30:
        recent_vol = np.mean(vols[-10:])
        prior_vol = np.mean(vols[-30:-10])
        vol_dry_ratio = recent_vol / prior_vol if prior_vol > 0 else 1.0
    else:
        vol_dry_ratio = 1.0

    # 高低点突破检测: 最近20日是否创新高/新低
    if len(closes) >= 40:
        recent_high = np.max(highs[-20:])
        prior_high = np.max(highs[-40:-20])
        recent_low = np.min(lows[-20:])
        prior_low = np.min(lows[-40:-20])
        new_high = recent_high > prior_high * 1.005  # 创新高
        new_low = recent_low < prior_low * 0.995     # 创新低
        lower_highs = np.max(highs[-20:]) < np.max(highs[-40:-20])  # 高点降低
        higher_lows = np.min(lows[-20:]) > np.min(lows[-40:-20])    # 低点抬高
    else:
        new_high = new_low = lower_highs = higher_lows = None

    # 成交量确认: 上涨日 vs 下跌日量能对比
    if len(closes) >= 20:
        gains_vol = np.mean([vols[i] for i in range(-20, 0) if closes[i] > closes[i-1]]) if any(closes[i] > closes[i-1] for i in range(-20, 0)) else 0
        losses_vol = np.mean([vols[i] for i in range(-20, 0) if closes[i] < closes[i-1]]) if any(closes[i] < closes[i-1] for i in range(-20, 0)) else 0
        vol_up_vs_down = gains_vol / losses_vol if losses_vol > 0 else 1.0
    else:
        vol_up_vs_down = 1.0

    # 两融变化率 (暂缺则0)
    margin_change = 0.0

    # 恐慌指数: 跌停家数 / 总家数 (~5000)
    panic = limit_down_count / 5000.0 * 100 if limit_down_count > 0 else 0

    # 赚钱效应: 上涨比例 (用传入的 advance_ratio)
    profit_effect = advance_ratio * 100 if advance_ratio is not None else 50

    return {
        "idx_trend_pct": round(idx_trend, 2),
        "ma50_slope": round(ma50_slope, 2),
        "ma50_above_200": ma50_above_200,
        "vol_ratio": round(vol_ratio, 3),
        "volatility_idx": round(vol_idx, 3),
        "vol_dry_ratio": round(vol_dry_ratio, 3),
        "vol_up_vs_down": round(vol_up_vs_down, 2),
        "new_high": new_high,
        "new_low": new_low,
        "lower_highs": lower_highs,
        "higher_lows": higher_lows,
        "margin_change": round(margin_change, 3),
        "panic_pct": round(panic, 3),
        "profit_effect_pct": round(profit_effect, 1),
        # ── 矛盾论框架预留字段（暂返回默认值，待实现） ──
        "policy_score": _calc_policy_score(),
        "leverage_score": _calc_leverage_score(),
        "sentiment_score": _calc_sentiment_score(
            panic_pct=panic, profit_effect_pct=profit_effect,
            limit_up_count=limit_up_count, limit_down_count=limit_down_count
        ),
        "valuation_score": _calc_valuation_score(),
        "liquidity_score": _calc_liquidity_score(
            vol_ratio=vol_ratio, limit_down_count=limit_down_count,
            total_stocks=total_stocks, advance_ratio=advance_ratio
        ),
    }


# ── 矛盾论框架：五大维度预留接口 ──
#
# 参考《A股的牛熊密码——用矛盾论重新理解市场》:
# 市场不是单一指标驱动的系统，而是一组矛盾的集合体。
# 在周期不同阶段，总有一组矛盾处于支配地位。
# 识别"主导矛盾切换"才是判断市场状态切换的真正信号。
#
# 当前 classifier 只用了技术指标（价格/量/波动），
# 以下 5 个维度暂返回默认值，后续逐步接入真实数据。
# 参见 memory/project/contradiction_theory_regime.md

# ==== 矛盾论 5 维度实现 ====

def _calc_policy_score():
    try:
        pro = _get_pro()
    except Exception:
        return 50.0
    score = 50.0
    # Shibor trend
    try:
        df_shibor = pro.shibor(start_date="20250501", end_date="20260531")
        if df_shibor is not None and not df_shibor.empty:
            for col in ["1w", "1m"]:
                if col in df_shibor.columns:
                    vals = df_shibor[col].dropna().astype(float).values
                    if len(vals) >= 20:
                        recent = vals[-10:]
                        earlier = vals[-20:-10]
                        if len(recent) and len(earlier):
                            chg = (recent.mean() - earlier.mean()) / max(earlier.mean(), 0.01)
                            if chg < -0.1:
                                score += 15
                            elif chg < -0.03:
                                score += 8
                            elif chg > 0.1:
                                score -= 15
                            elif chg > 0.03:
                                score -= 8
    except Exception:
        pass
    # Northbound flow
    try:
        df_hsgt = pro.moneyflow_hsgt(start_date="20260501", end_date="20260531")
        if df_hsgt is not None and not df_hsgt.empty:
            flows = None
            if "north_money" in df_hsgt.columns:
                flows = df_hsgt["north_money"].dropna().astype(float).values
            elif "ggt_ss" in df_hsgt.columns:
                if "ggt_sz" in df_hsgt.columns:
                    flows = (df_hsgt["ggt_ss"].astype(float) + df_hsgt["ggt_sz"].astype(float)).values
                else:
                    flows = df_hsgt["ggt_ss"].astype(float).values
            if flows is not None and len(flows) >= 5:
                recent5 = flows[-5:]
                if np.sum(recent5 > 0) >= 4:
                    score += 20
                elif np.sum(recent5 > 0) >= 3:
                    score += 10
                elif np.sum(recent5 < 0) >= 4:
                    score -= 15
                elif np.sum(recent5 < 0) >= 3:
                    score -= 8
    except Exception:
        pass
    return round(max(10, min(90, score)), 1)


def _calc_leverage_score():
    try:
        pro = _get_pro()
    except Exception:
        return 50.0
    try:
        df = pro.margin(start_date="20260401", end_date="20260531")
        if df is not None and not df.empty:
            bal_col = None
            for col in ["rzye", "margin_balance", "balance"]:
                if col in df.columns:
                    bal_col = col
                    break
            if bal_col is None:
                return 50.0
            vals = df[bal_col].dropna().astype(float).values
            if len(vals) < 20:
                return 50.0
            chg_5d = (vals[-1] - vals[-5]) / max(vals[-5], 1) * 100 if len(vals) >= 5 else 0
            chg_20d = (vals[-1] - vals[-20]) / max(vals[-20], 1) * 100 if len(vals) >= 20 else 0
            score = 50.0
            if chg_5d > 5:
                score += 25
            elif chg_5d > 3:
                score += 15
            elif chg_5d > 1:
                score += 5
            elif chg_5d < -3:
                score -= 25
            elif chg_5d < -1.5:
                score -= 15
            elif chg_5d < -0.5:
                score -= 5
            if chg_20d > 10:
                score += 10
            elif chg_20d < -8:
                score -= 10
            return round(max(10, min(90, score)), 1)
    except Exception:
        return 50.0
    return 50.0


def _calc_sentiment_score(panic_pct=None, profit_effect_pct=None,
                       limit_up_count=None, limit_down_count=None):
    score = 50.0
    pct = panic_pct if panic_pct is not None else 0
    if pct > 5:
        score -= 30
    elif pct > 3:
        score -= 15
    elif pct > 1.5:
        score -= 5
    pe = profit_effect_pct if profit_effect_pct is not None else 50
    if pe > 70:
        score += 20
    elif pe > 60:
        score += 10
    elif pe < 30:
        score -= 20
    elif pe < 40:
        score -= 10
    if limit_up_count is not None and limit_down_count is not None:
        up = max(limit_up_count, 1)
        down = max(limit_down_count, 1)
        ratio = up / down
        if ratio > 5:
            score += 15
        elif ratio > 3:
            score += 10
        elif ratio > 1.5:
            score += 5
        elif ratio < 0.3:
            score -= 15
        elif ratio < 0.5:
            score -= 10
    return round(max(10, min(90, score)), 1)


def _calc_valuation_score():
    try:
        pro = _get_pro()
    except Exception:
        return 50.0
    try:
        df = pro.index_dailybasic(
            trade_date="20260530", ts_code="000300.SH",
            fields="trade_date,pe,pe_ttm,pb"
        )
        if df is None or df.empty:
            return 50.0
        pe_val = None
        for col in ["pe_ttm", "pe"]:
            if col in df.columns:
                pe_val = float(df.iloc[0][col])
                break
        pb_val = float(df.iloc[0]["pb"]) if "pb" in df.columns else None
        if pe_val is None and pb_val is None:
            return 50.0
        df_hist = pro.index_dailybasic(
            ts_code="000300.SH", start_date="20240530", end_date="20260530",
            fields="trade_date,pe_ttm,pe,pb"
        )
        if df_hist is not None and not df_hist.empty:
            score_pe = 50.0
            score_pb = 50.0
            for col, cur_val in [("pe_ttm", pe_val), ("pe", pe_val)]:
                if col in df_hist.columns and cur_val is not None:
                    hist = df_hist[col].dropna().astype(float).values
                    if len(hist) >= 100:
                        pct_rank = (hist < cur_val).sum() / len(hist) * 100
                        if pct_rank < 20:
                            score_pe = 80.0
                        elif pct_rank < 40:
                            score_pe = 65.0
                        elif pct_rank > 80:
                            score_pe = 20.0
                        elif pct_rank > 60:
                            score_pe = 35.0
                        break
            if pb_val is not None and "pb" in df_hist.columns:
                hist_pb = df_hist["pb"].dropna().astype(float).values
                if len(hist_pb) >= 100:
                    pct_rank = (hist_pb < pb_val).sum() / len(hist_pb) * 100
                    if pct_rank < 20:
                        score_pb = 80.0
                    elif pct_rank < 40:
                        score_pb = 65.0
                    elif pct_rank > 80:
                        score_pb = 20.0
                    elif pct_rank > 60:
                        score_pb = 35.0
            return round((score_pe + score_pb) / 2, 1)
        return 50.0
    except Exception:
        return 50.0


def _calc_liquidity_score(vol_ratio=None, limit_down_count=None,
                         total_stocks=None, advance_ratio=None):
    score = 50.0
    vr = vol_ratio if vol_ratio is not None else 1.0
    if vr > 1.5:
        score += 20
    elif vr > 1.2:
        score += 10
    elif vr < 0.5:
        score -= 25
    elif vr < 0.7:
        score -= 15
    elif vr < 0.9:
        score -= 5
    if limit_down_count is not None and total_stocks is not None:
        if total_stocks > 0:
            down_ratio = limit_down_count / total_stocks * 100
            if down_ratio > 5:
                score -= 20
            elif down_ratio > 3:
                score -= 10
            elif down_ratio > 1:
                score -= 5
    ar = advance_ratio if advance_ratio is not None else 0.5
    if ar > 0.7:
        score += 10
    elif ar < 0.3:
        score -= 10
    return round(max(10, min(90, score)), 1)


# ==== 状态判定 (含矛盾论调节) ====

def _classify_regime(indicators):
    t = indicators["idx_trend_pct"]
    vr = indicators["vol_ratio"]
    mc = indicators["margin_change"]
    panic = indicators["panic_pct"]
    profit = indicators["profit_effect_pct"]
    vi = indicators["volatility_idx"]
    vi_80pct = 1.8

    regime = "震荡"
    conf = 0.5

    if t < -5 and panic > 3 and profit < 30:
        regime = "底部"
        conf = min(1.0, (abs(t) / 15 + panic / 5 + (100 - profit) / 80) / 3)
    elif t > 5 and vr > 1.2 and mc > 3:
        regime = "加速"
        conf = min(1.0, (t / 12 + vr / 2 + mc / 8) / 3)
    elif t > 8 and vr > 1.5 and vi > vi_80pct:
        regime = "顶部"
        conf = min(1.0, (t / 15 + vr / 2.5 + vi / 3) / 3)
    elif t > 0 and vr > 0.8 and panic < 2:
        regime = "上升"
        conf = min(1.0, (t / 8 + vr / 1.5 + (2 - panic) / 2) / 3)
    elif t < -3 and panic > 5:
        regime = "下跌"
        conf = min(1.0, (abs(t) / 8 + panic / 10) / 2)

    # 矛盾论5维度调节
    policy = indicators.get("policy_score", 50)
    leverage = indicators.get("leverage_score", 50)
    sentiment = indicators.get("sentiment_score", 50)
    valuation = indicators.get("valuation_score", 50)
    liquidity = indicators.get("liquidity_score", 50)
    contra_scores = [policy, leverage, sentiment, valuation, liquidity]

    # 1) 置信度加权
    if regime in ("上升", "加速"):
        supports = sum(1 for s in contra_scores if s > 60)
        opposes = sum(1 for s in contra_scores if s < 40)
        if supports >= 3:
            conf = min(1.0, conf + 0.12)
        elif opposes >= 3:
            conf = max(0.3, conf - 0.12)
        elif opposes >= 2:
            conf = max(0.35, conf - 0.06)
    elif regime in ("下跌", "底部"):
        supports_weak = sum(1 for s in contra_scores if s < 40)
        opposes_weak = sum(1 for s in contra_scores if s > 60)
        if supports_weak >= 3:
            conf = min(1.0, conf + 0.10)
        elif opposes_weak >= 3:
            conf = max(0.3, conf - 0.08)

    # 2) 主导矛盾偏离预警
    if policy < 30 and regime in ("上升", "加速"):
        regime_map = {"上升": "震荡", "加速": "上升"}
        regime = regime_map.get(regime, regime)
        conf = max(0.3, conf - 0.10)

    if leverage > 80 and regime == "上升":
        regime = "加速"
        conf = min(1.0, conf + 0.05)

    if valuation < 20 and regime in ("上升", "加速"):
        conf = max(0.3, conf - 0.08)

    if liquidity < 20 and regime == "下跌":
        conf = min(1.0, conf + 0.10)

    if sentiment > 90 and regime not in ("顶部", "加速"):
        if conf > 0.6:
            conf = max(0.4, conf - 0.08)

    # HMM 概率调节（叠加，不改变现有规则）
    try:
        from hmm_regime import HMMRegime
        _hmm = HMMRegime()
        if _hmm.load():
            hmm_state, hmm_conf, hmm_probs = _hmm.predict(indicators)
            # 两者一致 → 提升 confidence
            if hmm_state == regime:
                conf = min(1.0, conf + 0.08)
            # 两者矛盾 → 以规则为主，降 confidence
            elif hmm_conf > 0.6:
                conf = max(0.3, conf - 0.05)
    except Exception:
        pass

    return (regime, round(conf, 2))


# ==== 威科夫阶段分类 ====

def _classify_wyckoff_phase(indicators: dict) -> tuple:
    """
    基于指数量价数据将市场分为 Wyckoff 阶段。

    Returns:
        (phase_label, description)
        phase_label: Accum_A / Accum_B / Accum_C / Markup /
                     Distribute_A / Distribute_B / Markdown / Range
    """
    t = indicators.get("idx_trend_pct", 0)
    ma50_200 = indicators.get("ma50_above_200")
    vol_dry = indicators.get("vol_dry_ratio", 1.0)
    vol_uvd = indicators.get("vol_up_vs_down", 1.0)
    vol_r = indicators.get("vol_ratio", 1.0)
    new_h = indicators.get("new_high")
    new_l = indicators.get("new_low")
    lh = indicators.get("lower_highs")
    hl = indicators.get("higher_lows")
    vi = indicators.get("volatility_idx", 0)
    panic = indicators.get("panic_pct", 0)
    profit = indicators.get("profit_effect_pct", 50)
    ma50_slope = indicators.get("ma50_slope", 0)

    # 1) Markdown — 下跌阶段
    # 条件: 均线空头排列 + 创新低 + 恐慌
    if ma50_200 is False and new_l and t < -3:
        return ("Markdown", "均线空头+创新低+趋势向下")
    if ma50_200 is False and t < -5 and vol_r > 1.2:
        return ("Markdown", "均线空头+放量下跌")
    if panic > 5 and t < -3 and profit < 30:
        return ("Markdown", "恐慌抛售+趋势向下")

    # 2) Distribute_B — 派发末期
    # 条件: 均线可能还多头但高点降低+下跌放量
    if lh and vol_uvd < 0.8 and t < 0:
        return ("Distribute_B", "高点降低+下跌放量")
    if ma50_200 is not False and vol_r > 1.5 and t < -2 and profit < 40:
        return ("Distribute_B", "放量下跌+赚钱效应差")

    # 3) Distribute_A — 派发初期
    # 条件: 均线多头+低点没抬高+上涨缩量/下跌放量
    if ma50_200 and not hl and vol_uvd < 0.9 and t < 3:
        return ("Distribute_A", "均线多头但低点未抬高+量价背离")
    if ma50_200 and vol_r > 1.3 and profit < 45 and t > 0 and t < 5:
        return ("Distribute_A", "放量滞涨")

    # 4) Markup — 拉升阶段
    # 条件: 均线多头排列+低点抬高+上涨放量
    if ma50_200 and hl and vol_uvd > 1.1 and t > 2:
        return ("Markup", "均线多头+低点抬高+上涨放量")
    if ma50_200 and profit > 60 and t > 3 and vol_r > 1.0:
        return ("Markup", "赚钱效应好+趋势向上")

    # 5) Accum_C — 吸筹末期 (准备突破)
    # 条件: 均线转多头+量能萎缩后回升+低点抬高
    if ma50_200 and vol_dry < 0.8 and hl and t > 0 and t < 5:
        return ("Accum_C", "量能萎缩后回升+低点抬高")
    if ma50_200 and vi < 1.0 and vol_dry < 0.85 and profit > 50:
        return ("Accum_C", "低波动+缩量+赚钱效应中性")

    # 6) Accum_B — 吸筹中期
    # 条件: 均线走平+量能持续萎缩+低波动
    if ma50_200 is not False and vol_dry < 0.75 and vi < 1.2 and abs(t) < 3:
        return ("Accum_B", "量能持续萎缩+低波动+趋势不明")
    if vol_dry < 0.7 and panic < 2 and abs(t) < 2:
        return ("Accum_B", "地量+低波动")

    # 7) Accum_A — 吸筹初期/恐慌后修复
    # 条件: 恐慌后缩量+不再创新低+低点抬高开始
    if panic > 3 and vol_dry < 0.85 and not new_l:
        return ("Accum_A", "恐慌后缩量+不再创新低")
    if t < -5 and vol_dry < 0.8 and not new_l:
        return ("Accum_A", "大跌后缩量+不再创新低")

    # 8) Range — 震荡（兜底）
    return ("Range", "无明确威科夫阶段信号")


# ==== 对外接口 ====

def get_regime(date=None) -> dict:
    """
    获取当前市场状态

    Args:
      date: str "YYYYMMDD" 或 None(=最新)

    Returns:
      {
        "regime": "底部|上升|加速|顶部|下跌|震荡",
        "confidence": 0.0~1.0,
        "indicators": {...},
        "date": "20260530"
      }
    """
    pro = _get_pro()

    if date is None:
        try:
            df_idx = pro.index_daily(ts_code='000001.SH', start_date='20260501', end_date='20260530')
            if df_idx is not None and len(df_idx) > 0:
                date = str(df_idx.iloc[0]["trade_date"])
            else:
                date = "20260529"
        except Exception:
            date = "20260529"

    idx_data = _fetch_index_daily(pro, date, lookback=120)
    if idx_data is None:
        return {"regime": "震荡", "confidence": 0.3, "indicators": {},
                "date": date, "error": "指数数据拉取失败"}

    limit_down_count = 0
    limit_up_count = 0
    advance_count = 0
    decline_count = 0

    try:
        df_day = pro.daily(adj='qfq', trade_date=date)
        if df_day is not None and len(df_day) > 0:
            pct_chg = df_day["pct_chg"].astype(float)
            limit_down_count = int((pct_chg <= -9.5).sum())
            limit_up_count = int((pct_chg >= 9.5).sum())
            advance_count = int((pct_chg > 0).sum())
            decline_count = int((pct_chg < 0).sum())
    except Exception:
        pass

    total_stocks = advance_count + decline_count
    advance_ratio = advance_count / max(total_stocks, 1)

    indicators = _calc_all_indicators(idx_data, limit_down_count, advance_ratio,
                                       limit_up_count=limit_up_count, total_stocks=total_stocks)
    if indicators is None:
        return {"regime": "震荡", "confidence": 0.3, "indicators": {},
                "date": date, "error": "指标数据不足"}

    regime, confidence = _classify_regime(indicators)

    # 威科夫阶段分类
    wyckoff_phase, wyckoff_desc = _classify_wyckoff_phase(indicators)

    # HMM 概率（可选）
    hmm_probs = None
    try:
        from hmm_regime import HMMRegime
        _hmm = HMMRegime()
        if _hmm.load():
            _, _, hmm_probs = _hmm.predict(indicators)
    except Exception:
        pass

    result = {
        "regime": regime,
        "confidence": round(confidence, 2),
        "indicators": indicators,
        "wyckoff_phase": wyckoff_phase,
        "wyckoff_desc": wyckoff_desc,
        "date": date,
        "hmm_probs": hmm_probs,
    }
    try:
        from system_state import report_market_regime
        report_market_regime(result)
    except Exception:
        pass
    return result


def get_regime_history(start_date, end_date, step_days=30) -> list:
    """
    获取历史区间内每隔 step_days 的市场状态快照

    返回 [{date, regime, confidence, indicators}, ...]
    """
    pro = _get_pro()
    results = []

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    current = start_dt

    idx_data_full = _fetch_index_daily(pro, end_date, lookback=360)
    idx_map = _src_to_date_map(idx_data_full or [])

    while current <= end_dt:
        date_str = current.strftime("%Y%m%d")
        try:
            idx_slice = [r for r in (idx_data_full or [])
                         if r["trade_date"] <= date_str][-120:]

            if len(idx_slice) >= 20:
                indicators = _calc_all_indicators(idx_slice, 0, 0.5)
                if indicators:
                    regime, conf = _classify_regime(indicators)
                    results.append({
                        "date": date_str,
                        "regime": regime,
                        "confidence": round(conf, 2),
                        "indicators": indicators,
                    })
        except Exception:
            pass

        current += timedelta(days=step_days)
        time.sleep(0.1)

    return results


# ── 策略联动 ──

def get_regime_adjustments(regime_result):
    """
    根据市场状态返回评分调整参数

    Returns:
      {
        "veto_threshold": int,       # factor_veto 触发阈值
        "wyckoff_multiplier": float,
        "position_cap_pct": float,   # 仓位上限
        "trend_weight_multiplier": float,
        "volatility_weight_multiplier": float,
        "buy_threshold_adjust": float,  # 买阈偏移
      }
    """
    regime = regime_result.get("regime", "震荡")

    defaults = {
        "veto_threshold": 20,
        "wyckoff_multiplier": 1.0,
        "position_cap_pct": 0.8,
        "trend_weight_multiplier": 1.0,
        "volatility_weight_multiplier": 1.0,
        "buy_threshold_adjust": 0,
    }

    adjustments = {
        "底部": {
            "veto_threshold": 15,
            "wyckoff_multiplier": 1.5,
            "position_cap_pct": 0.7,
            "trend_weight_multiplier": 1.0,
            "volatility_weight_multiplier": 1.0,
            "buy_threshold_adjust": -8,  # 降低买阈
        },
        "上升": {
            "veto_threshold": 20,
            "wyckoff_multiplier": 1.1,
            "position_cap_pct": 0.8,
            "trend_weight_multiplier": 1.2,
            "volatility_weight_multiplier": 1.0,
            "buy_threshold_adjust": -3,
        },
        "加速": {
            "veto_threshold": 25,
            "wyckoff_multiplier": 0.8,
            "position_cap_pct": 0.6,
            "trend_weight_multiplier": 1.0,
            "volatility_weight_multiplier": 1.3,
            "buy_threshold_adjust": +5,  # 升高买阈防追高
        },
        "顶部": {
            "veto_threshold": 30,
            "wyckoff_multiplier": 0.5,
            "position_cap_pct": 0.4,
            "trend_weight_multiplier": 0.5,
            "volatility_weight_multiplier": 1.0,
            "buy_threshold_adjust": +12,
        },
        "下跌": {
            "veto_threshold": 25,
            "wyckoff_multiplier": 0.3,
            "position_cap_pct": 0.2,
            "trend_weight_multiplier": 0.3,
            "volatility_weight_multiplier": 1.0,
            "buy_threshold_adjust": +10,
        },
        "震荡": defaults,
    }

    return adjustments.get(regime, defaults)


# ── 自检 ──
if __name__ == "__main__":
    print("market_regime 自检:")
    r = get_regime()
    print(f"  Regime: {r['regime']} (conf={r['confidence']})")
    print(f"  Wyckoff Phase: {r.get('wyckoff_phase','N/A')} — {r.get('wyckoff_desc','')}")
    print(f"  Date: {r['date']}")
    if r.get("indicators"):
        ind = r["indicators"]
        print(f"  Index trend: {ind.get('idx_trend_pct',0):.2f}%")
        print(f"  Vol ratio: {ind.get('vol_ratio',0):.3f}")
        print(f"  Panic: {ind.get('panic_pct',0):.3f}%")
        print(f"  Profit effect: {ind.get('profit_effect_pct',0):.1f}%")

    adj = get_regime_adjustments(r)
    print(f"  Adjustments: veto={adj['veto_threshold']}, "
          f"wyckoff_x={adj['wyckoff_multiplier']:.1f}, "
          f"position_cap={adj['position_cap_pct']:.0%}, "
          f"buy_adj={adj['buy_threshold_adjust']:+d}")
    print("  OK")