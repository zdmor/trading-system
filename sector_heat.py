"""
板块景气度五维评分 — 四层漏斗第二层
数据源: Tushare Pro (5000积分)

Five dimensions:
  政策支持度 (15%) - 静态政策映射表
  资金流向   (25%) - Tushare moneyflow 按行业汇总
  板块强度   (25%) - Tushare ths_index 涨跌幅排名
  估值水位   (15%) - 行业PE/PB估值分位
  技术趋势   (20%) - 板块指数K线MA/RSI

All dimensions have graceful fallback to neutral (50) when data unavailable.
Results cached at module level with configurable TTL.
"""

import time
import math
from datetime import datetime, timedelta
from typing import Optional

# Module-level cache
_cache = {}
_cache_ttl = 3600  # 1 hour default


# ==================== 工具函数 ====================

def _get_pro():
    """获取 Tushare Pro API 单例"""
    import tushare as ts
    import json, os
    global _pro
    if _pro is None:
        cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        token = cfg.get('_tushare', {}).get('token', '')
        if not token:
            raise RuntimeError("Tushare token not found")
        ts.set_token(token)
        _pro = ts.pro_api()
    return _pro

_pro = None


def _cached(key: str, func, ttl: int = 3600):
    """模块级缓存: key → {data, timestamp}"""
    now = time.time()
    entry = _cache.get(key)
    if entry and now - entry["timestamp"] < ttl:
        return entry["data"]
    data = func()
    _cache[key] = {"data": data, "timestamp": now}
    return data


def _last_trade_date() -> str:
    """获取最近交易日（本地估算，不调 API）"""
    today = datetime.now()
    wd = today.weekday()
    if wd == 5:   # Saturday → 周五
        today -= timedelta(days=1)
    elif wd == 6:  # Sunday → 周五
        today -= timedelta(days=2)
    return today.strftime('%Y%m%d')


def _get_board_kline(industry: str) -> tuple:
    """
    获取板块指数的 K 线数据（最近 35 天）
    通过 Tushare ths_daily, 带缓存避免重复请求
    Returns (dates: list, closes: list) 按日期升序, 或 (None, None)
    """
    tdate = _last_trade_date()
    start = (datetime.now() - timedelta(days=35)).strftime('%Y%m%d')
    cache_key = f"kline_{industry}_{tdate}"
    cached = _cache.get(cache_key)
    if cached:
        return cached["dates"], cached["closes"]

    try:
        pro = _get_pro()
        df_idx = pro.ths_index(name=industry)
        if df_idx is None or df_idx.empty:
            return None, None
        idx_code = df_idx.iloc[0]['ts_code']
        df_k = pro.ths_daily(ts_code=idx_code, start_date=start, end_date=tdate)
        if df_k is None or df_k.empty:
            return None, None
        df_k = df_k.sort_values('trade_date')
        dates = df_k['trade_date'].tolist()
        closes = df_k['close'].astype(float).tolist()
        _cache[cache_key] = {"dates": dates, "closes": closes, "timestamp": time.time()}
        return dates, closes
    except Exception:
        return None, None

def _get_board_change(industry: str) -> dict:
    """获取单个板块的 5日/20日涨跌幅"""
    dates, closes = _get_board_kline(industry)
    if not closes or len(closes) < 2:
        return {"change_5d": 0, "change_20d": 0}
    n = len(closes)
    chg_5d = ((closes[-1] / closes[-6]) - 1) * 100 if n >= 6 else 0
    chg_20d = ((closes[-1] / closes[-21]) - 1) * 100 if n >= 21 else chg_5d
    return {"change_5d": round(chg_5d, 2), "change_20d": round(chg_20d, 2)}


def _get_sector_codes(industry: str) -> list:
    """
    获取板块成分股代码列表
    优先通过 ths_index + ths_member（支持概念板块如"人工智能"），
    回退到 stock_basic.industry（申万行业分类）
    """
    pro = _get_pro()

    # 方案1: 通过 ths_index 找板块代码 → ths_member 取成分股
    try:
        df_idx = pro.ths_index(name=industry)
        if df_idx is not None and not df_idx.empty:
            # 可能有多个同名的指数（不同分类），取第一个有成分股的
            for _, row in df_idx.iterrows():
                try:
                    ts_code = row['ts_code']
                    df_m = pro.ths_member(ts_code=ts_code)
                    if df_m is not None and not df_m.empty:
                        codes = df_m['con_code'].tolist()
                        if codes:
                            return codes
                except Exception:
                    continue
    except Exception:
        pass

    # 方案2: stock_basic.industry 过滤（申万行业）
    try:
        df = pro.stock_basic(industry=industry, list_status='L', fields='ts_code,symbol')
        if df is not None and not df.empty:
            return df['ts_code'].tolist()
    except Exception:
        pass

    # 方案3: 模糊匹配——stock_basic 全量数据中 industry 包含关键字
    try:
        df = pro.stock_basic(list_status='L', fields='ts_code,symbol,industry')
        if df is not None and not df.empty:
            matched = df[df['industry'].str.contains(industry, na=False)]
            if not matched.empty:
                return matched['ts_code'].tolist()
    except Exception:
        pass

    return []


# ==================== 政策支持度 ====================

POLICY_MAP = {
    "人工智能": 95, "半导体": 90, "芯片": 90, "信创": 85, "数字经济": 85,
    "新能源": 85, "光伏": 85, "锂电池": 85, "新能源汽车": 85,
    "机器人": 80, "军工": 80, "航空航天": 80,
    "创新药": 75, "生物医药": 75, "医疗器械": 75, "医疗服务": 70,
    "消费电子": 70, "通信": 70, "计算机": 70, "软件": 70, "5G": 70,
    "汽车": 70, "电力": 65, "电网": 65,
    "化工": 55, "有色": 55, "钢铁": 45, "煤炭": 45,
    "银行": 50, "保险": 50, "证券": 50,
    "地产": 30, "建筑": 45, "建材": 50,
    "农业": 50, "食品": 55, "医药": 65, "家电": 55,
    "纺织": 45, "服装": 45, "商贸": 45, "零售": 45,
    "环保": 60, "公用事业": 55, "交通": 50, "运输": 50,
    "传媒": 45, "游戏": 45, "文旅": 40, "酒店": 40,
}


def get_policy_score(industry: str) -> int:
    """政策支持度: 基于行业-政策映射表"""
    if not industry or industry == "其他":
        return 50
    for keyword, score in POLICY_MAP.items():
        if keyword in industry:
            return score
    return 50


def _policy_detail(industry: str) -> str:
    for keyword in POLICY_MAP:
        if keyword in industry:
            return f"政策支持: {keyword}"
    return "政策中性"


# ==================== 板块强度（Tushare 涨跌幅排名） ====================

def get_strength_score(industry: str) -> dict:
    """
    板块强度评分: 基于板块自身 5日/20日绝对涨跌幅
    不再拉全量板块排名（API 限频太高）, 改用绝对涨跌幅阈值打分
    Returns {score: int, rank_pct: float, change_pct: float}
    """
    try:
        chg = _get_board_change(industry)
        weighted = chg["change_5d"] * 0.6 + chg["change_20d"] * 0.4

        # 涨跌幅 → 分数 (不做排名分位, 用绝对涨跌幅阈值)
        if weighted >= 10:
            score = 90
        elif weighted >= 5:
            score = 75
        elif weighted >= 1:
            score = 60
        elif weighted >= -2:
            score = 45
        elif weighted >= -6:
            score = 30
        else:
            score = 15

        return {"score": score, "rank_pct": 0.5, "change_pct": round(weighted, 2)}

    except Exception:
        return {"score": 50, "rank_pct": 0.5, "change_pct": 0}


# ==================== 资金流向 ====================

def get_capital_flow_score(industry: str) -> dict:
    """
    资金流向评分: 板块内主力资金净流入率
    取行业板块前 15 只成分股，汇总资金流向
    Returns {score: int, net_flow_pct: float, positive_ratio: float}
    """
    try:
        pro = _get_pro()
        # 获取板块成分股
        codes = _get_sector_codes(industry)
        if not codes:
            return {"score": 50, "net_flow_pct": 0, "positive_ratio": 0.5}

        # 限前 15 只（兼顾性能和代表性）
        sample = codes[:15]

        total_net = 0.0
        total_amount = 0.0
        positive_count = 0

        for ts_code in sample:
            try:
                time.sleep(0.05)
                df_mf = pro.moneyflow(ts_code=ts_code, limit=1)
                if df_mf is not None and not df_mf.empty:
                    row = df_mf.iloc[0]
                    net = float(row.get('net_mf_amount', 0) or 0)
                    # 估算总成交额: 各档买入+卖出
                    buy = (float(row.get('buy_sm_amount', 0) or 0) +
                           float(row.get('buy_md_amount', 0) or 0) +
                           float(row.get('buy_lg_amount', 0) or 0) +
                           float(row.get('buy_elg_amount', 0) or 0))
                    sell = (float(row.get('sell_sm_amount', 0) or 0) +
                            float(row.get('sell_md_amount', 0) or 0) +
                            float(row.get('sell_lg_amount', 0) or 0) +
                            float(row.get('sell_elg_amount', 0) or 0))
                    turnover = buy + sell
                    total_net += net
                    total_amount += turnover
                    if net > 0:
                        positive_count += 1
            except Exception:
                continue

        if total_amount == 0:
            return {"score": 50, "net_flow_pct": 0, "positive_ratio": 0.5}

        net_flow_pct = (total_net / total_amount) * 100
        positive_ratio = positive_count / len(sample) if sample else 0.5

        # 净流入率 → 分数
        if net_flow_pct > 5:
            score = 90
        elif net_flow_pct > 2:
            score = 75
        elif net_flow_pct > 0:
            score = 60
        elif net_flow_pct > -2:
            score = 40
        elif net_flow_pct > -5:
            score = 25
        else:
            score = 10

        return {"score": score, "net_flow_pct": round(net_flow_pct, 2), "positive_ratio": round(positive_ratio, 2)}

    except Exception:
        return {"score": 50, "net_flow_pct": 0, "positive_ratio": 0.5}


# ==================== 估值水位 ====================

def get_valuation_score(industry: str) -> dict:
    """
    估值水位评分: 行业PE/PB估值分位
    取板块前 20 只成分股，计算中位数 PE/PB
    Returns {score: int, pe_pctl: float, pb_pctl: float}
    """
    try:
        pro = _get_pro()
        codes = _get_sector_codes(industry)
        if not codes:
            return {"score": 50, "pe_pctl": 0.5, "pb_pctl": 0.5}

        codes = codes[:20]

        pe_list = []
        pb_list = []

        for ts_code in codes:
            try:
                df_db = pro.daily_basic(ts_code=ts_code, fields='pe,pb')
                if df_db is not None and not df_db.empty:
                    pe = float(df_db.iloc[0].get('pe', 0) or 0)
                    pb = float(df_db.iloc[0].get('pb', 0) or 0)
                    if pe > 0 and pe < 500:
                        pe_list.append(pe)
                    if pb > 0 and pb < 50:
                        pb_list.append(pb)
            except Exception:
                continue

        if not pe_list:
            return {"score": 50, "pe_pctl": 0.5, "pb_pctl": 0.5}

        # 行业中位数 PE/PB
        pe_med = sorted(pe_list)[len(pe_list) // 2]
        pb_med = sorted(pb_list)[len(pb_list) // 2] if pb_list else 0

        # 将 PE 映射到分数（A 股经验值）
        if pe_med < 15:
            pe_score = 80   # 低估
        elif pe_med < 25:
            pe_score = 65   # 合理偏低
        elif pe_med < 40:
            pe_score = 50   # 合理
        elif pe_med < 60:
            pe_score = 35   # 偏高
        else:
            pe_score = 20   # 高估

        # PE 和 PB 综合
        if pb_med > 0:
            if pb_med < 1.5:
                pb_score = 80
            elif pb_med < 3:
                pb_score = 65
            elif pb_med < 5:
                pb_score = 50
            elif pb_med < 8:
                pb_score = 35
            else:
                pb_score = 20
        else:
            pb_score = 50

        score = (pe_score + pb_score) // 2

        return {"score": score, "pe_pctl": round(pe_med, 1), "pb_pctl": round(pb_med, 2)}

    except Exception:
        return {"score": 50, "pe_pctl": 0.5, "pb_pctl": 0.5}


# ==================== 技术趋势 ====================

def get_technical_score(industry: str) -> dict:
    """
    技术趋势评分: 板块指数 K 线均线排列 + RSI
    使用 Tushare ths_daily 长周期 K 线, 带独立缓存
    Returns {score: int, bull_ratio: float, avg_rsi: float}
    """
    try:
        pro = _get_pro()
        tdate = _last_trade_date()
        df_k = None

        try:
            df_idx = pro.ths_index(name=industry)
            if df_idx is not None and not df_idx.empty:
                idx_code = df_idx.iloc[0]['ts_code']
                cache_key = f"klong_{idx_code}"
                cached = _cache.get(cache_key)
                if cached:
                    df_k = cached["df"]
                else:
                    df_k = pro.ths_daily(ts_code=idx_code, start_date='20251201', end_date=tdate)
                    if df_k is not None and not df_k.empty:
                        _cache[cache_key] = {"df": df_k, "timestamp": time.time()}
        except Exception:
            pass

        if df_k is None or df_k.empty:
            return {"score": 50, "bull_ratio": 0.5, "avg_rsi": 0}

        # 按日期升序排列
        df_k = df_k.sort_values('trade_date')
        closes = df_k['close'].astype(float).tolist()
        volumes = df_k['vol'].astype(float).tolist()

        if len(closes) < 20:
            return {"score": 50, "bull_ratio": 0.5, "avg_rsi": 0}

        # 计算 MA
        def ma(data, n):
            if len(data) < n:
                return data[-1]
            return sum(data[-n:]) / n

        ma5 = ma(closes, 5)
        ma20 = ma(closes, 20)
        ma60 = ma(closes, 60) if len(closes) >= 60 else ma20

        # 均线排列评分
        bull = 0
        if ma5 > ma20:
            bull += 1
        if ma20 > ma60:
            bull += 1
        if closes[-1] > ma5:
            bull += 1
        if closes[-1] > ma20:
            bull += 1

        bull_ratio = bull / 4

        # RSI(14)
        if len(closes) >= 15:
            gains, losses = 0, 0
            for i in range(-14, 0):
                diff = closes[i] - closes[i - 1]
                if diff > 0:
                    gains += diff
                else:
                    losses -= diff
            avg_gain = gains / 14
            avg_loss = losses / 14
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50

        # 综合评分: 均线 40% + RSI 30% + 量能变化 30%
        ma_score = 20 + bull_ratio * 80

        if rsi > 70:
            rsi_score = 80   # 偏强但可能过热
        elif rsi > 60:
            rsi_score = 75
        elif rsi > 50:
            rsi_score = 60
        elif rsi > 40:
            rsi_score = 40
        elif rsi > 30:
            rsi_score = 25
        else:
            rsi_score = 15

        # 量能变化
        if len(volumes) >= 10:
            vol_avg5 = sum(volumes[-5:]) / 5
            vol_avg20 = sum(volumes[-20:]) / 20
            vol_ratio = vol_avg5 / vol_avg20 if vol_avg20 > 0 else 1
            if vol_ratio > 1.5:
                vol_score = 80   # 放量
            elif vol_ratio > 1.2:
                vol_score = 65
            elif vol_ratio > 0.8:
                vol_score = 50
            elif vol_ratio > 0.5:
                vol_score = 35
            else:
                vol_score = 20   # 缩量
        else:
            vol_score = 50

        score = int(ma_score * 0.4 + rsi_score * 0.3 + vol_score * 0.3)

        return {"score": score, "bull_ratio": round(bull_ratio, 2), "avg_rsi": round(rsi, 1)}

    except Exception:
        return {"score": 50, "bull_ratio": 0.5, "avg_rsi": 0}


# ==================== 综合评分 ====================

DEFAULT_WEIGHTS = {
    "policy": 0.15,
    "capital_flow": 0.25,
    "strength": 0.25,
    "valuation": 0.15,
    "technical": 0.20,
}


def analyze(industry: str, weights: Optional[dict] = None) -> dict:
    """
    板块五维景气度综合评分

    Args:
        industry: 行业名称（如"人工智能"、"半导体"）
        weights: 自定义权重，None 使用默认

    Returns:
        dict with composite, dimensions, level
    """
    if not industry or industry == "其他":
        return _neutral_result(industry)

    w = weights or DEFAULT_WEIGHTS

    # 各维度评分
    policy = get_policy_score(industry)
    strength = get_strength_score(industry)
    capital = get_capital_flow_score(industry)
    valuation = get_valuation_score(industry)
    technical = get_technical_score(industry)

    dimensions = {
        "policy": {"score": policy, "label": _label(policy),
                   "detail": _policy_detail(industry)},
        "strength": {"score": strength["score"], "label": _label(strength["score"]),
                     "detail": f"涨幅 {strength['change_pct']:+.1f}%"},
        "capital_flow": {"score": capital["score"], "label": _label(capital["score"]),
                         "detail": f"净流入率 {capital['net_flow_pct']:+.1f}%, 正向比 {capital['positive_ratio']:.0%}"},
        "valuation": {"score": valuation["score"], "label": _label(valuation["score"]),
                      "detail": f"中位数 PE={valuation['pe_pctl']}, PB={valuation['pb_pctl']}"},
        "technical": {"score": technical["score"], "label": _label(technical["score"]),
                      "detail": f"多头比 {technical['bull_ratio']:.0%}, RSI={technical['avg_rsi']:.0f}"},
    }

    # 加权综合
    composite = (
        policy * w.get("policy", 0.15) +
        capital["score"] * w.get("capital_flow", 0.25) +
        strength["score"] * w.get("strength", 0.25) +
        valuation["score"] * w.get("valuation", 0.15) +
        technical["score"] * w.get("technical", 0.20)
    )
    composite = round(composite)

    # 等级映射
    if composite >= 80:
        level = "强"
    elif composite >= 65:
        level = "较强"
    elif composite >= 45:
        level = "一般"
    elif composite >= 30:
        level = "较弱"
    else:
        level = "弱"

    return {
        "composite": composite,
        "dimensions": dimensions,
        "level": level,
    }


def _neutral_result(industry: str) -> dict:
    return {
        "composite": 50,
        "dimensions": {
            d: {"score": 50, "label": "中性", "detail": "无数据"}
            for d in ["policy", "capital_flow", "strength", "valuation", "technical"]
        },
        "level": "一般",
    }


def _label(score: int) -> str:
    if score >= 80:
        return "强"
    if score >= 65:
        return "较强"
    if score >= 45:
        return "一般"
    if score >= 30:
        return "较弱"
    return "弱"
