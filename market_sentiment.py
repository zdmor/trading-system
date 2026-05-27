"""
market_sentiment.py — A股市场情绪四指标
收盘后计算：涨停家数、连板高度、昨日涨停溢价、炸板率
数据源：Tushare Pro

用法:
  python market_sentiment.py                    # 最近交易日
  python market_sentiment.py 20260526           # 指定日期
  python -c "from market_sentiment import get_market_sentiment; print(get_market_sentiment())"
"""

import json
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

import tushare as ts
import pandas as pd

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def _load_token() -> str:
    """从 config.json 加载 Tushare token"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("tushare_token") or cfg.get("_tushare", {}).get("token", "")


def _get_pro():
    """获取 Tushare Pro 对象"""
    token = _load_token()
    if not token:
        raise RuntimeError("Tushare token 未配置，请检查 config.json")
    ts.set_token(token)
    return ts.pro_api()


def _get_recent_trade_date(pro, ref_date: Optional[str] = None) -> str:
    """
    获取最近交易日（非周末/节假日）
    若 ref_date 为 None，从 trade_cal 获取最近交易日
    否则返回 ref_date 本身
    """
    if ref_date:
        return ref_date

    # 往最近30天找
    today = datetime.now()
    for offset in range(30):
        d = today - timedelta(days=offset)
        ds = d.strftime("%Y%m%d")
        try:
            cal = pro.trade_cal(exchange="SSE", start_date=ds, end_date=ds)
            if not cal.empty and cal.iloc[0]["is_open"] == 1:
                return ds
        except Exception:
            continue
    return today.strftime("%Y%m%d")


def _is_limit_up(row: pd.Series, threshold: float = 9.5) -> bool:
    """判断是否涨停（涨幅 >= threshold%）"""
    try:
        return float(row.get("pct_chg", 0)) >= threshold
    except (ValueError, TypeError):
        return False


def _intraday_touched_limit(row: pd.Series, threshold: float = 9.5) -> bool:
    """
    盘中是否触及涨停（最高价涨幅 >= threshold%）
    pct_chg = (close/pre_close - 1) * 100
    high_pct = (high/pre_close - 1) * 100
    """
    try:
        high = float(row.get("high", 0))
        pre_close = float(row.get("pre_close", 0))
        if pre_close <= 0:
            return False
        return (high / pre_close - 1) * 100 >= threshold
    except (ValueError, TypeError):
        return False


def _calc_high_board(pro, today: str) -> int:
    """
    计算连板高度（最高连续涨停天数）
    简化版：对今天涨停的股票，逐一向前追溯至首板
    """
    # 获取最近15天的日线数据（覆盖可能的连板）
    start_d = (datetime.strptime(today, "%Y%m%d") - timedelta(days=30)).strftime("%Y%m%d")
    try:
        df_all = pro.daily(trade_date=today)
    except Exception:
        return 0

    if df_all is None or df_all.empty:
        return 0

    today_zt = df_all[df_all.apply(_is_limit_up, axis=1)]["ts_code"].tolist()
    if not today_zt:
        return 0

    today_zt_set = set(today_zt)

    # 从昨天开始往前查，看今天涨停的股票最大连续涨停天数
    max_streak = 1
    date = datetime.strptime(today, "%Y%m%d")
    for offset in range(1, 15):
        prev_date = (date - timedelta(days=offset)).strftime("%Y%m%d")
        try:
            prev_df = pro.daily(trade_date=prev_date)
        except Exception:
            break
        if prev_df is None or prev_df.empty:
            continue

        prev_zt = set(prev_df[prev_df.apply(_is_limit_up, axis=1)]["ts_code"].tolist())
        streak_set = today_zt_set & prev_zt
        if streak_set:
            max_streak = offset + 1
            today_zt_set = streak_set  # 递进交集
        else:
            break

    return max_streak


def _zting_count(pro, today: str) -> int:
    """涨停家数：pct_chg >= 9.5%"""
    try:
        df = pro.daily(trade_date=today)
    except Exception:
        return 0
    if df is None or df.empty:
        return 0
    return len(df[df.apply(_is_limit_up, axis=1)])


def _yesterday_zting_premium(pro, today: str) -> float:
    """昨日涨停溢价：昨天涨停股票今天的平均涨幅"""
    date = datetime.strptime(today, "%Y%m%d")
    yesterday = (date - timedelta(days=1)).strftime("%Y%m%d")

    # 昨天涨停列表
    try:
        df_y = pro.daily(trade_date=yesterday)
    except Exception:
        return 0.0
    if df_y is None or df_y.empty:
        return 0.0

    y_zt = df_y[df_y.apply(_is_limit_up, axis=1)]["ts_code"].tolist()
    if not y_zt:
        return 0.0

    # 今天这些股票的涨幅
    try:
        df_t = pro.daily(trade_date=today)
    except Exception:
        return 0.0
    if df_t is None or df_t.empty:
        return 0.0

    y_zt_today = df_t[df_t["ts_code"].isin(y_zt)]
    if y_zt_today.empty:
        return 0.0

    avg_chg = float(y_zt_today["pct_chg"].mean())
    return round(avg_chg, 2)


def _zha_ban_rate(pro, today: str) -> float:
    """
    炸板率：
    盘中触涨停（最高 >= 9.5%）但收盘未涨停（收盘 < 9.5%）的占比
    炸板率 = 触板未封数 / (封板数 + 触板未封数)
    """
    try:
        df = pro.daily(trade_date=today)
    except Exception:
        return 0.0
    if df is None or df.empty:
        return 0.0

    touched = df[df.apply(_intraday_touched_limit, axis=1)]
    if touched.empty:
        return 0.0

    sealed = touched[touched.apply(_is_limit_up, axis=1)]
    failed = len(touched) - len(sealed)
    total = len(sealed) + failed

    if total == 0:
        return 0.0

    return round(failed / total, 4)


def _compute_score(zting: int, high_board: int, prem: float, zha_ban: float) -> tuple:
    """综合评分 0-100 + 标签"""
    score = 0

    # 涨停家数
    if zting >= 80:
        score += 20
    elif zting >= 50:
        score += 10
    elif zting >= 20:
        score += 0
    else:
        score -= 10

    # 连板高度
    if high_board >= 7:
        score += 20
    elif high_board >= 4:
        score += 10
    elif high_board >= 2:
        score += 0
    else:
        score -= 10

    # 昨日涨停溢价
    if prem >= 3:
        score += 10
    elif prem >= 1:
        score += 5
    else:
        score -= 5

    # 炸板率
    if zha_ban < 0.2:
        score += 10
    elif zha_ban <= 0.4:
        score += 0
    else:
        score -= 10

    # 映射到 0-100
    score = max(0, min(100, score + 30))

    # 标签
    if score >= 70:
        label = "过热"
    elif score >= 50:
        label = "偏热"
    elif score >= 30:
        label = "正常"
    elif score >= 15:
        label = "冷"
    else:
        label = "冰点"

    return score, label


def get_market_sentiment(trade_date: Optional[str] = None) -> dict:
    """
    获取A股市场情绪四指标

    Args:
        trade_date: 交易日 YYYYMMDD，None=最近交易日

    Returns:
        dict with: date, zting_count, high_board, prem_yesterday_zting,
                   zha_ban_rate, score, label
    """
    pro = _get_pro()
    today = _get_recent_trade_date(pro, trade_date)

    zting = _zting_count(pro, today)
    high_board = _calc_high_board(pro, today)
    prem = _yesterday_zting_premium(pro, today)
    zha_ban = _zha_ban_rate(pro, today)
    score, label = _compute_score(zting, high_board, prem, zha_ban)

    return {
        "date": today,
        "zting_count": zting,
        "high_board": high_board,
        "prem_yesterday_zting": prem,
        "zha_ban_rate": zha_ban,
        "score": score,
        "label": label,
    }


def main():
    """CLI 入口"""
    trade_date = sys.argv[1] if len(sys.argv) > 1 else None
    result = get_market_sentiment(trade_date)

    print("=" * 50)
    print("  A股市场情绪")
    print("=" * 50)
    print(f"  交易日:    {result['date']}")
    print(f"  涨停家数:  {result['zting_count']}只")
    print(f"  连板高度:  {result['high_board']}板")
    print(f"  昨日涨停溢价: {result['prem_yesterday_zting']:+.2f}%")
    print(f"  炸板率:    {result['zha_ban_rate']:.1%}")
    print(f"  综合评分:  {result['score']}分 [{result['label']}]")
    print("=" * 50)


if __name__ == "__main__":
    main()
