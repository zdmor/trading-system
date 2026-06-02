"""
退出规则 — 行为约束代码化

两条规则:
  1. 逻辑破退出 — 入场逻辑关键位被跌破
  2. K线三根高点不抬高 — 趋势衰竭信号

用法:
    from exit_rules import check_exit_rules
    signals = check_exit_rules(df, entry_key_price)
    # 返回 {"logic_broken": bool, "trend_exhaust": bool, "messages": [...]}
"""
import numpy as np


def check_logic_broken(df, entry_key_price: float = None, lookback: int = 3):
    """
    规则1: 逻辑破退出

    检查最近N根K线的最低价是否跌破了入场逻辑关键位。
    关键位可以是: 三买确认价、Spring影线低点、支撑位。

    Args:
        df: 日线 DataFrame (须含 low 列)
        entry_key_price: 入场逻辑关键位。None=不检查
        lookback: 检查最近多少根K线

    Returns:
        (是否触发, 消息)
    """
    if entry_key_price is None or entry_key_price <= 0:
        return False, ""

    recent = df.tail(lookback)
    if recent["low"].min() < entry_key_price:
        broken_idx = recent[recent["low"] < entry_key_price].index[0]
        broken_low = df.loc[broken_idx, "low"]
        return True, f"逻辑破: 最低{broken_low:.2f}<关键位{entry_key_price:.2f} ({df.loc[broken_idx,'date']})"

    return False, ""


def check_trend_exhaustion(df, lookback: int = 3):
    """
    规则2: K线三根高点不抬高卖出

    连续 N 根K线高点依次降低 → 上升动力衰竭。

    Returns:
        (是否触发, 消息)
    """
    if len(df) < lookback:
        return False, ""

    recent = df.tail(lookback)
    highs = recent["high"].values

    if len(highs) >= 3 and highs[-3] > highs[-2] > highs[-1]:
        return True, f"K线衰竭: {lookback}根高点依次降低 ({highs[-3]:.2f}>{highs[-2]:.2f}>{highs[-1]:.2f})"

    return False, ""


def check_exit_rules(df, entry_key_price: float = None,
                     lookback_logic: int = 3, lookback_trend: int = 3):
    """
    同时检查两条退出规则。

    Returns:
        dict with keys:
          - triggered: bool (任一规则触发)
          - logic_broken: bool
          - trend_exhaust: bool
          - messages: [str]
    """
    result = {
        "triggered": False,
        "logic_broken": False,
        "trend_exhaust": False,
        "messages": [],
    }

    lb, msg1 = check_logic_broken(df, entry_key_price, lookback_logic)
    if lb:
        result["logic_broken"] = True
        result["messages"].append(msg1)

    te, msg2 = check_trend_exhaustion(df, lookback_trend)
    if te:
        result["trend_exhaust"] = True
        result["messages"].append(msg2)

    result["triggered"] = result["logic_broken"] or result["trend_exhaust"]
    return result


def check_time_stop(buy_date: str, max_hold_days: int = 20) -> dict:
    """
    时间止损：持仓超过最大允许天数强制退出

    联动 market_regime：
        震荡市 20 日，上升市 40 日，下跌市 10 日

    Args:
        buy_date: 买入日期 (YYYY-MM-DD 或 YYYYMMDD)
        max_hold_days: 最大持仓天数

    Returns:
        {"triggered": bool, "message": str}
    """
    from datetime import datetime

    # 标准化日期格式
    clean = buy_date.replace("-", "").replace(" ", "").replace("/", "")
    if len(clean) == 8:
        date_str = clean[:4] + "-" + clean[4:6] + "-" + clean[6:8]
    else:
        return {"triggered": False, "message": f"无法解析买入日期: {buy_date}"}

    try:
        buy_dt = datetime.strptime(date_str, "%Y-%m-%d")
        hold_days = (datetime.now() - buy_dt).days
        if hold_days > max_hold_days:
            return {"triggered": True,
                    "message": f"时间止损: 持仓{hold_days}日 > 最大{max_hold_days}日"}
        return {"triggered": False,
                "message": f"持仓{hold_days}/{max_hold_days}日"}
    except ValueError:
        return {"triggered": False, "message": f"日期解析失败: {buy_date}"}


def get_time_stop_days(regime: str = "震荡") -> int:
    """
    根据市场状态返回时间止损天数

    Args:
        regime: 市场状态 ("震荡" / "多头" / "空头")

    Returns:
        最大持仓天数
    """
    regime_map = {
        "上升": 40, "多头": 40, "bull": 40,
        "震荡": 20, "盘整": 20, "neutral": 20,
        "下跌": 10, "空头": 10, "bear": 10,
    }
    return regime_map.get(regime, 20)
