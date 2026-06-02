"""
risk_filter.py — 独立风控过滤器

与因子评分解耦的硬性风控门槛，在评分结果之后、下单决策之前执行。
规则:
  1. 趋势禁买 — MA200下方且MA50<MA200 → 买入降级为持有
  2. 连败冻结 — 同股票连续2次亏损 → 冻结3个月
  3. 累计亏损 > 15%初始本金 → 拉黑
  4. 大盘禁投 — 大盘评分 < 30 → 买入降级为持有

用法:
    rf = RiskFilter()
    result = rf.evaluate(symbol, price, daily, market_score, score, action)
    if result["blocked"]:
        print(f"风控拦截: {result['reason']}")
"""
import json
import os
import numpy as np
from datetime import datetime, timedelta

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_trade_log.json")

# ─── 阈值 ───
MAX_CONSECUTIVE_LOSSES = 2       # 连败次数上限
FREEZE_MONTHS = 3                # 冻结月数
MAX_LOSS_RATIO = 0.15            # 累计亏损比例上限
MARKET_FREEZE = 30               # 大盘评分下限
TREND_BAN = True                 # 启用趋势禁买


def _load():
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"trades": {}, "blacklist": {}}
    return {"trades": {}, "blacklist": {}}


def _save(data):
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class RiskFilter:
    def __init__(self):
        self.db = _load()

    def evaluate(self, symbol: str, price: float, daily,
                 market_score: float, score: float, action: str,
                 account_value: float = 100000) -> dict:
        """执行全部风控规则"""
        result = {
            "passed": True,
            "blocked": False,
            "reason": "",
            "adjusted_action": action,
            "adjusted_score": score,
        }

        # 规则 1
        if TREND_BAN and action in ("加仓", "强加仓", "建仓"):
            result = self._check_trend(symbol, price, daily, result)

        if result["blocked"]:
            return result

        # 规则 2
        result = self._check_freeze(symbol, result)
        if result["blocked"]:
            return result

        # 规则 3
        result = self._check_blacklist(symbol, result)
        if result["blocked"]:
            return result

        # 规则 4
        result = self._check_market(market_score, result)

        return result

    def _check_trend(self, symbol, price, daily, result):
        try:
            closes = daily["close"].values.astype(float)
            if len(closes) < 200:
                return result
            ma50 = float(daily["ma50"].iloc[-1]) if "ma50" in daily.columns else np.mean(closes[-50:])
            ma200 = float(daily["ma200"].iloc[-1]) if "ma200" in daily.columns else np.mean(closes[-200:])
            if price < ma200 and ma50 < ma200:
                result.update({
                    "passed": False, "blocked": True,
                    "reason": f"趋势禁买: 价{price:.2f}<MA200{ma200:.2f} MA50{ma50:.2f}<MA200",
                    "adjusted_action": "持有",
                    "adjusted_score": min(result["adjusted_score"], 49),
                })
        except Exception:
            pass
        return result

    def _check_freeze(self, symbol, result):
        trades = self.db.get("trades", {}).get(symbol, [])
        if len(trades) < MAX_CONSECUTIVE_LOSSES:
            return result
        recent = trades[-MAX_CONSECUTIVE_LOSSES:]
        if all(t.get("pnl", 0) < 0 for t in recent):
            last = recent[-1].get("date", "")
            if last:
                try:
                    dt = datetime.strptime(last, "%Y-%m-%d")
                    until = dt + timedelta(days=30 * FREEZE_MONTHS)
                    if datetime.now() < until:
                        result.update({
                            "passed": False, "blocked": True,
                            "reason": f"连败冻结: 连续{MAX_CONSECUTIVE_LOSSES}次亏损, 冻结至{until.strftime('%Y-%m-%d')}",
                            "adjusted_action": "离场",
                            "adjusted_score": min(result["adjusted_score"], 29),
                        })
                except ValueError:
                    pass
        return result

    def _check_blacklist(self, symbol, result):
        bl = self.db.get("blacklist", {})
        if symbol in bl:
            entry = bl[symbol]
            result.update({
                "passed": False, "blocked": True,
                "reason": f"累计亏损超限: {entry.get('total_loss', 0):,.0f}",
                "adjusted_action": "离场",
                "adjusted_score": min(result["adjusted_score"], 20),
            })
        return result

    def _check_market(self, market_score, result):
        if market_score < MARKET_FREEZE:
            if result["adjusted_action"] in ("加仓", "强加仓", "建仓"):
                result.update({
                    "passed": False, "blocked": True,
                    "reason": f"大盘禁投: {market_score}<{MARKET_FREEZE}",
                    "adjusted_action": "持有",
                    "adjusted_score": min(result["adjusted_score"], 49),
                })
        return result

    # ─── 交易记录 ───

    def record_trade(self, symbol: str, date: str, price: float,
                     pnl: float, action: str, score: int):
        self.db.setdefault("trades", {}).setdefault(symbol, []).append({
            "date": date, "price": price,
            "pnl": round(pnl, 2), "action": action, "score": score,
        })
        if pnl < 0:
            trades = self.db["trades"][symbol]
            total = sum(t.get("pnl", 0) for t in trades)
            if total < 0 and abs(total) > MAX_LOSS_RATIO * 100000:
                self.db.setdefault("blacklist", {})[symbol] = {
                    "total_loss": round(total, 2), "date": date,
                }
        _save(self.db)

    def record_full_trade(self, symbol: str, buy_date: str, buy_price: float,
                          sell_date: str, sell_price: float, shares: int,
                          score_on_buy: int):
        pnl = (sell_price - buy_price) * shares
        self.record_trade(symbol, sell_date, sell_price, pnl, "卖出", score_on_buy)

    def status(self, symbol: str) -> dict:
        trades = self.db.get("trades", {}).get(symbol, [])
        wins = sum(1 for t in trades if t["pnl"] > 0)
        total = len(trades)
        recent_losses = sum(1 for t in trades[-MAX_CONSECUTIVE_LOSSES:] if t["pnl"] < 0)
        return {
            "symbol": symbol,
            "trades": total,
            "win_rate": round(wins / max(total, 1), 2),
            "cumulative_pnl": round(sum(t.get("pnl", 0) for t in trades), 2),
            "blacklisted": symbol in self.db.get("blacklist", {}),
            "consecutive_losses": recent_losses,
        }


def check_take_profit_levels(price: float, buy_price: float,
                              levels: dict = None) -> dict:
    """
    分批止盈：达到阻力位分段卖出

    Args:
        price: 当前价格
        buy_price: 买入价格
        levels: 阻力位信息 {"resistances": [r1, r2, ...]}

    Returns:
        {
            "level_reached": int (0/1/2),
            "action": "持有/减1/3/减2/3/跟踪止盈",
            "target_price": float or None,
            "pct_gain": float,
        }
    """
    pct_gain = round((price - buy_price) / buy_price * 100, 1) if buy_price > 0 else 0
    resistances = levels.get("resistances", []) if levels else []

    # 如果没有阻力位，使用固定百分比目标
    if not resistances or len(resistances) < 2:
        # 默认目标：+10% / +20%
        t1 = buy_price * 1.10
        t2 = buy_price * 1.20
    else:
        t1 = resistances[0]
        t2 = resistances[1] if len(resistances) > 1 else resistances[0] * 1.1

    result = {"level_reached": 0, "action": "持有",
              "target_price": None, "pct_gain": pct_gain}

    if price >= t2:
        result.update({"level_reached": 2, "action": "减2/3",
                       "target_price": t2})
    elif price >= t1:
        result.update({"level_reached": 1, "action": "减1/3",
                       "target_price": t1})

    return result
