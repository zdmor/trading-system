"""
A股交易系统
基于趋势跟踪 + 价格行为 + ATR仓位管理
一键输出交易报告：方向判断、关键价位、仓位计算、操作建议

数据源: 腾讯财经 HTTP API（免费、无需 token、无需登录）

用法:
  python main.py <股票代码> [账户总资金] [--position 持仓股数,持仓均价]

示例:
  python main.py 002050 80000 --position 100,51.44
  python main.py 000858 200000
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import json
import os
import requests
import warnings
warnings.filterwarnings("ignore")

from data_providers import AkshareProvider


# ============================================================
# 数据获取 (腾讯财经 HTTP API)
# ============================================================

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,{days},qfq"


class DataFetcher:
    """A股数据获取 — 腾讯财经HTTP API（免费、免token、免登录）"""

    @staticmethod
    def _market(code):
        return "sh" if code.startswith("6") else "sz"

    @staticmethod
    def _get_daily_baostock(symbol, days=400):
        """从BaoStock获取日线（容灾备选）"""
        import baostock as bs
        bs.login()
        try:
            start = (datetime.now() - timedelta(days=days + 100)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            bs_code = f"sz.{symbol}" if DataFetcher._market(symbol) == "sz" else f"sh.{symbol}"
            rs = bs.query_history_k_data_plus(bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start, end_date=end, frequency="d", adjustflag="2")
            rows = []
            while rs.next():
                row = rs.get_row_data()
                try:
                    rows.append({
                        "date": row[0],
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]) if row[5] else 0,
                        "amount": float(row[6]) if row[6] else 0,
                    })
                except (ValueError, IndexError):
                    continue
            if len(rows) < 30:
                raise ValueError(f"BaoStock 仅获取到 {len(rows)} 条数据")
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df.tail(days)
        finally:
            bs.logout()

    @staticmethod
    def get_daily(symbol, days=400):
        """获取日线数据，腾讯 → BaoStock 两级容灾"""
        # 首选：腾讯
        try:
            market = DataFetcher._market(symbol)
            url = TENCENT_KLINE_URL.format(market=market, code=symbol, days=days)
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("code") == 0:
                klines = data.get("data", {}).get(market + symbol, {}).get("qfqday")
                if not klines:
                    klines = data.get("data", {}).get(market + symbol, {}).get("day")
                if klines and len(klines) >= 30:
                    rows = []
                    for k in klines:
                        rows.append({
                            "date": k[0],
                            "open": float(k[1]),
                            "high": float(k[3]),
                            "low": float(k[4]),
                            "close": float(k[2]),
                            "volume": float(k[5]) * 100,
                            "amount": 0.0,
                        })
                    df = pd.DataFrame(rows)
                    df["date"] = pd.to_datetime(df["date"])
                    df.sort_values("date", inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    return df
        except Exception:
            pass

        # 容灾：BaoStock
        return DataFetcher._get_daily_baostock(symbol, days)

    @staticmethod
    def get_name(symbol):
        """获取股票名称"""
        try:
            market = DataFetcher._market(symbol)
            url = TENCENT_KLINE_URL.format(market=market, code=symbol, days=1)
            r = requests.get(url, timeout=10)
            qt = r.json().get("data", {}).get(market + symbol, {}).get("qt", {}).get(market + symbol, [])
            if qt and len(qt) > 1 and qt[1]:
                return qt[1]
        except Exception:
            pass
        return symbol


# ============================================================
# 技术分析
# ============================================================

class Analyzer:
    """技术指标计算"""

    @staticmethod
    def calc_atr(df, period=14):
        """Wilder's平滑ATR"""
        df = df.copy()
        prev = df["close"].shift(1)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                (df["high"] - prev).abs(),
                (df["low"] - prev).abs()
            )
        )
        atr = np.zeros(len(df))
        if len(df) <= period:
            return df
        atr[period] = df["tr"].iloc[1:period+1].mean()
        for i in range(period + 1, len(df)):
            atr[i] = (atr[i-1] * (period - 1) + df["tr"].iloc[i]) / period
        df["atr"] = atr
        df["atr_pct"] = np.where(df["close"] > 0, df["atr"] / df["close"] * 100, 0)
        return df

    @staticmethod
    def calc_ma(df, periods):
        """计算多周期移动平均线"""
        df = df.copy()
        for p in periods:
            df[f"ma{p}"] = df["close"].rolling(window=p).mean()
        return df

    @staticmethod
    def detect_levels(df, lookback=60):
        """基于价格聚类检测关键支撑阻力位"""
        df = df.iloc[-lookback:]
        prices = np.concatenate([
            df["high"].values,
            df["low"].values
        ])

        # 价格聚类（简单密度估计）
        bins = np.linspace(prices.min(), prices.max(), 30)
        density, edges = np.histogram(prices, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2

        # 找密度峰值作为关键价位
        levels = []
        for i in range(1, len(density) - 1):
            if density[i] > density[i-1] and density[i] > density[i+1] and density[i] > 1:
                levels.append(centers[i])

        current = df["close"].iloc[-1]
        supports = sorted([l for l in levels if l < current], reverse=True)[:3]
        resistances = sorted([l for l in levels if l > current])[:3]

        return supports, resistances


# ============================================================
# 仓位计算
# ============================================================

class PositionSizer:
    """基于ATR和固定风险的仓位计算"""

    @staticmethod
    def calculate(account_value, entry_price, stop_price, risk_pct=0.02):
        """计算建议持仓量（取整到100股）"""
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share < 0.01:
            return 0, 0, 0
        max_risk = account_value * risk_pct
        shares = max_risk / risk_per_share
        shares = max(0, int(shares / 100) * 100)
        cost = shares * entry_price
        actual_risk = shares * risk_per_share
        return shares, cost, actual_risk


# ============================================================
# 策略引擎
# ============================================================

class Strategy:
    """整合趋势、入场、出场、仓位的决策引擎"""

    def __init__(self, df, account_value=80000, current_position=None):
        self.df = df
        self.latest = df.iloc[-1]
        self.account_value = account_value
        self.current_position = current_position or {"shares": 0, "avg_price": 0}
        self.price = self.latest["close"]

    def trend_analysis(self):
        """趋势判断"""
        result = {"direction": "未知", "strength": 0}
        has_ma50 = "ma50" in self.df.columns and not pd.isna(self.latest.get("ma50"))
        has_ma200 = "ma200" in self.df.columns and not pd.isna(self.latest.get("ma200"))

        if has_ma50 and has_ma200:
            ma50 = self.latest["ma50"]
            ma200 = self.latest["ma200"]
            if ma50 > ma200:
                result["direction"] = "多头"
                result["strength"] = round((ma50 / ma200 - 1) * 100, 2)
            else:
                result["direction"] = "空头"
                result["strength"] = round((ma50 / ma200 - 1) * 100, 2)

        # 价格相对位置
        if has_ma50:
            result["above_ma50"] = self.price > self.latest["ma50"]
        if has_ma200:
            result["above_ma200"] = self.price > self.latest["ma200"]

        return result

    def volatility_analysis(self):
        """波动率分析"""
        if "atr" not in self.df.columns or pd.isna(self.latest.get("atr")):
            return {"atr": 0, "atr_pct": 0}
        return {
            "atr": round(self.latest["atr"], 2),
            "atr_pct": round(self.latest["atr_pct"], 1),
            "lower_band": round(self.price - self.latest["atr"] * 2, 2),
            "upper_band": round(self.price + self.latest["atr"] * 2, 2),
        }

    def position_plan(self, stop_price, risk_pct=0.02, entry_status=None, market_health=None, composite_score=None):
        """生成加仓/建仓计划

        Args:
            stop_price: 止损价
            risk_pct: 单笔风险比例（默认2%）
            entry_status: 入场信号状态，None/未知/观望 时不建议加仓
            market_health: 大盘健康度，较差时强行禁止加仓
            composite_score: 多因子综合评分，None时不调整
        """
        # 入场信号弱时禁止加仓/建仓
        signal_blocked = entry_status in (None, "", "未知", "观望", "禁止开仓")

        # 大盘环境极差时强行禁止（系统性风险硬限制）
        if market_health and "较差" in market_health:
            risk_pct = 0
            signal_blocked = True

        # 贝叶斯仓位系数：分数→概率连续映射 + 历史校准
        if composite_score is not None and not signal_blocked:
            # 连续映射函数：在关键分位匹配原离散值
            #   30分→0   50分→0.25   65分→0.50   80分→1.0
            if composite_score >= 80:
                base_mult = 1.0
            elif composite_score >= 65:
                base_mult = 0.50 + (composite_score - 65) * (0.50 / 15)
            elif composite_score >= 50:
                base_mult = 0.25 + (composite_score - 50) * (0.25 / 15)
            elif composite_score >= 30:
                base_mult = (composite_score - 30) * (0.25 / 20)
            else:
                base_mult = 0.0

            # 历史校准：根据该分数段的实际胜率修正仓位
            if base_mult > 0:
                calib = _load_calibration()
                bracket = _score_bracket(composite_score)
                cal = calib.get(bracket, {})
                accuracy = cal.get("accuracy", 0.5)
                # 校准系数 = 实际胜率 / 期望胜率
                # 期望胜率以分数中值为基准（50分→0.5, 80分→0.8）
                mid = {"50": 50, "60": 60, "70": 70, "80": 85}.get(bracket, 50)
                expected = mid / 100
                calib_factor = accuracy / expected if expected > 0 else 1.0
                calib_factor = max(0.3, min(2.0, calib_factor))  # 限制范围
                base_mult = base_mult * calib_factor
                base_mult = max(0.0, min(1.0, base_mult))

            risk_pct = risk_pct * base_mult
            if risk_pct <= 0:
                signal_blocked = True

        # 硬约束：不在亏损头寸上加仓（不摊平）
        if self.current_position["shares"] > 0 and self.current_position["avg_price"] > 0:
            unrealized_pnl_pct = (self.price / self.current_position["avg_price"] - 1) * 100
            if unrealized_pnl_pct < -2 and not signal_blocked:
                signal_blocked = True

        if self.current_position["shares"] > 0:
            # 已有持仓：计算加仓
            shares, cost, risk = PositionSizer.calculate(
                self.account_value, self.price, stop_price, risk_pct
            )
            if signal_blocked:
                shares, cost, risk = 0, 0, 0
            total_shares = self.current_position["shares"] + shares
            total_cost = (self.current_position["shares"] * self.current_position["avg_price"]
                          + cost)
            avg_price = total_cost / total_shares if total_shares > 0 else 0

            current_value = self.current_position["shares"] * self.price
            position_ratio = current_value / self.account_value

            # 移动止盈计算
            trailing_stop = {"active": False, "trigger_price": 0, "message": ""}
            ap = self.current_position["avg_price"]
            sp = self.current_position["shares"]
            if ap > 0 and sp > 0:
                pnl_pct = (self.price - ap) / ap * 100
                lookback = min(20, len(self.df))
                recent_high = self.df["high"].iloc[-lookback:].max()
                dd_from_high = (recent_high - self.price) / recent_high * 100
                if pnl_pct > 10 and dd_from_high > 5:
                    trailing_stop["active"] = True
                    trailing_stop["trigger_price"] = round(recent_high * 0.95, 2)
                    trailing_stop["message"] = (f"从高点{recent_high:.2f}回撤{dd_from_high:.1f}%，"
                                                f"触发移动止盈于{trailing_stop['trigger_price']:.2f}")

            return {
                "current_shares": self.current_position["shares"],
                "current_avg_price": self.current_position["avg_price"],
                "current_value": round(current_value, 2),
                "position_ratio": round(position_ratio * 100, 1),
                "add_shares": shares,
                "add_cost": round(cost, 2),
                "suggested_total": total_shares,
                "suggested_avg": round(avg_price, 2),
                "total_ratio": round((current_value + cost) / self.account_value * 100, 1),
                "trailing_stop": trailing_stop,
            }
        else:
            # 空仓：新建仓位
            if signal_blocked:
                return {
                    "current_shares": 0, "current_avg_price": 0, "current_value": 0,
                    "position_ratio": 0, "add_shares": 0, "add_cost": 0,
                    "suggested_total": 0, "suggested_avg": 0, "total_ratio": 0,
                }
            shares, cost, risk = PositionSizer.calculate(
                self.account_value, self.price, stop_price, risk_pct
            )
            return {
                "current_shares": 0,
                "current_avg_price": 0,
                "current_value": 0,
                "position_ratio": 0,
                "add_shares": shares,
                "add_cost": round(cost, 2),
                "suggested_total": shares,
                "suggested_avg": round(self.price, 2),
                "total_ratio": round(cost / self.account_value * 100, 1),
            }

    def entry_check(self, levels, composite_score=None, trend=None):
        """检查入场条件是否触发"""
        result = {
            "signal": "观望",
            "reason": [],
            "conditions": []
        }

        # 硬约束1：空头趋势禁止开新仓
        if trend and trend.get("direction") == "空头":
            result["signal"] = "禁止开仓"
            result["conditions"].append("[N] 趋势为空头，禁止开仓")
            return result

        # 硬约束2：评分<50禁止开新仓
        if composite_score is not None and composite_score < 50:
            result["signal"] = "禁止开仓"
            result["conditions"].append(f"[N] 评分{composite_score}<50，禁止开仓")
            return result

        # 条件1：价格在支撑位附近
        if levels.get("supports"):
            nearest_support = levels["supports"][-1] if levels["supports"] else 0
            support_distance = (self.price - nearest_support) / self.price * 100
            if 0 <= support_distance < 2:
                result["conditions"].append(f"[Y] 价格接近支撑 {nearest_support:.2f}（距离 {support_distance:.1f}%）")
            else:
                result["conditions"].append(f"[N] 价格距支撑 {nearest_support:.2f} 较远（-{abs(support_distance):.1f}%）" if support_distance < 0 else f"[~] 价格距支撑 {nearest_support:.2f} 较远（{support_distance:.1f}%）")

        # 条件2：趋势为多头
        trend = self.trend_analysis()
        if trend["direction"] == "多头":
            result["conditions"].append(f"[Y] 趋势多头（MA50/MA200差值 {trend['strength']}%）")
        else:
            result["conditions"].append("[N] 趋势为空头，不宜做多")

        # 条件3：波动率适中
        vol = self.volatility_analysis()
        if vol["atr_pct"] > 0:
            if 1.5 <= vol["atr_pct"] <= 5:
                result["conditions"].append(f"[Y] 波动率适中（ATR% {vol['atr_pct']}%）")
            elif vol["atr_pct"] < 1.5:
                result["conditions"].append(f"[~] 波动率偏低，注意横盘突破（ATR% {vol['atr_pct']}%）")
            else:
                result["conditions"].append(f"[!] 波动率偏高，扩大止损范围（ATR% {vol['atr_pct']}%）")

        # 综合信号
        buy_conditions = sum(1 for c in result["conditions"] if c.startswith("[Y]"))
        total_check = len(result["conditions"])
        if total_check > 0 and buy_conditions >= total_check - 1:
            result["signal"] = "可入场"
        elif buy_conditions >= total_check / 2:
            result["signal"] = "等待确认"
        else:
            result["signal"] = "观望"

        return result


# ============================================================
# 报告生成
# ============================================================

class Report:
    """生成可读的交易报告"""

    def __init__(self, symbol, name, account_value, price_date=None):
        self.symbol = symbol
        self.name = name
        self.account_value = account_value
        self.price_date = price_date or datetime.now()

    def _line(self, char="="):
        """分隔线"""
        return char * 58

    def _section(self, title):
        """节标题"""
        return f"\n  {title}\n  {'-' * 50}"

    def trend_section(self, trend):
        """趋势分析段落"""
        lines = [self._section("趋势方向")]
        lines.append(f"  {trend['direction']}（MA50/MA200差值 {trend['strength']}%）")
        if "above_ma50" in trend:
            lines.append(f"  价格在MA50上方: {'是' if trend['above_ma50'] else '否'}")
        if "above_ma200" in trend:
            lines.append(f"  价格在MA200上方: {'是' if trend['above_ma200'] else '否'}")
        lines.append("")
        return "\n".join(lines)

    def volatility_section(self, vol):
        """波动率段落"""
        lines = [self._section("波动率")]
        if vol["atr"] > 0:
            lines.append(f"  ATR(14): {vol['atr']}  占比: {vol['atr_pct']}%")
            lines.append(f"  正常区间: {vol['lower_band']} ~ {vol['upper_band']}")
        else:
            lines.append("  数据不足，无法计算")
        return "\n".join(lines)

    def levels_section(self, levels):
        """关键价位段落"""
        lines = [self._section("关键价位")]
        for r in reversed(levels.get("resistances", [])):
            lines.append(f"  阻力: {r:.2f}")
        lines.append(f"  -----------------")
        lines.append(f"  当前: {levels.get('current', 0):.2f}")
        lines.append(f"  -----------------")
        for s in levels.get("supports", []):
            lines.append(f"  支撑: {s:.2f}")
        return "\n".join(lines)

    def scoring_section(self, scoring_result: dict):
        """多因子评分段落"""
        if not scoring_result:
            return ""

        lines = [self._section("多因子评分")]
        lines.append(f"  综合评分: {scoring_result['composite_score']}/100 → {scoring_result['action']}")
        lines.append("")
        for f in scoring_result.get("factors", []):
            name_map = {
                "wyckoff": "威科夫信号", "risk_reward": "盈亏比",
                "volume": "量能分析", "candlestick": "K线形态",
                "sector": "板块强度", "momentum": "趋势与动量",
                "market": "大盘趋势", "relative_strength": "相对强度",
            }
            cn = name_map.get(f["key"], f["key"])
            lines.append(f"  {cn:<8} [{f['weight_pct']:>2}%]  {f['score']:>2}分 {f['label']}  {f['detail']}")
        lines.append("")
        return "\n".join(lines)

    def position_section(self, pos, current_price=None):
        """仓位段落"""
        lines = [self._section("仓位管理")]
        lines.append(f"  账户资金: {self.account_value:,.0f}")
        lines.append(f"  单笔风险预算: {self.account_value * 0.02:,.0f} (2%)")
        lines.append(f"  单票上限: {self.account_value * 0.7:,.0f} (70%)")
        lines.append("")
        if current_price is None:
            current_price = pos["current_value"] / max(pos["current_shares"], 1) if pos["current_shares"] > 0 else 0
        if pos["current_shares"] > 0:
            lines.append(f"  当前持仓: {pos['current_shares']} 股 @ {pos['current_avg_price']}")
            lines.append(f"  持仓市值: {pos['current_value']:,.0f}  占比: {pos['position_ratio']}%")
            unrealized = (current_price - pos["current_avg_price"]) * pos["current_shares"]
            lines.append(f"  浮动盈亏: {unrealized:+,.0f}")
            # 亏损头寸不加仓标记
            if unrealized < 0 and pos["add_shares"] == 0:
                lines.append(f"  [规则] 浮亏头寸不加仓")
        else:
            lines.append("  当前持仓: 空仓")
        lines.append("")
        if pos["add_shares"] > 0:
            lines.append(f"  建议加仓: {pos['add_shares']} 股（约 {pos['add_cost']:,.0f} 元）")
            lines.append(f"  建议总仓: {pos['suggested_total']} 股（均价 {pos['suggested_avg']}）")
            lines.append(f"  总仓位占比: {pos['total_ratio']}%")
        else:
            lines.append("  暂不建议加仓")
        return "\n".join(lines)

    def signal_section(self, entry):
        """信号段落"""
        lines = [self._section("入场信号")]
        signal_map = {"可入场": "[Y] 可入场", "等待确认": "[-] 等待确认", "观望": "[N] 观望"}
        lines.append(f"  {signal_map.get(entry['signal'], entry['signal'])}")
        lines.append("")
        for c in entry["conditions"]:
            lines.append(f"  {c}")
        return "\n".join(lines)

    def recommendation_section(self, trend, entry, stop_price, exit_prices, pos, current_price=None):
        """操作建议段落"""
        lines = [self._section("操作建议")]
        current_price = current_price or exit_prices[0] * 0.9 if exit_prices else 0

        # 趋势不允许做空
        if trend["direction"] == "多头":
            # 有持仓
            if pos["current_shares"] > 0:
                lines.append("  [持仓] 继续持有")
                if entry["signal"] == "可入场" and pos["add_shares"] > 0:
                    lines.append(f"  [加仓] 条件满足，可加仓 {pos['add_shares']} 股")
                else:
                    lines.append("  [加仓] 等待更明确信号")
            else:
                if entry["signal"] == "可入场":
                    lines.append(f"  [建仓] 买入 {pos['add_shares']} 股")
                else:
                    lines.append("  [建仓] 等待入场条件满足")

            # 止损
            total_shares = max(pos["current_shares"], 1) if pos["add_shares"] == 0 else max(pos["current_shares"] + pos["add_shares"], 1)
            lines.append(f"  [止损] {stop_price:.2f}（亏损约 {abs(stop_price - current_price) * total_shares:,.0f}）")

            # 止盈
            if exit_prices:
                lines.append(f"  [止盈一] {exit_prices[0]:.2f}")
            if len(exit_prices) > 1:
                lines.append(f"  [止盈二] {exit_prices[1]:.2f}")
        else:
            lines.append("  当前为空头趋势，不宜做多")

        return "\n".join(lines)

    def financial_section(self, metrics: dict):
        """基本面指标段落（来自AKShare）"""
        if not metrics:
            return ""

        lines = [self._section("基本面指标")]
        if metrics.get("pe") is not None:
            lines.append(f"  PE(市盈率): {metrics['pe']:.2f}")
        if metrics.get("pb") is not None:
            lines.append(f"  PB(市净率): {metrics['pb']:.2f}")
        if metrics.get("roe") is not None:
            lines.append(f"  ROE(净资产收益率): {metrics['roe']:.2f}%")
        if metrics.get("total_mv") is not None:
            mv = metrics["total_mv"]
            lines.append(f"  总市值: {mv/1e8:.2f}亿" if mv > 1e8 else f"  总市值: {mv:.2f}")
        if metrics.get("float_mv") is not None:
            fmv = metrics["float_mv"]
            lines.append(f"  流通市值: {fmv/1e8:.2f}亿" if fmv > 1e8 else f"  流通市值: {fmv:.2f}")
        lines.append("")
        return "\n".join(lines)

    def moneyflow_section(self, flow: dict):
        """资金流向段落（来自AKShare）"""
        if not flow:
            return ""

        lines = [self._section("资金流向")]
        if flow.get("主力净流入") is not None:
            val = flow["主力净流入"]
            arr = "↑" if val > 0 else "↓"
            lines.append(f"  主力净流入: {arr} {abs(val)/1e4:.2f}万")
        if flow.get("主力净占比") is not None:
            lines.append(f"  主力净占比: {flow['主力净占比']:.2f}%")
        if flow.get("超大单净流入") is not None:
            val = flow["超大单净流入"]
            lines.append(f"  超大单净流入: {val/1e4:.2f}万" if val != 0 else "  超大单净流入: 0")
        lines.append("")
        return "\n".join(lines)

    def header(self):
        """报告头部"""
        lines = [
            self._line(),
            f"  {self.name} ({self.symbol})  交易报告",
            f"  {self.price_date.strftime('%Y-%m-%d %H:%M')}",
            self._line(),
        ]
        return "\n".join(lines)

    def footer(self):
        """报告尾部"""
        lines = [
            "",
            self._line("-"),
            "  风险提示: 本报告基于技术分析，仅供参考，不构成投资建议",
            "  股市有风险，投资需谨慎",
            self._line(),
        ]
        return "\n".join(lines)


# ============================================================
# 主程序
# ============================================================

def parse_position(arg):
    """解析持仓参数: --position 100,51.44"""
    try:
        parts = arg.split(",")
        if len(parts) == 2:
            return {"shares": int(parts[0]), "avg_price": float(parts[1])}
    except Exception:
        pass
    return {"shares": 0, "avg_price": 0}


def load_config(symbol):
    """从config.json加载该股票的个性化策略配置"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                configs = json.load(f)
            return configs.get(symbol, {})
        except Exception:
            pass
    return {}


# ============================================================
# 贝叶斯校准
# ============================================================

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration.json")


def _score_bracket(score):
    """评分→校准桶（每10分一档）"""
    if score >= 80: return "80"
    if score >= 70: return "70"
    if score >= 60: return "60"
    return "50"


def _load_calibration():
    """读取历史准确率校准数据"""
    try:
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def record_score_outcome(score, outcome_up):
    """记录评分→实际涨跌，更新校准数据

    Args:
        score: 综合评分
        outcome_up: True=上涨, False=下跌
    """
    bracket = _score_bracket(score)
    calib = _load_calibration()
    bucket = calib.setdefault(bracket, {"count": 0, "correct": 0, "accuracy": 0.5})
    bucket["count"] += 1
    if outcome_up:
        bucket["correct"] += 1
    bucket["accuracy"] = round(bucket["correct"] / bucket["count"], 4)
    try:
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(calib, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法:")
        print("  python main.py <股票代码> [账户总资金] [--position 持仓股数,持仓均价]")
        print("  python main.py scan [最小成交额(亿)] [--quick]")
        print("")
        print("示例:")
        print("  python main.py 002050              # 三花智控，默认资金8万")
        print("  python main.py 002050 80000         # 指定资金")
        print("  python main.py 002050 80000 --position 100,51.44   # 带持仓")
        print("  python main.py 000858 200000        # 五粮液")
        print("  python main.py scan                 # 全市场扫描，默认5亿成交额")
        print("  python main.py scan 10              # 全市场扫描，成交额≥10亿")
        print("  python main.py scan --quick         # 快速模式(跳过趋势检查)")
        sys.exit(0)

    # 子命令: scan
    if sys.argv[1] == "scan":
        min_amt = 5e8
        quick = False
        for arg in sys.argv[2:]:
            if arg.replace(".", "").isdigit():
                min_amt = float(arg) * 1e8
            elif arg == "--quick":
                quick = True
        from scanner import run_scanner
        run_scanner(min_amt, quick)
        return

    symbol = sys.argv[1].strip()
    account_value = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].replace(".", "").isdigit() else 80000

    # 解析可选参数
    position = {"shares": 0, "avg_price": 0}
    if "--position" in sys.argv:
        idx = sys.argv.index("--position")
        if idx + 1 < len(sys.argv):
            position = parse_position(sys.argv[idx + 1])

    # 加载策略配置
    config = load_config(symbol)

    try:
        print(f"获取数据 {symbol}...", end=" ", flush=True)

        fetcher = DataFetcher()
        daily = fetcher.get_daily(symbol, days=400)
        name = fetcher.get_name(symbol)

        print(f"{name}，{len(daily)} 个交易日")
        print("分析中...")

        # 价格
        price = daily["close"].iloc[-1]
        price_date = daily["date"].iloc[-1]

        # 技术指标
        daily = Analyzer.calc_atr(daily)
        daily = Analyzer.calc_ma(daily, [20, 50, 200])

        # 检测关键价位
        supports, resistances = Analyzer.detect_levels(daily)
        levels = {
            "supports": supports,
            "resistances": resistances,
            "current": price,
        }

        # 策略引擎
        strategy = Strategy(daily, account_value, position)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()
        entry_check = strategy.entry_check(levels)

        # 止损位（近3个月最低点下方一点）
        recent_lows = daily["low"].iloc[-60:].min()
        stop_price = round(recent_lows * 0.985, 2) if recent_lows > 0 else round(price * 0.93, 2)

        # 配置覆盖（如果config.json中有设定，优先使用）
        if "stop_price" in config:
            stop_price = config["stop_price"]
        if "exit_prices" in config:
            exit_prices = config["exit_prices"]
        else:
            exit_prices = [r for r in resistances] if resistances else [round(price * 1.08, 2)]
        if "supports" in config:
            levels["supports"] = config["supports"]
        if "resistances" in config:
            levels["resistances"] = config["resistances"]

        # AKShare 基本面 & 资金流（技能整合）
        fin_metrics = AkshareProvider.get_financial_indicators(symbol)
        moneyflow = AkshareProvider.get_stock_moneyflow(symbol)

        # 多因子评分
        scoring_result = None
        try:
            from scoring import StockScorer
            scorer = StockScorer(
                df=daily, price=price, trend=trend, vol=vol, levels=levels,
                stop_price=stop_price, exit_prices=exit_prices,
                position=position, symbol=symbol,
            )
            scoring_result = scorer.compute()
        except Exception:
            pass

        # 仓位计算（带评分）
        pos = strategy.position_plan(
            stop_price, entry_status=entry_check.get("signal"),
            composite_score=scoring_result["composite_score"] if scoring_result else None
        )

        # 生成报告
        report = Report(symbol, name, account_value, price_date)
        output = "\n".join([
            report.header(),
            report.financial_section(fin_metrics),
            report.moneyflow_section(moneyflow),
            report.trend_section(trend),
            report.volatility_section(vol),
            report.levels_section(levels),
            report.position_section(pos, current_price=price),
            report.signal_section(entry_check),
            report.scoring_section(scoring_result),
            report.recommendation_section(trend, entry_check, stop_price, exit_prices, pos, current_price=price),
            report.footer(),
        ])
        print(output)

    except ValueError as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n程序异常: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
