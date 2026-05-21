"""
多因子个股评分系统
对个股从6个维度综合评分，输出操作建议和理由
"""
import numpy as np
import pandas as pd
from data_providers import AkshareProvider


# 因子权重
WEIGHTS = {
    "wyckoff": 0.20,
    "risk_reward": 0.16,
    "volume": 0.12,
    "candlestick": 0.07,
    "sector": 0.11,
    "momentum": 0.12,
    "market": 0.12,
    "relative_strength": 0.10,
}

# 评分 → 操作映射
SCORE_LEVELS = [
    (80, "强加仓", 1.0),
    (65, "加仓", 0.7),
    (50, "持有", 0.0),
    (30, "减仓", -0.5),
    (0, "离场", -1.0),
]


def score_to_level(composite_score):
    """综合分 → (操作标签, 仓位系数)"""
    for threshold, action, factor in SCORE_LEVELS:
        if composite_score >= threshold:
            return action, factor
    return "离场", -1.0


class StockScorer:
    """多因子个股评分器"""

    def __init__(self, df, price, trend, vol, levels, stop_price, exit_prices,
                 wyckoff_signals=None, wyckoff_phase=None, industry=None,
                 position=None, symbol=None, default_sector_score=None,
                 market_df=None):
        """
        Args:
            df: DataFrame with OHLCV + ma20/ma50/ma200/atr/atr_pct
            price: 当前价格
            trend: trend_analysis() 返回的 dict
            vol: volatility_analysis() 返回的 dict
            levels: {"supports": [...], "resistances": [...], "current": price}
            stop_price: 止损价
            exit_prices: 止盈价位列表
            wyckoff_signals: analyze_all() 返回的 signal list，或 None
            wyckoff_phase: detect_phase() 返回的阶段标签，或 None
            industry: 行业名称，或 None（将自动获取）
            position: {"shares": int, "avg_price": float} 或 None
            market_df: 大盘指数DataFrame（含close列），用于大盘评分
        """
        self.df = df
        self.latest = df.iloc[-1] if len(df) > 0 else {}
        self.prev = df.iloc[-2] if len(df) > 1 else {}
        self.price = price
        self.trend = trend or {}
        self.vol = vol or {}
        self.levels = levels or {}
        self.supports = levels.get("supports", []) if levels else []
        self.resistances = levels.get("resistances", []) if levels else []
        self.stop_price = stop_price
        self.exit_prices = exit_prices or []
        self.wyckoff_signals = wyckoff_signals or []
        self.wyckoff_phase = wyckoff_phase or ""
        self.industry = industry
        self.symbol = symbol
        self.default_sector_score = default_sector_score
        self.position = position or {"shares": 0, "avg_price": 0}
        self.market_df = market_df

    # ─────────────────────── 工具方法 ───────────────────────

    @staticmethod
    def _calc_rsi(series, period=14):
        """计算 RSI"""
        if len(series) < period + 1:
            return 50.0
        deltas = np.diff(series)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - 100.0 / (1.0 + rs), 1)

    def _label(self, score):
        """分数 → 等级标签"""
        if score >= 80: return "优"
        if score >= 65: return "良"
        if score >= 50: return "中"
        if score >= 30: return "差"
        return "劣"

    # ─────────────────────── 因子1: 威科夫信号 ───────────────────────

    def score_wyckoff(self):
        """威科夫信号评分 (权重25%)"""
        if not self.wyckoff_signals:
            return {
                "score": 50, "label": "中",
                "detail": "无威科夫信号数据",
                "raw": "-", "raw_score": 0,
            }

        # 取最强信号
        best_sig, best_raw_score, best_detail = self.wyckoff_signals[0]

        # 信号类型 → 基准分
        type_map = {
            "SOS": 90, "Spring": 85, "LPS": 75,
            "Compression": 65, "Markup": 80,
            "Upthrust": 30, "EVR": 40,
        }

        # 检查是否为弱信号（"弱"前缀）
        is_weak = best_sig.startswith("弱")
        clean_sig = best_sig.replace("弱", "")

        base = type_map.get(clean_sig, 50)
        if is_weak:
            base -= 15

        # 用原始检测分微调 (±10)
        if 30 <= best_raw_score <= 100:
            if best_raw_score >= 80:
                base = min(100, base + 8)
            elif best_raw_score <= 50:
                base = max(0, base - 8)

        # 阶段加成
        phase = self.wyckoff_phase or ""
        if "Markup" in phase or "Phase D" in phase or "Phase E" in phase:
            base += 5
        elif "Phase B" in phase or "Phase C" in phase:
            base += 3
        elif "派发" in phase or "Phase" not in phase:
            base -= 5

        final = max(0, min(100, base))
        return {
            "score": final, "label": self._label(final),
            "detail": f"{best_sig} {best_raw_score}分" + (f" | {best_detail}" if best_detail else ""),
            "raw": best_sig, "raw_score": best_raw_score,
        }

    # ─────────────────────── 因子2: 盈亏比 ───────────────────────

    def score_risk_reward(self):
        """盈亏比评分 (权重20%)"""
        risk = self.price - self.stop_price
        if risk <= 0:
            return {"score": 10, "label": "劣", "detail": "止损价高于现价", "ratio": 0}

        # 找最近的阻力位作为止盈目标
        take_profit = None
        for r in self.resistances:
            if r > self.price:
                take_profit = r
                break
        if take_profit is None:
            if self.exit_prices:
                take_profit = self.exit_prices[0]
            else:
                take_profit = self.price * 1.08

        reward = take_profit - self.price
        rr_ratio = round(reward / risk, 2)

        # 空头趋势打折
        trend_dir = self.trend.get("direction", "未知")
        discount = 0.7 if trend_dir == "空头" else 1.0

        if rr_ratio >= 3.0:
            base = 95
        elif rr_ratio >= 2.0:
            base = 80
        elif rr_ratio >= 1.5:
            base = 60
        elif rr_ratio >= 1.0:
            base = 40
        else:
            base = 20

        final = max(5, min(100, int(base * discount)))

        return {
            "score": final, "label": self._label(final),
            "detail": f"{rr_ratio}:1 (风{risk:.2f} 酬{reward:.2f})",
            "ratio": rr_ratio,
        }

    # ─────────────────────── 因子3: 量能分析 ───────────────────────

    def score_volume(self):
        """量能分析评分 (权重15%)"""
        try:
            closes = self.df["close"].values.astype(float)
            volumes = self.df["volume"].values.astype(float)
        except Exception:
            return {"score": 50, "label": "中", "detail": "数据不足", "vol_ratio": 0}

        if len(volumes) < 25:
            return {"score": 50, "label": "中", "detail": "数据不足", "vol_ratio": 0}

        # 量比 = 当日量 / 20日均量
        latest_v = float(volumes[-1])
        avg_20 = float(np.mean(volumes[-21:-1])) if len(volumes) >= 21 else 1
        vol_ratio = latest_v / avg_20 if avg_20 > 0 else 1

        # 量趋势：近5日 / 前10日
        avg_5 = float(np.mean(volumes[-6:-1])) if len(volumes) >= 6 else 1
        avg_prior_10 = float(np.mean(volumes[-16:-6])) if len(volumes) >= 16 else 1
        vol_trend = avg_5 / avg_prior_10 if avg_prior_10 > 0 else 1.0

        # 价格涨跌
        price_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0

        # 评分
        if 1.2 <= vol_ratio <= 2.5:
            if vol_trend >= 1.1:
                base = 80  # 温和放量，量趋势上升
            else:
                base = 60  # 温和放量，量持平
        elif vol_ratio > 2.5:
            if price_chg < 1.0:
                base = 30  # 巨量滞涨（警惕派发）
            elif price_chg >= 3.0:
                base = 75  # 巨量大涨（SOS确认）
            else:
                base = 55  # 巨量小涨
        elif 0.5 <= vol_ratio <= 1.2:
            base = 50  # 量平
        else:  # < 0.5
            base = 40  # 缩量

        # 在支撑位附近缩量 → 抛压枯竭加分
        if self.supports and vol_ratio < 0.7:
            nearest_support = self.supports[-1]
            dist = (self.price - nearest_support) / self.price * 100
            if 0 <= dist < 3:
                base += 10

        final = max(0, min(100, base))
        vol_trend_str = f"{'上升' if vol_trend >= 1.05 else '下降' if vol_trend <= 0.95 else '持平'}"
        return {
            "score": final, "label": self._label(final),
            "detail": f"量比{vol_ratio:.1f}倍 量趋势{vol_trend_str}",
            "vol_ratio": round(vol_ratio, 2),
        }

    # ─────────────────────── 因子4: K线形态 ───────────────────────

    def score_candlestick(self):
        """K线形态评分 (权重10%)"""
        try:
            o = float(self.latest.get("open", self.price))
            h = float(self.latest.get("high", self.price))
            l = float(self.latest.get("low", self.price))
            c = float(self.latest.get("close", self.price))
            p_o = float(self.prev.get("open", o))
            p_h = float(self.prev.get("high", h))
            p_l = float(self.prev.get("low", l))
            p_c = float(self.prev.get("close", c))
        except Exception:
            return {"score": 50, "label": "中", "detail": "数据不足"}

        total_range = h - l
        if total_range < 0.01:
            return {"score": 50, "label": "中", "detail": "振幅过小"}

        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        is_green = c > o
        pos_in_range = (c - l) / total_range

        score = 50  # 基准分

        # 在支撑/阻力位附近的辅助判断
        near_support = False
        near_resistance = False
        if self.supports:
            nearest_s = self.supports[-1]
            if 0 <= (self.price - nearest_s) / self.price * 100 < 3:
                near_support = True
        if self.resistances:
            nearest_r = self.resistances[0]
            if 0 <= (nearest_r - self.price) / self.price * 100 < 3:
                near_resistance = True

        # 看涨吞没
        if is_green and not (p_c > p_o) and o <= p_c and c >= p_o:
            score += 20

        # 锤子线（下影线 ≥ 实体2倍，上影线短）
        if lower_wick >= body * 2 and upper_wick <= body * 0.5 and total_range > 0:
            score += 15

        # 十字星（实体极小）
        if body / total_range < 0.05:
            score += 10

        # 看跌吞没
        if not is_green and (p_c > p_o) and o >= p_c and c <= p_o:
            score -= 20

        # 射击之星（上影线 ≥ 实体2倍，下影线短）
        if upper_wick >= body * 2 and lower_wick <= body * 0.5 and total_range > 0:
            score -= 15

        # 收盘位置
        if pos_in_range >= 0.67:
            score += 5
        elif pos_in_range <= 0.33:
            score -= 5

        # 在支撑位出锤子线/十字星 → 额外加分
        if near_support and (lower_wick >= body * 2 or body / total_range < 0.05):
            score += 8

        # 在阻力位出射击之星/看跌吞没 → 额外减分
        if near_resistance and (upper_wick >= body * 2 or (not is_green and p_c > p_o)):
            score -= 8

        final = max(0, min(100, score))
        return {
            "score": final, "label": self._label(final),
            "detail": f"{'阳线' if is_green else '阴线'} 实体{body:.2f} 影线({upper_wick:.2f}/{lower_wick:.2f})",
        }

    # ─────────────────────── 因子5: 板块强度 ───────────────────────

    def score_sector(self):
        """板块强度评分 (权重15%)"""
        # 获取行业名称
        industry = self.industry
        if not industry and self.symbol:
            try:
                industry = AkshareProvider.get_stock_industry(self.symbol)
            except Exception:
                pass
        if not industry or industry == "其他":
            if self.default_sector_score is not None:
                return {"score": self.default_sector_score, "label": self._label(self.default_sector_score),
                        "detail": f"行业默认分{self.default_sector_score}", "rank_pct": 50}
            return {"score": 50, "label": "中", "detail": "行业未知", "rank_pct": 50}

        # 获取行业板块涨跌幅排行
        try:
            boards = AkshareProvider.get_industry_board_performance()
        except Exception:
            boards = []

        if not boards:
            if self.default_sector_score is not None:
                return {"score": self.default_sector_score, "label": self._label(self.default_sector_score),
                        "detail": f"{industry} (板块数据无,默认{self.default_sector_score})", "rank_pct": 50}
            return {"score": 50, "label": "中", "detail": f"{industry} (板块数据无)", "rank_pct": 50}

        # 找所属行业的排名
        target = None
        for i, b in enumerate(boards):
            if industry in b.get("name", ""):
                target = {"rank": i + 1, "total": len(boards), "change_pct": b.get("change_pct", 0)}
                break

        if not target:
            if self.default_sector_score is not None:
                return {"score": self.default_sector_score, "label": self._label(self.default_sector_score),
                        "detail": f"{industry} 未在排行中找到(默认{self.default_sector_score})", "rank_pct": 50}
            return {"score": 50, "label": "中", "detail": f"{industry} 未在板块排行中找到", "rank_pct": 50}

        rank_pct = target["rank"] / target["total"] * 100
        change = target["change_pct"]

        if rank_pct <= 10:
            base = 90
        elif rank_pct <= 25:
            base = 75
        elif rank_pct <= 50:
            base = 60
        elif rank_pct <= 75:
            base = 35
        else:
            base = 20

        final = max(0, min(100, base))
        return {
            "score": final, "label": self._label(final),
            "detail": f"{industry} {change:+.2f}% 排名{target['rank']}/{target['total']}",
            "rank_pct": round(rank_pct, 1),
            "change_pct": change,
        }

    # ─────────────────────── 因子6: 趋势与动量 ───────────────────────

    def score_momentum(self):
        """趋势与动量评分 (权重15%)"""
        trend_dir = self.trend.get("direction", "未知")

        # 基准：多头40，空头20
        base = 40 if trend_dir == "多头" else 20
        details = []

        # MA50/200 gap
        try:
            ma50 = float(self.latest.get("ma50", 0))
            ma200 = float(self.latest.get("ma200", 0))
            if ma200 > 0 and ma50 > 0:
                gap = (ma50 - ma200) / ma200 * 100
                if gap > 0:
                    bonus = min(20, int(gap * 2))
                    base += bonus
                    details.append(f"MA多头排列(gap={gap:.1f}%) +{bonus}")
                else:
                    base += max(-10, int(gap))
                    details.append(f"MA空头排列(gap={gap:.1f}%) {int(gap)}")
        except Exception:
            pass

        # RSI
        try:
            closes = self.df["close"].values.astype(float)
            rsi = self._calc_rsi(closes)
            if 40 <= rsi <= 60:
                details.append(f"RSI{rsi} 中性")
            elif 30 <= rsi < 40:
                base += 10
                details.append(f"RSI{rsi} 超卖反弹潜力 +10")
            elif 60 < rsi <= 70:
                base += 10
                details.append(f"RSI{rsi} 强势 +10")
            elif rsi < 30 or rsi > 70:
                base -= 10
                details.append(f"RSI{rsi} 极端区域 -10")
        except Exception:
            rsi = 50

        # 价格在MA20上方/下方
        try:
            ma20 = float(self.latest.get("ma20", 0))
            if ma20 > 0:
                if self.price > ma20:
                    base += 10
                    details.append("价格>MA20 +10")
                else:
                    base -= 10
                    details.append("价格<MA20 -10")
        except Exception:
            pass

        # ATR%波动率适中加分
        atr_pct = self.vol.get("atr_pct", 0)
        if 1.5 <= atr_pct <= 4.0:
            base += 10
            details.append(f"ATR%{atr_pct}% 适中 +10")
        elif atr_pct > 4.0:
            details.append(f"ATR%{atr_pct}% 偏高")

        final = max(0, min(100, base))
        detail_str = " | ".join(details) if details else trend_dir

        return {
            "score": final, "label": self._label(final),
            "detail": detail_str,
            "rsi": rsi,
        }

    # ─────────────────────── 因子7: 大盘趋势 ───────────────────────

    def score_market(self):
        """大盘指数趋势评分 (权重14%)"""
        mdf = self.market_df
        if mdf is None or len(mdf) < 30:
            return {"score": 50, "label": "中", "detail": "大盘数据不足", "trend": "未知"}

        closes = mdf["close"].values.astype(float)
        current = closes[-1]

        # 计算MA
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else current
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else current
        ma200 = np.mean(closes[-200:]) if len(closes) >= 200 else current

        # 趋势方向
        trend_dir = "多头" if current > ma50 > ma200 else ("空头" if current < ma50 < ma200 else "震荡")

        score = 50  # 基准
        details = []

        # MA排列评分
        if current > ma50 > ma200:
            score += 20
            details.append("多头排列 +20")
        elif current < ma50 < ma200:
            score -= 15
            details.append("空头排列 -15")
        else:
            details.append("震荡整理")

        # MA gap
        if ma200 > 0:
            gap = (ma50 - ma200) / ma200 * 100
            if gap > 5:
                score += 10
                details.append(f"MA gap+{gap:.1f}% +10")
            elif gap > 0:
                score += 5
                details.append(f"MA gap+{gap:.1f}% +5")
            elif gap > -5:
                score -= 5
                details.append(f"MA gap{gap:.1f}% -5")
            else:
                score -= 10
                details.append(f"MA gap{gap:.1f}% -10")

        # 近20日涨幅
        if len(closes) >= 20:
            ret_20d = (closes[-1] / closes[-20] - 1) * 100
            if ret_20d > 5:
                score += 10
                details.append(f"近20日+{ret_20d:.1f}% +10")
            elif ret_20d > 0:
                score += 3
            elif ret_20d > -5:
                score -= 3
            else:
                score -= 8
                details.append(f"近20日{ret_20d:.1f}% -8")

        # 成交量确认（涨有量跌缩量 → 健康）
        if len(mdf) >= 40:
            volumes = mdf["volume"].values.astype(float) if "volume" in mdf.columns else np.ones_like(closes) * np.mean(closes)
            vol_20 = np.mean(volumes[-20:])
            vol_prior = np.mean(volumes[-40:-20]) if len(volumes) >= 40 else vol_20
            if vol_20 > vol_prior * 1.2 and ret_20d if len(closes) >= 20 else 0:
                pass  # 放量上涨已经体现在涨幅加分中

        final = max(0, min(100, score))
        detail_str = " | ".join(details) if details else trend_dir
        return {
            "score": final, "label": self._label(final),
            "detail": detail_str, "trend": trend_dir,
        }

    # ─────────────────────── 因子8: 相对强度 ───────────────────────

    def score_relative_strength(self):
        """相对强度评分: 个股 vs 大盘 (权重10%)

        逆势上涨加分，顺势下跌扣分。衡量个股在市场中的相对表现。
        """
        mdf = self.market_df
        df = self.df
        if mdf is None or df is None or len(df) < 2 or len(mdf) < 2:
            return {"score": 50, "label": "中", "detail": "数据不足"}

        try:
            s_close = df["close"].values.astype(float)
            m_close = mdf["close"].values.astype(float)

            # 取各序列最后一笔，计算1日/5日/20日相对收益
            def rel_ret(s_arr, m_arr, n):
                if len(s_arr) < n + 1 or len(m_arr) < n + 1:
                    return 0
                s_ret = (s_arr[-1] / s_arr[-(n+1)] - 1) * 100
                m_ret = (m_arr[-1] / m_arr[-(n+1)] - 1) * 100
                return s_ret - m_ret

            d1 = rel_ret(s_close, m_close, 1)
            d5 = rel_ret(s_close, m_close, 5)  if max(len(s_close), len(m_close)) >= 6 else 0
            d20 = rel_ret(s_close, m_close, 20) if max(len(s_close), len(m_close)) >= 21 else 0

            def diff_score(diff):
                if diff > 5: return 15
                if diff > 2: return 8
                if diff > 0.5: return 3
                if diff > -0.5: return 0
                if diff > -2: return -3
                if diff > -5: return -8
                return -15

            total = diff_score(d1) * 0.5 + diff_score(d5) * 0.3 + diff_score(d20) * 0.2
            score = max(0, min(100, 50 + total))

            parts = []
            if d1 != 0: parts.append(f"1日{d1:+.1f}%")
            if d5 != 0: parts.append(f"5日{d5:+.1f}%")
            if d20 != 0: parts.append(f"20日{d20:+.1f}%")
            detail = " | ".join(parts) if parts else "与大盘持平"
            return {"score": score, "label": self._label(score), "detail": detail}
        except Exception:
            return {"score": 50, "label": "中", "detail": "计算异常"}

    # ─────────────────────── 综合评分 ───────────────────────

    def compute(self):
        """运行所有因子评分，返回综合结果"""
        factors = {}

        # 逐个计算，每个失败都降级为50分
        for key, method in [
            ("wyckoff", self.score_wyckoff),
            ("risk_reward", self.score_risk_reward),
            ("volume", self.score_volume),
            ("candlestick", self.score_candlestick),
            ("sector", self.score_sector),
            ("momentum", self.score_momentum),
            ("market", self.score_market),
            ("relative_strength", self.score_relative_strength),
        ]:
            try:
                factors[key] = method()
            except Exception as e:
                factors[key] = {"score": 50, "label": "中", "detail": f"计算异常: {e}"}

        # 加权综合
        composite = 0.0
        breakdown_lines = []
        for key, weight in WEIGHTS.items():
            f = factors[key]
            contribution = f["score"] * weight
            composite += contribution
            breakdown_lines.append({
                "key": key,
                "weight": weight,
                "score": f["score"],
                "contribution": round(contribution, 1),
                "label": f["label"],
                "detail": f["detail"],
                "weight_pct": int(weight * 100),
            })

        composite = round(composite, 1)
        action, pos_factor = score_to_level(composite)

        return {
            "composite_score": composite,
            "action": action,
            "pos_factor": pos_factor,
            "factors": breakdown_lines,
            "level": self._label(composite),
        }
