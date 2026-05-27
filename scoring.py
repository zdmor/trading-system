"""
多因子个股评分系统
对个股从6个维度综合评分，输出操作建议和理由
"""
import os
import numpy as np
import pandas as pd
from data_providers import AkshareProvider

# 可选依赖: 板块景气度五维评分
try:
    from sector_heat import analyze as sector_heat_analyze
    _SECTOR_HEAT_OK = True
except ImportError:
    _SECTOR_HEAT_OK = False

# 可选依赖: 仓位决策矩阵
try:
    from position_matrix import get_position_factor
    _POSITION_MATRIX_OK = True
except ImportError:
    _POSITION_MATRIX_OK = False

# 可选依赖: 深度研究个股
try:
    from research_watchlist import ResearchTracker
    _RESEARCH_TRACKER_OK = True
except ImportError:
    _RESEARCH_TRACKER_OK = False

# Kelly 仓位模块
try:
    from kelly_position import compute_kelly_position
    _KELLY_OK = True
except ImportError:
    _KELLY_OK = False


try:
    from buzz_monitor import get_stock_buzz
    _BUZZ_TOOL = True
except ImportError:
    _BUZZ_TOOL = False

try:
    from volume_momentum import VolumeMomentum
    _VOL_MOMENTUM = True
except ImportError:
    _VOL_MOMENTUM = False

# 动态因子权重系统（ICIR驱动）
try:
    from factor_weights import get_weights
    _DYNAMIC_WEIGHTS_OK = True
except ImportError:
    _DYNAMIC_WEIGHTS_OK = False

# 多因子多空辩论模块
try:
    from bull_bear_debate import debate_factors
    _DEBATE_OK = True
except ImportError:
    _DEBATE_OK = False

# 板块数据缓存（跨股票共享，避免重复抓取）
_SECTOR_BOARD_CACHE = {'data': [], 'time': 0.0}
_SECTOR_CACHE_TTL = 3600

WEIGHTS = {
    "risk_reward": 0.37,           # 盈亏比 — 唯一全周期正IC (+0.064, ICIR 0.33)
    "tech_strength": 0.18,         # 威科夫+趋势动量 — 反转市有效(IC+0.022, 胜率58%)
    "relative_strength": 0.14,     # 相对强度 — 反转市改善
    "volume": 0.12,                # 量比动量 — 反转市IC+0.011, 胜率58%
    "candlestick": 0.11,           # K线形态 — 反转市ICIR改善(-0.32→-0.09)
    "volatility": 0.08,            # 波动率 — 全周期正IC(+0.051, ICIR+0.31)
}
# 权重基于 144 截面 × 1972 只股票 IC 回测 (2026-05-27)。
# sector 因子从评分权重中移除（IC=-0.032），降级为仓位约束。
# 大盘趋势不参与个股评分，在 position_matrix.py 做仓位门槛。
#
# 关键发现（2026-05-27 市态分组IC评估）：
#   - 趋势市(52%截面)：全部因子IC为负，选股有效度低
#   - 反转市(17%截面)：因子IC明显改善，动量IC+0.022、量能IC+0.011、互证IC+0.011
#   - 反转市胜率全部>50%(动量58%/量能58%/均线58%)
#   - 波动因子反转市IC最强(-0.095, ICIR=-0.32)，全周期正IC
#   - 结论：因子是环境依赖型有效，反转市有效≠需降权
#   - 修正：删除之前"反转市下调技术因子"的错误做法
# 互证因子(多信号一致性)作为评分调节，不参与加权。
# 质_动量内嵌在动量评分中做量价确认。
# Δ因子(Δ动量/Δ量能/Δ均线): IC近零,不纳入评分。

#
# 评分 → 操作映射
SCORE_LEVELS = [
    (80, "强加仓", 1.0),
    (65, "加仓", 0.7),
    (50, "持有", 0.0),
    (30, "减仓", -0.5),
    (0, "离场", -1.0),
]

# 校准期望准确率（用于自动纠偏乐观/悲观偏差）
_CALIB_EXPECTED = {"80": 0.62, "70": 0.56, "60": 0.52, "50": 0.48}
_CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")


def _score_bracket(score):
    if score >= 80: return "80"
    if score >= 70: return "70"
    if score >= 60: return "60"
    return "50"


def _apply_calibration(composite):
    """基于历史校准数据修正评分。

    如果某档位历史准确率低于期望，自动下调评分（系统性乐观偏误修正）；
    高于期望则上调（保守偏误修正）。
    """
    bracket = _score_bracket(composite)
    try:
        if not os.path.exists(_CALIB_FILE):
            return composite, ""
        import json
        with open(_CALIB_FILE, encoding="utf-8") as f:
            calib = json.load(f)
        bucket = calib.get(bracket, {})
        count = bucket.get("count", 0)
        accuracy = bucket.get("accuracy", 0.5)
        if count < 10:
            return composite, "校准样本不足"
        expected = _CALIB_EXPECTED.get(bracket, 0.5)
        gap = expected - accuracy
        if gap > 0.05:
            penalty = int(gap * 100)
            adjusted = max(0, composite - penalty)
            return adjusted, f"校准{penalty:+.0f}({accuracy:.0%}<{expected:.0%})"
        elif gap < -0.05:
            bonus = int(abs(gap) * 100)
            adjusted = min(100, composite + bonus)
            return adjusted, f"校准+{bonus}({accuracy:.0%}>{expected:.0%})"
        return composite, ""
    except Exception:
        return composite, ""


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

    # ─────────────────────── 因子: 技术强势 (Wyckoff+趋势动量合并) ───────────────────────

    def score_tech_strength(self):
        """技术强势评分 (权重28%)
        P0-3 证明 Wyckoff 与趋势动量 ρ=0.73 高度共线，合并为一个因子。
        取两者等权平均，消除冗余但保留各自的信息增量。
        """
        w = self.score_wyckoff()
        m = self.score_momentum()

        combo_score = round((w["score"] + m["score"]) / 2)
        combo_label = self._label(combo_score)
        combo_detail = f"W:{w['score']}({w.get('raw','-')}) M:{m['score']}({m.get('rsi',50)})"

        return {
            "score": combo_score,
            "label": combo_label,
            "detail": combo_detail,
            "wyckoff_score": w["score"],
            "wyckoff_raw": w.get("raw", ""),
            "momentum_score": m["score"],
            "momentum_rsi": m.get("rsi", 50),
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

    # ─────────────────────── 因子3: 量能分析（量比动量系统） ───────────────────────

    def score_volume(self):
        """量比动量评分 (权重9%)
        基于 VolumeMomentum 的量比均线+斜率+综合评分
        """
        try:
            volumes = self.df["volume"].values.astype(float)
            closes = self.df["close"].values.astype(float)
        except Exception:
            return {"score": 50, "label": "中", "detail": "数据不足", "vol_ratio": 0}

        if len(volumes) < 10:
            return {"score": 50, "label": "中", "detail": "数据不足", "vol_ratio": 0}

        # 市态判定（用已有大盘数据或默认中性）
        regime = "neutral"
        if hasattr(self, "market_df") and self.market_df is not None:
            try:
                idx_closes = self.market_df["close"].values.astype(float)
                regime = VolumeMomentum.recommend_regime(list(idx_closes))
            except Exception:
                pass

        # 量比动量分析（传入历史量比序列用于百分位归一化）
        # 构建最近120日量比序列（不含当日）作为历史参考
        vol_history = None
        if len(volumes) >= 21:
            lookback = min(120, len(volumes) - 1)
            hist_ratios = []
            for i in range(len(volumes) - lookback, len(volumes)):
                avg = np.mean(volumes[max(0, i-20):i]) if i >= 20 else np.mean(volumes[:max(1,i)])
                hist_ratios.append(volumes[i] / avg if avg > 0 else 1.0)
            vol_history = hist_ratios[-lookback:]

        vm = VolumeMomentum(list(volumes), regime=regime, closes=list(closes),
                           vol_history=vol_history)
        result = vm.analyze()

        # 百分制映射：优先百分位，回退绝对阈值
        pct = result.get("percentile")
        if pct is not None:
            if pct >= 90:    base = 92
            elif pct >= 75:  base = 80
            elif pct >= 50:  base = 65
            elif pct >= 25:  base = 45
            else:            base = 20
            # 放量上涨加成、放量下跌打折
            direction = result.get("direction", "正常")
            if direction == "放量上涨" and pct >= 75:
                base = min(100, base + 8)
            elif direction == "放量下跌" and pct >= 75:
                base = max(0, base - 15)
        else:
            # 回退：绝对阈值
            cs = result["composite_score"]
            direction = result.get("direction", "正常")
            if cs >= 3.5:
                base = 85
                if direction == "放量上涨":
                    base = 92
                elif direction == "放量下跌":
                    base = 50
            elif cs >= 2.0:
                base = 65 + (cs - 2.0) / 1.5 * 15  # 65-80
            elif cs >= 1.3:
                base = 45 + (cs - 1.3) / 0.7 * 20  # 45-65
            elif cs >= 0.5:
                base = 30 + (cs - 0.5) * 15  # 30-45
            else:
                base = 20

        # 三重过滤调整
        checks_passed = result["checks_passed"]
        if not checks_passed:
            base -= 15  # 过滤不通过扣分
        elif result["slope"] > 0.1:
            base += 10  # 加速放量加分

        # 支撑/阻力附加判断
        price_chg = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
        vol_ratio = result["vol_ratio"]

        if self.supports:
            nearest_s = self.supports[-1]
            dist_s = (nearest_s - self.price) / self.price * 100
            if dist_s < 0 and abs(dist_s) < 3:
                if vol_ratio >= 1.5:
                    base -= 15
                elif vol_ratio < 0.8:
                    base += 12

        if self.resistances:
            nearest_r = self.resistances[0]
            dist_r = (self.price - nearest_r) / self.price * 100
            if dist_r > 0 and abs(dist_r) < 3:
                if vol_ratio >= 1.5:
                    base += 15
                elif vol_ratio < 0.8:
                    base -= 12

        final = max(0, min(100, base))
        detail_parts = [
            f"量比{result['vol_ratio']:.1f}",
        ]
        if pct is not None:
            detail_parts.append(f"百分位P{pct:.0f}")
        detail_parts.extend([
            f"均线{result['vol_ratio_ma']:.1f}",
            f"斜率{result['slope']:.2f}",
            f"综合{result['composite_score']:.1f}",
            result['direction'],
            result['signal'],
        ])

        return {
            "score": final, "label": self._label(final),
            "detail": " | ".join(detail_parts),
            "vol_ratio": round(vol_ratio, 2),
            "vol_momentum": result,
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

        # 获取行业板块涨跌幅排行（带缓存）
        import time as _time
        boards = []
        now = _time.time()
        if now - _SECTOR_BOARD_CACHE["time"] < _SECTOR_CACHE_TTL:
            boards = _SECTOR_BOARD_CACHE["data"]
        else:
            try:
                boards = AkshareProvider.get_industry_board_performance()
                if boards:
                    _SECTOR_BOARD_CACHE["data"] = boards
                    _SECTOR_BOARD_CACHE["time"] = now
            except Exception:
                boards = _SECTOR_BOARD_CACHE["data"] if _SECTOR_BOARD_CACHE["data"] else []

        if not boards:
            # 板块排行数据不可用 → 尝试五维景气度评分
            if _SECTOR_HEAT_OK:
                try:
                    heat = sector_heat_analyze(industry)
                    if heat and heat.get("composite"):
                        hs = heat["composite"]
                        return {"score": hs, "label": self._label(hs),
                                "detail": f"{industry} 景气{hs} {heat.get('level','')}",
                                "rank_pct": 50, "heat_score": hs, "heat_level": heat.get("level", "")}
                except Exception:
                    pass
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

        # 板块景气度五维增强（可选）
        heat_score = None
        if _SECTOR_HEAT_OK:
            try:
                heat = sector_heat_analyze(industry)
                if heat and heat.get("composite"):
                    heat_score = heat["composite"]
            except Exception:
                pass

        if heat_score is not None:
            blended = round(final * 0.6 + heat_score * 0.4)
            detail_extra = f" 景气{heat_score} {heat.get('level','')}"
            return {
                "score": blended, "label": self._label(blended),
                "detail": f"{industry} {change:+.2f}% 排名{target['rank']}/{target['total']}{detail_extra}",
                "rank_pct": round(rank_pct, 1),
                "heat_score": heat_score,
                "heat_level": heat.get("level", ""),
            }

        return {
            "score": final, "label": self._label(final),
            "detail": f"{industry} {change:+.2f}% 排名{target['rank']}/{target['total']}",
            "rank_pct": round(rank_pct, 1),
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

        # RSI — 方向修正 2026-05-24: 历史IC证明RSI是动量指标不是反转指标
        # 高RSI(>60)=强势延续，低RSI(<40)=弱势延续
        try:
            closes = self.df["close"].values.astype(float)
            rsi = self._calc_rsi(closes)
            if rsi > 70:
                base += 15
                details.append(f"RSI{rsi} 强势延续 +15")
            elif 60 < rsi <= 70:
                base += 10
                details.append(f"RSI{rsi} 偏强 +10")
            elif 40 <= rsi <= 60:
                details.append(f"RSI{rsi} 中性")
            elif 30 <= rsi < 40:
                base -= 5
                details.append(f"RSI{rsi} 偏弱 -5")
            else:  # rsi < 30
                base -= 10
                details.append(f"RSI{rsi} 弱势 -10")
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

        # 质_动量：量能验证趋势质量 — 趋势需要成交量确认才可信
        try:
            volumes = self.df["volume"].values.astype(float)
            vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            vol_ratio = volumes[-1] / max(vol_ma20, 1)
            vol_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
            vol_prev5 = np.mean(volumes[-10:-5]) if len(volumes) >= 10 else 1
            vol_trend = vol_5 / max(vol_prev5, 1)

            # 多头趋势+放量=趋势确认(加码)，多头趋势+缩量=趋势存疑(减码)
            if base > 55 and vol_ratio > 1.2 and vol_trend > 1.1:
                base += 10
                details.append(f"量比{vol_ratio:.1f}趋势{vol_trend:.2f}趋势确认 +10")
            elif base > 55 and vol_ratio < 0.7:
                base -= 8
                details.append(f"缩量上涨(量比{vol_ratio:.1f}) -8")
            # 空头趋势+放量=趋势确认向下
            elif base < 35 and vol_ratio > 1.3:
                base -= 8
                details.append(f"放量下跌(量比{vol_ratio:.1f}) -8")
        except Exception:
            pass

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

    # ─────────────────────── 因子9: 波动率 ───────────────────────

    def score_volatility(self):
        """波动率评分 (权重8%)

        全周期正IC(+0.051, ICIR+0.31)，反转市区分度最强(IC=-0.095, ICIR=-0.32)。
        逻辑：低波动=筹码稳定有主力，高波动=情绪化交易=负期望。
        """
        try:
            closes = self.df["close"].values.astype(float)
            n = len(closes)
            if n < 21:
                return {"score": 50, "label": "中", "detail": "数据不足", "vol_pct": 0}
        except Exception:
            return {"score": 50, "label": "中", "detail": "数据不足", "vol_pct": 0}

        # 20日收益率日波动率 → 年化
        rets_daily = np.diff(closes[-21:]) / closes[-21:-1]
        vol_annual = float(np.std(rets_daily) * np.sqrt(252))

        # A股典型年化波动区间 15%-40%
        if vol_annual <= 0.15:
            score = 85
        elif vol_annual <= 0.20:
            score = 75
        elif vol_annual <= 0.25:
            score = 60
        elif vol_annual <= 0.30:
            score = 40
        elif vol_annual <= 0.35:
            score = 25
        else:
            score = 15

        # ATR%微调
        atr_pct = self.vol.get("atr_pct", 0)
        if atr_pct < 1.5:
            score = min(100, score + 5)
        elif atr_pct > 5.0:
            score = max(0, score - 10)

        final = max(0, min(100, score))
        return {
            "score": final, "label": self._label(final),
            "detail": f"年化波动{vol_annual:.1%} ATR%{atr_pct:.1f}%",
            "vol_pct": round(vol_annual * 100, 1),
        }

    # ─────────────────────── 互证因子（多信号一致性调节）───────────────────────

    def _calc_mutual_confirmation_bonus(self):
        """计算互证因子调节分，返回 (bonus, info_dict)
        bonus ∈ [-15, 15] 加到综合分，正值=多信号看涨一致，负值=多信号看跌一致

        验证量价、趋势、RSI三组信号的方向一致性。
        信号越一致，方向越可信，加分/减分幅度越大。
        """
        try:
            closes = self.df["close"].values.astype(float)
            volumes = self.df["volume"].values.astype(float)
            n = len(closes)
            if n < 20:
                return 0, {}
        except Exception:
            return 0, {}

        price = closes[-1]

        # 量比
        vol_ma20 = np.mean(volumes[-20:]) if n >= 20 else np.mean(volumes)
        vol_ratio = volumes[-1] / max(vol_ma20, 1)

        # 阴阳
        is_up = closes[-1] >= self.df.iloc[-1]["open"]

        # MA多头
        ma20 = np.mean(closes[-20:]) if n >= 20 else price
        ma50 = np.mean(closes[-50:]) if n >= 50 else price
        ma200 = np.mean(closes[-200:]) if n >= 200 else price
        ma_bull = ma20 > ma50 > ma200

        # 5日收益
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n >= 6 else 0

        # RSI
        rsi = self._calc_rsi(closes)

        consensus = 0
        signals = []

        # 1. 量价互证：放量+阳线=买盘真实，放量+阴线=卖压真实
        if vol_ratio > 1.2 and is_up:
            consensus += 1
            signals.append("量价齐升")
        elif vol_ratio > 1.2 and not is_up:
            consensus -= 1
            signals.append("放量下跌")

        # 2. 趋势互证：MA多头+趋势向上 vs MA空头+趋势向下
        if ma_bull and ret_5d > 0:
            consensus += 1
            signals.append("多头趋势")
        elif not ma_bull and ret_5d < 0:
            consensus -= 1
            signals.append("空头趋势")

        # 3. RSI方向：高RSI+涨=强势延续，低RSI+跌=弱势延续
        if rsi > 60 and ret_5d > 0:
            consensus += 1
            signals.append("RSI强势")
        elif rsi < 40 and ret_5d < 0:
            consensus -= 1
            signals.append("RSI弱势")

        # consensus ∈ [-3, 3], 映射到 [-15, 15]
        bonus = consensus * 5

        if consensus >= 2:
            tag = "强互证看涨"
        elif consensus >= 1:
            tag = "偏多互证"
        elif consensus <= -2:
            tag = "强互证看跌"
        elif consensus <= -1:
            tag = "偏空互证"
        else:
            tag = "信号中性"

        return bonus, {"consensus": consensus, "tag": tag, "signals": signals}

    # ─────────────────────── 综合评分 ───────────────────────

    def compute(self):
        """运行所有因子评分，返回综合结果"""
        factors = {}

        # 逐个计算，每个失败都降级为50分
        for key, method in [
            ("tech_strength", self.score_tech_strength),
            ("risk_reward", self.score_risk_reward),
            ("volume", self.score_volume),
            ("candlestick", self.score_candlestick),
            ("sector", self.score_sector),
            ("relative_strength", self.score_relative_strength),
            ("volatility", self.score_volatility),
        ]:
            try:
                factors[key] = method()
            except Exception as e:
                factors[key] = {"score": 50, "label": "中", "detail": f"计算异常: {e}"}

        # 大盘评分（不参与加权，只给仓位矩阵用）
        try:
            factors["market"] = self.score_market()
        except Exception:
            factors["market"] = {"score": 50, "label": "中", "detail": "计算异常"}

        # 加权综合（动态权重优先）
        composite = 0.0
        breakdown_lines = []
        active_weights = get_weights() if _DYNAMIC_WEIGHTS_OK else WEIGHTS

        for key, weight in active_weights.items():
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

        # 互证因子调节（多信号一致性加分/减分）
        mf_bonus, mf_info = self._calc_mutual_confirmation_bonus()
        if mf_bonus:
            composite = max(0, min(100, composite + mf_bonus))
            breakdown_lines.append({
                "key": "mutual_confirmation", "weight": 0, "score": mf_bonus,
                "contribution": mf_bonus, "label": mf_info.get("tag", ""),
                "detail": " | ".join(mf_info.get("signals", [])),
                "weight_pct": 0,
            })

        # 校准纠偏（基于历史评分准确率自动修正系统性偏差）
        calib_adjust, calib_note = _apply_calibration(composite)
        if calib_note:
            if calib_adjust != composite:
                composite = round(calib_adjust, 1)
            breakdown_lines.append({
                "key": "calibration", "weight": 0, "score": 0,
                "contribution": 0, "label": "校准",
                "detail": calib_note, "weight_pct": 0,
            })

        # 深度研究个股加分（可选）
        research_bonus = 0
        if _RESEARCH_TRACKER_OK and self.symbol:
            try:
                research_bonus = ResearchTracker.get_score_bonus(self.symbol)
                if research_bonus:
                    bonus_factor = next((f for f in breakdown_lines if f["key"] == "wyckoff"), None)
                    if bonus_factor:
                        composite += research_bonus
                        breakdown_lines.append({
                            "key": "research_bonus", "weight": 0, "score": research_bonus,
                            "contribution": research_bonus, "label": "研",
                            "detail": f"深度研究确认 +{research_bonus}",
                            "weight_pct": 0,
                        })
            except Exception:
                pass

        action, _legacy_factor = score_to_level(composite)

        # ── Kelly 仓位计算 (优先) ──
        pos_factor = 0.0
        kelly_detail = ""
        if _KELLY_OK:
            try:
                market_score = factors.get("market", {}).get("score", 50)
                sector_score = factors.get("sector", {}).get("heat_score",
                                  factors.get("sector", {}).get("score", 50))
                kelly_result = compute_kelly_position(
                    composite, breakdown_lines, market_score, sector_score
                )
                pos_factor = kelly_result["position_factor"]
                kelly_detail = kelly_result["detail"]
            except Exception:
                # Kelly 失败回退到旧映射
                _, pos_factor = score_to_level(composite)

        # 仓位决策矩阵（最终约束层）
        if _POSITION_MATRIX_OK:
            try:
                market_score = factors.get("market", {}).get("score", 50)
                sector_score = factors.get("sector", {}).get("heat_score",
                                  factors.get("sector", {}).get("score", 50))
                matrix_factor = get_position_factor(market_score, sector_score)
                pos_factor = min(pos_factor, matrix_factor)
                pos_factor = max(0.0, min(1.0, pos_factor))
            except Exception:
                pass

        # 多因子多空辩论（修正仓位）
        if _DEBATE_OK:
            try:
                debate_result = debate_factors(breakdown_lines, composite)
                if debate_result.get("veto_triggered"):
                    pos_factor = min(pos_factor, 0.3)
                elif debate_result.get("verdict") == "争议":
                    pos_factor = min(pos_factor, 0.5)
            except Exception:
                pass

        # sector 约束：板块强度弱时降仓位（sector 已从评分权重移除）
        sector_score = factors.get("sector", {}).get("score", 50)
        if sector_score < 30:
            pos_factor *= 0.5
        elif sector_score < 40:
            pos_factor *= 0.75

        return {
            "composite_score": composite,
            "action": action,
            "pos_factor": round(pos_factor, 4),
            "factors": breakdown_lines,
            "level": self._label(composite),
            "kelly_detail": kelly_detail,
            "stop_price": self.stop_price,
            "stop_pct": round((self.price - self.stop_price) / self.price * 100, 1),
        }
