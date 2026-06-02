"""
买入前交易计划检查脚本
每次 scanner 出票后，对每一只有信号的股票自动输出检查清单

数据源: 腾讯财经 HTTP API（与 scanner.py 一致，免费、免token、免登录）
"""

import sys
import os
import json
import argparse
import requests
import numpy as np

# Windows 终端 UTF-8 输出支持
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,60,qfq"


# ============================================================
# 数据获取
# ============================================================

def _code_to_tencent(code: str) -> str:
    """将股票代码转为腾讯API格式: sh600001 / sz000001 / sz300001"""
    code = code.strip().zfill(6)
    if code.startswith("6"):
        return f"sh{code}"
    elif code.startswith("0") or code.startswith("3"):
        return f"sz{code}"
    elif code.startswith("8") or code.startswith("4"):
        return f"bj{code}"
    return code


def fetch_kline_tencent(code: str, days: int = 60):
    """从腾讯财经获取日K线

    Returns:
        (dates, opens, closes, highs, lows, volumes) 各为 list[float]
        或 (None, None) 获取失败
    """
    key = _code_to_tencent(code)
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            return None, None

        klines = (data.get("data", {}).get(key, {}).get("qfqday") or
                  data.get("data", {}).get(key, {}).get("day") or [])

        if not klines:
            return None, None

        dates = []
        opens = []
        closes = []
        highs = []
        lows = []
        volumes = []

        for k in klines:
            if len(k) < 6:
                continue
            try:
                dates.append(str(k[0]))
                opens.append(float(k[1]))
                closes.append(float(k[2]))
                highs.append(float(k[3]))
                lows.append(float(k[4]))
                volumes.append(float(k[5]))
            except (ValueError, IndexError):
                continue

        return (dates, opens, closes, highs, lows, volumes)
    except Exception:
        return None, None


def calc_atr(highs, lows, closes, period=14):
    """计算 ATR(period)"""
    arr_h = np.array(highs, dtype=float)
    arr_l = np.array(lows, dtype=float)
    arr_c = np.array(closes, dtype=float)
    n = len(arr_h)
    if n < period + 1:
        return 0.0
    trs = []
    for i in range(1, n):
        hl = arr_h[i] - arr_l[i]
        hc = abs(arr_h[i] - arr_c[i - 1])
        lc = abs(arr_l[i] - arr_c[i - 1])
        trs.append(max(hl, hc, lc))
    return float(np.mean(trs[-period:]))


def calc_ma(closes, period):
    """计算移动平均线的最新值"""
    arr = np.array(closes, dtype=float)
    if len(arr) < period:
        return float(arr[-1]) if len(arr) > 0 else 0.0
    return float(np.mean(arr[-period:]))


def _fmt_pct(val):
    """格式化百分比显示, e.g. -9.4%"""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


# ============================================================
# 信号质量映射（简化版，与 scanner 核心逻辑对齐）
# ============================================================

# 信号评分到中文描述的映射
SIGNAL_NAMES = {
    "Spring": "Spring (弹簧)",
    "SOS": "SOS (强势信号)",
    "LPS": "LPS (最后支撑点)",
    "LPS5": "LPS5 (缩量LPS)",
    "UT": "UT (上冲回落)",
    "UTAD": "UTAD (上冲回落吸筹尾声)",
    "BU": "BU (突破上涨)",
    "BC": "BC (初次供应)",
    "SC": "SC (恐慌抛售)",
    "AR": "AR (自动反弹)",
    "ST": "ST (二次测试)",
    "LPS2": "LPS2 (二次LPS)",
}

# 威科夫阶段中英文映射
PHASE_NAMES = {
    "Accum_A": "吸筹初期 (Accum A)",
    "Accum_B": "吸筹中期 (Accum B)",
    "Accum_C": "吸筹末期 (Accum C)",
    "Markup": "上涨阶段 (Markup)",
    "Distribute_A": "派发初期 (Distribute A)",
    "Distribute_B": "派发末期 (Distribute B)",
    "Markdown": "下跌阶段 (Markdown)",
    "Range": "震荡区间 (Range)",
}

WYCKOFF_PASS_SIGNALS = {"SOS", "LPS", "LPS5", "BU", "Spring", "LPS2", "BC"}

# ── 2D 门控矩阵: signal_ratio × wyckoff_phase ──
# 每个矩阵元: (允许买入?, 备注)
GATE_MATRIX = {
    #               Red(<1.0)          Yellow(1.0-2.0)      Green(>=2.0)
    "Accum_A":   [(False, "不开新仓"),  (True, "轻仓试探"),  (True, "正常执行")],
    "Accum_B":   [(False, "不开新仓"),  (True, "轻仓试探"),  (True, "正常执行")],
    "Accum_C":   [(True,  "准备"),      (True, "正常执行"),  (True, "正常执行")],
    "Markup":    [(True,  "仅减仓"),    (True, "正常执行"),  (True, "正常执行")],
    "Dist_A":    [(False, "不开新仓"),  (False, "不开新仓"), (True,  "仅减仓")],
    "Dist_B":    [(False, "不开新仓"),  (False, "不开新仓"), (False, "不开新仓")],
    "Markdown":  [(False, "不开新仓"),  (False, "不开新仓"), (False, "不开新仓")],
    "Range":     [(False, "不开新仓"),  (True, "精选R/R>=3"),(True, "正常执行")],
}

# ── CAN SLIM M 状态 ──
M_STATUS = {
    "Accum_A":   "修正中",
    "Accum_B":   "筑底中",
    "Accum_C":   "转为上升",
    "Markup":    "确认上升",
    "Distribute_A": "上升受压",
    "Distribute_B": "进入修正",
    "Markdown":  "下跌趋势",
    "Range":     "方向不明",
}

M_WEIGHT = {
    "确认上升": 1.0,
    "转为上升": 0.8,
    "上升受压": 0.5,
    "筑底中":   0.5,
    "修正中":   0.3,
    "进入修正": 0.2,
    "下跌趋势": 0.0,
    "方向不明": 0.5,
}


# ============================================================
# 主检查类
# ============================================================

class PreBuyChecklist:
    """买入前交易计划检查清单

    集成:
      - 2D 门控（信号比×Wyckoff阶段）
      - CAN SLIM M 大盘方向判断
      - 威科夫信号确认
      - 3 种止损方法 (结构/ATR/硬止损)
      - 盈亏比检查（阶段依赖）
      - ATR 仓位管理
    """

    def __init__(self, account_size=100000, risk_per_trade=0.02):
        self.account_size = max(account_size, 1)
        self.risk_per_trade = risk_per_trade
        self.max_loss = self.account_size * risk_per_trade

    def run(self, code, name, price, signal, score, phase="",
            buy_count=0, sell_count=0, wyckoff_phase=None,
            closes=None, highs=None, lows=None):
        """全流程检查，返回结构化结果 dict

        参数:
            code: 股票代码
            name: 股票名称
            price: 入场价
            signal: 威科夫信号
            score: 信号评分 (0-100)
            phase: 个股威科夫阶段
            buy_count: 当日全市场买入信号数
            sell_count: 当日全市场卖出信号数
            wyckoff_phase: 大盘威科夫阶段 (market_regime 输出)
            closes/highs/lows: K线价格序列
        """
        steps = {}
        all_pass = True
        reasons = []

        # ─── 第一步: 2D 门控 (信号比 × 大盘阶段) ───
        step1 = self._step_market_gate(buy_count, sell_count, wyckoff_phase)
        steps["step1_market_gate"] = step1

        terminated = step1.get("terminated", False)

        if terminated:
            all_pass = False
            reasons.append(step1.get("gate_reason", "不开新仓"))
            steps["step2_wyckoff"] = {"name": "Wyckoff 信号确认", "skip": True}
            steps["step3_stop_loss"] = {"name": "止损位计算", "skip": True}
            steps["step4_risk_reward"] = {"name": "盈亏比检查", "skip": True}
            steps["step5_position"] = {"name": "仓位建议", "skip": True}
            return self._build_result(code, name, price, signal, score, phase,
                                      steps, all_pass, reasons,
                                      terminated=terminated)

        # ─── 第二步: 威科夫信号确认 ───
        step2 = self._step_wyckoff_signal(signal, score, phase, step1.get("gate", "绿灯"))
        steps["step2_wyckoff"] = step2
        if not step2["pass"]:
            all_pass = False
            reasons.append(step2.get("reason", "威科夫信号不达标"))

        # ─── 第三步: 止损位计算 ───
        step3 = self._step_stop_loss(price, closes, highs, lows)
        steps["step3_stop_loss"] = step3
        if not step3["pass"]:
            all_pass = False
            reasons.append("止损位计算异常")

        # ─── 第四步: 盈亏比检查 ───
        target = self._calc_target(price, closes, highs, lows, phase)
        gate_mode = step1.get("gate", "绿灯")
        # 如果止损未通过，用 price*0.9 作为兜底止损
        fallback_stop = step3.get("recommended_stop", price * 0.9)
        step4 = self._step_risk_reward(price, fallback_stop, target, gate_mode, phase)
        steps["step4_risk_reward"] = step4
        if not step4["pass"]:
            all_pass = False
            reasons.append(step4.get("reason", "盈亏比不足"))

        # ─── 第五步: ATR 仓位管理 ───
        atr_val = calc_atr(highs or [], lows or [], closes or [], period=14)
        gate_label = step1.get("gate", "绿灯")
        m_weight = step1.get("m_weight", 1.0)
        fallback_stop_pos = step3.get("recommended_stop", price * 0.9)
        step5 = self._step_position_sizing(price, fallback_stop_pos, all_pass,
                                           atr_value=atr_val, gate=gate_label,
                                           m_weight=m_weight)
        steps["step5_position"] = step5

        # ─── 最终判定 ───
        if not all_pass:
            final_verdict = "不买入"
            final_reason = " + ".join(reasons) if reasons else "检查未通过"
        else:
            final_verdict = "建议买入"
            final_reason = f"五项检查全部通过, 建议仓位 {step5['suggested_shares']}股"

        return self._build_result(code, name, price, signal, score, phase,
                                  steps, all_pass, reasons,
                                  final_verdict, final_reason,
                                  step3, step4, step5, terminated=False)

    # ─────────────── 各步骤 ───────────────

    def _step_market_gate(self, buy_count, sell_count, wyckoff_phase=None):
        """第一步: 2D 门控 — 信号比 × 大盘威科夫阶段"""
        signal_ratio = buy_count / sell_count if sell_count > 0 else 999

        # 信号比维度
        if signal_ratio >= 2.0:
            ratio_band = 2  # 绿灯
            ratio_label = "绿灯"
        elif signal_ratio >= 1.0:
            ratio_band = 1  # 黄灯
            ratio_label = "黄灯"
        else:
            ratio_band = 0  # 红灯
            ratio_label = "红灯"

        # CAN SLIM M 状态
        m_status = M_STATUS.get(wyckoff_phase, "方向不明")
        m_weight = M_WEIGHT.get(m_status, 0.5)

        # 查 2D 矩阵
        matrix_row = GATE_MATRIX.get(wyckoff_phase)
        if matrix_row:
            can_buy, action_note = matrix_row[ratio_band]
        else:
            # 无阶段信息 → 退回到纯信号比判断
            can_buy = ratio_band >= 1
            action_note = {"绿灯": "正常执行", "黄灯": "精选", "红灯": "不开新仓"}[ratio_label]

        terminated = not can_buy

        # 构建理由
        phase_str = wyckoff_phase or "N/A"
        gate_reason = (f"信号比 {signal_ratio:.2f}({ratio_label}) × "
                       f"大盘阶段 {phase_str}({m_status}) → {action_note}")

        return {
            "name": "市场情绪门禁 (2D)",
            "pass": can_buy,
            "terminated": terminated,
            "signal_ratio": round(signal_ratio, 2),
            "gate": ratio_label,
            "wyckoff_phase": wyckoff_phase,
            "m_status": m_status,
            "m_weight": round(m_weight, 2),
            "action_note": action_note,
            "gate_reason": gate_reason,
        }

    def _step_wyckoff_signal(self, signal, score, phase, gate):
        """第二步: 威科夫信号确认"""
        signal_ok = signal.upper() in {s.upper() for s in WYCKOFF_PASS_SIGNALS}
        score_ok = score >= 60 if score is not None else False

        # 黄灯模式下提高信号门槛
        if gate == "黄灯":
            score_ok = score >= 75 if score is not None else False

        phase_label = PHASE_NAMES.get(phase, phase)
        signal_label = SIGNAL_NAMES.get(signal, signal)

        if not signal_ok:
            return {
                "name": "威科夫信号确认",
                "pass": False,
                "signal": signal_label,
                "score": score,
                "phase": phase_label,
                "reason": f"信号 {signal} 不在买入信号列表内",
            }
        if not score_ok:
            return {
                "name": "威科夫信号确认",
                "pass": False,
                "signal": signal_label,
                "score": score,
                "phase": phase_label,
                "reason": f"信号评分 {score} 不足{'75' if gate == '黄灯' else '60'}",
            }
        return {
            "name": "威科夫信号确认",
            "pass": True,
            "signal": signal_label,
            "score": score,
            "phase": phase_label,
        }

    def _step_stop_loss(self, price, closes, highs, lows, phase=""):
        """第三步: 止损位计算（四种方法）

        A: 结构止损（最近20日最低价）
        B: ATR 止损 (2×ATR)
        C: 硬止损 (CAN SLIM 7-8%)
        D: 均线止损 (MA20)

        推荐: 取最紧止损（保护本金），但不超过 10%。
        阶段依赖: 吸筹阶段用 A/B（宽松），拉升阶段用 C/D（收紧）。
        """
        if closes is None or highs is None or lows is None:
            return {
                "name": "止损位计算",
                "pass": False,
                "reason": "缺少K线数据",
            }
        closes_a = np.array(closes, dtype=float)
        highs_a = np.array(highs, dtype=float)
        lows_a = np.array(lows, dtype=float)

        # A: 结构止损 (最近20日最低价)
        recent_low = float(np.min(lows_a[-20:])) if len(lows_a) >= 20 else float(np.min(lows_a))
        stop_a = recent_low

        # B: ATR 止损
        atr_val = calc_atr(highs, lows, closes, period=14)
        if atr_val > 0 and price > 0:
            stop_b = price - 2 * atr_val
        else:
            stop_b = price * 0.92

        # C: 硬止损 (CAN SLIM 7-8%, 用8%)
        hard_stop_pct = 0.08
        stop_c = price * (1 - hard_stop_pct)

        # D: 均线止损 (MA20)
        ma20 = calc_ma(closes, 20)
        stop_d = ma20

        # 阶段依赖的止损选择
        phase_stops = {
            "Accum_A": max,       # 宽止损
            "Accum_B": max,
            "Accum_C": max,
            "Markup": min,        # 紧止损
            "Distribute_A": min,
            "Distribute_B": min,
            "Markdown": max,
            "Range": max,
        }
        stop_selector = phase_stops.get(phase, max)

        # 推荐止损: 根据阶段选择宽松或收紧
        recommended_stop = stop_selector(stop_a, stop_b, stop_d)

        # 硬止损作为最终防线: 推荐止损不能比硬止损宽太多
        if recommended_stop < stop_c:
            recommended_stop = stop_c

        # 确保止损低于入场价
        if recommended_stop >= price:
            recommended_stop = min(stop_a, stop_b, stop_c)
            if recommended_stop >= price:
                recommended_stop = stop_c

        loss_pct = (price - recommended_stop) / price * 100

        return {
            "name": "止损位计算",
            "pass": True,
            "stop_a_structure": round(stop_a, 2),
            "stop_b_atr": round(stop_b, 2),
            "stop_c_hard": round(stop_c, 2),
            "stop_d_ma": round(stop_d, 2),
            "atr_value": round(atr_val, 4),
            "recommended_stop": round(recommended_stop, 2),
            "loss_pct": round(abs(loss_pct), 1),
            "method": "结构" if recommended_stop == stop_a else
                      "ATR" if recommended_stop == stop_b else
                      "硬止损" if recommended_stop == stop_c else
                      "均线",
        }

    def _calc_target(self, price, closes, highs, lows, phase=""):
        """计算目标位（阶段依赖）

        方法:
          1. 前高（最近30日最高价）
          2. ATR 扩展目标 (入场 + 3×ATR)
          3. CAN SLIM 利润目标 (20-25%)

        阶段依赖:
          Accum: 取前高 (突破区间)
          Markup: 取 ATR 或 CAN SLIM (趋势延续)
          Dist: 不设目标 (尽快离场)
        """
        if closes is None or highs is None or lows is None:
            return round(price * 1.15, 2)

        highs_a = np.array(highs, dtype=float)

        # 方法1: 前高
        if len(highs_a) >= 30:
            recent_high = float(np.max(highs_a[-30:]))
        else:
            recent_high = float(np.max(highs_a))

        # 方法2: ATR 目标
        atr_val = calc_atr(highs, lows, closes, period=14)
        atr_target = price + 3 * atr_val if atr_val > 0 else price * 1.15

        # 方法3: CAN SLIM 20-25%
        can_slim_target1 = price * 1.20  # 卖1/3
        can_slim_target2 = price * 1.25  # 卖1/3

        # 阶段依赖
        if phase in ("Accum_A", "Accum_B", "Accum_C"):
            # 吸筹: 突破前高是第一目标
            target = max(recent_high, atr_target)
        elif phase in ("Markup",):
            # 拉升: ATR 或 CAN SLIM
            target = max(atr_target, can_slim_target2)
        elif phase in ("Distribute_A", "Distribute_B", "Markdown"):
            # 派发/下跌: 短目标
            target = min(recent_high, can_slim_target1)
        else:
            target = max(recent_high, atr_target)

        return round(target, 2)

    def _step_risk_reward(self, price, stop_loss, target, gate, phase=""):
        """第四步: 盈亏比检查（阶段依赖）"""
        if stop_loss is None or target is None or price is None:
            return {
                "name": "盈亏比检查",
                "pass": False,
                "reason": "缺少价格数据",
            }

        risk = price - stop_loss
        reward = target - price

        if risk <= 0:
            return {
                "name": "盈亏比检查",
                "pass": False,
                "reason": "止损价高于入场价",
            }

        rr_ratio = reward / risk if risk > 0 else 0

        # 阶段依赖的盈亏比门槛
        phase_min_rr = {
            "Accum_A": 3.0,   # 吸筹初期不确定性大
            "Accum_B": 3.0,
            "Accum_C": 2.5,   # 吸筹末期可以放宽
            "Markup": 2.0,    # 趋势中正常 2:1
            "Distribute_A": 4.0,  # 派发期要求极高
            "Distribute_B": 5.0,
            "Markdown": 5.0,
            "Range": 3.0,
        }
        phase_min = phase_min_rr.get(phase, 2.5)

        # 灯色门槛 (绿灯 2:1, 黄灯 3:1)
        gate_min = {"绿灯": 2.0, "黄灯": 3.0, "红灯": 3.0}.get(gate, 2.0)

        min_rr = max(phase_min, gate_min)
        rr_pass = rr_ratio >= min_rr

        # CAN SLIM 建议: 累积 20-25% 利润时分批卖
        sell_zones = []
        if price > 0:
            sell_zones.append(f"卖1/3@{price*1.20:.2f}(+20%)")
            sell_zones.append(f"卖1/3@{price*1.25:.2f}(+25%)")
            sell_zones.append("余下移动止盈")

        return {
            "name": "盈亏比检查",
            "pass": rr_pass,
            "target": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "potential_gain": round(reward, 2),
            "potential_gain_pct": round(reward / price * 100, 1),
            "potential_loss": round(risk, 2),
            "potential_loss_pct": round(risk / price * 100, 1),
            "rr_ratio": round(rr_ratio, 2),
            "min_rr": min_rr,
            "phase_min_rr": phase_min,
            "sell_zones": sell_zones,
            "reason": f"盈亏比 {rr_ratio:.2f}:1 < {min_rr}:1" if not rr_pass else None,
        }

    def _step_position_sizing(self, price, stop_loss, all_previous_pass,
                              atr_value=None, gate="绿灯", m_weight=1.0):
        """第五步: ATR仓位管理

        综合三种方法:
          1. ATR 波动率仓位 (海龟公式): shares = (账户×风险%) / (2×ATR)
          2. 固定比例仓位: shares = (账户×风险%) / (price - stop)
          3. CAN SLIM M 折扣 × 阶段折扣

        最终取三者中较低值，保证风险可控。
        """
        if not all_previous_pass or stop_loss is None or price is None:
            return {
                "name": "仓位建议 (ATR)",
                "pass": False,
                "reason": "不满足入场条件，不计算仓位",
                "suggested_shares": 0,
                "suggested_amount": 0.0,
            }

        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return {
                "name": "仓位建议 (ATR)",
                "pass": False,
                "reason": "止损价不低于入场价，无法计算仓位",
                "suggested_shares": 0,
                "suggested_amount": 0.0,
            }

        max_risk_amount = self.max_loss

        # ── 方法1: 等风险仓位（原方法） ──
        shares_risk = int(max_risk_amount / risk_per_share)

        # ── 方法2: ATR 波动率仓位（海龟） ──
        if atr_value and atr_value > 0:
            # 海龟: 1单位 = 账户1% / (2×ATR)
            # A股没有期货乘数，直接用: base = 账户风险额 / (2×ATR)
            base_units = max_risk_amount / (2 * atr_value)
            shares_atr = int(base_units / price * 100) * 100 if price > 0 else 99999
        else:
            shares_atr = 99999  # ATR 不可用时回退

        # ── 取两种方法中较低值 ──
        suggested_shares = min(shares_risk, shares_atr)

        # ── CAN SLIM M 折扣 ──
        suggested_shares = int(suggested_shares * m_weight)

        # ── 阶段折扣 ──
        # 按整手取整（A股100股/手）
        suggested_shares = max(0, (suggested_shares // 100) * 100)
        suggested_amount = round(suggested_shares * price, 2)

        # 校验: 不超过账户总资金
        if suggested_amount > self.account_size:
            suggested_shares = int(self.account_size / price // 100 * 100)
            suggested_amount = round(suggested_shares * price, 2)

        return {
            "name": "仓位建议 (ATR)",
            "pass": True,
            "max_loss": round(self.max_loss, 2),
            "risk_per_share": round(risk_per_share, 3),
            "method": "ATR海龟" if atr_value and atr_value > 0 else "等风险",
            "shares_by_risk": shares_risk,
            "shares_by_atr": int(shares_atr) if shares_atr < 99999 else None,
            "m_weight": round(m_weight, 2),
            "suggested_shares": suggested_shares,
            "suggested_amount": suggested_amount,
        }

    def _build_result(self, code, name, price, signal, score, phase,
                      steps, all_pass, reasons,
                      final_verdict=None, final_reason=None,
                      stop_data=None, rr_data=None, pos_data=None,
                      terminated=False):
        """组装最终结果"""
        if final_verdict is None:
            final_verdict = "不买入" if not all_pass else "建议买入"
        if final_reason is None:
            final_reason = " + ".join(reasons) if reasons else "检查未通过"

        return {
            "code": code,
            "name": name,
            "price": price,
            "signal": signal,
            "score": score,
            "phase": phase,
            "steps": steps,
            "all_pass": all_pass,
            "terminated": terminated,
            "final_verdict": final_verdict,
            "final_reason": final_reason,
        }

    # ─────────────── 报告输出 ───────────────

    def print_report(self, result):
        """打印格式化检查报告"""
        code = result["code"]
        name = result["name"]
        price = result["price"]
        signal = result["signal"]
        score = result.get("score")
        phase = result.get("phase", "")
        terminated = result.get("terminated", False)

        steps = result["steps"]

        # 步骤数据提取
        step1 = steps.get("step1_market_gate", {})
        step2 = steps.get("step2_wyckoff", {})
        step3 = steps.get("step3_stop_loss", {})
        step4 = steps.get("step4_risk_reward", {})
        step5 = steps.get("step5_position", {})

        sep = "=" * 48
        line = chr(9472) * 48

        print()
        print(sep)
        print("  买入前检查清单")
        print(sep)
        print()
        print(f"  股票: {name} ({code})  价格: {price:.2f}")
        print(line)

        # ─── 第一步: 2D 门控 ───
        s1_pass = "OK" if step1.get("pass") else ("STOP" if not step1.get("skip") else "SKIP")
        s1_ratio = step1.get("signal_ratio", "N/A")
        s1_gate = step1.get("gate", "N/A")
        s1_phase = step1.get("wyckoff_phase", "N/A")
        s1_m = step1.get("m_status", "N/A")
        s1_note = step1.get("action_note", "")

        print(f"  [{s1_pass}] 第一步: 2D 门控")
        print(f"      信号比         : {s1_ratio}  <- {s1_gate}")
        print(f"      大盘威科夫阶段 : {s1_phase}  <- CAN SLIM M: {s1_m}")
        print(f"      门控动作       : {s1_note}")
        if step1.get("pass"):
            print(f"      判定           : v 2D门控通过")
        else:
            print(f"      判定           : x 不开新仓")
            print(f"      >> 终止")

        # ─── 第二步: 威科夫信号确认 ───
        if step2.get("skip"):
            print(f"  [SKIP] 第二步: Wyckoff 信号确认 (因第一步终止跳过)")
        else:
            s2_pass = "OK" if step2.get("pass") else "STOP"
            s2_signal = step2.get("signal", signal)
            s2_score = step2.get("score", score)
            s2_phase = step2.get("phase", phase)

            print(f"  [{s2_pass}] 第二步: Wyckoff 信号确认")
            print(f"      信号           : {s2_signal}")
            if s2_score is not None:
                print(f"      评分           : {s2_score}分")
            if s2_phase:
                print(f"      阶段           : {s2_phase}")
            if step2.get("pass"):
                print(f"      判定           : v 买入信号有效")
            else:
                reason = step2.get("reason", "信号不达标")
                print(f"      判定           : x {reason}")

        # ─── 第三步: 止损位计算 ───
        if step3.get("skip"):
            print(f"  [SKIP] 第三步: 止损位计算 (因第一步终止跳过)")
        else:
            s3_pass = "OK" if step3.get("pass") else "STOP"
            print(f"  [{s3_pass}] 第三步: 止损位计算")
            if step3.get("pass"):
                stop_a = step3.get("stop_a_structure", 0)
                stop_b = step3.get("stop_b_atr", 0)
                stop_c = step3.get("stop_c_hard", 0)
                stop_d = step3.get("stop_d_ma", 0)
                method = step3.get("method", "")
                loss_pct = step3.get("loss_pct", 0)
                rec_stop = step3.get("recommended_stop", 0)

                if stop_a:
                    a_pct = (stop_a / price - 1) * 100 if price > 0 else 0
                    print(f"      A 结构止损     : {stop_a:.2f} ({_fmt_pct(a_pct)})")
                if stop_b:
                    b_pct = (stop_b / price - 1) * 100 if price > 0 else 0
                    atr_val = step3.get("atr_value", 0)
                    print(f"      B ATR止损      : {stop_b:.2f} ({_fmt_pct(b_pct)})  [ATR={atr_val:.3f}]")
                if stop_c:
                    c_pct = (stop_c / price - 1) * 100 if price > 0 else 0
                    print(f"      C 硬止损(CANSLIM): {stop_c:.2f} ({_fmt_pct(c_pct)})")
                if stop_d:
                    d_pct = (stop_d / price - 1) * 100 if price > 0 else 0
                    print(f"      D 均线止损     : {stop_d:.2f} ({_fmt_pct(d_pct)})")

                print(f"      推荐止损       : {rec_stop:.2f} (亏损 {loss_pct}%)  [{method}]")
            else:
                print(f"      判定           : x {step3.get('reason', '计算失败')}")

        # ─── 第四步: 盈亏比检查 ───
        if step4.get("skip"):
            print(f"  [SKIP] 第四步: 盈亏比检查 (因第一步终止跳过)")
        else:
            s4_pass = "OK" if step4.get("pass") else "STOP"
            print(f"  [{s4_pass}] 第四步: 盈亏比检查")
            if step4.get("pass") is not None:
                target = step4.get("target", 0)
                stop = step4.get("stop_loss", 0)
                gain_pct = step4.get("potential_gain_pct", 0)
                loss_pct = step4.get("potential_loss_pct", 0)
                rr = step4.get("rr_ratio", 0)
                min_rr = step4.get("min_rr", 2)

                print(f"      目标位         : {target:.2f}")
                print(f"      止损位         : {stop:.2f}")
                print(f"      潜在盈利       : {step4.get('potential_gain', 0):.2f} ({_fmt_pct(gain_pct)})")
                print(f"      潜在亏损       : {step4.get('potential_loss', 0):.2f} ({_fmt_pct(-loss_pct)})")
                print(f"      盈亏比         : {rr:.2f}:1  (门槛 {min_rr}:1)")

                sell_zones = step4.get("sell_zones", [])
                if sell_zones:
                    print(f"      CAN SLIM 分批卖: {', '.join(sell_zones)}")

                if step4.get("pass"):
                    print(f"      判定           : v 盈亏比达标")
                else:
                    print(f"      判定           : x 盈亏比 {rr:.2f}:1 < {min_rr}:1，不符合条件")
            else:
                print(f"      判定           : x {step4.get('reason', '数据不足')}")

        # ─── 第五步: 仓位建议 ───
        if step5.get("skip"):
            print(f"  [SKIP] 第五步: 仓位建议 (因第一步终止跳过)")
        else:
            s5_pass = "OK" if step5.get("pass") else "STOP"
            print(f"  [{s5_pass}] 第五步: ATR 仓位管理")
            if step5.get("pass"):
                max_loss = step5.get("max_loss", 0)
                shares = step5.get("suggested_shares", 0)
                amount = step5.get("suggested_amount", 0)
                risk_per = step5.get("risk_per_share", 0)
                method = step5.get("method", "等风险")
                m_weight = step5.get("m_weight", 1.0)
                print(f"      账户资金       : {self.account_size:,.0f}")
                print(f"      单笔风险上限   : {max_loss:,.0f} ({self.risk_per_trade*100:.0f}%)")
                print(f"      每股风险       : {risk_per:.3f}")
                print(f"      仓位方法       : {method}  (M权重={m_weight})")
                print(f"      建议仓位       : {shares}股 ({amount:,.0f})")
            else:
                print(f"      判定           : x {step5.get('reason', '不计算仓位')}")
                print(f"      账户资金       : {self.account_size:,.0f}")
                print(f"      单笔风险上限   : {self.max_loss:,.0f} ({self.risk_per_trade*100:.0f}%)")
                print(f"      建议仓位       : 不满足入场条件，不计算仓位")

        # ─── 最终判定 ───
        print(line)
        sym = "v" if result["all_pass"] else "x"
        print(f"  最终判定: {sym} {result['final_verdict']}")
        print(f"  原因: {result['final_reason']}")
        print(sep)
        print()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="买入前交易计划检查清单",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pre_buy_checklist.py --code 600011 --name 华能国际 --price 9.60
      --signal SOS --score 85 --phase Markup
      --buy-count 80 --sell-count 60
      --account 100000 --risk 0.02
        """,
    )
    parser.add_argument("--code", required=True, help="股票代码 (如 600011)")
    parser.add_argument("--name", required=True, help="股票名称")
    parser.add_argument("--price", type=float, required=True, help="入场价")
    parser.add_argument("--signal", default="SOS", help="信号名 (SOS/Spring/LPS等)")
    parser.add_argument("--score", type=int, default=80, help="信号评分 (0-100)")
    parser.add_argument("--phase", default="", help="个股威科夫阶段")
    parser.add_argument("--market-phase", default="",
                        help="大盘威科夫阶段 (wyckoff_phase, 从 market_regime 获取)")
    parser.add_argument("--buy-count", type=int, default=0,
                        help="当日全市场买入信号数 (从scanner获取)")
    parser.add_argument("--sell-count", type=int, default=0,
                        help="当日全市场卖出信号数")
    parser.add_argument("--account", type=float, default=100000,
                        help="账户总资金 (默认 100000)")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="单笔风险比率 (默认 0.02=2%%)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="不自动获取K线数据（仅用于测试）")

    args = parser.parse_args()

    checklist = PreBuyChecklist(
        account_size=args.account,
        risk_per_trade=args.risk,
    )

    closes = highs = lows = None

    if not args.no_fetch:
        # 获取K线数据
        result = fetch_kline_tencent(args.code, days=60)
        if result and result[0] is not None:
            _dates, _opens, closes, highs, lows, _volumes = result
        else:
            print(f"[警告] 无法获取 {args.code} K线数据，使用模拟数据")
            # 用简单模拟数据保证脚本能运行
            np.random.seed(42)
            base = args.price
            closes = list(base * (1 + np.random.randn(60) * 0.03))
            highs = [c * 1.02 for c in closes]
            lows = [c * 0.98 for c in closes]

    # 自动获取大盘 Wyckoff 阶段（如果没指定）
    market_phase = args.market_phase
    if not market_phase:
        try:
            from market_regime import get_regime
            regime_r = get_regime()
            market_phase = regime_r.get("wyckoff_phase", "")
            if market_phase:
                print(f"  [auto] 大盘 Wyckoff 阶段: {market_phase} ({regime_r.get('wyckoff_desc','')})")
        except Exception:
            pass

    result = checklist.run(
        code=args.code,
        name=args.name,
        price=args.price,
        signal=args.signal,
        score=args.score,
        phase=args.phase,
        wyckoff_phase=market_phase,
        buy_count=args.buy_count,
        sell_count=args.sell_count,
        closes=closes,
        highs=highs,
        lows=lows,
    )

    checklist.print_report(result)

    # JSON 输出到 stdout 方便管道
    if os.environ.get("PRE_BUY_JSON"):
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return result


if __name__ == "__main__":
    main()
