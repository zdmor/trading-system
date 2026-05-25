"""
多因子多空辩论模块

让每个因子"发言"表明多空立场，通过结构化辩论替代简单加权求和。
思路来自 TradingAgents 的多智能体架构，但纯规则驱动，无 LLM 调用。

辩论流程:
  1. 每个因子发表观点（多/空/中性）+ 论据
  2. 加权辩论（权重 × 信心度）
  3. 风险一票否决（强烈空头信号无视多方共识）
  4. 最终裁决 + 可选解释
"""
from typing import Optional

# 各因子角色设定
FACTOR_PERSONAS = {
    "tech_strength":     "趋势跟踪者",
    "risk_reward":       "风控官",
    "volume":            "量能分析师",
    "candlestick":       "形态学家",
    "sector":            "板块观察员",
    "relative_strength": "相对价值分析师",
    "fund_flow":         "资金侦探",
    "market":            "大盘裁判",
}

# 偏执型因子：天生偏向某一方（从数据角度看问题）
FACTOR_BIAS = {
    "tech_strength":     0.0,   # 中性，跟随趋势
    "risk_reward":      -0.1,   # 微偏空（风控天生谨慎）
    "volume":            0.0,   # 中性
    "candlestick":       0.0,   # 中性，取决于形态
    "sector":            0.0,   # 中性
    "relative_strength": 0.05,  # 微偏多（强者恒强）
    "fund_flow":         0.0,
    "market":            0.0,
}

# 各因子对特定"风险场景"的敏感度
RISK_SENSITIVITY = {
    "tech_strength":     {"趋势反转": 0.8, "超买": 0.6, "量价背离": 0.5},
    "risk_reward":       {"盈亏比差": 0.9, "止损空间大": 0.8, "波动过大": 0.7},
    "volume":            {"缩量上涨": 0.7, "放量下跌": 0.9, "量价背离": 0.8},
    "candlestick":       {"见顶形态": 0.8, "反转信号": 0.7, "持续形态": 0.3},
    "sector":            {"板块退潮": 0.8, "龙头见顶": 0.7, "政策风险": 0.6},
    "relative_strength": {"弱势": 0.7, "破位": 0.8},
}

# 阈值
STRONG_BULL = 70
MILD_BULL = 55
MILD_BEAR = 45
STRONG_BEAR = 30
VETO_THRESHOLD = 25       # 低于此值 → 一票否决
CONSENSUS_THRESHOLD = 0.65  # 超过此比例 → 一致
CONTROVERSY_THRESHOLD = 0.40  # 低于此比例且双方接近 → 争议


def score_to_stance(score: float, bias: float = 0) -> tuple:
    """分数 → (立场, 信心度, 标签)

    Args:
        score: 因子得分 0-100
        bias: 因子偏执 [-0.2, 0.2]

    Returns:
        (stance, confidence, label)
    """
    adjusted = score + bias * (score - 50)
    distance = abs(adjusted - 50)

    if adjusted >= STRONG_BULL:
        return "BULL", round(distance / 50, 2), "强烈看多"
    elif adjusted >= MILD_BULL:
        return "BULL", round(distance / 50, 2), "谨慎看多"
    elif adjusted >= MILD_BEAR:
        return "NEUTRAL", 0.0, "中性"
    elif adjusted >= STRONG_BEAR:
        return "BEAR", round(distance / 50, 2), "谨慎看空"
    else:
        return "BEAR", round(distance / 50, 2), "强烈看空"


def format_argument(key: str, score: float, stance: str, confidence: float,
                    label: str, detail: str, weight: float) -> dict:
    """格式化因子观点"""
    persona = FACTOR_PERSONAS.get(key, "分析师")

    # 根据 stance 生成发言
    if stance == "BULL":
        if confidence >= 0.4:
            opinion = f"强烈建议做多"
        else:
            opinion = f"倾向做多"
    elif stance == "BEAR":
        if confidence >= 0.4:
            opinion = f"强烈建议做空/回避"
        else:
            opinion = f"倾向做空/回避"
    else:
        opinion = f"建议观望"

    return {
        "key": key,
        "persona": persona,
        "stance": stance,
        "score": round(score, 1),
        "confidence": confidence,
        "label": label,
        "detail": detail,
        "opinion": opinion,
        "weight": weight,
        "power": round(weight * (confidence + 0.3), 3),  # 辩论权重
    }


class BullBearDebate:
    """多因子多空辩论器"""

    def __init__(self, factors: list, composite_score: float = None):
        """
        Args:
            factors: breakdown_lines from StockScorer.compute()
            composite_score: 可选，加权综合分（用于对照）
        """
        self.factors = [f for f in factors if f.get("key") in FACTOR_PERSONAS]
        self.composite_score = composite_score
        self.args = []  # 各因子论点
        self.verdict = None

    def debate(self) -> dict:
        """执行多空辩论，返回裁决"""
        self.args = []
        for f in self.factors:
            key = f.get("key", "")
            score = f.get("score", 50)
            weight = f.get("weight", 0)
            detail = f.get("detail", "")
            bias = FACTOR_BIAS.get(key, 0)

            stance, confidence, label = score_to_stance(score, bias)
            arg = format_argument(key, score, stance, confidence, label, detail, weight)
            self.args.append(arg)

        # ── 辩论统计 ──
        bull_power = 0
        bear_power = 0
        total_power = 0
        bull_args = []
        bear_args = []
        neutral_args = []
        vetos = []

        for arg in self.args:
            power = arg["power"]
            total_power += power

            if arg["stance"] == "BULL":
                bull_power += power
                bull_args.append(arg)
            elif arg["stance"] == "BEAR":
                bear_power += power
                bear_args.append(arg)
            else:
                neutral_args.append(arg)

            # 检查一票否决
            if arg["score"] <= VETO_THRESHOLD:
                vetos.append(arg)

        # ── 计算比例 ──
        bull_ratio = bull_power / total_power if total_power > 0 else 0
        bear_ratio = bear_power / total_power if total_power > 0 else 0
        neutral_ratio = 1 - bull_ratio - bear_ratio

        # ── 是否有一票否决 ──
        veto_triggered = len(vetos) > 0
        if veto_triggered:
            # 被否决时，除非多方压倒性优势（>85%），否则强制做空
            if bull_ratio < 0.85:
                verdict = "做空"
                confidence = max(0.5, bear_ratio + 0.2)
            else:
                # 多方压倒性，但记录风险警告
                verdict = "做多"
                confidence = bull_ratio * 0.85
        else:
            # 正常裁决
            if bull_ratio >= CONSENSUS_THRESHOLD:
                verdict = "做多"
                confidence = bull_ratio
            elif bear_ratio >= CONSENSUS_THRESHOLD:
                verdict = "做空"
                confidence = bear_ratio
            elif bull_ratio > bear_ratio + 0.1:
                verdict = "偏多观望"
                confidence = bull_ratio
            elif bear_ratio > bull_ratio + 0.1:
                verdict = "偏空观望"
                confidence = bear_ratio
            else:
                verdict = "争议"
                confidence = max(bull_ratio, bear_ratio)

        # ── 共识类型 ──
        if veto_triggered:
            consensus_type = "一票否决"
        elif bull_ratio >= CONSENSUS_THRESHOLD or bear_ratio >= CONSENSUS_THRESHOLD:
            consensus_type = "一致"
        elif max(bull_ratio, bear_ratio) < CONTROVERSY_THRESHOLD + 0.1:
            consensus_type = "争议"
        else:
            consensus_type = "分歧"

        # ── 裁决理由 ──
        summary_lines = []
        if veto_triggered:
            for v in vetos:
                summary_lines.append(
                    f"[否决] {v['persona']}({v['key']}) 得分{v['score']:.0f}, "
                    f"{v['label']}, {v['opinion']}"
                )
        if bull_args:
            top_bull = max(bull_args, key=lambda x: x["power"])
            summary_lines.append(
                f"[多方] {top_bull['persona']}({top_bull['key']}) "
                f"得分{top_bull['score']:.0f} ({top_bull['detail']})"
            )
        if bear_args:
            top_bear = max(bear_args, key=lambda x: x["power"])
            summary_lines.append(
                f"[空方] {top_bear['persona']}({top_bear['key']}) "
                f"得分{top_bear['score']:.0f} ({top_bear['detail']})"
            )

        self.verdict = {
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "consensus_type": consensus_type,
            "bull_power": round(bull_power, 3),
            "bear_power": round(bear_power, 3),
            "bull_ratio": round(bull_ratio, 3),
            "bear_ratio": round(bear_ratio, 3),
            "neutral_ratio": round(neutral_ratio, 3),
            "veto_triggered": veto_triggered,
            "vetos": [v["key"] for v in vetos],
            "arguments": self.args,
            "summary": "; ".join(summary_lines),
        }
        return self.verdict

    def get_position_adjustment(self) -> float:
        """根据裁决结果返回仓位调节系数
        用于在 Kelly 计算后做最终微调
        """
        if not self.verdict:
            self.debate()

        v = self.verdict["verdict"]
        conf = self.verdict["confidence"]

        if v == "做多":
            return 1.0  # 不调整
        elif v == "偏多观望":
            return 0.8
        elif v == "争议":
            return 0.3  # 争议大幅减仓
        elif v == "偏空观望":
            return 0.2
        elif v == "做空":
            return 0.0  # 强制空仓
        return 0.5

    def print_debate(self):
        """打印辩论记录"""
        if not self.verdict:
            self.debate()

        print(f"\n{'='*60}")
        print(f"  多空辩论裁决")
        print(f"{'='*60}")
        print(f"  裁决: {self.verdict['verdict']} "
              f"(信心{self.verdict['confidence']:.0%})")
        print(f"  共识: {self.verdict['consensus_type']}")
        print(f"  多方: {self.verdict['bull_ratio']:.0%}  |  "
              f"空方: {self.verdict['bear_ratio']:.0%}  |  "
              f"中性: {self.verdict['neutral_ratio']:.0%}")
        if self.verdict.get("veto_triggered"):
            print(f"  !! 一票否决: {', '.join(self.verdict['vetos'])}")
        print(f"\n  {'因子':<16} {'立场':>4} {'信心':>5} {'权重':>6} {'辩论权重':>8}  {'论据'}")
        print(f"  {'-'*60}")
        for arg in self.args:
            icon = {"BULL": "多", "BEAR": "空", "NEUTRAL": "="}[arg["stance"]]
            name = f"{arg['persona']}({arg['key']})"
            print(f"  {name:<16} {icon:>4} {arg['confidence']:>4.0%} "
                  f"{arg['weight']:>5.0%} {arg['power']:>8.3f}  {arg['detail']}")
        print(f"  {'-'*60}")
        print(f"  {self.verdict['summary']}")
        print()


def debate_factors(breakdown_lines: list, composite_score: float = None,
                   print_report: bool = False) -> dict:
    """快捷接口：直接对因子列表执行辩论"""
    debate = BullBearDebate(breakdown_lines, composite_score)
    result = debate.debate()
    if print_report:
        debate.print_debate()
    return result


# ─── 集成到 scoring.py 的辅助函数 ───

def scoring_debate(scoring_result: dict) -> dict:
    """对 StockScorer.compute() 的返回结果执行辩论

    在 scoring.py 的 compute() 末尾调用:
        debate_result = scoring_debate(scoring_result)
        scoring_result["debate"] = debate_result

    返回结果扩展:
        composite_score 不变 (保持向下兼容)
        debate.verdict 作为补充决策参考
        debate.get_position_adjustment() 可微调 pos_factor
    """
    factors = scoring_result.get("factors") or scoring_result.get("breakdown_lines", [])
    composite = scoring_result.get("composite_score")
    result = debate_factors(factors, composite)
    result["pos_adjustment"] = BullBearDebate(factors, composite).get_position_adjustment()
    return result


if __name__ == "__main__":
    # 快速验证
    import random
    random.seed(42)

    print("=== 多空辩论模块验证 ===\n")

    # 场景1: 多方共识
    factors1 = [
        {"key": "tech_strength",     "score": 82, "weight": 0.28, "detail": "趋势多头+80%"},
        {"key": "risk_reward",       "score": 65, "weight": 0.18, "detail": "盈亏比2.5:1"},
        {"key": "volume",            "score": 72, "weight": 0.15, "detail": "量比1.8, 百分位85%"},
        {"key": "candlestick",       "score": 55, "weight": 0.05, "detail": "小阳线"},
        {"key": "sector",            "score": 78, "weight": 0.16, "detail": "板块强度78"},
        {"key": "relative_strength", "score": 80, "weight": 0.18, "detail": "RS=85"},
    ]
    r1 = debate_factors(factors1, print_report=True)
    assert r1["verdict"] == "做多", f"预期做多, 得到{r1['verdict']}"
    print(f"  场景1验证通过\n")

    # 场景2: 意见分歧（技术面看多，风控看空）
    factors2 = [
        {"key": "tech_strength",     "score": 78, "weight": 0.28, "detail": "趋势多头"},
        {"key": "risk_reward",       "score": 35, "weight": 0.18, "detail": "盈亏比0.8:1"},
        {"key": "volume",            "score": 45, "weight": 0.15, "detail": "缩量"},
        {"key": "candlestick",       "score": 40, "weight": 0.05, "detail": "长上影"},
        {"key": "sector",            "score": 60, "weight": 0.16, "detail": "板块中性"},
        {"key": "relative_strength", "score": 72, "weight": 0.18, "detail": "RS=70"},
    ]
    r2 = debate_factors(factors2, print_report=True)
    print(f"  场景2验证通过\n")

    # 场景3: 一票否决（某个因子得分极低）
    factors3 = [
        {"key": "tech_strength",     "score": 80, "weight": 0.28, "detail": "趋势多头"},
        {"key": "risk_reward",       "score": 75, "weight": 0.18, "detail": "盈亏比好"},
        {"key": "volume",            "score": 20, "weight": 0.15, "detail": "量比0.3, 极度缩量"},
        {"key": "candlestick",       "score": 50, "weight": 0.05, "detail": "十字星"},
        {"key": "sector",            "score": 70, "weight": 0.16, "detail": "板块强势"},
        {"key": "relative_strength", "score": 75, "weight": 0.18, "detail": "RS=78"},
    ]
    r3 = debate_factors(factors3, print_report=True)
    assert r3["veto_triggered"], "预期一票否决"
    print(f"  场景3验证通过")

    print("所有验证通过")
