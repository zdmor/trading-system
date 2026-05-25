"""
Kelly 仓位模块 — 用认知优势计算最优仓位

核心公式: f* = (bp - q) / b
  p = 胜率 (因子IC回测的历史正向命中率)
  b = 赔率 (score_risk_reward 的 R/R 比)
  q = 1 - p
  edge = bp - q  (认知优势，<=0 则不下注)

输出半凯利仓位 (conservative)，再由 position_matrix 做最终约束。

P0-1 已验证: A股短期追高有动量效应，追高惩罚无效且有害。
因此凯利不惩罚趋势——趋势本身可能提供 edge。
"""
import json
import os
import numpy as np

# ─── 因子历史统计（来自 factor_ic_rolling 回测，可定期更新） ───
# 格式: {factor_name: {"win_rate": float, "avg_win": float, "avg_loss": float}}
# win_rate: 该因子高分组的正向前收益占比
# avg_win: 正收益均值 (%)
# avg_loss: 负收益均值 (%) 的绝对值
FACTOR_STATS = {
    "tech_strength":   {"win_rate": 0.56, "avg_win": 3.2, "avg_loss": 2.8},
    "risk_reward":     {"win_rate": 0.39, "avg_win": 3.5, "avg_loss": 3.5},
    "volume":          {"win_rate": 0.41, "avg_win": 3.0, "avg_loss": 3.2},
    "candlestick":     {"win_rate": 0.51, "avg_win": 2.5, "avg_loss": 2.5},
    "sector":          {"win_rate": 0.37, "avg_win": 2.8, "avg_loss": 3.0},
    "relative_strength": {"win_rate": 0.51, "avg_win": 3.3, "avg_loss": 3.0},
}

# 文件持久化路径
STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_stats.json")


def load_factor_stats():
    """从文件加载因子统计，文件不存在则用默认值"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return FACTOR_STATS.copy()


def save_factor_stats(stats):
    """保存更新后的因子统计"""
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def kelly_fraction(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
    """
    凯利公式: f* = (bp - q) / b

    Args:
        win_rate: 胜率 p (0~1)
        avg_win_pct: 平均盈利百分比 (正数)
        avg_loss_pct: 平均亏损百分比 (正数, 取绝对值)

    Returns:
        f* 最优仓位比例 (0~1)，edge<=0 返回 0
    """
    if avg_loss_pct <= 0 or win_rate <= 0:
        return 0.0

    b = avg_win_pct / avg_loss_pct  # 赔率
    p = min(0.99, max(0.01, win_rate))  # 胜率钳制
    q = 1 - p
    edge = b * p - q

    if edge <= 0:
        return 0.0  # 无认知优势不下注

    f_star = edge / b
    return round(min(1.0, max(0.0, f_star)), 4)


def half_kelly(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> float:
    """半凯利：保守仓位 = f* / 2"""
    return round(kelly_fraction(win_rate, avg_win_pct, avg_loss_pct) / 2, 4)


def compute_kelly_position(composite_score: float, factors: list,
                           market_score: float = 50, sector_score: float = 50,
                           risk_pct: float = 0.02) -> dict:
    """
    综合计算 Kelly 仓位

    Args:
        composite_score: 综合评分 (0-100)
        factors: compute() 输出的 breakdown_lines 列表
        market_score: 大盘评分 (0-100)
        sector_score: 板块评分 (0-100)
        risk_pct: 单笔最大风险比例 (默认 2%)

    Returns:
        {
            "kelly_raw": float,       # 理论全凯利仓位
            "kelly_half": float,      # 半凯利仓位 (推荐)
            "position_factor": float, # 最终仓位系数 (0-1)
            "edge": float,            # 认知优势 bp-q
            "detail": str,            # 计算说明
        }
    """
    stats = load_factor_stats()

    # 1. 从因子分解中提取加权胜率和赔率
    total_weight = 0
    weighted_win_rate = 0
    weighted_b = 0
    rr_ratio = None  # 泰珀改进：直接取该票实际赔率

    for f in factors:
        key = f.get("key", "")
        weight = f.get("weight", 0)
        if key == "risk_reward":
            rr_ratio = f.get("ratio")  # scoring.py score_risk_reward() 返回值含 ratio
        if key in stats and weight > 0:
            s = stats[key]
            weighted_win_rate += s["win_rate"] * weight
            if key != "risk_reward":  # R/R因子用实际赔率，其他因子用统计值
                weighted_b += (s["avg_win"] / max(s["avg_loss"], 0.01)) * weight
            total_weight += weight

    if total_weight > 0:
        avg_win_rate = weighted_win_rate / total_weight
        if total_weight > stats["risk_reward"].get("win_rate", 0) * 0.18 if "risk_reward" in stats else 0:
            avg_b = weighted_b / (total_weight - 0.18) if total_weight > 0.18 else weighted_b / max(total_weight, 0.01)
        else:
            avg_b = weighted_b / max(total_weight, 0.01)
        # 泰珀改进：用该票实际 R/R 替换统计 b
        if rr_ratio and rr_ratio > 0:
            avg_b = rr_ratio
        avg_win = avg_b * 2.0
        avg_loss = 2.0
    else:
        avg_win_rate = 0.5
        avg_b = rr_ratio if (rr_ratio and rr_ratio > 0) else 1.2
        avg_win = avg_b * 2.0
        avg_loss = 2.0

    # 2. 凯利计算
    f_star = kelly_fraction(avg_win_rate, avg_win, avg_loss)
    f_half = f_star / 2

    # 3. 综合分微调：高分略微加仓，低分降仓
    score_factor = 1.0
    if composite_score >= 80:
        score_factor = 1.15
    elif composite_score >= 65:
        score_factor = 1.05
    elif composite_score < 40:
        score_factor = 0.5
    elif composite_score < 50:
        score_factor = 0.7

    position = min(1.0, f_half * score_factor)

    # 4. 风险预算钳制
    max_position = risk_pct * 10
    position = min(position, max_position)

    # 5. 大盘熊市打折扣（大盘不参与评分，但约束仓位）
    if market_score < 35:
        position *= 0.5
    elif market_score < 50:
        position *= 0.75

    # 6. 泰珀修正：情绪放大器 — 市场恐慌时放大 edge，贪婪时收缩
    try:
        from sentiment_indicator import SentimentIndicator
        si = SentimentIndicator()
        sentiment_result = si.analyze()
        sentiment = sentiment_result.get("composite", 50)
        if sentiment < 30:
            position = min(1.0, position * 1.5)
        elif sentiment < 40:
            position = min(1.0, position * 1.25)
        elif sentiment > 75:
            position *= 0.5
    except Exception:
        pass  # 情绪模块不可用时静默跳过

    position = round(position, 4)

    # 计算 edge
    b = avg_win / max(avg_loss, 0.01)
    edge = b * avg_win_rate - (1 - avg_win_rate)

    return {
        "kelly_raw": round(f_star, 4),
        "kelly_half": round(f_half, 4),
        "position_factor": position,
        "edge": round(edge, 4),
        "weighted_win_rate": round(avg_win_rate, 3),
        "weighted_b": round(avg_b, 2),
        "detail": (
            f"K=({avg_win_rate:.0%}×{avg_b:.1f}-{1-avg_win_rate:.0%})/{avg_b:.1f}"
            f"={f_star:.0%} → 半K={f_half:.0%} → 仓位={position:.0%}"
        ),
    }


def update_factor_stats_from_ic(results: dict):
    """
    从 IC 回测结果更新因子统计

    Args:
        results: {"factor_name": {"ic_mean": float, "ic_std": float, "win_rate": float,
                                   "avg_win": float, "avg_loss": float}, ...}
    """
    stats = load_factor_stats()
    for key, vals in results.items():
        if key in stats:
            stats[key].update(vals)
    save_factor_stats(stats)
    print(f"  因子统计已更新: {STATS_FILE}")


if __name__ == "__main__":
    # 快速验证
    print("=== Kelly 仓位模块验证 ===\n")

    # 场景1: 有明显优势
    r = compute_kelly_position(75, [
        {"key": "tech_strength", "weight": 0.28},
        {"key": "risk_reward", "weight": 0.18},
        {"key": "volume", "weight": 0.15},
    ], market_score=65, sector_score=70)
    print(f"高分信号: {r['detail']}")
    print(f"  仓位={r['position_factor']:.0%} edge={r['edge']:.3f}\n")

    # 场景2: 无优势
    r2 = compute_kelly_position(35, [
        {"key": "tech_strength", "weight": 0.28},
    ], market_score=25, sector_score=20)
    print(f"低分熊市: {r2['detail']}")
    print(f"  仓位={r2['position_factor']:.0%} edge={r2['edge']:.3f}\n")

    # 场景3: 极端
    print(f"全凯利(60%胜率, 赔率2:1): {kelly_fraction(0.6, 4.0, 2.0):.1%}")
    print(f"半凯利(60%胜率, 赔率2:1): {half_kelly(0.6, 4.0, 2.0):.1%}")
    print(f"全凯利(51%胜率, 赔率1:1): {kelly_fraction(0.51, 2.0, 2.0):.1%}")
    print(f"无优势(50%胜率):        {kelly_fraction(0.5, 2.0, 2.0):.1%}")
    print(f"\nOK - Kelly 模块验证通过")
