"""
仓位决策矩阵
基于 大盘环境(多/震/空) x 板块景气度(强/中/弱) 决定仓位系数
纯计算模块，无API调用
"""

# 3x3 Position Matrix
# Rows: market_trend (bull/neutral/bear)
# Columns: sector_heat_level (strong/moderate/weak)
POSITION_MATRIX = {
    "bull":    {"strong": 1.0, "moderate": 0.8, "weak": 0.4},
    "neutral": {"strong": 0.7, "moderate": 0.5, "weak": 0.2},
    "bear":    {"strong": 0.3, "moderate": 0.1, "weak": 0.0},
}

# Thresholds
MARKET_BULL = 65
MARKET_BEAR = 35
SECTOR_STRONG = 70
SECTOR_WEAK = 35


def get_position_factor(market_score: int, sector_heat_score: int) -> float:
    """
    仓位决策矩阵

    Args:
        market_score: 大盘评分 (0-100)
        sector_heat_score: 板块景气度评分 (0-100)

    Returns:
        float: 仓位系数 0.0 (空仓) ~ 1.0 (满仓)
    """
    if market_score >= MARKET_BULL:
        market_level = "bull"
    elif market_score <= MARKET_BEAR:
        market_level = "bear"
    else:
        market_level = "neutral"

    if sector_heat_score >= SECTOR_STRONG:
        sector_level = "strong"
    elif sector_heat_score <= SECTOR_WEAK:
        sector_level = "weak"
    else:
        sector_level = "moderate"

    return POSITION_MATRIX[market_level][sector_level]


def get_position_suggestion(market_score: int, sector_heat_score: int) -> str:
    """返回可读的仓位建议"""
    factor = get_position_factor(market_score, sector_heat_score)
    if factor >= 0.8:
        return "重仓 (80-100%)"
    if factor >= 0.5:
        return "中等仓位 (50-80%)"
    if factor >= 0.2:
        return "轻仓 (20-50%)"
    return "空仓/观望 (0-20%)"
