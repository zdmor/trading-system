"""
数据源整合层：AKShare + Tushare Pro + 腾讯财经 API

策略:
  1. 优先使用 AKShare（带 timeout 保护）
  2. AKShare 超时/失败 → 自动降级到 Tushare Pro
  3. 两者都失败 → 返回空/None, 不影响主流程

提示: Tushare Pro 需 5000 积分, token 已在模块内配置
"""
from .akshare_provider import AkshareProvider
from .tushare_provider import TushareProvider

# tushare 是否可用（懒加载）
_tushare_ok = None


def _check_tushare():
    global _tushare_ok
    if _tushare_ok is None:
        _tushare_ok = TushareProvider.health_check()
    return _tushare_ok


# ---------- 自适应接口（尝试 AKShare → 降级 Tushare） ----------

def get_industry_map() -> dict:
    """行业分类: 优选 AKShare(带缓存), 降级 Tushare"""
    try:
        result = AkshareProvider.get_industry_map()
        if result:
            return result
    except Exception:
        pass
    if _check_tushare():
        return TushareProvider.get_industry_map()
    return {}


def get_financial_indicators(symbol: str) -> dict:
    """基本面: 优选 Tushare(稳定), 降级 AKShare"""
    if _check_tushare():
        result = TushareProvider.get_financial_indicators(symbol)
        if result:
            return result
    try:
        return AkshareProvider.get_financial_indicators(symbol)
    except Exception:
        return {}


def get_stock_moneyflow(symbol: str) -> dict:
    """资金流向: 优选 Tushare, 降级 AKShare"""
    if _check_tushare():
        result = TushareProvider.get_stock_moneyflow(symbol)
        if result:
            return result
    try:
        return AkshareProvider.get_stock_moneyflow(symbol)
    except Exception:
        return {}


def get_concept_boards() -> list:
    """概念板块: AKShare"""
    try:
        return AkshareProvider.get_concept_boards()
    except Exception:
        return []


def get_macro_dashboard() -> dict:
    """宏观数据: AKShare"""
    try:
        return AkshareProvider.get_macro_dashboard()
    except Exception:
        return {}


def get_northbound_flow() -> dict:
    """北向资金: AKShare"""
    try:
        return AkshareProvider.get_northbound_flow()
    except Exception:
        return {}


def get_stock_industry(symbol: str) -> str:
    """个股行业: 优选 Tushare"""
    if _check_tushare():
        result = TushareProvider.get_stock_industry(symbol)
        if result != "其他":
            return result
    try:
        return AkshareProvider.get_stock_industry(symbol)
    except Exception:
        return "其他"


def get_stock_quality(symbol: str) -> dict:
    """个股质地全面检查: Tushare only"""
    if _check_tushare():
        return TushareProvider.get_stock_quality(symbol)
    return {}


def get_pledge_ratio(symbol: str) -> float:
    """个股质押率: Tushare only"""
    if _check_tushare():
        return TushareProvider.get_pledge_ratio(symbol)
    return 0.0


def get_institutional_coverage(symbol: str, months: int = 3) -> dict:
    """个股机构覆盖: Tushare only"""
    if _check_tushare():
        return TushareProvider.get_institutional_coverage(symbol, months)
    return {'org_count': 0, 'report_count': 0}


def is_st(symbol: str) -> bool:
    """ST检查: Tushare only"""
    if _check_tushare():
        return TushareProvider.is_st(symbol)
    return False


def get_sector_heat(industry: str) -> dict:
    """板块景气度五维评分"""
    try:
        from sector_heat import analyze
        return analyze(industry)
    except Exception:
        return {"composite": 50, "level": "一般", "dimensions": {}}


__all__ = [
    "AkshareProvider", "TushareProvider",
    "get_industry_map", "get_financial_indicators",
    "get_stock_moneyflow", "get_concept_boards",
    "get_macro_dashboard", "get_northbound_flow",
    "get_stock_industry",
    "get_stock_quality", "get_pledge_ratio",
    "get_institutional_coverage", "is_st",
    "get_sector_heat",
]
