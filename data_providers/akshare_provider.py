"""
AKShare 数据提供器 — 带 timeout 保护 + Tushare Pro 降级
所有接口调用都加了线程超时，防止网络卡死
AKShare 失败时自动降级到 TusharePro

设计:
  _ak_call(timeout, lambda: ak.function(...)) → 线程执行+超时保护
  每个方法: _ak_call 返回 None → 自动降级 TushareProvider
"""
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from typing import Optional, Callable, Any

# 延迟导入 Tushare
_tushare_provider = None
def _get_tushare():
    global _tushare_provider
    if _tushare_provider is None:
        from data_providers.tushare_provider import TushareProvider
        _tushare_provider = TushareProvider
    return _tushare_provider

_POOL = ThreadPoolExecutor(max_workers=1)

def _ak_call(timeout: int, func: Callable) -> Any:
    """在线程中执行 func，超时返回 None"""
    fut = _POOL.submit(func)
    try:
        return fut.result(timeout=timeout)
    except (TimeoutError, Exception):
        fut.cancel()
        return None


class AkshareProvider:
    """AKShare 数据封装层 — 所有接口均有 timeout + Tusahre 降级"""

    # ==================== 个股基本信息 ====================

    @staticmethod
    def get_stock_info(symbol: str) -> dict:
        import akshare as ak
        try:
            df = _ak_call(10, lambda: ak.stock_individual_info_em(symbol=symbol))
            if df is not None and not df.empty:
                return {str(row.iloc[0]).strip(): row.iloc[1] for _, row in df.iterrows()}
        except Exception:
            pass
        return {"error": "timeout or failed"}

    @staticmethod
    def get_stock_zh_spot() -> list:
        import akshare as ak
        try:
            df = _ak_call(15, lambda: ak.stock_zh_a_spot_em())
            if df is not None and not df.empty:
                return df.to_dict(orient="records")
        except Exception:
            pass
        return [{"error": "timeout"}]

    # ==================== 个股财务指标 ====================

    @staticmethod
    def get_financial_metrics(symbol: str) -> dict:
        import akshare as ak
        try:
            df = _ak_call(10, lambda: ak.stock_financial_abstract_ths(symbol=symbol))
            if df is not None and not df.empty:
                return {str(k): v for k, v in df.iloc[0].to_dict().items()}
        except Exception:
            pass
        return {}

    @staticmethod
    def _get_latest_price(symbol: str) -> Optional[float]:
        import akshare as ak
        try:
            df = _ak_call(10, lambda: ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                                         start_date="20260101", adjust="qfq"))
            if df is not None and not df.empty:
                return float(df["收盘"].iloc[-1])
        except Exception:
            pass
        return None

    @staticmethod
    def get_financial_indicators(symbol: str) -> dict:
        """基本面指标，失败降级 Tushare"""
        import akshare as ak
        result = {}
        try:
            df = _ak_call(10, lambda: ak.stock_individual_info_em(symbol=symbol))
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    key, val = str(row.iloc[0]).strip(), row.iloc[1]
                    if "总市值" in key: result["total_mv"] = float(val)
                    elif "流通市值" in key: result["float_mv"] = float(val)
        except Exception:
            pass
        try:
            df = _ak_call(10, lambda: ak.stock_financial_abstract_ths(symbol=symbol, indicator="主要指标"))
            if df is not None and not df.empty:
                for key, val in df.iloc[0].items():
                    ks, vs = str(key).strip(), str(val).strip() if val else ""
                    if "每股收益" in ks and "基本" in ks:
                        try: result["eps"] = float(vs)
                        except ValueError: pass
                    elif "每股净资产" in ks:
                        try: result["bvps"] = float(vs)
                        except ValueError: pass
                    elif "净资产收益率" in ks and "摊薄" not in ks and "roe" not in result:
                        try: result["roe"] = float(vs.replace("%", ""))
                        except: pass
                    elif "毛利率" in ks:
                        try: result["gross_margin"] = float(vs.replace("%", ""))
                        except: pass
                    elif "资产负债率" in ks:
                        try: result["debt_ratio"] = float(vs.replace("%", ""))
                        except: pass
        except Exception:
            pass
        if result.get("eps") or result.get("total_mv"):
            price = AkshareProvider._get_latest_price(symbol)
            if price and result.get("eps") and result["eps"] > 0:
                result["pe"] = round(price / result["eps"], 2)
            if price and result.get("bvps") and result["bvps"] > 0:
                result["pb"] = round(price / result["bvps"], 2)
            return result
        # 降级 Tushare
        return _get_tushare().get_financial_indicators(symbol)

    # ==================== 行业分类 ====================

    @staticmethod
    def get_industry_map() -> dict:
        """全市场行业分类（遍历板块，60s 超时），失败降级 Tushare"""
        import akshare as ak
        try:
            boards_df = _ak_call(15, lambda: ak.stock_board_industry_name_em())
            if boards_df is None or boards_df.empty:
                raise ValueError("empty")
            industry_map = {}
            for _, board in boards_df.iterrows():
                board_name = board.get("板块名称", "")
                board_code = board.get("板块代码", "")
                if not board_name or not board_code:
                    continue
                cons = _ak_call(5, lambda bc=board_code: ak.stock_board_industry_cons_em(symbol=bc))
                if cons is not None and not cons.empty:
                    for _, row in cons.iterrows():
                        code = str(row.get("代码", "")).zfill(6)
                        if code:
                            industry_map[code] = board_name
                time.sleep(0.15)
            if industry_map:
                return industry_map
        except Exception:
            pass
        return _get_tushare().get_industry_map()

    @staticmethod
    def get_industry_board_performance() -> list:
        import akshare as ak
        try:
            df = _ak_call(15, lambda: ak.stock_board_industry_name_em())
            if df is not None and not df.empty:
                return [{"name": str(r.get("板块名称", "")), "change_pct": float(r.get("涨跌幅", 0))}
                        for _, r in df.iterrows() if r.get("涨跌幅") is not None]
        except Exception:
            pass
        return []

    @staticmethod
    def get_stock_industry(symbol: str) -> str:
        import akshare as ak
        try:
            df = _ak_call(10, lambda: ak.stock_board_industry_cons_em(symbol=symbol))
            if df is not None and not df.empty:
                return str(df.iloc[0].get("板块名称", "其他"))
        except Exception:
            pass
        return _get_tushare().get_stock_industry(symbol)

    # ==================== 概念板块 ====================

    @staticmethod
    def get_concept_boards() -> list:
        import akshare as ak
        try:
            df = _ak_call(15, lambda: ak.stock_board_concept_name_em())
            if df is not None and not df.empty:
                return df.sort_values("涨跌幅", ascending=False).to_dict(orient="records")
        except Exception:
            pass
        return []

    # ==================== 资金流向 ====================

    @staticmethod
    def get_stock_moneyflow(symbol: str) -> dict:
        import akshare as ak
        try:
            market = "sh" if symbol.startswith("6") else "sz"
            df = _ak_call(10, lambda: ak.stock_individual_fund_flow(stock=symbol, market=market))
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                result = {
                    "主力净流入": float(latest.get("主力净流入-净额", 0)),
                    "主力净占比": float(latest.get("主力净流入-净占比", 0)),
                    "超大单净流入": float(latest.get("超大单净流入-净额", 0)),
                    "大单净流入": float(latest.get("大单净流入-净额", 0)),
                    "中单净流入": float(latest.get("中单净流入-净额", 0)),
                    "小单净流入": float(latest.get("小单净流入-净额", 0)),
                    "日期": str(latest.get("日期", "")),
                }
                if result.get("主力净流入") != 0:
                    return result
        except Exception:
            pass
        return _get_tushare().get_stock_moneyflow(symbol)

    # ==================== 北向资金 ====================

    @staticmethod
    def get_northbound_flow() -> dict:
        import akshare as ak
        sh_flow = sz_flow = 0
        try:
            df = _ak_call(10, lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="沪股通"))
            if df is not None and not df.empty:
                sh_flow = float(df.iloc[-1].get("value", 0))
        except Exception:
            pass
        try:
            df = _ak_call(10, lambda: ak.stock_hsgt_north_net_flow_in_em(symbol="深股通"))
            if df is not None and not df.empty:
                sz_flow = float(df.iloc[-1].get("value", 0))
        except Exception:
            pass
        return {"沪股通": round(sh_flow, 2), "深股通": round(sz_flow, 2), "合计": round(sh_flow + sz_flow, 2)}

    @staticmethod
    def get_southbound_flow() -> dict:
        import akshare as ak
        sh = sz = 0
        try:
            df = _ak_call(10, lambda: ak.stock_hsgt_south_net_flow_in_em(symbol="沪市港股通"))
            if df is not None and not df.empty:
                sh = float(df.iloc[-1].get("value", 0))
        except Exception:
            pass
        try:
            df = _ak_call(10, lambda: ak.stock_hsgt_south_net_flow_in_em(symbol="深市港股通"))
            if df is not None and not df.empty:
                sz = float(df.iloc[-1].get("value", 0))
        except Exception:
            pass
        return {"沪市港股通": round(sh, 2), "深市港股通": round(sz, 2), "合计": round(sh + sz, 2)}

    # ==================== 宏观数据 ====================

    @staticmethod
    def get_macro_dashboard() -> dict:
        import akshare as ak
        result = {}
        try:
            df = _ak_call(10, lambda: ak.macro_china_pmi())
            if df is not None and not df.empty:
                for col in df.columns:
                    if "制造业" in str(col) and "指数" in str(col):
                        result["pmi"] = float(df.iloc[-1].get(col, 0))
                        break
        except Exception: pass
        try:
            df = _ak_call(10, lambda: ak.macro_china_cpi_monthly())
            if df is not None and not df.empty:
                for col in df.columns:
                    if "同比增长" in str(col) or "当月同比" in str(col):
                        v = df.iloc[-1].get(col)
                        if v is not None and not (isinstance(v, float) and str(v) == "nan"):
                            result["cpi"] = float(v)
                        break
        except Exception: pass
        try:
            df = _ak_call(10, lambda: ak.macro_china_money_supply())
            if df is not None and not df.empty:
                for col in df.columns:
                    if "M2" in str(col) and "同比" in str(col):
                        result["m2"] = float(df.iloc[-1].get(col, 0))
                        break
        except Exception: pass
        try:
            df = _ak_call(10, lambda: ak.macro_china_gdp_yearly())
            if df is not None and not df.empty:
                for col in df.columns:
                    if "同比" in str(col) or "增长率" in str(col):
                        v = df.iloc[-1].get(col)
                        if v is not None:
                            result["gdp"] = float(v)
                        break
        except Exception: pass
        return {k: v for k, v in result.items() if v not in (None, 0)}

    # ==================== 涨停板块 ====================

    @staticmethod
    def get_limit_up_pool(date: str = None) -> list:
        import akshare as ak
        date = date or datetime.now().strftime("%Y%m%d")
        try:
            df = _ak_call(10, lambda: ak.stock_zt_pool_em(date=date))
            if df is not None and not df.empty:
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []
