"""
AKShare 数据提供器：A股基本面、财务指标、宏观数据
从 finskills (findata-toolkit-cn) 提取核心能力，整合到交易系统

安装: pip install akshare pandas numpy
"""
import time
from datetime import datetime, timedelta
from typing import Optional


class AkshareProvider:
    """AKShare 数据封装层 — 为交易系统提供基本面/宏观/行业数据"""

    # ---------- 个股基本信息 ----------

    @staticmethod
    def get_stock_info(symbol: str) -> dict:
        """
        获取个股基本面信息：名称、行业、PE、PB、ROE、总市值、流通市值
        symbol: 6位数字代码，如 "002050"
        akshare 接口: stock_individual_info_em
        """
        import akshare as ak
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            info = {}
            for _, row in df.iterrows():
                key = str(row.iloc[0]).strip()
                val = row.iloc[1]
                info[key] = val
            return info
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def get_stock_zh_spot() -> list:
        """
        获取A股实时行情全览（含涨跌幅、成交量、换手率、PE、总市值等）
        akshare 接口: stock_zh_a_spot_em
        """
        import akshare as ak
        try:
            df = ak.stock_zh_a_spot_em()
            return df.to_dict(orient="records")
        except Exception as e:
            return [{"error": str(e)}]

    # ---------- 个股财务指标 ----------

    @staticmethod
    def get_financial_metrics(symbol: str) -> dict:
        """
        获取个股完整财务指标：
        营收增长率、净利润增长率、ROE、毛利率、资产负债率等
        akshare 接口: stock_financial_abstract_ths
        """
        import akshare as ak
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol)
            if df is None or df.empty:
                return {}
            latest = df.iloc[0].to_dict()
            return {str(k): v for k, v in latest.items()}
        except Exception:
            return {}

    @staticmethod
    def _get_latest_price(symbol: str) -> Optional[float]:
        """获取个股最新价格（辅助方法）"""
        import akshare as ak
        try:
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20260101", adjust="qfq")
            if df is not None and not df.empty:
                return float(df["收盘"].iloc[-1])
        except Exception:
            pass
        return None

    @staticmethod
    def get_financial_indicators(symbol: str) -> dict:
        """
        获取估值/盈利/成长指标摘要
        使用 stock_individual_info_em（市值）+ stock_financial_abstract_ths（ROE/毛利率等）
        """
        import akshare as ak
        result = {}

        # 基本信息（总市值、流通市值、行业）
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    key = str(row.iloc[0]).strip()
                    val = row.iloc[1]
                    if "总市值" in key:
                        result["total_mv"] = float(val)
                    elif "流通市值" in key:
                        result["float_mv"] = float(val)
        except Exception:
            pass

        # 财务指标（ROE、毛利率、资产负债率等）
        try:
            df = ak.stock_financial_abstract_ths(symbol=symbol, indicator="主要指标")
            if df is not None and not df.empty:
                latest = df.iloc[0]
                for key, val in latest.items():
                    ks = str(key).strip()
                    vs = str(val).strip() if val else ""
                    if "每股收益" in ks and "基本" in ks:
                        try:
                            result["eps"] = float(vs)
                        except ValueError:
                            pass
                    elif "每股净资产" in ks:
                        try:
                            result["bvps"] = float(vs)
                        except ValueError:
                            pass
                    elif "净资产收益率" in ks and "摊薄" not in ks and result.get("roe") is None:
                        try:
                            result["roe"] = float(vs.replace("%", ""))
                        except (ValueError, AttributeError):
                            pass
                    elif "毛利率" in ks:
                        try:
                            result["gross_margin"] = float(vs.replace("%", ""))
                        except (ValueError, AttributeError):
                            pass
                    elif "资产负债率" in ks:
                        try:
                            result["debt_ratio"] = float(vs.replace("%", ""))
                        except (ValueError, AttributeError):
                            pass
        except Exception:
            pass

        # 如果能拿到 EPS/BVPS，用实时价估算 PE/PB
        price = AkshareProvider._get_latest_price(symbol)
        if price and result.get("eps") and result["eps"] > 0:
            result["pe"] = round(price / result["eps"], 2)

        if price and result.get("bvps") and result["bvps"] > 0:
            result["pb"] = round(price / result["bvps"], 2)

        return result

    # ---------- 行业分类 ----------

    @staticmethod
    def get_industry_map() -> dict:
        """
        获取全市场股票行业分类映射 {股票代码: 行业名称}
        akshare 接口: stock_board_industry_cons_em
        替换 scanner.py 中的 baostock 行业分类
        """
        import akshare as ak
        industry_map = {}
        try:
            # 获取行业板块列表
            boards_df = ak.stock_board_industry_name_em()
            if boards_df is None or boards_df.empty:
                return {}

            for _, board in boards_df.iterrows():
                board_name = board.get("板块名称", "")
                board_code = board.get("板块代码", "")
                if not board_name or not board_code:
                    continue
                try:
                    cons = ak.stock_board_industry_cons_em(symbol=board_code)
                    if cons is not None and not cons.empty:
                        for _, row in cons.iterrows():
                            code = str(row.get("代码", ""))
                            if code:
                                # 统一成 6 位代码
                                code = code.zfill(6)
                                industry_map[code] = board_name
                except Exception:
                    continue
                time.sleep(0.15)  # 限速

        except Exception:
            pass

        return industry_map

    @staticmethod
    def get_industry_board_performance() -> list:
        """
        获取所有行业板块涨跌幅排行
        返回: [{"name": str, "change_pct": float}, ...] 按涨跌幅降序
        """
        import akshare as ak
        try:
            df = ak.stock_board_industry_name_em()
            if df is not None and not df.empty:
                results = []
                for _, row in df.iterrows():
                    try:
                        results.append({
                            "name": str(row.get("板块名称", "")),
                            "change_pct": float(row.get("涨跌幅", 0)),
                        })
                    except (ValueError, TypeError):
                        continue
                return results
        except Exception:
            pass
        return []

    @staticmethod
    def get_stock_industry(symbol: str) -> str:
        """获取单只股票行业归属"""
        import akshare as ak
        try:
            df = ak.stock_board_industry_cons_em(symbol=symbol)
            if df is not None and not df.empty:
                return str(df.iloc[0].get("板块名称", "其他"))
        except Exception:
            pass
        return "其他"

    # ---------- 概念板块 ----------

    @staticmethod
    def get_concept_boards() -> list:
        """
        获取概念板块列表及涨跌幅排序
        akshare 接口: stock_board_concept_name_em
        """
        import akshare as ak
        try:
            df = ak.stock_board_concept_name_em()
            if df is not None and not df.empty:
                return df.sort_values("涨跌幅", ascending=False).to_dict(orient="records")
        except Exception:
            pass
        return []

    # ---------- 北向资金 ----------

    @staticmethod
    def get_northbound_flow() -> dict:
        """
        获取北向资金（沪深港通）当日流入/流出
        akshare 接口: stock_hsgt_north_net_flow_in_em
        """
        import akshare as ak
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="沪股通")
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                sh_flow = float(latest.get("value", 0))
            else:
                sh_flow = 0
        except Exception:
            sh_flow = 0

        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="深股通")
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                sz_flow = float(latest.get("value", 0))
            else:
                sz_flow = 0
        except Exception:
            sz_flow = 0

        return {
            "沪股通": round(sh_flow, 2),
            "深股通": round(sz_flow, 2),
            "合计": round(sh_flow + sz_flow, 2),
        }

    @staticmethod
    def get_southbound_flow() -> dict:
        """南向资金（港股通）"""
        import akshare as ak
        try:
            df = ak.stock_hsgt_south_net_flow_in_em(symbol="沪市港股通")
            sh = float(df.iloc[-1].get("value", 0)) if df is not None and not df.empty else 0
        except Exception:
            sh = 0
        try:
            df = ak.stock_hsgt_south_net_flow_in_em(symbol="深市港股通")
            sz = float(df.iloc[-1].get("value", 0)) if df is not None and not df.empty else 0
        except Exception:
            sz = 0
        return {"沪市港股通": round(sh, 2), "深市港股通": round(sz, 2), "合计": round(sh + sz, 2)}

    # ---------- 宏观数据 ----------

    @staticmethod
    def get_macro_dashboard() -> dict:
        """宏观经济仪表盘：CPI、PMI、M2、社融（最佳努力）"""
        import akshare as ak
        result = {}

        # PMI
        try:
            df = ak.macro_china_pmi()
            if df is not None and not df.empty:
                for col in df.columns:
                    if "制造业" in str(col) and "指数" in str(col):
                        result["pmi"] = float(df.iloc[-1].get(col, 0))
                        break
        except Exception:
            pass

        # CPI (月度)
        try:
            df = ak.macro_china_cpi_monthly()
            if df is not None and not df.empty:
                for col in df.columns:
                    if "同比增长" in str(col) or "当月同比" in str(col):
                        val = df.iloc[-1].get(col)
                        if val is not None and not (isinstance(val, float) and str(val) == "nan"):
                            result["cpi"] = float(val)
                        break
        except Exception:
            pass

        # M2 同比增长
        try:
            df = ak.macro_china_money_supply()
            if df is not None and not df.empty:
                for col in df.columns:
                    if "M2" in str(col) and "同比" in str(col):
                        result["m2"] = float(df.iloc[-1].get(col, 0))
                        break
        except Exception:
            pass

        # GDP
        try:
            df = ak.macro_china_gdp_yearly()
            if df is not None and not df.empty:
                for col in df.columns:
                    if "同比" in str(col) or "增长率" in str(col):
                        val = df.iloc[-1].get(col)
                        if val is not None:
                            result["gdp"] = float(val)
                        break
        except Exception:
            pass

        return {k: v for k, v in result.items() if v not in (None, 0)}

    # ---------- 涨停板块 ----------

    @staticmethod
    def get_limit_up_pool(date: str = None) -> list:
        """
        获取涨停股池
        date: YYYYMMDD，默认今日
        akshare 接口: stock_zt_pool_em
        """
        import akshare as ak
        date = date or datetime.now().strftime("%Y%m%d")
        try:
            df = ak.stock_zt_pool_em(date=date)
            if df is not None and not df.empty:
                return df.to_dict(orient="records")
        except Exception:
            pass
        return []

    # ---------- 资金流向 ----------

    @staticmethod
    def get_stock_moneyflow(symbol: str) -> dict:
        """
        个股资金流向：主力、超大单、大单、中单、小单
        akshare 接口: stock_individual_fund_flow
        """
        import akshare as ak
        try:
            df = ak.stock_individual_fund_flow(stock=symbol, market="sh" if symbol.startswith("6") else "sz")
            if df is not None and not df.empty:
                latest = df.iloc[-1]
                return {
                    "主力净流入": float(latest.get("主力净流入-净额", 0)),
                    "主力净占比": float(latest.get("主力净流入-净占比", 0)),
                    "超大单净流入": float(latest.get("超大单净流入-净额", 0)),
                    "大单净流入": float(latest.get("大单净流入-净额", 0)),
                    "中单净流入": float(latest.get("中单净流入-净额", 0)),
                    "小单净流入": float(latest.get("小单净流入-净额", 0)),
                    "日期": str(latest.get("日期", "")),
                }
        except Exception:
            pass
        return {}
