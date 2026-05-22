"""
Tushare Pro 数据提供器 — 备用/主要数据源
积分门槛: 5000, 频次: 500次/分钟

用途:
  - 行业分类（替代 AKShare stock_board_industry_cons_em）
  - 基本面 PE/PB/ROE（替代 AKShare stock_financial_abstract_ths）
  - 资金流向（替代 AKShare stock_individual_fund_flow）
  - 概念板块

使用前设置 token:
  import tushare as ts
  ts.set_token('your_token')
"""
import math
import time
from datetime import datetime, timedelta
from typing import Optional

# 单例 pro_api
_pro = None

def _get_pro():
    global _pro
    if _pro is None:
        import tushare as ts
        ts.set_token('6209f704e5762ae2e03088c18b08c22d5d9350d36962a0ee067e108f')
        _pro = ts.pro_api()
    return _pro


class TushareProvider:
    """Tushare Pro 数据封装层 — 5000积分档"""

    # ---------- 行业分类 ----------

    @staticmethod
    def get_industry_map() -> dict:
        """
        获取全市场股票行业分类 {6位代码: 行业名称}
        基于 stock_basic 的 industry 字段，比 AKShare 更稳定
        """
        pro = _get_pro()
        try:
            df = pro.stock_basic(exchange='', list_status='L',
                                 fields='ts_code,symbol,name,industry')
            if df is None or df.empty:
                return {}
            industry_map = {}
            for _, row in df.iterrows():
                code = str(row['symbol']).zfill(6)
                ind = str(row.get('industry', '') or '其他')
                industry_map[code] = ind
            return industry_map
        except Exception:
            return {}

    @staticmethod
    def get_stock_industry(symbol: str) -> str:
        """获取单只股票行业归属"""
        pro = _get_pro()
        try:
            df = pro.stock_basic(symbol=symbol + '.SZ' if not symbol.startswith('6')
                                 else symbol + '.SH',
                                 fields='industry')
            if df is not None and not df.empty:
                return str(df.iloc[0].get('industry', '其他'))
        except Exception:
            pass
        # 尝试从全量 map 查
        im = TushareProvider.get_industry_map()
        return im.get(symbol, '其他')

    # ---------- 基本面指标 ----------

    @staticmethod
    def get_financial_indicators(symbol: str) -> dict:
        """
        获取估值/盈利指标: PE, PB, ROE, EPS, 总市值, 流通市值
        合并 daily_basic + fina_indicator
        """
        pro = _get_pro()
        result = {}

        # 1) daily_basic: 实时 PE/PB/市值
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        try:
            df = pro.daily_basic(ts_code=ts_code, fields='pe,pb,total_mv,circ_mv')
            if df is not None and not df.empty:
                result['pe'] = float(df.iloc[0]['pe']) if df.iloc[0]['pe'] else None
                result['pb'] = float(df.iloc[0]['pb']) if df.iloc[0]['pb'] else None
                result['total_mv'] = float(df.iloc[0]['total_mv']) if df.iloc[0]['total_mv'] else None
                result['float_mv'] = float(df.iloc[0]['circ_mv']) if df.iloc[0]['circ_mv'] else None
        except Exception:
            pass

        # 2) fina_indicator: ROE/EPS/BPS + 扣非利润/负债率
        try:
            df = pro.fina_indicator(ts_code=ts_code,
                                    fields='end_date,roe,eps,bps,profit_dedt,debt_to_assets')
            if df is not None and not df.empty:
                latest = df.iloc[0]
                result['roe'] = float(latest['roe']) if latest['roe'] else None
                result['eps'] = float(latest['eps']) if latest['eps'] else None
                result['bvps'] = float(latest['bps']) if latest['bps'] else None
                result['profit_dedt'] = float(latest['profit_dedt']) if latest['profit_dedt'] else None
                result['debt_to_assets'] = float(latest['debt_to_assets']) if latest['debt_to_assets'] else None

                # 连续ROE检查：找年报数据（5年，兼顾周期股）
                annual = df[df['end_date'].str.endswith('1231')]
                if len(annual) >= 3:
                    roes = []
                    for _, row in annual.head(5).iterrows():
                        if row['roe'] is not None:
                            roes.append(float(row['roe']))
                    if len(roes) >= 3:
                        result['roe_3y_min'] = min(roes[:3])
                        result['roe_3y_avg'] = sum(roes[:3]) / len(roes[:3])
                        result['roe_3y_all_above_10'] = all(r > 10 for r in roes[:3])
                    if len(roes) >= 5:
                        result['roe_5y_avg'] = sum(roes) / len(roes)
                        result['roe_5y_all_above_10'] = all(r > 10 for r in roes)

                # 近两季度利润环比
                if len(df) >= 2:
                    def _get_float(row, field):
                        v = row[field]
                        return float(v) if v else None
                    q_now, q_prev = df.iloc[0], df.iloc[1]
                    pn, pp = _get_float(q_now, 'profit_dedt'), _get_float(q_prev, 'profit_dedt')
                    if pn and pp and pp:
                        result['profit_qoq'] = (pn / pp - 1) * 100
        except Exception:
            pass

        return result

    # ---------- 质押率 ----------

    @staticmethod
    def get_pledge_ratio(symbol: str) -> float:
        """获取个股质押比例（%）"""
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        try:
            pro = _get_pro()
            df = pro.pledge_stat(ts_code=ts_code, fields='pledge_ratio')
            if df is not None and not df.empty:
                return float(df.iloc[0]['pledge_ratio'])
        except Exception:
            pass
        return 0.0

    # ---------- 机构覆盖 ----------

    @staticmethod
    def get_institutional_coverage(symbol: str, months: int = 3) -> dict:
        """
        近N个月机构研报覆盖数
        返回: {org_count: int, report_count: int}
        注意: report_rc 接口限制 10次/小时，无数据/限频时返回空 {}
        """
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        try:
            pro = _get_pro()
            start = (datetime.now() - timedelta(days=months * 30)).strftime('%Y%m%d')
            end = datetime.now().strftime('%Y%m%d')
            time.sleep(0.6)
            df = pro.report_rc(ts_code=ts_code, fields='org_name',
                               start_date=start, end_date=end)
            if df is not None and not df.empty:
                orgs = df['org_name'].dropna().unique()
                return {
                    'org_count': len(orgs),
                    'report_count': len(df),
                }
        except Exception:
            pass
        return {}

    # ---------- ST检查 ----------

    @staticmethod
    def is_st(symbol: str) -> bool:
        """判断是否ST/*ST"""
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        try:
            pro = _get_pro()
            df = pro.stock_basic(ts_code=ts_code, fields='name')
            if df is not None and not df.empty:
                name = str(df.iloc[0]['name'])
                return 'ST' in name or '*ST' in name or '退' in name
        except Exception:
            pass
        return False

    # ---------- 综合质地评估 ----------

    @staticmethod
    def get_stock_quality(symbol: str) -> dict:
        """
        个股质地全面检查（四层漏斗第三层）
        返回所有指标 + 每项是否达标，调用方用结果做过滤或展示
        """
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        pro = _get_pro()
        result = {}

        # --- 1) fina_indicator: ROE/扣非利润/负债率（多期） ---
        try:
            df = pro.fina_indicator(
                ts_code=ts_code,
                fields='end_date,roe,eps,bps,profit_dedt,debt_to_assets'
            )
            if df is not None and not df.empty:
                # 去重
                df = df.drop_duplicates(subset='end_date').reset_index(drop=True)
                latest = df.iloc[0]
                result['roe'] = float(latest['roe']) if latest['roe'] else None
                result['eps'] = float(latest['eps']) if latest['eps'] else None
                result['bvps'] = float(latest['bps']) if latest['bps'] else None
                result['profit_dedt'] = float(latest['profit_dedt']) if latest['profit_dedt'] else None
                result['debt_to_assets'] = float(latest['debt_to_assets']) if latest['debt_to_assets'] else None

                # ROE检查: 优先连续三年>10%, 周期股看五年均值
                annual = df[df['end_date'].str.endswith('1231')]
                roes = []
                for _, row in annual.head(5).iterrows():
                    if row['roe'] is not None:
                        roes.append(float(row['roe']))
                if len(roes) >= 3:
                    result['roe_3y_min'] = min(roes[:3])
                    result['roe_3y_avg'] = sum(roes[:3]) / len(roes[:3])
                    result['roe_3y_ok'] = all(r > 10 for r in roes[:3])
                if len(roes) >= 5:
                    result['roe_5y_min'] = min(roes)
                    result['roe_5y_avg'] = sum(roes) / len(roes)
                    result['roe_5y_ok'] = sum(roes) / len(roes) > 10

                # 扣非利润同比（同季度 vs 去年同期）
                if len(df) >= 4:
                    q_now, q_lastyr = df.iloc[0], df.iloc[3]
                    def _g(r, f):
                        return float(r[f]) if r[f] else None
                    pn, pp = _g(q_now, 'profit_dedt'), _g(q_lastyr, 'profit_dedt')
                    if pn and pp and pp:
                        result['profit_yoy'] = (pn / pp - 1) * 100
        except Exception:
            pass

        # --- 2) income: 营收（累计值，需去重+单季还原） ---
        try:
            df = pro.income(ts_code=ts_code, fields='end_date,revenue')
            if df is not None and not df.empty:
                df = df.drop_duplicates(subset='end_date').sort_values('end_date', ascending=False).reset_index(drop=True)
                def _rev(row):
                    return float(row['revenue']) if row['revenue'] else None
                result['revenue'] = _rev(df.iloc[0])

                # 单季营收: 累计值减去上一期累计值
                if len(df) >= 2:
                    r0, r1 = _rev(df.iloc[0]), _rev(df.iloc[1])
                    if r0 and r1:
                        result['revenue_single_q'] = r0 - r1 if r0 > r1 else r0
                        result['revenue_qoq'] = (result['revenue_single_q'] / r1 - 1) * 100
                # 年度营收同比
                annual = df[df['end_date'].str.endswith('1231')]
                if len(annual) >= 2:
                    r_this, r_prev = _rev(annual.iloc[0]), _rev(annual.iloc[1])
                    if r_this and r_prev:
                        result['revenue_yoy'] = (r_this / r_prev - 1) * 100
        except Exception:
            pass

        # --- 3) balancesheet: 总资产/商誉 ---
        try:
            df = pro.balancesheet(ts_code=ts_code, fields='end_date,total_assets,goodwill')
            if df is not None and not df.empty:
                df = df.drop_duplicates(subset='end_date').reset_index(drop=True)
                latest = df.iloc[0]
                result['total_assets'] = float(latest['total_assets']) if latest['total_assets'] else None
                # 商誉: 处理 NaN
                gw = latest['goodwill']
                if gw is not None and not (isinstance(gw, float) and math.isnan(gw)):
                    result['goodwill'] = float(gw)

                # 商誉/净资产
                if result.get('goodwill') and result.get('total_assets') and result.get('debt_to_assets') is not None:
                    na = result['total_assets'] * (1 - result['debt_to_assets'] / 100)
                    result['goodwill_ratio'] = result['goodwill'] / na * 100 if na > 0 else 0
                else:
                    result['goodwill_ratio'] = 0  # 无商誉=安全
        except Exception:
            pass

        # --- 质押率 ---
        result['pledge_ratio'] = TushareProvider.get_pledge_ratio(symbol)

        # --- 机构覆盖（可选，受限接口可能拿不到） ---
        coverage = TushareProvider.get_institutional_coverage(symbol)
        if coverage:
            result['org_count'] = coverage.get('org_count', 0)
            result['report_count'] = coverage.get('report_count', 0)

        # --- ST检查 ---
        result['is_st'] = TushareProvider.is_st(symbol)

        # --- 综合结论 ---
        checks = {}
        checks['st_excluded'] = not result.get('is_st', False)
        checks['pledge_excluded'] = result.get('pledge_ratio', 0) <= 50
        checks['goodwill_excluded'] = result.get('goodwill_ratio', 0) <= 30
        checks['unlock_excluded'] = True  # 解禁暂缺，默认通过
        checks['roe_qualified'] = result.get('roe_3y_ok', False) or result.get('roe_5y_ok', False)
        # 营收增长: 三级判断
        rev_yoy = result.get('revenue_yoy')
        if rev_yoy is None:
            checks['revenue_growing'] = True
        elif rev_yoy > 0:
            checks['revenue_growing'] = True
        elif rev_yoy > -15:
            roe_val = result.get('roe', 0) or 0
            debt = result.get('debt_to_assets', 100) or 100
            gw_ratio = result.get('goodwill_ratio', 0) or 0
            checks['revenue_growing'] = (roe_val > 15 and debt < 50 and gw_ratio < 15)
        else:
            checks['revenue_growing'] = False
        # 机构覆盖: 有数据时>=2家算通过, 无数据时(org_count不存在)不拦截
        oc = result.get('org_count')
        checks['institution_covered'] = oc >= 2 if oc is not None else True
        result['checks'] = checks
        result['all_passed'] = all(checks.values())
        return result

    # ---------- 资金流向 ----------

    @staticmethod
    def get_stock_moneyflow(symbol: str) -> dict:
        """
        个股资金流向（最近一天）
        字段: 主力净流入, 超大单净流入, 大单净流入, 中单净流入, 小单净流入
        """
        pro = _get_pro()
        ts_code = symbol + '.SZ' if not symbol.startswith('6') else symbol + '.SH'
        try:
            df = pro.moneyflow(ts_code=ts_code,
                               fields='trade_date,net_mf_amount,buy_elg_amount,'
                                      'sell_elg_amount,buy_lg_amount,sell_lg_amount,'
                                      'buy_md_amount,sell_md_amount,'
                                      'buy_sm_amount,sell_sm_amount')
            if df is not None and not df.empty:
                latest = df.iloc[0]
                net = float(latest.get('net_mf_amount', 0) or 0)
                buy_elg = float(latest.get('buy_elg_amount', 0) or 0)
                sell_elg = float(latest.get('sell_elg_amount', 0) or 0)
                buy_lg = float(latest.get('buy_lg_amount', 0) or 0)
                sell_lg = float(latest.get('sell_lg_amount', 0) or 0)
                buy_md = float(latest.get('buy_md_amount', 0) or 0)
                sell_md = float(latest.get('sell_md_amount', 0) or 0)
                buy_sm = float(latest.get('buy_sm_amount', 0) or 0)
                sell_sm = float(latest.get('sell_sm_amount', 0) or 0)
                return {
                    "主力净流入": round(net, 2),
                    "超大单净流入": round(buy_elg - sell_elg, 2),
                    "大单净流入": round(buy_lg - sell_lg, 2),
                    "中单净流入": round(buy_md - sell_md, 2),
                    "小单净流入": round(buy_sm - sell_sm, 2),
                    "日期": str(latest.get('trade_date', '')),
                }
        except Exception:
            pass
        return {}

    # ---------- 概念板块 ----------

    @staticmethod
    def get_concept_boards() -> list:
        """
        获取概念板块列表及涨跌幅
        返回: [{"板块名称": str, "涨跌幅": float}, ...]
        """
        pro = _get_pro()
        try:
            df = pro.concept(fields='code,name')
            if df is not None and not df.empty:
                results = []
                # Tushare concept 不直接返回涨跌幅, 只返回概念列表
                for _, row in df.iterrows():
                    results.append({
                        "板块名称": str(row.get('name', '')),
                        "涨跌幅": 0,
                    })
                return results
        except Exception:
            pass
        return []

    # ---------- 筹码分布（5000档可用） ----------

    @staticmethod
    def get_chip_distribution(ts_code: str, date: str = None) -> dict:
        """
        获取个股筹码分布，返回关键密集区信息
        ts_code: 如 '002050.SZ'
        date: YYYYMMDD，默认最近交易日
        返回: {
          "dense_below": [{"price": float, "percent": float}, ...],  # 当前价下方密集区
          "dense_above": [{"price": float, "percent": float}, ...],  # 当前价上方密集区
          "max_dense_below": {"price": float, "percent": float},     # 下方最大密集峰
          "max_dense_above": {"price": float, "percent": float},     # 上方最大密集峰
          "current_price": float,                                     # 当前价
          "chip_support": float,                                       # 密集支撑位
          "chip_resistance": float,                                    # 密集阻力位
        }
        """
        pro = _get_pro()
        try:
            df = pro.cyq_chips(ts_code=ts_code, date=date or datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                return {}
        except Exception:
            return {}

        # 获取当前价
        try:
            price_df = pro.daily(ts_code=ts_code, fields='close', limit=1)
            current_price = float(price_df.iloc[0]['close']) if price_df is not None and not price_df.empty else None
        except Exception:
            current_price = None

        if current_price is None:
            return {}

        # 按价格排序
        df = df.sort_values('price')
        all_prices = []
        for _, row in df.iterrows():
            p, pr = float(row['price']), float(row['percent'])
            if pr > 0.5:  # 过滤 < 0.5% 的噪声
                all_prices.append({'price': p, 'percent': pr})

        below = [x for x in all_prices if x['price'] <= current_price]
        above = [x for x in all_prices if x['price'] > current_price]

        max_below = max(below, key=lambda x: x['percent']) if below else None
        max_above = max(above, key=lambda x: x['percent']) if above else None

        return {
            "dense_below": below[-5:] if len(below) > 5 else below,
            "dense_above": above[:5] if len(above) > 5 else above,
            "max_dense_below": max_below,
            "max_dense_above": max_above,
            "current_price": current_price,
            "chip_support": max_below['price'] if max_below else None,
            "chip_resistance": max_above['price'] if max_above else None,
        }

    # ---------- 便捷方法 ----------

    @staticmethod
    def health_check() -> bool:
        """测试 Tushare 连接是否正常"""
        pro = _get_pro()
        try:
            df = pro.stock_basic(exchange='', list_status='L', fields='ts_code', limit=1)
            return df is not None and not df.empty
        except Exception:
            return False
