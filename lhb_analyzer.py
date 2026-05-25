"""
龙虎榜分析模块
数据源: Tushare Pro top_list + top_inst
额度: top_list 1次/天 ≈ 80条, top_inst 按需调用
"""
import time
from datetime import datetime, timedelta
from typing import Optional

_cache = {}
_cache_ttl = 7200  # 2h


def _get_pro():
    import tushare as ts, json, os
    cfg_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    ts.set_token(cfg['_tushare']['token'])
    return ts.pro_api()


def _last_trade_date() -> str:
    """本地估算最近交易日"""
    today = datetime.now()
    wd = today.weekday()
    if wd == 5:
        today -= timedelta(days=1)
    elif wd == 6:
        today -= timedelta(days=2)
    return today.strftime('%Y%m%d')


def _cached(key: str, func, ttl: int = _cache_ttl):
    now = time.time()
    entry = _cache.get(key)
    if entry and (now - entry['timestamp']) < ttl:
        return entry['data']
    data = func()
    _cache[key] = {'data': data, 'timestamp': now}
    return data


def fetch_top_list(trade_date: str = None) -> dict:
    """
    获取单日龙虎榜，返回 {ts_code: {...}} 字典
    API: top_list, 1次调用 ≈ 80条, 约2000次/天额度
    """
    if trade_date is None:
        trade_date = _last_trade_date()

    def _fetch():
        pro = _get_pro()
        df = pro.top_list(trade_date=trade_date)
        if df is None or df.empty:
            return {}
        result = {}
        for _, row in df.iterrows():
            code = row['ts_code']
            result[code] = {
                'trade_date': row['trade_date'],
                'name': row['name'],
                'close': row['close'],
                'pct_change': row['pct_change'],
                'turnover_rate': row['turnover_rate'],
                'amount': row['amount'],
                'l_buy': row['l_buy'],
                'l_sell': row['l_sell'],
                'l_amount': row['l_amount'],
                'net_amount': row['net_amount'],
                'net_rate': row['net_rate'],
                'reason': row['reason'],
            }
        return result

    return _cached(f"top_list_{trade_date}", _fetch)


def fetch_recent(days: int = 5) -> dict:
    """
    获取最近 N 个交易日的龙虎榜数据
    返回 {ts_code: {trade_date, net_amount, ...}}
    如果同一只股多日上榜, 保留最新一天
    """
    tdate = _last_trade_date()
    results = {}
    for i in range(days * 2):  # 最多推 10 个自然日
        d = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
        wd = datetime.strptime(d, '%Y%m%d').weekday()
        if wd >= 5:
            continue
        day_data = fetch_top_list(d)
        if day_data:
            for code, info in day_data.items():
                if code not in results:
                    results[code] = info
        if len(results) > 0 and i >= days:
            break
    return results


def fetch_stock_detail(ts_code: str, trade_date: str = None) -> list:
    """
    获取个股龙虎榜席位明细
    API: top_inst, 返回席位列表 [{exalter, buy, sell, net_buy, side}]
    """
    if trade_date is None:
        trade_date = _last_trade_date()

    def _fetch():
        pro = _get_pro()
        df = pro.top_inst(trade_date=trade_date)
        if df is None or df.empty:
            return []
        df_stock = df[df['ts_code'] == ts_code]
        result = []
        for _, row in df_stock.iterrows():
            side = str(row.get('side', '0'))
            exalter = str(row.get('exalter', ''))
            result.append({
                'exalter': exalter,
                'buy': float(row.get('buy', 0) or 0),
                'sell': float(row.get('sell', 0) or 0),
                'net_buy': float(row.get('net_buy', 0) or 0),
                'is_inst': '机构专用' in exalter,
                'is_north': '股通' in exalter,
                'buy_rate': float(row.get('buy_rate', 0) or 0),
                'sell_rate': float(row.get('sell_rate', 0) or 0),
            })
        return result

    return _cached(f"top_inst_{trade_date}", _fetch)


def classify_investors(detail: list) -> dict:
    """
    将龙虎榜席位明细分类汇总
    返回 {inst: {buy, sell, net}, north: {buy, sell, net}, hot_spring: {buy, sell, net}, other: {...}}
    """
    cats = {'inst': {'buy': 0, 'sell': 0, 'net': 0, 'count': 0},
            'north': {'buy': 0, 'sell': 0, 'net': 0, 'count': 0},
            'hot_spring': {'buy': 0, 'sell': 0, 'net': 0, 'count': 0},
            'other': {'buy': 0, 'sell': 0, 'net': 0, 'count': 0}}

    for item in detail:
        if item['is_inst']:
            cat = 'inst'
        elif item['is_north']:
            cat = 'north'
        elif '拉萨' in item['exalter']:
            cat = 'hot_spring'  # 东方财富拉萨团结路/东环路 = 散户大本营
        else:
            cat = 'other'

        cats[cat]['buy'] += item['buy']
        cats[cat]['sell'] += item['sell']
        cats[cat]['net'] += item['net_buy']
        cats[cat]['count'] += 1

    return cats


def analyze_stock(ts_code: str) -> Optional[dict]:
    """
    一站式分析: 某只股票最近是否上龙虎榜 + 资金结构
    返回 {on_list: bool, top_list: {...}, detail: [...], classified: {...}}
    """
    recent = fetch_recent(days=5)
    entry = recent.get(ts_code)
    if not entry:
        return None

    detail = fetch_stock_detail(ts_code, entry['trade_date'])
    classified = classify_investors(detail) if detail else {}

    return {
        'on_list': True,
        'trade_date': entry['trade_date'],
        'top_list': entry,
        'detail': detail,
        'classified': classified,
    }


def enrich_stock_list(stock_codes: list) -> dict:
    """
    批量查询: 给一组股票代码, 返回 {ts_code: lhb_info}
    只查一次 top_list, 内存匹配 — 0 额外 API 调用
    """
    recent = fetch_recent(days=5)
    result = {}
    for code in stock_codes:
        if code in recent:
            result[code] = recent[code]
    return result


# ══════════════════════════════════════════
# 席位溢价分析（概念1）
# ══════════════════════════════════════════

# 席位标签库（按需扩展）
_SEAT_TAGS = {
    "拉萨": "散户", "东方财富": "散户",
    "机构专用": "机构",
    "沪股通": "北向", "深股通": "北向",
}


def _tag_seat(exalter: str) -> str:
    """给席位打标签"""
    for keyword, tag in _SEAT_TAGS.items():
        if keyword in exalter:
            return tag
    return "游资"  # 未匹配默认为游资


def classify_seats_detail(detail: list) -> dict:
    """
    按席位标签分类汇总（比 classify_investors 更细）
    返回 {tag: {buy, sell, net, count}}
    """
    cats = {}
    for item in detail:
        tag = _tag_seat(item["exalter"])
        if tag not in cats:
            cats[tag] = {"buy": 0, "sell": 0, "net": 0, "count": 0}
        cats[tag]["buy"] += item["buy"]
        cats[tag]["sell"] += item["sell"]
        cats[tag]["net"] += item["net_buy"]
        cats[tag]["count"] += 1
    return cats


def calc_seat_premium(exalter: str, n_days: int = 5) -> dict:
    """
    计算某席位的历史溢价（席位买入后 N 日胜率）
    需要历史 top_list + top_inst 数据
    返回 {appearances, avg_net_buy, win_rate_est, tag}
    """
    tag = _tag_seat(exalter)
    # 拉最近 30 个交易日的龙虎榜
    from datetime import timedelta, datetime
    pro = _get_pro()
    end = time.strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

    try:
        df_list = pro.top_list(start_date=start, end_date=end)
        if df_list is None or df_list.empty:
            return {"exalter": exalter, "tag": tag, "appearances": 0,
                    "avg_net_buy": 0, "win_rate_est": "数据不足"}

        # 找该席位的所有上榜记录
        df_inst = pro.top_inst(start_date=start, end_date=end)
        if df_inst is None or df_inst.empty:
            return {"exalter": exalter, "tag": tag, "appearances": 0,
                    "avg_net_buy": 0, "win_rate_est": "数据不足"}

        seat_records = df_inst[df_inst["exalter"] == exalter]
        if seat_records.empty:
            return {"exalter": exalter, "tag": tag, "appearances": 0,
                    "avg_net_buy": 0, "win_rate_est": "无记录"}

        avg_buy = float(seat_records["buy"].mean()) if "buy" in seat_records.columns else 0
        avg_sell = float(seat_records["sell"].mean()) if "sell" in seat_records.columns else 0
        total_appearances = len(seat_records)

        # 简化胜率估算：净买入为正的比例
        if "net_buy" in seat_records.columns:
            wins = (seat_records["net_buy"] > 0).sum()
            win_rate = wins / total_appearances * 100
        else:
            win_rate = None

        return {
            "exalter": exalter,
            "tag": tag,
            "appearances": total_appearances,
            "avg_buy": round(avg_buy, 2),
            "avg_sell": round(avg_sell, 2),
            "avg_net_buy": round(avg_buy - avg_sell, 2),
            "win_rate_est": f"{win_rate:.0f}%" if win_rate else "数据不足",
        }
    except Exception as e:
        return {"exalter": exalter, "tag": tag, "appearances": 0,
                "avg_net_buy": 0, "win_rate_est": f"错误: {e}"}


def batch_seat_analysis(detail: list) -> list:
    """批量分析一批席位的历史溢价"""
    seen = set()
    results = []
    for item in detail:
        exalter = item["exalter"]
        if exalter in seen:
            continue
        seen.add(exalter)
        results.append(calc_seat_premium(exalter))
    return results
