"""
新闻舆情服务：为威科夫信号提供新闻上下文
数据源：东方财富个股新闻 + 财联社快讯（均免费，无需 API Key）
"""
import logging
import re
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================
# 东方财富个股新闻
# ============================================================

def get_stock_news(symbol: str, limit: int = 3):
    """
    获取指定股票的近期新闻

    Args:
        symbol: 股票代码，如 '002050', '600519'
        limit: 返回条数

    Returns:
        [{"title": str, "date": str, "source": str, "url": str}, ...]
    """
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return []
        # 列: 关键词, 新闻标题, 摘要内容, 发布时间, 文章来源, 文章链接
        news_list = []
        for _, row in df.head(limit).iterrows():
            news_list.append({
                "title": str(row.iloc[1]),
                "date": str(row.iloc[3])[:10],
                "source": str(row.iloc[4]),
                "url": str(row.iloc[5]),
            })
        return news_list
    except Exception as e:
        logger.debug(f"stock_news_em({symbol}) failed: {e}")
        return []


# ============================================================
# 财联社大盘新闻
# ============================================================

def get_market_news(limit: int = 5):
    """
    获取财联社大盘快讯（市场热点、宏观政策）

    Returns:
        [{"title": str, "summary": str, "url": str}, ...]
    """
    try:
        import akshare as ak
        df = ak.stock_info_global_cls()
        if df is None or df.empty:
            return []
        news_list = []
        for _, row in df.head(limit).iterrows():
            content = str(row.iloc[1])
            # 截取前80字作为摘要
            summary = content[:80] + "..." if len(content) > 80 else content
            news_list.append({
                "title": str(row.iloc[0]),
                "summary": summary,
                "time": str(row.iloc[3]) if len(row) > 3 else "",
            })
        return news_list
    except Exception as e:
        logger.debug(f"get_market_news failed: {e}")
        return []


# ============================================================
# 概念板块新闻热度（基于东方财富概念板块新闻）
# ============================================================

def get_top_signal_news(signal_results, limit_per_stock=2):
    """
    为所有有信号的股票批量获取新闻

    Args:
        signal_results: scanner results 中筛选出的有信号股票列表
        limit_per_stock: 每只股票取几条新闻

    Returns:
        {code: [news_item, ...], ...}
    """
    result = {}
    for r in signal_results:
        # code格式 "sz.002050" → "002050"
        code = r["code"].split(".")[1] if "." in r["code"] else r["code"]
        news = get_stock_news(code, limit_per_stock)
        if news:
            result[r["code"]] = news
    return result


def format_news_block(news_dict, stock_name_map):
    """
    将新闻数据格式化为字符串，供 run.py 输出

    Args:
        news_dict: {code: [news_item, ...], ...}
        stock_name_map: {code: name, ...}

    Returns:
        格式化的新闻文本
    """
    if not news_dict:
        return ""
    lines = ["  新闻舆情参考"]
    lines.append(f"  {'─'*62}")
    for code, items in news_dict.items():
        sym = code.split(".")[1]
        name = stock_name_map.get(code, sym)
        lines.append(f"  [{name} {sym}]")
        for item in items[:2]:
            title = item["title"][:50]
            date = item["date"]
            src = item["source"][:6]
            lines.append(f"    {title:<52} {date} {src}")
    return "\n".join(lines)


def format_market_news_block(news_list):
    """
    将大盘新闻格式化为字符串
    """
    if not news_list:
        return ""
    lines = ["  【市场快讯】"]
    for item in news_list:
        lines.append(f"  · {item['summary']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    print("=== 三花智控新闻 ===")
    for n in get_stock_news("002050", 3):
        print(f"  {n['date']} | {n['source']} | {n['title'][:40]}")

    print("\n=== 财联社快讯 ===")
    for n in get_market_news(3):
        print(f"  {n['title']} | {n['summary'][:50]}...")
