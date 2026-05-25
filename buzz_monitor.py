"""
热度增长率监控（概念2）
数据源: 东方财富股吧综合评分（综合得分/关注指数/排名）+ 关键词热度

功能:
- 个股综合热度评分
- 热度增长率（综合得分偏离均值程度）
- 热度过热预警（高分 + 高关注 + 大涨）
"""
import time

_CACHE = {}
_CACHE_TTL = 180

try:
    import akshare as ak
    _AK = True
except ImportError:
    _AK = False


def _cached(key, func, ttl=None):
    ttl = ttl or _CACHE_TTL
    now = time.time()
    entry = _CACHE.get(key)
    if entry and (now - entry["ts"]) < ttl:
        return entry["data"]
    data = func()
    _CACHE[key] = {"data": data, "ts": now}
    return data


def _get_comment_scores():
    """全量股吧评分数据（缓存180s）"""
    if not _AK:
        return []
    def _fetch():
        try:
            df = ak.stock_comment_em()
            if df is None or df.empty:
                return []
            result = []
            for _, row in df.iterrows():
                result.append({
                    "code": str(row["代码"]),
                    "name": str(row["名称"]),
                    "score": float(row["综合得分"]) if row["综合得分"] else 0,
                    "attention": float(row["关注指数"]) if row["关注指数"] else 0,
                    "rank": int(row["目前排名"]) if row["目前排名"] else 9999,
                    "pct_chg": float(row["涨跌幅"]) if row["涨跌幅"] else 0,
                })
            return result
        except Exception:
            return []
    return _cached("comment_scores", _fetch, ttl=120)


def get_hot_rank(top_n: int = 30) -> list:
    """按排名排序的热榜（排名越靠前越热）"""
    raw = _get_comment_scores()
    if not raw:
        return []
    sorted_data = sorted(raw, key=lambda x: x["rank"])
    return sorted_data[:top_n]


def get_stock_buzz(code: str) -> dict:
    """个股综合热度"""
    raw = _get_comment_scores()
    if not raw:
        return {"code": code, "score": 0, "rank": 9999, "attention": 0, "sentiment": "未知"}

    # 匹配个股
    clean = code.replace(".", "").replace("SH", "").replace("SZ", "")
    found = None
    for s in raw:
        sc = s["code"].replace(".", "").replace("SH", "").replace("SZ", "")
        if clean == sc:
            found = s
            break

    if not found:
        return {"code": code, "score": 0, "rank": 9999, "attention": 0, "sentiment": "无数据"}

    # 计算热度分
    buzz_score = found["score"] * 0.6 + found["attention"] * 0.4

    if buzz_score >= 75:
        sentiment = "过热"
    elif buzz_score >= 55:
        sentiment = "偏热"
    else:
        sentiment = "正常"

    return {
        "code": found["code"],
        "name": found["name"],
        "score": round(buzz_score, 1),
        "rank": found["rank"],
        "attention": round(found["attention"], 1),
        "pct_chg": found["pct_chg"],
        "sentiment": sentiment,
    }


def get_hot_keywords(top_stock: str = None) -> list:
    """热门关键词热度"""
    if not _AK:
        return []
    def _fetch():
        try:
            df = ak.stock_hot_keyword_em()
            if df is None or df.empty:
                return []
            result = []
            for _, row in df.iterrows():
                result.append({
                    "code": row["股票代码"],
                    "concept": row["概念名称"],
                    "heat": int(row["热度"]),
                })
            return result
        except Exception:
            return []
    data = _cached("hot_keywords", _fetch, ttl=180)
    if top_stock:
        return [d for d in data if top_stock in d["code"]]
    return data


def get_buzz_growth(top_n: int = 10) -> list:
    """热度增长率：综合分偏离均值比例
    用当前综合分与全市场均值的比值衡量"热度异常程度"
    """
    raw = _get_comment_scores()
    if not raw or len(raw) < 50:
        return []
    sorted_by_score = sorted(raw, key=lambda x: x["score"], reverse=True)
    mean_score = sum(s["score"] for s in raw) / len(raw)
    std_score = (sum((s["score"] - mean_score)**2 for s in raw) / len(raw))**0.5

    results = []
    for s in sorted_by_score[:30]:
        z = (s["score"] - mean_score) / std_score if std_score > 0 else 0
        signal = "过热" if z > 3 else ("偏热" if z > 2 else ("略热" if z > 1.5 else ""))
        if signal:
            results.append({
                "code": s["code"],
                "name": s["name"],
                "rank": s["rank"],
                "score": s["score"],
                "attention": s["attention"],
                "z_score": round(z, 1),
                "signal": signal,
                "pct_chg": s["pct_chg"],
            })
    return results[:top_n]


def scan_overheat(top_n: int = 20) -> list:
    """过热预警：综合分 ≥ 70 + 排名前30 + 涨幅异常 ≥ 5%
    或者综合分 ≥ 80 + 排名前10
    """
    raw = _get_comment_scores()
    if not raw:
        return []
    sorted_data = sorted(raw, key=lambda x: x["rank"])

    alerts = []
    for s in sorted_data[:top_n]:
        if s["score"] >= 80 and s["rank"] <= 10:
            alerts.append({
                "code": s["code"], "name": s["name"],
                "rank": s["rank"], "score": s["score"],
                "pct_chg": s["pct_chg"], "alert": "过热预警",
            })
        elif s["score"] >= 72 and s["rank"] <= 15 and abs(s["pct_chg"]) >= 7:
            alerts.append({
                "code": s["code"], "name": s["name"],
                "rank": s["rank"], "score": s["score"],
                "pct_chg": s["pct_chg"], "alert": "过热预警",
            })
        elif s["score"] >= 75 and s["rank"] <= 20:
            alerts.append({
                "code": s["code"], "name": s["name"],
                "rank": s["rank"], "score": s["score"],
                "pct_chg": s["pct_chg"], "alert": "关注",
            })
    return alerts


if __name__ == "__main__":
    print("═══ 热榜 Top 10 ═══")
    for s in get_hot_rank(10):
        print(f"  #{s['rank']} {s['name']} ({s['code']}) 综合{s['score']:.0f} 关注{s['attention']:.0f} {s['pct_chg']:+.2f}%")

    print("\n═══ 热度增长异常 ═══")
    for s in get_buzz_growth(5):
        print(f"  {s['name']} ({s['code']}): {s['signal']} (z={s['z_score']})")

    print("\n═══ 过热预警 ═══")
    for a in scan_overheat(20):
        print(f"  {a['name']} ({a['code']}): {a['alert']} ({a['score']:.0f}分, #{a['rank']})")
