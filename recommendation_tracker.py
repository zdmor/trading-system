"""
推荐回顾追踪器
保存每次扫描结果，支持回顾昨日/5日前/10日前推荐的个股表现。
"""
import json
import os
import requests
from datetime import datetime, timedelta

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "recommendation_history.json")


def _load():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(data):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_scan_result(results, pool, holdings, recommendations, indices, top_n=15):
    """保存当日扫描结果到历史记录"""
    today = datetime.now().strftime("%Y-%m-%d")

    # 推荐的票 = 质地通过+有信号的，保存推荐价
    recs = []
    for r in recommendations[:10]:
        code = r["code"].split(".")[1] if "." in r["code"] else r["code"]
        recs.append({
            "code": code,
            "name": r.get("name", ""),
            "signal": r["wyckoff_sig"],
            "score": r["wyckoff_score"],
            "price": r.get("price", 0),
            "suggest": _suggest_label(r),
        })

    # Top N 全部保存（用于全览）
    tops = []
    for r in results[:top_n]:
        code = r["code"].split(".")[1] if "." in r["code"] else r["code"]
        tops.append({
            "code": code,
            "name": r.get("name", ""),
            "signal": r["wyckoff_sig"],
            "score": r["wyckoff_score"],
            "price": r.get("price", 0),
            "trend": r.get("trend", ""),
            "quality": r.get("quality_passed"),
        })

    # 股票池
    pools = []
    for s in pool:
        pools.append({
            "code": s.get("code", ""),
            "name": s.get("name", ""),
            "signal": s["wyckoff_sig"],
            "score": s["wyckoff_score"],
            "price": s.get("price", 0),
            "trend": s.get("trend", ""),
            "phase": s.get("phase", ""),
        })

    # 持仓
    holds = []
    for s in holdings:
        holds.append({
            "code": s.get("code", ""),
            "name": s.get("name", ""),
            "signal": s["wyckoff_sig"],
            "score": s["wyckoff_score"],
            "price": s.get("price", 0),
            "cost": s.get("cost", 0),
            "pnl": s.get("pnl_pct"),
            "suggest": s.get("suggest", ""),
        })

    # 市值风云研究股票池
    research_cfg_path = os.path.join(os.path.dirname(__file__), "research_watchlist.json")
    research_stocks = []
    if os.path.exists(research_cfg_path):
        try:
            with open(research_cfg_path, encoding="utf-8") as f:
                rw = json.load(f)
            # 对每一只研究票，检查它在本次扫描中的状态
            for item in rw.get("stocks", []):
                if not item.get("active", True):
                    continue
                code = item["code"]
                name = item.get("name", "")
                # 在 Top N 中找
                found = None
                for r in results:
                    c = r["code"].split(".")[1] if "." in r["code"] else r["code"]
                    if c == code:
                        found = r
                        break
                if found:
                    research_stocks.append({
                        "code": code,
                        "name": name,
                        "signal": found["wyckoff_sig"],
                        "score": found["wyckoff_score"],
                        "price": found.get("price", 0),
                        "trend": found.get("trend", ""),
                        "quality": found.get("quality_passed"),
                        "in_top": True,
                    })
                else:
                    # 未进入分析范围，查实时快照
                    bs_code = f"sh.{code}" if code.startswith("6") else f"sz.{code}"
                    snap = {}
                    for r in results:
                        if r["code"] == bs_code:
                            snap = r
                            break
                    if snap:
                        research_stocks.append({
                            "code": code, "name": name,
                            "signal": snap.get("wyckoff_sig", "-"),
                            "score": snap.get("wyckoff_score", 0),
                            "price": snap.get("price", 0),
                            "trend": snap.get("trend", ""),
                            "quality": snap.get("quality_passed"),
                            "in_top": False,
                        })
                    else:
                        research_stocks.append({
                            "code": code, "name": name,
                            "signal": "-", "score": 0,
                            "price": 0, "trend": "-",
                            "quality": None, "in_top": False,
                        })
        except Exception:
            pass

    # 大盘
    market = {}
    if indices:
        for k, v in indices.items():
            market[k] = {"name": v.get("name", ""), "price": v.get("price", 0),
                         "change_pct": v.get("change_pct", 0)}

    history = _load()
    history[today] = {
        "market": market,
        "recommendations": recs,
        "top": tops,
        "pool": pools,
        "holdings": holds,
        "research": research_stocks,
        "update_time": datetime.now().strftime("%H:%M"),
    }
    _save(history)


def _suggest_label(r):
    """从扫描结果推断建议标签"""
    sig = r["wyckoff_sig"]
    trend = r.get("trend", "")
    if sig in ("Spring", "SOS", "LPS"):
        return "关注"
    if sig == "Upthrust":
        return "不追/减仓"
    if trend == "空头":
        return "回避"
    return "观察"


def _fetch_current_prices(codes):
    """批量获取当前价格，返回 {code: price}"""
    if not codes:
        return {}
    # 腾讯 API 格式: sh600183 / sz300857 (无点号)
    grouped = []
    for c in codes:
        prefix = "sh" if c.startswith("6") else "sz"
        grouped.append(f"{prefix}{c}")
    url = f"http://qt.gtimg.cn/q={','.join(grouped)}"
    result = {}
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        for line in r.text.strip().split(";"):
            line = line.strip()
            if "=" not in line or "none_match" in line:
                continue
            # key 是 v_sz300857 格式, 去掉 v_ 前缀
            key = line.split("=")[0].replace("v_", "")
            parts = line.split("=", 1)[1].strip('"').split("~")
            if len(parts) < 4:
                continue
            code = parts[2]  # 纯数字代码
            price = float(parts[3]) if parts[3] else 0
            change_pct = float(parts[32]) if len(parts) > 32 and parts[32] else 0
            name = parts[1] if len(parts) > 1 else ""
            result[code] = {"price": price, "change_pct": change_pct, "name": name}
    except Exception:
        pass
    return result


def _find_business_days(history, target_days):
    """在历史记录中找到距离今天 target_days 个交易日的历史日期列表
    返回 [(date_str, label), ...]
    """
    today = datetime.now().date()
    dates = sorted(history.keys(), reverse=True)

    result = []
    for d in dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            continue
        if dt >= today:
            continue  # skip today and future
        diff = (today - dt).days
        if diff in target_days or _is_approximate(diff, target_days):
            result.append((d, f"{diff}日前"))
    return result


def _is_approximate(diff, target_days):
    """允许1天的偏差（处理周末/节假日）"""
    for t in target_days:
        if abs(diff - t) <= 1:
            return True
    return False


def print_review():
    """打印推荐回顾：昨日/5日前/10日前"""
    history = _load()
    if not history:
        print("  【推荐回顾】无历史记录（首次运行）")
        print()
        return

    today = datetime.now().strftime("%Y-%m-%d")
    dates = sorted(history.keys(), reverse=True)

    # 找三个时间点: 最近的非今天记录, 大约5日前, 大约10日前
    review_dates = []
    seen = set()
    for d in dates:
        if d == today:
            continue
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
            td = datetime.now().date()
            diff = (td - dt).days
        except ValueError:
            continue

        # 归类
        bucket = None
        if diff <= 2:
            bucket = ("昨日", 0)
        elif 4 <= diff <= 7:
            bucket = ("5日前", 1)
        elif 9 <= diff <= 14:
            bucket = ("10日前", 2)

        if bucket and bucket[1] not in seen:
            review_dates.append((d, bucket[0]))
            seen.add(bucket[1])

    if not review_dates:
        print("  【推荐回顾】无足够历史数据")
        print()
        return

    print(f"  【推荐回顾】")
    print()

    # 收集所有需要查询的股票代码
    all_codes = set()
    for date_str, _ in review_dates:
        day_data = history[date_str]
        for rec in day_data.get("recommendations", []):
            all_codes.add(rec["code"])
        for h in day_data.get("holdings", []):
            all_codes.add(h["code"])
        for p in day_data.get("pool", []):
            all_codes.add(p["code"])
        for rs in day_data.get("research", []):
            all_codes.add(rs["code"])

    current_prices = _fetch_current_prices(list(all_codes))

    for date_str, label in sorted(review_dates, key=lambda x: x[0], reverse=True):
        day_data = history[date_str]
        recs = day_data.get("recommendations", [])

        # 提炼市场背景
        market_info = ""
        if day_data.get("market"):
            names = []
            for v in day_data["market"].values():
                n = v.get("name", "")
                chg = v.get("change_pct", 0)
                names.append(f"{n}{chg:+.2f}%")
            market_info = " | ".join(names[:3])

        print(f"  【{label} {date_str}】市场: {market_info}")

        if recs:
            print(f"  { '信号':<10} {'代码':<8} {'名称':<7} {'推荐价':<8} {'现价':<8} {'涨跌幅':<8} {'结论':<12}")
            print(f"  " + "-" * 64)
            for rec in recs:
                code = rec["code"]
                name = rec.get("name", "")[:6]
                signal = f"{rec.get('signal','-')}({rec.get('score',0)})"
                old_price = rec.get("price", 0)
                suggest = rec.get("suggest", "")

                # 找现价
                current = current_prices.get(code, {})
                new_price = current.get("price", 0)

                if old_price > 0 and new_price > 0:
                    pnl = (new_price / old_price - 1) * 100
                    pnl_str = f"{pnl:+.2f}%"
                    # 表现标记
                    if pnl > 5:
                        pnl_str += " +"
                    elif pnl < -5:
                        pnl_str += " -"
                else:
                    pnl_str = "N/A"

                price_str = f"{old_price:.2f}" if old_price else "-"
                new_str = f"{new_price:.2f}" if new_price else "-"
                print(f"  {signal:<10} {code:<8} {name:<7} {price_str:<8} {new_str:<8} {pnl_str:<8} {suggest:<12}")
        else:
            print(f"  当日无推荐")

        # 持仓表现
        holdings = day_data.get("holdings", [])
        if holdings:
            print(f"  (持仓: ", end="")
            items = []
            for h in holdings:
                code = h["code"]
                current = current_prices.get(code, {})
                new_price = current.get("price", 0)
                old_price = h.get("price", 0)
                cost = h.get("cost", 0)
                if old_price > 0 and new_price > 0:
                    chg = (new_price / old_price - 1) * 100
                    items.append(f"{h['name']}{chg:+.1f}%")
                else:
                    items.append(f"{h['name']}-")
            print(" | ".join(items), end="")
            print(")")

        # 市值风云研究股表现
        research = day_data.get("research", [])
        if research:
            print(f"  (研究票: ", end="")
            items = []
            for rs in research:
                code = rs.get("code", "")
                current = current_prices.get(code, {})
                new_price = current.get("price", 0)
                old_price = rs.get("price", 0)
                name = rs.get("name", "")[:4]
                if old_price > 0 and new_price > 0:
                    chg = (new_price / old_price - 1) * 100
                    items.append(f"{name}{chg:+.1f}%")
                else:
                    items.append(f"{name}-")
            print(" | ".join(items), end="")
            print(")")
        print()

    print()
