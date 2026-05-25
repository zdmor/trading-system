"""
基本面分析系统 — 长线选股

独立于技术面评分系统，专注公司质地评估。
输出 "可交易股票池"，供技术面系统筛选入场时机。

数据源: Tushare Pro (优先) / AKShare (降级)
种子池: 市值风云研报股票池
"""
import json
import os
import sys
from datetime import datetime
from typing import Optional

from data_providers import get_stock_quality, is_st

# ─── 持久化 ───
POOL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fundamental_pool.json")

# ─── 评级 ───
RATING_PASS = "PASS"     # 基本面合格，进入技术面筛选
RATING_WATCH = "WATCH"   # 边缘，需持续观察
RATING_FAIL = "FAIL"     # 不合格
RATING_UNKNOWN = "--"    # 数据不足

# ─── 评分阈值（长线选股标准） ───
THRESHOLDS = {
    "roe_min": 10.0,            # ROE连续三年 >= 10%
    "revenue_yoy_min": 0.0,     # 营收增长 > 0%（不要求高增长，但不能萎缩）
    "debt_max": 65.0,           # 负债率 <= 65%
    "roe_single_min": 8.0,      # 最近一期ROE >= 8%（宽松）
    "gross_margin_min": 15.0,   # 毛利率 >= 15%
}

# ─── 种子池：市值风云当前覆盖股票 ───
SEED_POOL = {
    # 市值风云研报股票池 (2026-05-22)
    "000988": "华工科技",  "002648": "卫星化学",  "603713": "密尔克卫",
    "600285": "羚锐制药",  "605099": "共创草坪",  "300628": "亿联网络",
    "300274": "阳光电源",  "688120": "华海清科",  "300750": "宁德时代",
    "300308": "中际旭创",  "600415": "小商品城",  "000920": "沃顿科技",
    "300976": "达瑞电子",  "002709": "天赐材料",  "600968": "海油发展",
    "603871": "嘉友国际",  "603993": "洛阳钼业",  "002714": "牧原股份",
    "688603": "天承科技",  "002158": "汉钟精机",  "600529": "山东药玻",
    "600580": "卧龙电驱",  "600745": "闻泰科技",
    # 市值风云研报深度覆盖 (2026-05)
    "600183": "生益科技",  "688183": "生益电子",  "600150": "中国船舶",
    "000725": "京东方A",   "000100": "TCL科技",   "002064": "华峰化学",
    # 当前持仓
    "002050": "三花智控",  "600038": "中直股份",  "600416": "湘电股份",
}


def load_pool() -> dict:
    """加载股票池"""
    if os.path.exists(POOL_FILE):
        try:
            with open(POOL_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_pool(pool: dict):
    """保存股票池"""
    with open(POOL_FILE, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)


def init_pool():
    """用种子池初始化（不覆盖已有评级）"""
    pool = load_pool()
    changed = False
    for code, name in SEED_POOL.items():
        if code not in pool:
            pool[code] = {
                "name": name,
                "rating": RATING_UNKNOWN,
                "composite": 0,
                "source": "市值风云" if code not in ("002050", "600038", "600416") else "持仓",
                "added": datetime.now().strftime("%Y-%m-%d"),
                "updated": "",
                "financials": {},
                "notes": "",
            }
            changed = True
    if changed:
        save_pool(pool)
    return pool


def evaluate_stock(code: str, pool: dict) -> dict:
    """对单只股票执行基本面评估，返回更新后的 entry"""
    entry = pool.get(code, {})
    name = entry.get("name", code)

    fin = get_stock_quality(code)
    st = is_st(code)

    # ── 提取关键指标 ──
    roe_3y_ok = fin.get("roe_3y_ok", False)
    roe_3y_avg = fin.get("roe_3y_avg")
    roe_current = fin.get("roe")
    revenue_yoy = fin.get("revenue_yoy")
    debt = fin.get("debt_to_assets")
    profit_yoy = fin.get("profit_yoy")
    eps = fin.get("eps")
    bvps = fin.get("bvps")
    pe = fin.get("pe")
    pb = fin.get("pb")

    # ── 评级逻辑 ──
    reasons = []
    rating = RATING_PASS

    if st:
        rating = RATING_FAIL
        reasons.append("ST")

    if roe_3y_avg is not None and roe_3y_avg < THRESHOLDS["roe_min"]:
        if rating == RATING_PASS:
            rating = RATING_FAIL
        reasons.append(f"ROE3年均值{roe_3y_avg:.1f}%<{THRESHOLDS['roe_min']}%")
    elif roe_3y_avg is None:
        if roe_current is not None and roe_current < THRESHOLDS["roe_single_min"]:
            if rating == RATING_PASS:
                rating = RATING_WATCH
            reasons.append(f"ROE单期{roe_current:.1f}%（数据不足3年）")

    if revenue_yoy is not None and revenue_yoy < THRESHOLDS["revenue_yoy_min"]:
        if rating == RATING_PASS:
            rating = RATING_WATCH
        reasons.append(f"营收同比{revenue_yoy:.1f}%")
    elif revenue_yoy is None and roe_3y_avg is None and rating == RATING_PASS:
        rating = RATING_UNKNOWN
        reasons.append("数据不足")

    if debt is not None and debt > THRESHOLDS["debt_max"]:
        if rating == RATING_PASS:
            rating = RATING_WATCH
        reasons.append(f"负债率{debt:.1f}%>{THRESHOLDS['debt_max']}%")

    if rating == RATING_PASS and not reasons:
        reasons.append("基本面合格")

    # ── 综合评分（用于 PASS 内部排序） ──
    composite = _calc_composite_score(fin)

    # ── 更新 entry ──
    entry["name"] = name
    entry["rating"] = rating
    entry["composite"] = round(composite, 1)
    entry["updated"] = datetime.now().strftime("%Y-%m-%d")
    entry["financials"] = {
        "roe": _f(roe_current),
        "roe_3y_avg": _f(roe_3y_avg),
        "roe_3y_ok": roe_3y_ok,
        "revenue_yoy": _f(revenue_yoy),
        "profit_yoy": _f(profit_yoy),
        "debt_to_assets": _f(debt),
        "eps": _f(eps),
        "bvps": _f(bvps),
        "pe": _f(pe),
        "pb": _f(pb),
        "st": st,
    }
    entry["reasons"] = "; ".join(reasons)

    return entry


def _calc_composite_score(fin: dict) -> float:
    """综合质地评分（仅用于 PASS 内排序）"""
    score = 50.0
    roe = fin.get("roe_3y_avg") or fin.get("roe") or 0
    score += min(roe, 30) * 1.5  # ROE贡献最高45分

    rev = fin.get("revenue_yoy") or 0
    score += min(max(rev, -20), 50) * 0.3  # 营收贡献最高15分

    debt = fin.get("debt_to_assets") or 50
    score += max(0, 100 - debt) * 0.2  # 低负债贡献最高20分

    return min(100, max(0, score))


def _f(v) -> Optional[float]:
    """安全取浮点数"""
    return round(float(v), 2) if v is not None else None


def update_all(force: bool = False) -> dict:
    """更新池中所有股票的基本面评估

    Args:
        force: 是否强制刷新（默认只刷 UNKNOWN/过期的）

    Returns:
        pool dict
    """
    pool = init_pool()
    today = datetime.now().strftime("%Y-%m-%d")

    codes = sorted(pool.keys())
    total = len(codes)

    print(f"\n基本面评估: {total} 只")
    print(f"{'='*50}")

    passed = failed = watch = unknown = 0
    for i, code in enumerate(codes, 1):
        entry = pool[code]
        # 跳过已评估且未过期（一周内评估过、非强制）
        updated = entry.get("updated") or ""
        if not force and entry.get("rating") != RATING_UNKNOWN and updated >= _days_ago(7):
            # 只计数不重复请求
            r = entry.get("rating", RATING_UNKNOWN)
            if r == RATING_PASS:
                passed += 1
            elif r == RATING_FAIL:
                failed += 1
            else:
                watch += 1
            continue

        name = entry.get("name", code)
        print(f"  [{i}/{total}] {code} {name}...", end="", flush=True)
        entry = evaluate_stock(code, pool)
        pool[code] = entry
        r = entry.get("rating", RATING_UNKNOWN)
        reasons = entry.get("reasons", "")
        print(f" {r} {reasons}")

        if r == RATING_PASS:
            passed += 1
        elif r == RATING_FAIL:
            failed += 1
        else:
            watch += 1

        # Tushare 限频保护
        if i < total:
            import time
            time.sleep(0.5)

    # 更新元数据
    today_str = today
    pool["_meta"] = {
        "last_update": today_str,
        "total": total,
        "passed": passed,
        "failed": failed,
        "watch": watch,
        "version": 3,
    }
    save_pool(pool)

    print(f"\n{'='*50}")
    print(f"PASS {passed} | FAIL {failed} | WATCH {watch} | {total} 只共计")
    return pool


def _days_ago(n: int) -> str:
    """返回 N 天前的日期字符串"""
    from datetime import timedelta
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")


def print_report(pool: dict = None):
    """打印可读报告"""
    if pool is None:
        pool = load_pool()

    meta = pool.pop("_meta", {})
    print(f"\n{'='*60}")
    print(f"  基本面股票池报告")
    print(f"  更新: {meta.get('last_update', '未知')}")
    print(f"  总计: {meta.get('total', len(pool))} 只 | "
          f"PASS {meta.get('passed', '?')} | "
          f"FAIL {meta.get('failed', '?')} | "
          f"WATCH {meta.get('watch', '?')}")
    print(f"{'='*60}")

    # 按评级分组
    groups = {"PASS": [], "WATCH": [], "FAIL": [], "--": []}
    for code, entry in pool.items():
        if code == "_meta":
            continue
        rating = entry.get("rating", RATING_UNKNOWN)
        groups.setdefault(rating, []).append((code, entry))

    for rating, label, highlight in [
        ("PASS", "合格 (可交易)", True),
        ("WATCH", "待观察", False),
        ("FAIL", "不合格", False),
        (RATING_UNKNOWN, "数据不足", False),
    ]:
        items = groups.get(rating, [])
        if not items:
            continue
        print(f"\n{'─'*40}")
        print(f"  {label} ({len(items)} 只)")
        print(f"{'─'*40}")

        items.sort(key=lambda x: x[1].get("composite", 0), reverse=True)
        for code, entry in items:
            name = entry.get("name", "")
            composite = entry.get("composite", 0)
            fin = entry.get("financials", {})
            reasons = entry.get("reasons", "")
            notes = entry.get("notes", "")

            roe = fin.get("roe_3y_avg") or fin.get("roe", 0)
            roe_str = f"ROE={roe}%" if roe else ""
            rev = fin.get("revenue_yoy", 0)
            rev_str = f"营收={rev}%" if rev else ""
            debt = fin.get("debt_to_assets", 0)
            debt_str = f"负债={debt}%" if debt else ""

            line = f"  {code} {name}"
            if rating == "PASS":
                line += f" [评分{composite}]"
            line += f" | {roe_str} {rev_str} {debt_str}"
            if reasons:
                line += f" | {reasons}"
            if notes:
                line += f" | 备注: {notes}"
            print(line)
            print()

    pool["_meta"] = meta


def add_stock(code: str, name: str, source: str = "手动", notes: str = ""):
    """手动添加股票到池中"""
    pool = load_pool()
    pool[code] = {
        "name": name,
        "rating": RATING_UNKNOWN,
        "composite": 0,
        "source": source,
        "added": datetime.now().strftime("%Y-%m-%d"),
        "updated": "",
        "financials": {},
        "notes": notes,
    }
    save_pool(pool)
    print(f"已添加: {code} {name}")


def remove_stock(code: str):
    """从池中移除"""
    pool = load_pool()
    if code in pool:
        name = pool[code].get("name", code)
        del pool[code]
        save_pool(pool)
        print(f"已移除: {code} {name}")
    else:
        print(f"股票 {code} 不在池中")


def note_stock(code: str, notes: str):
    """给股票添加备注"""
    pool = load_pool()
    if code in pool:
        pool[code]["notes"] = notes
        save_pool(pool)
        print(f"备注已更新: {code} {pool[code].get('name', '')}")
    else:
        print(f"股票 {code} 不在池中")


def get_approved_list() -> list:
    """获取评级为 PASS 的股票列表（供 scanner 调用）"""
    pool = load_pool()
    return [
        {"code": code, "name": entry.get("name", ""), "score": entry.get("composite", 0)}
        for code, entry in pool.items()
        if code != "_meta" and entry.get("rating") == RATING_PASS
    ]


def export_report(pool: dict = None, filepath: str = None) -> str:
    """导出 Markdown 报告到 vault"""
    if pool is None:
        pool = load_pool()
    meta = pool.pop("_meta", {})

    lines = []
    lines.append(f"# 基本面股票池报告\n")
    lines.append(f"- **更新日期**: {meta.get('last_update', '未知')}")
    lines.append(f"- **总计**: {meta.get('total', len(pool))} 只 | "
                 f"PASS {meta.get('passed', 0)} | "
                 f"FAIL {meta.get('failed', 0)} | "
                 f"WATCH {meta.get('watch', 0)}\n")

    groups = {"PASS": [], "WATCH": [], "FAIL": [], RATING_UNKNOWN: []}
    for code, entry in pool.items():
        if code == "_meta":
            continue
        groups.setdefault(entry.get("rating", RATING_UNKNOWN), []).append((code, entry))

    for rating, label in [("PASS", "合格 (可交易)"), ("WATCH", "待观察"),
                           ("FAIL", "不合格"), (RATING_UNKNOWN, "数据不足")]:
        items = groups.get(rating, [])
        if not items:
            continue
        lines.append(f"## {label} ({len(items)} 只)\n")
        lines.append("| 代码 | 名称 | 评分 | ROE | 营收增长 | 负债率 | 备注 |")
        lines.append("|------|------|------|-----|---------|-------|------|")
        items.sort(key=lambda x: x[1].get("composite", 0), reverse=True)
        for code, entry in items:
            name = entry.get("name", "")
            cs = entry.get("composite", 0)
            fin = entry.get("financials", {})
            roe = fin.get("roe_3y_avg") or fin.get("roe", "")
            rev = fin.get("revenue_yoy", "")
            debt = fin.get("debt_to_assets", "")
            notes = entry.get("notes", "")
            lines.append(
                f"| {code} | {name} | {cs} | {roe}% | {rev}% | {debt}% | {notes} |"
            )
        lines.append("")

    pool["_meta"] = meta
    text = "\n".join(lines)

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已导出: {filepath}")

    return text


# ─── CLI ───

def main():
    import argparse
    parser = argparse.ArgumentParser(description="基本面分析系统")
    parser.add_argument("--update", "-u", action="store_true", help="更新所有股票基本面评估")
    parser.add_argument("--force", "-f", action="store_true", help="强制刷新（忽略缓存）")
    parser.add_argument("--report", "-r", action="store_true", help="打印报告")
    parser.add_argument("--add", nargs=2, metavar=("CODE", "NAME"), help="添加股票")
    parser.add_argument("--remove", metavar="CODE", help="移除股票")
    parser.add_argument("--note", nargs=2, metavar=("CODE", "TEXT"), help="添加备注")
    parser.add_argument("--export", metavar="FILE", help="导出 Markdown 报告")
    parser.add_argument("--approved", action="store_true", help="列出 PASS 股票")
    args = parser.parse_args()

    if args.add:
        add_stock(args.add[0], args.add[1])
    elif args.remove:
        remove_stock(args.remove)
    elif args.note:
        note_stock(args.note[0], args.note[1])
    elif args.approved:
        approved = get_approved_list()
        print(f"\nPASS 股票 ({len(approved)} 只):")
        for s in approved:
            print(f"  {s['code']} {s['name']} (评分{s['score']})")
    elif args.export:
        pool = load_pool()
        export_report(pool, args.export)
    elif args.report:
        pool = load_pool()
        print_report(pool)
    elif args.update or args.force:
        pool = update_all(force=args.force)
        print_report(pool)
    else:
        # 默认：打印当前报告
        pool = load_pool()
        if pool:
            print_report(pool)
        else:
            print("股票池为空，运行 python fundamental_analyzer.py --update 初始化")
            init_pool()
            print("种子池已初始化，运行 --update 获取数据")


if __name__ == "__main__":
    main()
