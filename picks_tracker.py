"""
选股历史追踪 & 表现评估
记录每日选股信号，追踪1日/5日/10日后表现，输出到 Obsidian
"""
import os, json
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "picks_history.json")
OBSIDIAN_FILE = r"D:\DiskMigration\MySecondBrain\选股表现.md"


def append_history(picks_list):
    """将今日选股追加到历史记录"""
    if not picks_list:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass

    existing = {(p["date"], p["code"]) for p in history}
    new_entries = []
    for r in picks_list[:8]:
        code = r["code"].split(".")[1]
        key = (today, code)
        if key not in existing:
            new_entries.append({
                "date": today,
                "code": code,
                "name": r["name"],
                "signal": r["wyckoff_sig"],
                "score": r["wyckoff_score"],
                "price": r["price"],
            })
    if new_entries:
        history.extend(new_entries)
        history.sort(key=lambda x: (x["date"], x["code"]))
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _calc_return(daily, entry_date_str, offset):
    """计算 entry_date 之后 offset 个交易日的涨跌幅"""
    dates = daily["date"].values.astype(str)
    closes = daily["close"].values.astype(float)
    for i, d in enumerate(dates):
        if d.startswith(entry_date_str):
            idx = i + offset
            if idx < len(dates):
                ret = (closes[idx] / closes[i] - 1) * 100
                return round(ret, 2), closes[i], closes[idx]
            return None, closes[i], None
    return None, None, None


def generate_report():
    """读取历史选股，计算1/5/10日表现，生成 Obsidian 报告"""
    from main import DataFetcher

    if not os.path.exists(HISTORY_FILE):
        return

    with open(HISTORY_FILE, encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        return

    fetcher = DataFetcher()
    lines = ["# 选股表现", "", "> 每日选股信号的1日/5日/10日表现追踪", "", "---", ""]

    # 按日期分组（倒序）
    dates = sorted(set(p["date"] for p in history), reverse=True)

    for date in dates:
        day_picks = [p for p in history if p["date"] == date]
        lines.append(f"## {date}")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 信号 | 评分 | 入场价 | 1日后 | 5日后 | 10日后 |")
        lines.append("|---|------|------|------|------|--------|-------|-------|--------|")

        for i, p in enumerate(day_picks, 1):
            try:
                daily = fetcher.get_daily(p["code"], days=200)
            except Exception:
                daily = None

            def fmt_offset(offset):
                if daily is None:
                    return "—"
                ret, entry_c, target_c = _calc_return(daily, date, offset)
                if ret is None:
                    return "待确认"
                arrow = "↑" if ret >= 0 else "↓"
                return f"{arrow}{abs(ret):.1f}%"

            d1 = fmt_offset(1)
            d5 = fmt_offset(5)
            d10 = fmt_offset(10)

            lines.append(
                f"| {i} | {p['code']} | {p['name']} | {p['signal']} | {p['score']} "
                f"| {p['price']:.2f} | {d1} | {d5} | {d10} |"
            )
        lines.append("")

    lines.append("---")
    lines.append(f"*更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    content = "\n".join(lines)

    # 写入 Obsidian
    try:
        os.makedirs(os.path.dirname(OBSIDIAN_FILE), exist_ok=True)
        with open(OBSIDIAN_FILE, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass

    # 同时打印到终端
    print(f"\n{'─'*62}")
    print(f"  选股表现追踪")
    print(f"{'─'*62}")
    for date in dates[:3]:  # 最多显示最近3天
        day_picks = [p for p in history if p["date"] == date]
        print(f"\n  {date}:")
        print(f"  {'#':<3} {'名称':<8} {'信号':<8} {'入场':<7} {'1日后':<8} {'5日后':<8} {'10日后':<8}")
        for i, p in enumerate(day_picks, 1):
            try:
                daily = fetcher.get_daily(p["code"], days=200)
            except Exception:
                daily = None

            def fmt_offset_term(offset):
                if daily is None:
                    return "—"
                ret, _, _ = _calc_return(daily, date, offset)
                if ret is None:
                    return "待确认"
                arr = "↑" if ret >= 0 else "↓"
                return f"{arr}{abs(ret):.1f}%"

            d1t = fmt_offset_term(1)
            d5t = fmt_offset_term(5)
            d10t = fmt_offset_term(10)
            print(f"  {i:<3} {p['name']:<8} {p['signal']:<8} {p['price']:<7.2f} {d1t:<8} {d5t:<8} {d10t:<8}")
    print()
    return content
