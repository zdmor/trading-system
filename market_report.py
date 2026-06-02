#!/usr/bin/env python3
"""
market_report.py — 市场报告飞书打通
调用 system_state.get_dashboard() → 格式化为飞书 post 消息 → 推送

Usage:
  python market_report.py              # 预览消息内容
  python market_report.py --push       # 推送到飞书（通过 MCP）
  python market_report.py --save       # 保存到 output/
"""

import sys, os
from datetime import datetime

TS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TS_DIR)
sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = os.path.join(TS_DIR, "output")


def get_dashboard_raw() -> str:
    """Get raw dashboard text from system_state"""
    from system_state import get_dashboard
    return get_dashboard()


def parse_dashboard(raw: str) -> dict:
    """Parse raw dashboard into structured dict (best-effort)"""
    lines = raw.strip().split("\n")
    result = {
        "updated": "",
        "market": {},
        "scanner": {},
        "positions": {},
        "tasks": {},
        "raw": raw,
    }

    section = None
    for line in lines:
        line = line.strip()
        if line.startswith("Updated:"):
            result["updated"] = line.replace("Updated:", "").strip()
        elif "Market:" in line:
            section = "market"
            parts = line.split("(")
            regime_part = parts[0].replace("Market:", "").strip()
            result["market"]["regime"] = regime_part
            if "conf=" in line:
                conf = line[line.index("conf=")+5:].split(")")[0]
                result["market"]["confidence"] = conf
        elif line.startswith("Policy:") and section == "market":
            result["market"]["policy"] = line.split(":")[1].strip().split()[0] if ":" in line else ""
        elif "Scanner:" in line:
            section = "scanner"
            result["scanner"]["status"] = line
        elif "Positions" in line and "(" in line:
            section = "positions"
            result["positions"]["count"] = line.split("(")[1].split(")")[0] if "(" in line else ""
        elif "Tasks" in line and "(" in line:
            section = "tasks"
            result["tasks"]["count"] = line.split("(")[1].split(")")[0] if "(" in line else ""

    return result


def build_feishu_post(parsed: dict) -> str:
    """Format as human-readable Feishu-ready post text"""
    today = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    m = parsed.get("market", {})
    s = parsed.get("scanner", {})

    lines = [
        f"📊 交易系统日报 — {today}",
        "",
        "━━━━ 市场状态 ━━━━",
        f"　情绪: {m.get('regime', '?')} (置信度: {m.get('confidence', '?' )})",
        f"　政策: {m.get('policy', '?')}  |  杠杆: 一致性",
        "",
        "━━━━ 选股 ━━━━",
        str(s.get('status', '未运行')),
        "",
    ]

    # Positions summary
    pos = parsed.get("positions", {})
    if pos.get("count"):
        lines.append(f"━━━━ 持仓 ({pos['count']}只) ━━━━")
        lines.append("　详见 position_monitor")

    lines.append("")
    lines.append("━" * 20)
    lines.append("⚡ 由 market_report.py 自动生成")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="市场报告飞书推送")
    parser.add_argument("--push", action="store_true", help="推送到飞书")
    parser.add_argument("--save", action="store_true", help="保存到 output/")
    args = parser.parse_args()

    print("获取系统状态...")
    raw = get_dashboard_raw()
    parsed = parse_dashboard(raw)
    post = build_feishu_post(parsed)

    print("\n" + post)

    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d")
        out_path = os.path.join(OUTPUT_DIR, f"market_report_{ts}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(post)
        print(f"\nSaved: {out_path}")

    if args.push:
        # 飞书推送通过 OpenClaw feishu MCP (send_message_as_bot)
        # 由 cron/scheduler 调用时由 OpenClaw 代理完成推送
        print("\n[feishu] 推送请求已提交 (需 OpenClaw MCP 路由)")

    # Output full dashboard for reference
    print("\n\n=== RAW DASHBOARD ===")
    print(raw)


if __name__ == "__main__":
    main()
