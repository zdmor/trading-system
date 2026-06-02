#!/usr/bin/env python3
"""
daily_report.py — 每日汇总看板
调用 system_state.get_dashboard() 获取全系统状态，格式化后输出。
可选：推送飞书 bot

Usage:
  python daily_report.py              # 输出到 stdout
  python daily_report.py --feishu     # 推送到飞书
  python daily_report.py --save       # 保存到 output/
"""

import sys, os
from datetime import datetime

TS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TS_DIR)


def build_report() -> str:
    """生成 Markdown 格式的每日看板"""
    from system_state import get_dashboard

    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    raw = get_dashboard()

    lines = [
        "# 交易系统每日看板",
        f"**生成时间**: {today}",
        "",
        "```",
        raw,
        "```",
        "",
        "---",
        f"*自动生成 by daily_report.py*",
    ]
    return "\n".join(lines)


def push_feishu(text: str):
    """推送飞书（通过 MCP send_message_as_bot）"""
    print("[feishu] 推送中...")
    # 飞书推送需通过 OpenClaw 的 feishu MCP tool
    # 此脚本由 cron/手动触发时，由 OpenClaw 代理完成推送
    print(text)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="每日汇总看板")
    parser.add_argument("--feishu", action="store_true", help="推送到飞书")
    parser.add_argument("--save", action="store_true", help="保存到 output/")
    args = parser.parse_args()

    report = build_report()

    if args.save:
        out_dir = os.path.join(TS_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"dashboard_{datetime.now().strftime('%Y-%m-%d')}.md"
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved: {out_path}")

    if args.feishu:
        push_feishu(report)
    elif not args.save:
        print(report)


if __name__ == "__main__":
    main()
