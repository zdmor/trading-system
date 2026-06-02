"""
系统状态更新 - 在 scanner/position_monitor 运行后更新 _system_state.md
确保选股、持仓信息始终一致，伸手就能看到。
"""
import json, os
from datetime import datetime

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_system_state.md")

def update(scan_date=None, top_picks=None, holdings=None):
    """更新 _system_state.md

    Args:
        scan_date: 扫描日期字符串
        top_picks: [(rank, code, name, trend, signal, ws, ss, volume, industry)]
        holdings: [(code, name, cost, price, pnl, score, level, trend, signal)]
    """
    lines = []
    lines.append("# 系统最新状态 (auto-updated)\n")
    lines.append(f"_更新: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    lines.append("---\n")

    # 选股
    if top_picks:
        lines.append(f"## 最新选股 ({scan_date or '未知'})\n")
        lines.append("| 排名 | 代码 | 名称 | 趋势 | 信号 | WS | SS | 成交额 | 行业 |\n")
        lines.append("|------|------|------|------|------|-----|-----|--------|------|\n")
        for p in top_picks:
            industry = p[8] if len(p) > 8 else ""
            lines.append(f"| {p[0]} | {p[1]} | {p[2]} | {p[3]} | {p[4]} | {p[5]} | {p[6]} | {p[7]} | {industry} |\n")
        lines.append("\n")

    # 持仓
    if holdings:
        lines.append("## 持仓状态\n")
        lines.append("| 代码 | 名称 | 成本 | 现价 | 盈亏 | 评分 | 等级 | 趋势 | 信号 |\n")
        lines.append("|------|------|------|------|------|------|------|------|------|\n")
        for h in holdings:
            lines.append(f"| {h[0]} | {h[1]} | {h[2]} | {h[3]} | {h[4]} | {h[5]} | {h[6]} | {h[7]} | {h[8]} |\n")
        lines.append("\n")

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)
