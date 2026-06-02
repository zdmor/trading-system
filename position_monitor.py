#!/usr/bin/env python3
"""
position_monitor.py — 持仓每日监控
对三只持仓逐日评分，输出因子明细 + 与昨日对比变化 + 预警

Usage:
  python position_monitor.py              # 输出到 stdout
  python position_monitor.py --feishu     # 推送飞书
  python position_monitor.py --save       # 保存到 output/
"""

import sys, os, json, time
from datetime import datetime

TS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TS_DIR)

from update_state import update as update_state
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import requests

from main import Analyzer, Strategy
from scanner import WyckoffAnalyzer
from scoring import StockScorer

# ── 持仓定义 ──
POSITIONS = [
    {"symbol": "002050", "name": "三花智控", "shares": 100, "avg_price": 51.44},
    {"symbol": "600038", "name": "中直股份", "shares": 300, "avg_price": 35.37},
    {"symbol": "600011", "name": "华能国际", "shares": 500, "avg_price": 9.60},
]

CACHE_PATH = os.path.join(TS_DIR, "position_cache.json")
OUTPUT_DIR = os.path.join(TS_DIR, "output")

TENCENT_KLINE_URL = (
    "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,320,qfq"
)

# ── 预警阈值 ──
SCORE_DROP_THRESHOLD = 10   # 较昨日骤降 >10 触发预警
STOP_SIGNALS = {"Upthrust", "UTAD", "BC", "SOW"}  # 威科夫止损信号


def _tx_code(symbol: str) -> str:
    """Convert to Tencent API code format"""
    return f"sh{symbol}" if symbol.startswith(("6", "9")) else f"sz{symbol}"


def fetch_kline(symbol: str) -> pd.DataFrame | None:
    """Fetch daily K-line from Tencent API (~320 rows)"""
    code = _tx_code(symbol)
    url = TENCENT_KLINE_URL.format(code=code)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        stock_data = data.get("data", {})
        if not isinstance(stock_data, dict):
            return None
        days = stock_data.get(code, {})
        if not isinstance(days, dict):
            return None
        klines = days.get("qfqday") or days.get("day") or []
        if not klines:
            return None
    except Exception:
        return None

    rows = []
    for k in klines:
        try:
            rows.append({
                "date": pd.Timestamp(k[0]),
                "open": float(k[1]), "close": float(k[2]),
                "high": float(k[3]), "low": float(k[4]),
                "volume": float(k[5]),
            })
        except (ValueError, IndexError):
            continue
    if len(rows) < 30:
        return None
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(data: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def score_one(symbol: str, name: str, pos: dict) -> dict | None:
    """Score a single position, returns scoring dict or None"""
    df = fetch_kline(symbol)
    if df is None or len(df) < 50:
        return None

    try:
        df = Analyzer.calc_atr(df)
        df = Analyzer.calc_ma(df, [20, 50, 200])
    except Exception:
        return None

    price = float(df["close"].iloc[-1])
    supports, resistances = Analyzer.detect_levels(df)
    levels = {"supports": supports, "resistances": resistances, "current": price}

    strategy = Strategy(df, 100000)
    trend = strategy.trend_analysis()
    vol = strategy.volatility_analysis()

    # Wyckoff
    trend_dir = trend.get("direction", "未知")
    wyckoff_sigs = []
    phase_label = ""
    if len(df) >= 50 and trend_dir != "未知":
        c_arr = df["close"].values.astype(float)
        h_arr = df["high"].values.astype(float)
        l_arr = df["low"].values.astype(float)
        v_arr = df["volume"].values.astype(float)
        phase_label, _, _ = WyckoffAnalyzer.detect_phase(
            c_arr.tolist(), h_arr.tolist(), l_arr.tolist(), v_arr.tolist(),
            trend_dir, []
        )
        wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
            c_arr.tolist(), h_arr.tolist(), l_arr.tolist(),
            df["open"].values.astype(float).tolist(), v_arr.tolist()
        )

    atr_val = df["atr"].iloc[-1] if "atr" in df.columns else price * 0.02
    stop_price = round(price - atr_val * 1.5, 2)
    exit_prices = [round(price * 1.08, 2)]

    scorer = StockScorer(
        df=df, price=price, trend=trend, vol=vol, levels=levels,
        stop_price=stop_price, exit_prices=exit_prices,
        wyckoff_signals=wyckoff_sigs, wyckoff_phase=phase_label,
        position=pos, symbol=symbol,
    )
    result = scorer.compute()

    best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")

    return {
        "symbol": symbol,
        "name": name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "price": price,
        "composite": result["composite_score"],
        "level": result.get("level", "?"),
        "trend": trend_dir,
        "phase": phase_label,
        "wyckoff_signal": best_sig[0] if best_sig else "-",
        "breakdown": [
            {
                "key": b["key"],
                "score": b["score"],
                "contribution": b.get("contribution", 0),
                "label": b.get("label", ""),
                "detail": b.get("detail", ""),
                "weight_pct": b.get("weight_pct", 0),
            }
            for b in result.get("factors", [])
        ],
    }


def compare_delta(today: dict, yesterday: dict) -> dict:
    """Compute day-over-day changes"""
    delta_composite = today["composite"] - yesterday.get("composite", today["composite"])
    delta_price = today["price"] - yesterday.get("price", today["price"])

    factor_deltas = {}
    tb = {b["key"]: b["score"] for b in today.get("breakdown", [])}
    yb = {b["key"]: b["score"] for b in yesterday.get("breakdown", [])}
    for key in tb:
        if key in yb:
            factor_deltas[key] = tb[key] - yb[key]

    return {
        "delta_composite": delta_composite,
        "delta_price": delta_price,
        "delta_market": delta_market,
        "factor_deltas": factor_deltas,
    }


def check_alerts(scores: list[dict], cache: dict) -> list[str]:
    """Generate alert messages"""
    alerts = []
    today_date = datetime.now().strftime("%Y-%m-%d")

    for s in scores:
        sym = s["symbol"]
        name = s["name"]
        prev = cache.get("last_scores", {}).get(sym, {})

        # 1. Score drop >10
        if prev:
            delta = s["composite"] - prev.get("composite", s["composite"])
            if delta <= -SCORE_DROP_THRESHOLD:
                alerts.append(
                    f"⚠️ {name}({sym}) 评分骤降: "
                    f"{prev['composite']:.0f}→{s['composite']:.0f} ({delta:+.0f})"
                )

        # 2. Stop signal from Wyckoff
        sig = s.get("wyckoff_signal", "-")
        if sig in STOP_SIGNALS:
            alerts.append(
                f"🛑 {name}({sym}) 威科夫止损信号: {sig} "
                f"(phase={s.get('phase', '?')})"
            )

        # 3. Low composite score warning
        if s["composite"] < 30:
            alerts.append(
                f"🔻 {name}({sym}) 评分过低: {s['composite']:.0f} "
                f"(level={s['level']})"
            )

    return alerts


def build_report(scores: list[dict], cache: dict, alerts: list[str]) -> str:
    today_str = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"=== 持仓监控 {today_str} ===",
        f"生成时间: {datetime.now().strftime('%H:%M')}",
        "",
    ]

    if alerts:
        lines.append("## ⚠️ 预警")
        for a in alerts:
            lines.append(f"  {a}")
        lines.append("")

    lines.append(f"| 股票 | 价格 | 综合评分 | 等级 | 趋势 | 威科夫 | 涨跌 |")
    lines.append(f"|------|------|----------|------|------|--------|------|")

    for s in scores:
        prev = cache.get("last_scores", {}).get(s["symbol"], {})
        delta = s["composite"] - prev.get("composite", s["composite"])
        delta_s = f"{delta:+.0f}" if prev else "new"
        pnl_pct = (s["price"] - prev.get("price", s["price"])) / max(prev.get("price", s["price"]), 0.01) * 100 if prev else 0
        pnl_s = f"{pnl_pct:+.1f}%" if prev else "—"

        lines.append(
            f"| {s['name']}({s['symbol']}) | {s['price']:.2f} | "
            f"{s['composite']:.1f} | {s['level']} | {s['trend']} | "
            f"{s['wyckoff_signal']} | {delta_s} / {pnl_s} |"
        )

    # Factor breakdown per stock
    lines.append("")
    lines.append("## 因子明细")
    for s in scores:
        lines.append(f"\n### {s['name']}({s['symbol']}) — composite={s['composite']:.1f} | {s['level']}")
        lines.append(f"价格: {s['price']:.2f} | 趋势: {s['trend']} | 威科夫: {s['wyckoff_signal']}({s['phase']})")

        prev = cache.get("last_scores", {}).get(s["symbol"], {})
        prev_bd = {b["key"]: b["score"] for b in prev.get("breakdown", [])}

        lines.append(f"| 因子 | 权重% | 评分 | 贡献 | 标签 | 变化 |")
        lines.append(f"|------|-------|------|------|------|------|")
        for b in s.get("breakdown", []):
            prev_score = prev_bd.get(b["key"], b["score"])
            delta_f = b["score"] - prev_score if prev_bd else 0
            delta_fs = f"{delta_f:+.0f}" if prev_bd else "—"
            lines.append(
                f"| {b['key']} | {b['weight_pct']} | {b['score']:.0f} | "
                f"{b['contribution']} | {b['label']} | {delta_fs} |"
            )

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="持仓每日监控")
    parser.add_argument("--feishu", action="store_true", help="推送飞书")
    parser.add_argument("--save", action="store_true", help="保存到 output/")
    args = parser.parse_args()

    cache = load_cache()
    today_str = datetime.now().strftime("%Y-%m-%d")

    print(f"评估 {len(POSITIONS)} 只持仓...")
    all_scores = []
    holdings = []

    for p in POSITIONS:
        print(f"  {p['name']}({p['symbol']})...", end=" ", flush=True)
        t0 = time.time()
        result = score_one(
            p["symbol"], p["name"],
            {"shares": p["shares"], "avg_price": p["avg_price"]},
        )
        if result:
            all_scores.append(result)
            cost = p["avg_price"]
            price = result["price"]
            pnl = (price - cost) / cost * 100
            holdings.append((
                p["symbol"], p["name"], cost, price,
                f"{pnl:+.1f}%", result["composite"], result["level"],
                result["trend"], result["wyckoff_signal"],
            ))
            print(f"composite={result['composite']:.1f} ({time.time()-t0:.1f}s)")
        else:
            print("数据不足")

    if not all_scores:
        print("ERROR: 无有效评分数据")
        return

    # Update cache
    cache["last_run"] = today_str
    cache["last_scores"] = {
        s["symbol"]: {
            "composite": s["composite"],
            "price": s["price"],
            "level": s["level"],
            "trend": s["trend"],
            "wyckoff_signal": s["wyckoff_signal"],
            "breakdown": s["breakdown"],
        }
        for s in all_scores
    }
    save_cache(cache)

    alerts = check_alerts(all_scores, cache)
    report = build_report(all_scores, cache, alerts)
    print("\n" + report)

    # 更新 _system_state.md
    try:
        update_state(holdings=holdings)
    except Exception as e:
        print(f"[state] 更新 _system_state.md 失败: {e}")

    if args.save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, f"position_monitor_{today_str}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved: {out_path}")

    if args.feishu:
        # 通过 MCP send_message_as_bot 推送（由 OpenClaw 代理完成）
        print("[feishu] 待推送...")


if __name__ == "__main__":
    main()
