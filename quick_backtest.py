"""
最近5日回测 — 复用 Scanner 管道进行历史信号回溯
"""
import sys, os, time, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

from main import Analyzer, Strategy
from scanner import WyckoffAnalyzer, Scanner
from scoring import StockScorer
from data_providers import get_financial_indicators

# 腾讯K线API
TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"

def get_klines(code):
    """获取个股K线"""
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        if data.get("code") != 0:
            return None
        klines = data.get("data", {}).get(key, {}).get("qfqday") or \
                 data.get("data", {}).get(key, {}).get("day") or []
        if not klines:
            return None
        rows = []
        for k in klines:
            rows.append({
                "date": pd.Timestamp(k[0]),
                "open": float(k[1]), "close": float(k[2]),
                "high": float(k[3]), "low": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows)
        return df
    except Exception:
        return None

def analyze_stock_at_date(df, code, name, cutoff_date):
    """在 cutoff_date 截断数据并分析"""
    try:
        mask = df["date"] <= pd.Timestamp(cutoff_date)
        sdf = df[mask].copy()
        if len(sdf) < 60:
            return None

        sdf = Analyzer.calc_atr(sdf)
        sdf = Analyzer.calc_ma(sdf, [20, 50, 200])
        supports, resistances = Analyzer.detect_levels(sdf)
        price = float(sdf["close"].iloc[-1])
        levels = {"supports": supports, "resistances": resistances, "current": price}

        strategy = Strategy(sdf, 100000)
        trend = strategy.trend_analysis()
        vol = strategy.volatility_analysis()

        trend_dir = trend["direction"]
        wyckoff_sigs = []
        phase_label = ""
        if len(sdf) >= 50 and trend_dir != "未知":
            c = sdf["close"].values.astype(float)
            h = sdf["high"].values.astype(float)
            l = sdf["low"].values.astype(float)
            v = sdf["volume"].values.astype(float)
            phase_label, _, _ = WyckoffAnalyzer.detect_phase(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, []
            )
            wyckoff_sigs, _ = WyckoffAnalyzer.analyze_all(
                c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir
            )

        best_sig = wyckoff_sigs[0] if wyckoff_sigs else ("-", 0, "")
        wyckoff_score = best_sig[1] if len(best_sig) > 1 else 0

        atr_val = sdf["atr"].iloc[-1] if "atr" in sdf.columns else price * 0.02
        stop_price = round(price - atr_val * 1.5, 2) if atr_val > 0 else round(price * 0.93, 2)
        exit_prices = [r for r in resistances] if resistances else [round(price * 1.08, 2)]

        scorer = StockScorer(
            df=sdf, price=price, trend=trend, vol=vol, levels=levels,
            stop_price=stop_price, exit_prices=exit_prices,
            wyckoff_signals=wyckoff_sigs, wyckoff_phase=phase_label,
        )
        scoring = scorer.compute()

        buy_signals = {"SOS", "Spring", "LPS", "EVR", "Compression"}
        return {
            "code": code, "name": name,
            "date": cutoff_date,
            "signal": best_sig[0], "score": wyckoff_score,
            "composite": scoring["composite_score"],
            "trend": trend_dir, "phase": phase_label,
            "price": price,
            "is_buy": best_sig[0] in buy_signals and trend_dir == "多头",
        }
    except Exception as e:
        return None

def forward_return(df, analyze_date, days=2):
    """analyze_date 后第 days 天的涨幅"""
    mask = df["date"] <= pd.Timestamp(analyze_date)
    idx = len(df[mask])
    if idx + days >= len(df):
        return None
    buy_p = float(df.iloc[idx]["close"])
    sell_p = float(df.iloc[idx + days]["close"])
    return round((sell_p - buy_p) / buy_p * 100, 2)


# ====== 主流程 ======
print("=" * 70)
print("  最近5交易日回测 — 信号回溯")
print("=" * 70)

# 1. 获取候选股票（用Scanner的今日快照）
print("  获取候选股票池...")
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
candidates = sc.candidates[:80]
print(f"  候选: {len(candidates)}只")

# 2. 交易日范围
today = datetime.now()
dates = []
for i in range(10):
    d = today - timedelta(days=i)
    if d.weekday() < 5:
        dates.append(d.strftime("%Y-%m-%d"))
dates = sorted(dates[-6:])
print(f"  回测区间: {dates[0]} ~ {dates[-1]} ({len(dates)}个交易日)\n")

# 3. 获取每只股票的K线并缓存
print("  批量获取K线数据...")
stock_data = {}
for i, stk in enumerate(candidates):
    df = get_klines(stk["code"])
    if df is not None and len(df) >= 60:
        stock_data[stk["code"]] = {"df": df, "name": stk["name"]}
    if (i + 1) % 20 == 0:
        print(f"  K线缓存: {i+1}/{len(candidates)}", end="\r")
print(f"  K线缓存: {len(candidates)}/{len(candidates)} — 有效: {len(stock_data)}只")

# 4. 逐日回测
all_results = {}
for d in dates:
    print(f"\n{'─'*50}")
    print(f"  ▶ 回测日期: {d}")
    print(f"{'─'*50}")

    day_picks = []
    total = len(stock_data)
    for i, (code, info) in enumerate(stock_data.items()):
        result = analyze_stock_at_date(info["df"], code, info["name"], d)
        if result and result["signal"] not in ("-", "无信号", "数据不足"):
            day_picks.append(result)
        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{total}", end="\r")
    print(f"  进度: {total}/{total}")

    buy_picks = [r for r in day_picks if r["is_buy"]]
    buy_picks.sort(key=lambda x: -x["composite"])

    print(f"  全部信号: {len(day_picks)}只 | 买入信号: {len(buy_picks)}只")

    if buy_picks:
        for r in buy_picks[:6]:
            info = stock_data.get(r["code"])
            fwd = forward_return(info["df"], r["date"], days=2) if info else None
            fwd_str = f"{fwd:+.2f}%" if fwd is not None else "N/A"
            print(f"    {r['signal']:<10} {r['code']:<8} {r['name']:<6} "
                  f"评分{r['composite']:<5.0f} 现价{r['price']:<8.2f} "
                  f"→2日{fwd_str}")

    all_results[d] = {"total": len(day_picks), "buys": buy_picks}
    time.sleep(0.3)

# 5. 汇总
print(f"\n\n{'='*70}")
print(f"  回测汇总")
print(f"{'='*70}")
print(f"  {'日期':<14} {'信号总数':<10} {'买入信号':<10} {'Top3平均综合分':<16} {'Top3平均2日收益':<16}")
print(f"  {'-'*60}")
for d in dates:
    info = all_results[d]
    buys = info["buys"]
    avg_score = np.mean([r["composite"] for r in buys[:3]]) if buys else 0
    avg_fwd = 0
    fwd_count = 0
    for r in buys[:3]:
        sd = stock_data.get(r["code"])
        if sd:
            fwd = forward_return(sd["df"], r["date"], days=2)
            if fwd is not None:
                avg_fwd += fwd
                fwd_count += 1
    avg_fwd = round(avg_fwd / fwd_count, 2) if fwd_count > 0 else 0
    print(f"  {d:<14} {info['total']:<10} {len(buys):<10} {avg_score:<16.1f} {avg_fwd:<+16.2f}")

# 6. 汇总所有买入信号的胜率
all_buys = []
for d in dates:
    for r in all_results[d]["buys"]:
        sd = stock_data.get(r["code"])
        if sd:
            fwd = forward_return(sd["df"], r["date"], days=2)
            if fwd is not None:
                all_buys.append({"date": r["date"], "code": r["code"], "name": r["name"],
                               "signal": r["signal"], "score": r["composite"], "fwd_2d": fwd})
all_count = len(all_buys)
if all_count > 0:
    win_count = sum(1 for r in all_buys if r["fwd_2d"] > 0)
    avg_return = np.mean([r["fwd_2d"] for r in all_buys])
    print(f"\n  {'─'*50}")
    print(f"  总买入信号: {all_count}次 | 胜率: {win_count/all_count*100:.1f}%")
    print(f"  平均2日收益: {avg_return:+.2f}%")
    print(f"  最佳: {max(all_buys, key=lambda x: x['fwd_2d'])['fwd_2d']:+.2f}%")
    print(f"  最差: {min(all_buys, key=lambda x: x['fwd_2d'])['fwd_2d']:+.2f}%")

# 7. 保存
out = {"dates": dates, "signals_by_date": {}}
for d in dates:
    out["signals_by_date"][d] = {
        "total_signals": all_results[d]["total"],
        "buy_signals": len(all_results[d]["buys"]),
        "top_picks": [{"code": r["code"], "name": r["name"],
                       "signal": r["signal"], "score": r["composite"],
                       "price": r["price"]} for r in all_results[d]["buys"][:5]],
    }
out["summary"] = {
    "total_buy_signals": all_count,
    "win_rate": round(win_count/all_count*100, 1) if all_count > 0 else 0,
    "avg_2d_return": round(avg_return, 2),
}
with open("quick_backtest_result.json", "w", encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n  结果已保存到 quick_backtest_result.json")
print(f"{'='*70}\n")
