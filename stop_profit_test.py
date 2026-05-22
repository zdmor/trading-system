"""
止损止盈参数优化 — 不看形态，纯机械规则回测
核心问题: 买进后，什么止损/止盈组合长期最优？
"""
import sys, os, time, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,500,qfq"

def get_klines(code):
    key = code.replace(".", "")
    url = TENCENT_KLINE_URL.format(code=key)
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get(key, {}).get("qfqday") or data.get("data", {}).get(key, {}).get("day") or []
        if not klines: return None
        rows = [{"date": pd.Timestamp(k[0]), "close": float(k[2]),
                 "high": float(k[3]), "low": float(k[4])} for k in klines]
        return pd.DataFrame(rows)
    except: return None


def simulate_trades(closes, highs, lows, stop_loss_pct, take_profit_pct, trailing_stop_pct=0):
    """
    模拟交易：每次满足入场条件就买入，到止损/止盈/收盘卖出
    返回所有交易的收益率列表
    """
    profits = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    trailing_high = 0

    for i in range(1, len(closes)):
        if not in_trade:
            # 入场条件：收盘价 > 前一日收盘价（最简单的趋势入场）
            if closes[i] > closes[i-1]:
                entry_price = closes[i]
                entry_idx = i
                trailing_high = closes[i]
                in_trade = True
        else:
            # 更新跟踪最高价
            if highs[i] > trailing_high:
                trailing_high = highs[i]

            current_price = closes[i]

            # 止损检查
            stop_price = entry_price * (1 - stop_loss_pct / 100)
            if trailing_stop_pct > 0:
                # 移动止损：从最高点回撤
                trailing_stop = trailing_high * (1 - trailing_stop_pct / 100)
                stop_price = max(stop_price, trailing_stop)

            if lows[i] <= stop_price:
                # 止损触发，以止损价卖出
                exit_price = stop_price
                profit = (exit_price - entry_price) / entry_price * 100
                profits.append(profit)
                in_trade = False
                continue

            # 止盈检查
            if take_profit_pct > 0:
                target_price = entry_price * (1 + take_profit_pct / 100)
                if highs[i] >= target_price:
                    exit_price = target_price
                    profit = (exit_price - entry_price) / entry_price * 100
                    profits.append(profit)
                    in_trade = False
                    continue

    return profits


def calc_atr(closes, highs, lows, period=14):
    """简单ATR计算"""
    if len(closes) < period + 1:
        return np.mean(highs - lows)
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return np.mean(trs[-period:])


print("=" * 70)
print("  止损止盈参数优化回测")
print("=" * 70)
print("  规则: 收盘涨就买入 → 止损/止盈/移动止损卖出")
print("  数据: 80只流动性最好的股票 × 500天\n")

# 获取股票列表
from scanner import Scanner
sc = Scanner()
sc.fetch_all_stocks()
sc.filter_candidates()
codes = [c["code"] for c in sc.candidates[:80]]

# 获取K线
stock_dfs = []
for i, code in enumerate(codes):
    df = get_klines(code)
    if df is not None and len(df) > 100:
        stock_dfs.append((code, df))
    if (i+1) % 20 == 0:
        print(f"  数据加载: {i+1}/{len(codes)}", end="\r")
print(f"  有效股票: {len(stock_dfs)}只\n")

# ====== 参数扫描 ======
stop_losses = [2, 3, 5, 7, 10]        # 固定止损 %
take_profits = [3, 5, 7, 10, 15, 20]  # 止盈 %
trailings = [0, 3, 5, 8]              # 移动止损(0=不使用)

results = []
t0 = time.time()

for sl in stop_losses:
    for tp in take_profits:
        for tr in trailings:
            if tr > 0 and tr >= sl:
                continue  # 移动止损必须小于固定止损才有意义
            all_profits = []
            trade_count = 0
            for code, df in stock_dfs:
                c = df["close"].values.astype(float)
                h = df["high"].values.astype(float)
                l = df["low"].values.astype(float)
                profits = simulate_trades(c, h, l, sl, tp, tr)
                all_profits.extend(profits)
                trade_count += len(profits)

            if trade_count < 10:
                continue

            avg_profit = np.mean(all_profits)
            win_rate = sum(1 for p in all_profits if p > 0) / len(all_profits) * 100
            total_return = sum(all_profits)
            avg_win = np.mean([p for p in all_profits if p > 0]) if any(p > 0 for p in all_profits) else 0
            avg_loss = np.mean([p for p in all_profits if p <= 0]) if any(p <= 0 for p in all_profits) else 0
            profit_factor = abs(sum(p for p in all_profits if p > 0) / sum(p for p in all_profits if p < 0)) if any(p < 0 for p in all_profits) else 999

            results.append({
                "stop_loss": sl, "take_profit": tp, "trailing": tr,
                "trades": trade_count, "avg_profit": round(avg_profit, 2),
                "win_rate": round(win_rate, 1), "total_return": round(total_return, 2),
                "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
            })

# 排序：按平均每笔收益排序
results.sort(key=lambda x: -x["avg_profit"])

print(f"\n{'='*100}")
print(f"  按平均每笔收益排名")
print(f"{'='*100}")
print(f"  {'止损%':>5} {'止盈%':>5} {'移动止损%':>8} {'交易数':>6} {'胜率':>6} {'平均收益':>8} {'总收益':>8} {'盈亏比':>7}")
print(f"  {'-'*60}")

for r in results[:15]:
    tr_label = str(r["trailing"]) if r["trailing"] > 0 else "-"
    print(f"  {r['stop_loss']:>5} {r['take_profit']:>5} {tr_label:>8} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_profit']:>+7.2f}% {r['total_return']:>+8.2f}% {r['profit_factor']:>6.2f}")

# 按胜率排序
results.sort(key=lambda x: -x["win_rate"])
print(f"\n{'='*100}")
print(f"  按胜率排名")
print(f"{'='*100}")
print(f"  {'止损%':>5} {'止盈%':>5} {'移动止损%':>8} {'交易数':>6} {'胜率':>6} {'平均收益':>8} {'总收益':>8} {'盈亏比':>7}")
print(f"  {'-'*60}")
for r in results[:15]:
    tr_label = str(r["trailing"]) if r["trailing"] > 0 else "-"
    print(f"  {r['stop_loss']:>5} {r['take_profit']:>5} {tr_label:>8} {r['trades']:>6} {r['win_rate']:>5.1f}% {r['avg_profit']:>+7.2f}% {r['total_return']:>+8.2f}% {r['profit_factor']:>6.2f}")

# ATR-based 止损
print(f"\n{'='*100}")
print(f"  ATR自适应止损（按波动率动态调整）")
print(f"{'='*100}")

def simulate_atr(closes, highs, lows, atr_mult_stop, atr_mult_target):
    """ATR-based 止损止盈"""
    atr = calc_atr(closes, highs, lows)
    profits = []
    in_trade = False
    entry_price = 0
    for i in range(1, len(closes)):
        if not in_trade:
            if closes[i] > closes[i-1]:
                entry_price = closes[i]
                atr = calc_atr(closes[max(0,i-20):i+1], highs[max(0,i-20):i+1], lows[max(0,i-20):i+1])
                in_trade = True
        else:
            stop_price = entry_price - atr * atr_mult_stop
            target_price = entry_price + atr * atr_mult_target
            if lows[i] <= stop_price:
                profits.append((stop_price - entry_price) / entry_price * 100)
                in_trade = False
            elif highs[i] >= target_price:
                profits.append((target_price - entry_price) / entry_price * 100)
                in_trade = False
    return profits

atr_results = []
for stop_mult in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for target_mult in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        all_p = []
        for code, df in stock_dfs:
            c = df["close"].values.astype(float)
            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            p = simulate_atr(c, h, l, stop_mult, target_mult)
            all_p.extend(p)
        if len(all_p) < 10: continue
        avg = np.mean(all_p)
        wr = sum(1 for x in all_p if x > 0) / len(all_p) * 100
        pf = abs(sum(x for x in all_p if x>0) / sum(x for x in all_p if x<0)) if any(x<0 for x in all_p) else 999
        atr_results.append({
            "stop": stop_mult, "target": target_mult, "trades": len(all_p),
            "avg": round(avg, 2), "wr": round(wr, 1), "pf": round(pf, 2),
        })

atr_results.sort(key=lambda x: -x["avg"])
print(f"  {'ATR止损':>7} {'ATR止盈':>7} {'交易数':>6} {'胜率':>6} {'平均收益':>8} {'盈亏比':>7}")
for r in atr_results[:10]:
    print(f"  {r['stop']:>5.1f}x {r['target']:>5.1f}x {r['trades']:>6} {r['wr']:>5.1f}% {r['avg']:>+7.2f}% {r['pf']:>6.2f}")

print(f"\n  总耗时: {time.time()-t0:.0f}s")
print(f"{'='*70}")
