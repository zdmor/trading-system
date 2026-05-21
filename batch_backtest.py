"""多股票回测 — 评分系统通用性验证"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import Backtester

stocks = [
    ("000858", "五粮液(白酒)"),
    ("600519", "贵州茅台(白酒)"),
    ("300750", "宁德时代(新能源)"),
    ("000333", "美的集团(家电)"),
    ("600036", "招商银行(银行)"),
    ("002594", "比亚迪(汽车)"),
    ("601318", "中国平安(保险)"),
    ("002415", "海康威视(安防)"),
]

print("=" * 100)
print(f"  {'股票':<20} {'区间':<22} {'策略':>8} {'持有':>8} {'年化':>7} {'交易':>4} {'胜率':>6} {'评分范围':<16} {'跑赢?':>6}")
print(f"  {'-' * 93}")

results = []
for code, name in stocks:
    try:
        bt = Backtester(code, 100000, buy_threshold=50, strong_buy=65, sell_threshold=40)
        bt.run_all()
        m = bt.calc_metrics()
        scores = [r['composite_score'] for r in bt.results]
        score_rng = f"{min(scores):.0f}-{max(scores):.0f}({np.mean(scores):.1f})"
        period = f"{m['start_date'].strftime('%Y%m')}-{m['end_date'].strftime('%Y%m%d')}"
        beat = "是" if m["total_return"] > m["hold_return"] else "否"
        line = f"  {name+code:<20} {period:<22} {m['total_return']:>+7.2f}% {m['hold_return']:>+7.2f}% {m['annual_return']:>+6.2f}% {m['trade_count']:>3}次 {m['win_rate']:>5.1f}% {score_rng:<16} {beat:>6}"
        print(line)
        results.append({"name": name, "code": code, **m, "score_rng": score_rng})
        time.sleep(0.3)
    except Exception as e:
        print(f"  {name+code:<20} 失败: {e}")
        time.sleep(0.3)

if results:
    print()
    # 汇总统计
    win = sum(1 for r in results if r.get("total_return", 0) > r.get("hold_return", 0))
    print(f"  跑赢买入持有: {win}/{len(results)} 只")
    print(f"  平均策略收益: {np.mean([r['total_return'] for r in results]):.2f}%")
    print(f"  平均持有收益: {np.mean([r['hold_return'] for r in results]):.2f}%")
    print(f"  平均年化:     {np.mean([r['annual_return'] for r in results]):.2f}%")
    avg_trades = np.mean([r['trade_count'] for r in results])
    avg_winrate = np.mean([r['win_rate'] for r in results if r['trade_count'] > 0])
    print(f"  平均交易:     {avg_trades:.1f}次  平均胜率: {avg_winrate:.1f}%")
    print("=" * 100)
