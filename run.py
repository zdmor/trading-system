"""
交易系统主入口 — 全市场扫描 + 个股分析 一站式输出
"""
import sys, os, time
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from scanner import Scanner, WyckoffAnalyzer
from main import DataFetcher, Analyzer, Strategy, Report, load_config
from market import MarketAnalyzer


def analyze_stock(symbol, account_value=80000, position=None):
    """分析单只股票，返回报告"""
    fetcher = DataFetcher()
    daily = fetcher.get_daily(symbol, days=400)
    name = fetcher.get_name(symbol)
    price = daily["close"].iloc[-1]
    price_date = daily["date"].iloc[-1]

    daily = Analyzer.calc_atr(daily)
    daily = Analyzer.calc_ma(daily, [20, 50, 200])
    supports, resistances = Analyzer.detect_levels(daily)

    levels = {"supports": supports, "resistances": resistances, "current": price}
    strategy = Strategy(daily, account_value, position or {"shares": 0, "avg_price": 0})
    trend = strategy.trend_analysis()
    vol = strategy.volatility_analysis()
    entry_check = strategy.entry_check(levels)

    recent_lows = daily["low"].iloc[-60:].min()
    stop_price = round(recent_lows * 0.985, 2) if recent_lows > 0 else round(price * 0.93, 2)

    config = load_config(symbol)
    if "stop_price" in config: stop_price = config["stop_price"]
    if "exit_prices" in config: exit_prices = config["exit_prices"]
    else: exit_prices = [r for r in resistances] if resistances else [round(price * 1.08, 2)]
    if "supports" in config: levels["supports"] = config["supports"]
    if "resistances" in config: levels["resistances"] = config["resistances"]

    pos = strategy.position_plan(stop_price)

    # 威科夫阶段分析
    phase_str = ""
    trend_dir = trend["direction"]
    if len(daily) >= 50 and trend_dir != "未知":
        c = daily["close"].values.astype(float)
        h = daily["high"].values.astype(float)
        l = daily["low"].values.astype(float)
        v = daily["volume"].values.astype(float)
        phase, phase_detail, _ = WyckoffAnalyzer.detect_phase(
            c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, []
        )
        phase_str = f"\n  【威科夫阶段】\n  {phase}\n  {phase_detail}"

    report = Report(symbol, name, account_value, price_date)

    output = report.header()
    output += phase_str
    output += report.trend_section(trend)
    output += report.volatility_section(vol)
    output += report.levels_section(levels)
    output += report.position_section(pos, current_price=price)
    output += report.signal_section(entry_check)
    output += report.recommendation_section(trend, entry_check, stop_price, exit_prices, pos, current_price=price)
    output += report.footer()
    return output, name, price


def run(watch_stocks=None, min_amount=5e8, top_n=15, max_scan=80):
    """一站式：扫盘 + 威科夫 + 板块 + 个股分析"""
    watch_stocks = watch_stocks or ["002050"]

    today = datetime.now().strftime('%Y-%m-%d %H:%M')

    # ====== 交易时间 ======
    can_trade, trade_status = MarketAnalyzer.is_trading_time()
    time_icon = "🔄" if can_trade else "⏹"
    print(f"\n{'='*62}")
    print(f"  A股交易分析系统  {today}")
    print(f"  {time_icon} {trade_status}")
    print(f"{'='*62}\n")

    # ====== 1. 扫描 ======
    print("  ▶ 全市场扫描中...")

    s = Scanner()
    t0 = time.time()
    s.fetch_all_stocks()
    s.filter_candidates(min_amount)
    s.build_industry_map()
    # 直接跑80只的趋势+威科夫分析（scanner内部会调Tencent API）
    s.run_analysis(max_scan)

    print(f"  ✓ 全市场{len(s.snapshot)}只，流动性通过{len(s.candidates)}只")
    print(f"  ✓ 已完成Top{max_scan}趋势+威科夫分析 ({time.time()-t0:.1f}s)\n")

    # ====== 2. 板块分布 ======
    print(f"{'─'*62}")
    print(f"  板块分布")
    print(f"{'─'*62}")
    sectors = {}
    for r in s.results:
        ind = r.get("industry", "其他")
        if ind not in sectors:
            sectors[ind] = {"count": 0, "sig": 0}
        sectors[ind]["count"] += 1
        if r["wyckoff_sig"] not in ("-", "无信号", "数据不足"):
            sectors[ind]["sig"] += 1
    sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["count"])
    for ind, info in sorted_sec[:12]:
        tag = f" [{info['sig']}信号]" if info['sig'] > 0 else ""
        print(f"  {ind:<26} {info['count']:>2}只{tag}")

    # ====== 3. 信号精选 ======
    print(f"\n{'─'*62}")
    print(f"  威科夫信号精选（全部信号）")
    print(f"{'─'*62}")
    valid = [r for r in s.results if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据")]
    if valid:
        print(f"  {'信号':<12} {'代码':<8} {'名称':<7} {'板块':<12} {'得分':<5} {'阶段':<22} 细节")
        print(f"  {'-'*70}")
        for r in valid[:10]:
            sym = r["code"].split(".")[1]
            sig = r["wyckoff_sig"]
            ind = r.get("industry", "")[:10]
            ph = r.get("phase", "")[:20]
            detail = r.get("wyckoff_detail", "")[:22]
            print(f"  {sig:<12} {sym:<8} {r['name']:<7} {ind:<12} {r['wyckoff_score']:<5} {ph:<22} {detail}")
    else:
        print(f"  （今日无显著信号）")

    # ====== 3b. 今日选股推荐 ======
    buy_signals = {"SOS", "Spring", "LPS", "EVR", "Compression"}
    picks = [r for r in s.results if r["wyckoff_sig"] in buy_signals
             and r["trend"] == "多头"]
    picks.sort(key=lambda x: -x["wyckoff_score"])
    if picks:
        print(f"\n{'─'*62}")
        print(f"  今日选股推荐（买入信号+多头趋势）")
        print(f"{'─'*62}")
        print(f"  {'#':<3} {'代码':<8} {'名称':<7} {'板块':<12} {'信号':<10} {'得分':<5} {'现价':<8} {'成交额':<10}")
        print(f"  {'-'*62}")
        for i, r in enumerate(picks[:8]):
            sym = r["code"].split(".")[1]
            ind = r.get("industry", "")[:10]
            sig = r["wyckoff_sig"]
            amt = f"{r['amount']/1e8:.1f}亿"
            print(f"  {i+1:<3} {sym:<8} {r['name']:<7} {ind:<12} {sig:<10} {r['wyckoff_score']:<5} {r['price']:<8.2f} {amt:<10}")
    else:
        print(f"\n  （今日无符合条件的买入信号）")

    # ====== 4. 大盘分析 ======
    print(f"\n{'─'*62}")
    print(f"  大盘分析")
    print(f"{'─'*62}")

    indices = MarketAnalyzer.fetch_indices()
    breadth = MarketAnalyzer.analyze_breadth(s.snapshot)
    limitup_q = MarketAnalyzer.analyze_limitup_quality(s.snapshot)

    # ── 指数行情 ──
    exch_amt = MarketAnalyzer.get_exchange_amounts(s.snapshot)
    if indices:
        print(f"\n  【主要指数】")
        for code, idx in indices.items():
            arrow = "↑" if idx["change_pct"] >= 0 else "↓"
            print(f"  {idx['name']:<8} {idx['price']:>8.2f}  {arrow} {idx['change_pct']:+6.2f}%")
    else:
        print(f"\n  （指数数据获取失败）")

    # ── 量价对比 ──
    if exch_amt:
        print(f"\n  【量价对比】")
        print(f"  沪市成交 {exch_amt['sh']:.0f}亿  深市成交 {exch_amt['sz']:.0f}亿  "
              f"合计 {exch_amt['total']:.0f}亿")
        sh_trend = MarketAnalyzer.get_index_trend("sh000001", "上证指数")
        if sh_trend:
            vcp = sh_trend["vol_change_pct"]
            arr = "↑" if vcp > 0 else "↓"
            print(f"  上证量 {arr} {abs(vcp):.1f}%（今日vs昨日）")

    # ── 涨跌统计 ──
    if breadth:
        print(f"\n  【涨跌统计】")
        print(f"  上涨 {breadth['advance']:<5}  下跌 {breadth['decline']:<5}  平盘 {breadth['flat']}")
        print(f"  涨停 {breadth['limit_up']:<4}  跌停 {breadth['limit_down']:<4}"
              f"  涨幅≥5% {breadth['rise_5']:<4}  跌幅≤-5% {breadth['fall_5']:<4}")
        print(f"  涨跌比 {breadth['ad_ratio']:<6}  涨跌停比 {breadth['ul_ratio']:<6}"
              f"  全市场成交 {breadth['total_amount']:.0f}亿")

    # ── 涨停质量 ──
    if limitup_q:
        print(f"\n  【涨停质量】")
        print(f"  今日涨停共 {limitup_q['total']} 只")
        print(f"  ┌ 一字板 {limitup_q['one_word']:>3}只 ({limitup_q['one_word_pct']:.1f}%)"
              f"  ─ 开盘即封涨停（最强）")
        print(f"  ├ 高开涨停 {limitup_q['gap_up']:>3}只 ({limitup_q['gap_up_pct']:.1f}%)"
              f"  ─ 跳空高开后涨停（较强）")
        print(f"  └ 低开拉起 {limitup_q['pulled']:>3}只 ({limitup_q['pulled_pct']:.1f}%)"
              f"  ─ 盘中拉涨停（偏弱）")

    # ── 大盘健康度 ──
    health, pos, warns = MarketAnalyzer.judge(indices, breadth, limitup_q)
    print(f"\n  【大盘健康度】")
    print(f"  ▶ {health}")
    for p in pos:
        print(f"    ✓ {p}")
    for w in warns:
        print(f"    ⚠ {w}")

    # ====== 5. 关注个股深度分析 ======
    print(f"\n{'─'*62}")
    print(f"  关注个股分析")
    print(f"{'─'*62}")

    for sym in watch_stocks:
        try:
            account = 80000
            pos = None
            label = sym
            if sym == "002050":
                pos = {"shares": 100, "avg_price": 51.44}
                label = "三花智控"

            print(f"\n  ▼ {label}（{sym}）{'持仓100股@51.44' if pos else ''}")
            print(f"  {'─'*58}")

            report, name, price = analyze_stock(sym, account, pos)
            for line in report.split('\n'):
                stripped = line.strip()
                if stripped:
                    print(f"  {stripped}")

        except Exception as e:
            print(f"  ✗ {sym} 分析失败: {e}")

    # ====== 6. 总结 ======
    bullish = sum(1 for r in s.results if r["trend"] == "多头")
    print(f"\n{'='*62}")
    print(f"  总结")
    print(f"{'='*62}")
    if breadth:
        ad_dir = "上涨" if breadth['advance'] > breadth['decline'] else "下跌"
        total_amt = round(sum(v.get('amount', 0) for v in s.snapshot.values()) / 1e8, 0)
        print(f"  大盘: {ad_dir}  涨跌比{breadth['ad_ratio']} | 涨停{breadth['limit_up']}跌停{breadth['limit_down']}")
        print(f"       成交{total_amt:.0f}亿 | 健康度: {health}")
    print(f"  全市场: {len(s.snapshot)}只 | 流动性达标: {len(s.candidates)}只 | 多头趋势: {bullish}只")
    print(f"  威科夫信号: SOS {sum(1 for r in valid if 'SOS' in r['wyckoff_sig'])}只"
          f"  Spring {sum(1 for r in valid if 'Spring' in r['wyckoff_sig'])}只"
          f"  Upthrust {sum(1 for r in valid if 'Upthrust' in r['wyckoff_sig'])}只"
          f"  LPS {sum(1 for r in valid if 'LPS' in r['wyckoff_sig'])}只"
          f"  EVR {sum(1 for r in valid if 'EVR' in r['wyckoff_sig'])}只"
          f"  Compression {sum(1 for r in valid if 'Compression' in r['wyckoff_sig'])}只")
    # 阶段分布
    phases = {
        r["phase"][:10] for r in s.results if r.get("phase") and r["phase"] not in ("数据不足",)
    }
    if phases:
        print(f"  阶段分布: {', '.join(sorted(phases))}")
    print(f"  关注: {', '.join(watch_stocks)}")
    print(f"\n  ⚠️  基于技术分析，仅供参考，不构成投资建议")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run(watch_stocks=["002050"])
