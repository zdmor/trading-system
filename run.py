"""
交易系统主入口 — 全市场扫描 + 个股分析 一站式输出
"""
import sys, os, time, yaml, json
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

TRACKING_FILE = os.path.join(os.path.dirname(__file__), "last_picks.json")

from scanner import Scanner, WyckoffAnalyzer
from main import DataFetcher, Analyzer, Strategy, Report, load_config
from market import MarketAnalyzer
from data_providers import AkshareProvider
from news_service import get_top_signal_news, format_news_block, get_market_news, format_market_news_block
from notifier import push_report, send_serverchan, send_pushplus, send_card, make_div, make_hr, make_note


def analyze_stock(symbol, account_value=80000, position=None, market_health=None):
    """分析单只股票，返回 (report_text, name, price, op_panel, trade_data)"""
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

    # ── 威科夫阶段分析 ──
    phase_label, phase_detail = "", ""
    wyckoff_all_signals = []
    trend_dir = trend["direction"]
    if len(daily) >= 50 and trend_dir != "未知":
        c = daily["close"].values.astype(float)
        h = daily["high"].values.astype(float)
        l = daily["low"].values.astype(float)
        v = daily["volume"].values.astype(float)
        phase_label, phase_detail, _ = WyckoffAnalyzer.detect_phase(
            c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir, []
        )
        # 获取全部威科夫信号供评分使用
        wyckoff_all_signals, _ = WyckoffAnalyzer.analyze_all(
            c.tolist(), h.tolist(), l.tolist(), v.tolist(), trend_dir
        )

    # ── 多因子评分 ──
    scoring_result = None
    try:
        from scoring import StockScorer
        scorer = StockScorer(
            df=daily, price=price, trend=trend, vol=vol, levels=levels,
            stop_price=stop_price, exit_prices=exit_prices,
            wyckoff_signals=wyckoff_all_signals, wyckoff_phase=phase_label,
            position=position, symbol=symbol,
        )
        scoring_result = scorer.compute()
    except Exception:
        pass

    # ── 仓位计算（带评分） ──
    pos = strategy.position_plan(
        stop_price, entry_status=entry_check.get("signal"),
        market_health=market_health,
        composite_score=scoring_result["composite_score"] if scoring_result else None
    )

    # ── 操作面板（紧凑摘要） ──
    status = entry_check.get("signal", "未知")
    shares = pos.get("current_shares", 0)
    avg_p = position["avg_price"] if position else 0
    pnl = (price - avg_p) * shares if avg_p > 0 else 0
    rec_action = trend.get("recommendation", "观望")
    pnl_icon = "↑" if pnl >= 0 else "↓"

    # 评分摘要
    score_str = ""
    if scoring_result:
        score_str = f"  评分:{scoring_result['composite_score']}/100 {scoring_result['action']}"

    # total_ratio 可能是比值(0.82)或百分比(82)，统一处理
    raw_ratio = pos.get("total_ratio", 0)
    ratio_pct = raw_ratio if raw_ratio > 1 else raw_ratio * 100

    op_panel = (
        f"    现价: {price:<8.2f}  持仓: {shares}股@{avg_p:.2f}  "
        f"盈亏: {pnl_icon} {abs(pnl):.0f}\n"
        f"    威科夫: {phase_label} | 趋势: {trend_dir}\n"
        f"    止损: {stop_price:<8.2f}  止盈: {exit_prices[0]:<8.2f}  "
        f"(仓位: {ratio_pct:.0f}%)\n"
        f"    信号: {status} → {rec_action}{score_str}"
    )

    # ── 结构化交易数据（供推送使用） ──
    supports_str = ", ".join(str(s) for s in levels["supports"][:3]) if levels.get("supports") else "无"
    exit_str = ", ".join(str(e) for e in exit_prices[:3])
    add_shares = pos.get("add_shares", 0) or max(0, pos.get("max_shares", 0) - shares)
    total_shares = shares + add_shares
    avg_cost = ((avg_p * shares) + (price * add_shares)) / total_shares if total_shares > 0 else 0
    stop_loss_amt = (price - stop_price) * total_shares

    trade_data = {
        "name": name,
        "symbol": symbol,
        "price": price,
        "shares": shares,
        "avg_price": avg_p,
        "pnl": pnl,
        "pnl_pct": (pnl / (avg_p * shares) * 100) if avg_p * shares > 0 else 0,
        "stop_price": stop_price,
        "exit_prices": exit_str,
        "ratio_pct": ratio_pct,
        "phase": phase_label,
        "phase_detail": phase_detail,
        "trend": trend_dir,
        "signal": status,
        "action": rec_action,
        "supports": supports_str,
        "add_shares": add_shares,
        "total_shares": total_shares,
        "stop_loss_amt": stop_loss_amt,
        "score": scoring_result["composite_score"] if scoring_result else None,
        "score_action": scoring_result["action"] if scoring_result else None,
        "score_level": scoring_result["level"] if scoring_result else None,
        "scoring_result": scoring_result,
    }

    # ── 完整报告 ──
    report = Report(symbol, name, account_value, price_date)
    phase_str = f"\n  【威科夫阶段】\n  {phase_label}\n  {phase_detail}" if phase_label else ""
    output = report.header()
    fin_metrics = AkshareProvider.get_financial_indicators(symbol)
    moneyflow = AkshareProvider.get_stock_moneyflow(symbol)
    output += report.financial_section(fin_metrics)
    output += report.moneyflow_section(moneyflow)
    output += phase_str
    output += report.trend_section(trend)
    output += report.volatility_section(vol)
    output += report.levels_section(levels)
    output += report.position_section(pos, current_price=price)
    output += report.signal_section(entry_check)
    output += report.scoring_section(scoring_result)
    output += report.recommendation_section(trend, entry_check, stop_price, exit_prices, pos, current_price=price)
    output += report.footer()
    return output, name, price, op_panel, trade_data


def _track_prev_picks():
    """读取上次选股记录，获取最新价格计算涨跌幅

    Returns:
        (date_str, [{"name", "code", "signal", "prev_price", "current_price", "change_pct"}, ...])
    """
    if not os.path.exists(TRACKING_FILE):
        return None, []
    try:
        with open(TRACKING_FILE) as f:
            data = json.load(f)
    except Exception:
        return None, []

    fetcher = DataFetcher()
    results = []
    for p in data.get("picks", []):
        try:
            daily = fetcher.get_daily(p["code"], days=5)
            current = float(daily["close"].iloc[-1])
            prev = p["price"]
            change_pct = (current - prev) / prev * 100
            results.append({
                "name": p.get("name", p["code"]),
                "code": p["code"],
                "signal": p.get("signal", ""),
                "score": p.get("score", 0),
                "prev_price": prev,
                "current_price": current,
                "change_pct": round(change_pct, 2),
            })
        except Exception:
            results.append({
                "name": p.get("name", p["code"]),
                "code": p["code"],
                "signal": p.get("signal", ""),
                "score": p.get("score", 0),
                "prev_price": p["price"],
                "current_price": 0,
                "change_pct": None,
            })
    return data.get("date"), results


def _save_picks(picks_list):
    """保存今日选股推荐到文件，供次日跟踪"""
    if not picks_list:
        return
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "picks": [
            {
                "code": r["code"].split(".")[1],
                "name": r["name"],
                "signal": r["wyckoff_sig"],
                "score": r["wyckoff_score"],
                "price": r["price"],
            }
            for r in picks_list[:8]
        ],
    }
    try:
        with open(TRACKING_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


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
    print("  ▶ 获取全市场数据...")

    s = Scanner()
    t0 = time.time()
    s.fetch_all_stocks()
    s.filter_candidates(min_amount)
    s.build_industry_map()
    # 直接跑80只的趋势+威科夫分析（scanner内部会调Tencent API）
    s.run_analysis(max_scan)

    print(f"  ✓ 全市场{len(s.snapshot)}只 | 流动性通过{len(s.candidates)}只 | Top{max_scan}分析完成")
    print(f"  ✓ 用时 {time.time()-t0:.1f}s\n")

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

    # ====== 3a. 昨日选股回顾 ======
    prev_date, track_results = _track_prev_picks()
    if track_results:
        print(f"\n{'─'*62}")
        print(f"  昨日选股回顾 ({prev_date})")
        print(f"{'─'*62}")
        print(f"  {'#':<3} {'名称':<8} {'信号':<10} {'昨收':<8} {'现价':<8} {'涨幅':<8}")
        print(f"  {'-'*55}")
        for i, r in enumerate(track_results[:8]):
            if r["change_pct"] is not None:
                arr = "↑" if r["change_pct"] >= 0 else "↓"
                print(f"  {i+1:<3} {r['name']:<8} {r['signal']:<10} {r['prev_price']:<8.2f} {r['current_price']:<8.2f} {arr}{abs(r['change_pct']):<7.2f}%")
            else:
                print(f"  {i+1:<3} {r['name']:<8} {r['signal']:<10} {r['prev_price']:<8.2f} {'N/A':<8} {'数据失败':<8}")

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

    # ====== 3c. 选股新闻背景 ======
    if picks:
        name_map = {r["code"]: r["name"] for r in picks[:8]}
        news_dict = get_top_signal_news(picks[:8], limit_per_stock=2)
        if news_dict:
            print(f"\n{format_news_block(news_dict, name_map)}")

    # 保存今日选股供次日跟踪
    _save_picks(picks)

    # ====== 4. 大盘分析（提前，供持仓分析参考） ======
    print(f"\n{'─'*62}")
    print(f"  大盘分析")
    print(f"{'─'*62}")

    indices = MarketAnalyzer.fetch_indices()
    breadth = MarketAnalyzer.analyze_breadth(s.snapshot)
    limitup_q = MarketAnalyzer.analyze_limitup_quality(s.snapshot)

    # ── 宏观数据（AKShare） ──
    macro = MarketAnalyzer.get_macro_context()
    if macro:
        print(f"\n  【宏观速览】")
        if macro.get("lpr_1y") is not None: print(f"  LPR 1Y: {macro['lpr_1y']:.2f}%  ", end="")
        if macro.get("lpr_5y") is not None: print(f"LPR 5Y: {macro['lpr_5y']:.2f}%  ", end="")
        if macro.get("cpi") is not None: print(f"CPI: {macro['cpi']:.1f}%  ", end="")
        if macro.get("pmi") is not None: print(f"PMI: {macro['pmi']:.1f}%  ", end="")
        if macro.get("m2") is not None: print(f"M2: {macro['m2']:.1f}%", end="")
        print()

    # ── 市场快讯（财联社） ──
    market_news = get_market_news(5)
    if market_news:
        print(f"\n{format_market_news_block(market_news)}")

    # ── 北向资金（AKShare） ──
    nb = MarketAnalyzer.get_northbound_context()
    if nb and nb.get("合计") != 0:
        sh_arr = "↑" if nb.get("沪股通", 0) > 0 else "↓"
        sz_arr = "↑" if nb.get("深股通", 0) > 0 else "↓"
        total_arr = "↑" if nb.get("合计", 0) > 0 else "↓"
        print(f"  【北向资金】")
        print(f"  沪股通 {sh_arr} {abs(nb['沪股通']):.1f}亿  "
              f"深股通 {sz_arr} {abs(nb['深股通']):.1f}亿  "
              f"合计 {total_arr} {abs(nb['合计']):.1f}亿")

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
    has_realtime = any(v.get("amount", 0) > 0 for v in s.snapshot.values())
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
    if breadth and has_realtime:
        print(f"\n  【涨跌统计】")
        print(f"  上涨 {breadth['advance']:<5}  下跌 {breadth['decline']:<5}  平盘 {breadth['flat']}")
        print(f"  涨停 {breadth['limit_up']:<4}  跌停 {breadth['limit_down']:<4}"
              f"  涨幅≥5% {breadth['rise_5']:<4}  跌幅≤-5% {breadth['fall_5']:<4}")
        print(f"  涨跌比 {breadth['ad_ratio']:<6}  涨跌停比 {breadth['ul_ratio']:<6}"
              f"  全市场成交 {breadth['total_amount']:.0f}亿")

    # ── 涨停质量 ──
    if limitup_q and has_realtime:
        print(f"\n  【涨停质量】")
        print(f"  今日涨停共 {limitup_q['total']} 只")
        print(f"  ┌ 一字板 {limitup_q['one_word']:>3}只 ({limitup_q['one_word_pct']:.1f}%)"
              f"  ─ 开盘即封涨停（最强）")
        print(f"  ├ 高开涨停 {limitup_q['gap_up']:>3}只 ({limitup_q['gap_up_pct']:.1f}%)"
              f"  ─ 跳空高开后涨停（较强）")
        print(f"  └ 低开拉起 {limitup_q['pulled']:>3}只 ({limitup_q['pulled_pct']:.1f}%)"
              f"  ─ 盘中拉涨停（偏弱）")

    # ── 大盘健康度 ──
    health, pos_reasons, warns = MarketAnalyzer.judge(indices, breadth, limitup_q)
    print(f"\n  【大盘健康度】")
    print(f"  ▶ {health}")
    for p in pos_reasons:
        print(f"    ✓ {p}")
    for w in warns:
        print(f"    ⚠ {w}")

    # ── 概念板块 Top5（AKShare） ──
    concepts = MarketAnalyzer.get_concept_boards_top(5)
    if concepts:
        print(f"\n  【热门概念】")
        for name, pct in concepts:
            arr = "↑" if pct > 0 else "↓"
            print(f"  {arr} {name:<16} {abs(pct):+.2f}%")

    # ====== 5. 关注个股深度分析 ======
    print(f"\n{'─'*62}")
    print(f"  我的持仓")
    print(f"{'─'*62}")

    last_price = 0
    last_pnl = 0
    last_trade = {}
    for sym in watch_stocks:
        try:
            account = 80000
            pos = None
            label = sym
            if sym == "002050":
                pos = {"shares": 100, "avg_price": 51.44}
                label = "三花智控"

            report, name, price, op_panel, trade_data = analyze_stock(sym, account, pos, market_health=health)
            # 存持仓数据供后续推送使用
            last_price = price
            last_trade = trade_data
            if pos:
                last_pnl = (price - pos["avg_price"]) * pos["shares"]
            # ── 操作面板（紧凑） ──
            lines = op_panel.split("\n")
            print(f"\n  ▼ {label}（{sym}）")
            for line in lines:
                print(f"  {line}")

            # ── 详细报告（可折叠感） ──
            print(f"\n  {'·'*58}")
            print(f"  详细报告")
            print(f"  {'·'*58}")
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

    # ====== 7. 推送通知 ======
    cfg_path = os.path.join(os.path.dirname(__file__), "config", "scanner.yaml")
    try:
        with open(cfg_path, encoding="utf-8") as f:
            scfg = yaml.safe_load(f)
        notify = scfg.get("notify", {})
        pushplus_token = notify.get("pushplus_token", "")
        serverchan_key = notify.get("serverchan_key", "")
        webhook_url = notify.get("webhook_url", "")
    except Exception:
        pushplus_token = ""
        serverchan_key = ""
        webhook_url = ""

    if pushplus_token or serverchan_key or webhook_url:
        ad_dir = "上涨" if breadth and breadth['advance'] > breadth['decline'] else "下跌"
        health_str = health if health else "N/A"
        signal_counts = (
            f"SOS {sum(1 for r in valid if 'SOS' in r['wyckoff_sig'])} | "
            f"Spring {sum(1 for r in valid if 'Spring' in r['wyckoff_sig'])} | "
            f"Upthrust {sum(1 for r in valid if 'Upthrust' in r['wyckoff_sig'])} | "
            f"LPS {sum(1 for r in valid if 'LPS' in r['wyckoff_sig'])} | "
            f"EVR {sum(1 for r in valid if 'EVR' in r['wyckoff_sig'])} | "
            f"Compression {sum(1 for r in valid if 'Compression' in r['wyckoff_sig'])}"
        )
        pick_lines = ""
        for i, r in enumerate(picks[:5]):
            sym = r["code"].split(".")[1]
            pick_lines += f"\n  {i+1}. {r['name']}({sym}) {r['wyckoff_sig']} 评分{r['wyckoff_score']}"

        # ── 大盘概要 ──
        detail = (
            f"大盘: {ad_dir} | 健康度: {health_str}\n"
            f"信号分布: {signal_counts}\n"
            f"多头趋势: {bullish}只\n"
            f"选股推荐:{pick_lines}"
        )

        # ── 选股回顾 ──
        if track_results:
            detail += "\n\n── 昨日选股 ──"
            for r in track_results[:5]:
                if r["change_pct"] is not None:
                    arr = "+" if r["change_pct"] >= 0 else ""
                    detail += f"\n{r['name']} {arr}{r['change_pct']:.1f}%"
                else:
                    detail += f"\n{r['name']} 数据失败"

        # ── 持仓详情 ──
        if last_trade:
            t = last_trade
            pnl_icon = "+" if t["pnl"] >= 0 else ""
            detail += (
                f"\n\n──── 持仓 ────\n"
                f"{t['name']}({t['symbol']})\n"
                f"现价: {t['price']:.2f} | 持仓: {t['shares']}股@{t['avg_price']:.2f}\n"
                f"盈亏: {pnl_icon}{t['pnl']:.0f}({pnl_icon}{t['pnl_pct']:.1f}%)\n"
                f"仓位: {t['ratio_pct']:.0f}% | 止损: {t['stop_price']:.2f}\n"
                f"止盈: {t['exit_prices']} | 支撑: {t['supports']}\n"
                f"威科夫: {t['phase']} | 趋势: {t['trend']}\n"
                f"信号: {t['signal']}\n"
            )
            if t.get("score") is not None:
                detail += f"评分: {t['score']}/100 {t['score_action']}\n"
            detail += (
                f"\n── 交易计划 ──\n"
                f"操作: {t['action']}"
            )
            if t["add_shares"] > 0:
                detail += f"\n加仓: {t['add_shares']}股 (总计{t['total_shares']}股, 均价{t['avg_price']:.2f}→{(t['avg_price']*t['shares']+t['price']*t['add_shares'])/(t['shares']+t['add_shares']):.2f})"
            detail += f"\n止损线: {t['stop_price']:.2f} (亏{t['stop_loss_amt']:.0f})\n止盈: {t['exit_prices']}"

        if pushplus_token:
            send_pushplus(pushplus_token, f"A股交易分析 {today}", detail)
        elif serverchan_key:
            send_serverchan(serverchan_key, f"A股交易分析 {today}", detail)
        elif webhook_url:
            push_report(webhook_url, f"A股交易分析 {today}", detail)


if __name__ == "__main__":
    run(watch_stocks=["002050"])
