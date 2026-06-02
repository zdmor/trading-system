"""
Report engine: modular quick/full output for the trading system.
Usage:
    from reporter import quick_report, full_report, holdings_status_only
"""
import os, json, time
from datetime import datetime, timedelta
from typing import Optional

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

# ── helpers ──
def _div(char="=", n=58):
    return char * n

def _section(title):
    return f"\n  {title}\n  {'-' * 48}"

def _save_module(key, text):
    d = os.path.join(REPORTS_DIR, "modules")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{key}.md"), "w", encoding="utf-8") as f:
        f.write(text)

def _today():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def _sep():
    return f"\n{'─'*58}"

def _sub(name):
    return f"\n   {name}\n   {'·'*42}"

# ============================================================
# 快速模块 —— 纯渲染/无内部API调用
# ============================================================

def header(can_trade=True, trade_status="交易中"):
    return (
        f"\n{_div()}\n"
        f"  A股交易系统  {_today()}\n"
        f"  {'>' if can_trade else '|'} {trade_status}\n"
        f"{_div()}\n"
    )


def market_status(indices=None, breadth=None, health=None):
    """大盘状态判断 —— 指数 + 涨跌统计 + 健康度"""
    lines = [_section("大盘状态判断")]

    if indices:
        for code, idx in indices.items():
            arrow = "↑" if idx["change_pct"] >= 0 else "↓"
            lines.append(f"  {idx['name']:<6} {idx['price']:>8.2f}  {arrow} {idx['change_pct']:+6.2f}%")

    if breadth:
        lines.append(f"  上涨 {breadth['advance']}  下跌 {breadth['decline']}  涨跌比 {breadth['ad_ratio']}")
        lines.append(f"  涨停 {breadth['limit_up']}  跌停 {breadth['limit_down']}  成交 {breadth['total_amount']:.0f}亿")

    if health:
        lines.append(f"  大盘健康度: {health[0]}")
        for r in health[1][:3]:
            lines.append(f"    + {r}")

    text = "\n".join(lines) + "\n"
    _save_module("market_status", text)
    return text


def holdings_status(holdings_data: list, scanner=None, analysis_map=None,
                    exit_map=None):
    """持仓操作建议 —— 现价/盈亏/趋势/评分/建议
    exit_map: {code: {"triggered": bool, "logic_broken": bool, "trend_exhaust": bool, "messages": [str]}}
    """
    lines = [_sep(), "  持仓操作建议", f"{'─'*58}"]
    alerts = []

    for h in holdings_data:
        name = h.get("name", "")
        code = h.get("code", "")
        price = h.get("price", 0)
        shares = h.get("shares", 0)
        avg_p = h.get("avg_price", 0)
        pnl = (price - avg_p) * shares if avg_p > 0 else 0
        pnl_pct = (price / avg_p - 1) * 100 if avg_p > 0 else 0
        pnl_arrow = "↑" if pnl >= 0 else "↓"

        lines.append(f"\n  {name} {code}")
        lines.append(f"  现价 {price:<8.2f}  {pnl_arrow} {abs(pnl_pct):.1f}%  浮亏 {pnl_arrow}{abs(pnl):.0f}")

        # 今日涨幅
        day_info = ""
        if scanner:
            snap = scanner.snapshot.get(
                f"sh.{code}" if code.startswith("6") else f"sz.{code}", {})
            day_chg = snap.get("change_pct", 0)
            if day_chg:
                darr = "↑" if day_chg >= 0 else "↓"
                day_info = f"今日 {darr} {abs(day_chg):.2f}%"

        # 分析结论 (来自 analysis_map)
        analysis_info = ""
        if analysis_map and code in analysis_map:
            a = analysis_map[code]
            trend = a.get("trend", "")
            sig = a.get("signal", "")
            score = a.get("score")
            action = a.get("action", "")
            stop = a.get("stop_price", 0)
            phase = a.get("phase", "")
            detail = a.get("wyckoff_detail", "")

            parts = []
            if trend: parts.append(f"趋势:{trend}")
            if sig and sig not in ("-", "无信号"): parts.append(f"信号:{sig}")
            if phase and phase not in ("数据不足",): parts.append(f"阶段:{phase[:12]}")
            analysis_info = "  |  ".join(parts)

            score_str = f"  评分 {score}/100  {action}" if score is not None else ""
            stop_str = f"  止损 {stop:.2f}" if stop > 0 else ""
            extra = f"{score_str}{stop_str}"
        else:
            extra = ""

        info_line = "  ".join(filter(None, [day_info, analysis_info]))
        if info_line:
            lines.append(f"  {info_line}")
        if extra:
            lines.append(f"  {extra}")

        # 退出规则状态
        if exit_map and code in exit_map:
            es = exit_map[code]
            if es.get("triggered"):
                for msg in es.get("messages", []):
                    lines.append(f"  [!] {msg}")

        # 常规预警
        if pnl_pct < -15:
            alerts.append(f"  @ {name}: 深套 {pnl_pct:.0f}%，关注止损")
        elif pnl_pct < -5:
            alerts.append(f"  ! {name}: 浮亏 {pnl_pct:.0f}%")

    if alerts:
        lines.append("")
        for a in alerts:
            lines.append(a)

    return "\n".join(lines) + "\n"


def stock_picks(scanner, top_n=10):
    """选股推荐 —— 信号列表 + 三买 + 威科夫信号排名"""
    lines = [_sep(), "  选股推荐", f"{'─'*58}"]

    if not hasattr(scanner, 'results') or not scanner.results:
        lines.append("  数据收集中...")
        return "\n".join(lines) + "\n"

    valid = [r for r in scanner.results if r.get("wyckoff_sig") not in ("-", "无信号", "数据不足", "无数据")]
    if not valid:
        lines.append("  今日无显著信号")
        return "\n".join(lines) + "\n"

    # 三买
    chan_buy = [r for r in scanner.results if r.get("chan_third_buy") and r["chan_third_buy"].get("signal") == "三买"]
    if chan_buy:
        names = ", ".join(f"{r['name']}({r['code'].split('.')[1]})" for r in chan_buy[:3])
        lines.append(f"  三买信号: {len(chan_buy)}只 → {names}")
        if len(chan_buy) > 3:
            lines.append(f"            ...等{len(chan_buy)}只")

    # 威科夫信号计数
    sig_counts = {}
    for r in valid:
        sig = r["wyckoff_sig"]
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    sig_str = "  ".join(f"{k} {v}只" for k, v in sorted(sig_counts.items()))
    if sig_str:
        lines.append(f"  威科夫: {sig_str}")

    # Top N 明细
    lines.append("")
    lines.append(f"  {'信号':<10} {'代码':<8} {'名称':<8} {'得分':<4} {'现价':<7} {'行业':<12}")
    lines.append(f"  {'-'*58}")
    for r in valid[:top_n]:
        sym = r["code"].split(".")[1]
        sig = r["wyckoff_sig"]
        ind = (r.get("industry", "") or "")[:10]
        lines.append(f"  {sig:<10} {sym:<8} {r['name']:<8} {r['wyckoff_score']:<4} {r['price']:<7.2f} {ind:<12}")

    # 买入信号+多头 → 推荐关注
    buy_signals = {"SOS", "Spring", "LPS", "EVR", "Compression"}
    buys = [r for r in valid if r["wyckoff_sig"] in buy_signals and r["trend"] == "多头"]
    if buys:
        lines.append(f"\n  关注池 (买入信号+多头): {len(buys)}只")
        for r in buys[:5]:
            sym = r["code"].split(".")[1]
            lines.append(f"  {r['name']}({sym}) {r['wyckoff_sig']} {r['wyckoff_score']}分")

    return "\n".join(lines) + "\n"


def sector_analysis(scanner):
    """板块分析 —— 行业分布统计"""
    lines = [_sep(), "  板块分析", f"{'─'*58}"]

    if not hasattr(scanner, 'results') or not scanner.results:
        lines.append("  数据收集中...")
        return "\n".join(lines) + "\n"

    sectors = {}
    for r in scanner.results:
        ind = r.get("industry", "其他")
        if ind not in sectors:
            sectors[ind] = {"count": 0, "signals": 0}
        sectors[ind]["count"] += 1
        if r["wyckoff_sig"] not in ("-", "无信号", "数据不足", "无数据"):
            sectors[ind]["signals"] += 1

    sorted_sec = sorted(sectors.items(), key=lambda x: -x[1]["signals"])
    lines.append(f"  {'板块':<30} {'总数':<4} {'信号':<4}")
    lines.append(f"  {'-'*42}")
    for ind, info in sorted_sec[:15]:
        tag = f" [{info['signals']}信号]" if info['signals'] > 0 else ""
        lines.append(f"  {ind:<30} {info['count']:<4}{tag}")

    return "\n".join(lines) + "\n"


def picks_performance(days_lookback=10):
    """选股表现 —— 1/5/10天 历史选股涨跌幅追踪"""
    lines = [_sep(), "  选股表现回顾", f"{'─'*58}"]

    TRACKING_FILE = os.path.join(os.path.dirname(__file__), "last_picks.json")
    if not os.path.exists(TRACKING_FILE):
        lines.append("  无历史选股记录")
        return "\n".join(lines) + "\n"

    try:
        with open(TRACKING_FILE) as f:
            data = json.load(f)
    except Exception:
        lines.append("  读取记录失败")
        return "\n".join(lines) + "\n"

    picks = data.get("picks", [])
    if not picks:
        lines.append("  无历史选股记录")
        return "\n".join(lines) + "\n"

    try:
        from data_providers.tushare_provider import _get_pro
        pro = _get_pro()
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=days_lookback + 5)).strftime("%Y%m%d")

        ts_codes = ','.join(
            p['code'] + ('.SH' if p['code'].startswith('6') else '.SZ')
            for p in picks[:10]
        )
        df = pro.daily(adj='qfq', ts_code=ts_codes, start_date=start, end_date=end)
    except Exception:
        lines.append("  查询历史数据失败")
        return "\n".join(lines) + "\n"

    if df is None or df.empty:
        lines.append("  无历史数据")
        return "\n".join(lines) + "\n"

    lines.append(f"  {'名称':<8} {'信号':<10} {'选股价':<8} {'1天':<8} {'5天':<8} {'10天':<8}")
    lines.append(f"  {'-'*58}")

    for p in picks[:10]:
        ts = p['code'] + ('.SH' if p['code'].startswith('6') else '.SZ')
        sd = df[df['ts_code'] == ts].sort_values('trade_date')
        if len(sd) < 2:
            continue
        closes = sd['close'].astype(float).tolist()
        pick_price = p.get('price', closes[-1])

        def perf(offset):
            if offset >= len(closes):
                return None
            px = closes[-1 - offset]
            return (px / pick_price - 1) * 100 if pick_price > 0 else None

        d1 = perf(0)
        d5 = perf(min(4, len(closes)-1))
        d10 = perf(min(9, len(closes)-1))

        def fmt(v):
            if v is None: return "N/A".center(8)
            arr = "↑" if v >= 0 else "↓"
            return f"{arr}{abs(v):.1f}%"

        lines.append(f"  {p.get('name','?'):<8} {p.get('signal',''):<10} {pick_price:<8.2f} {fmt(d1):<8} {fmt(d5):<8} {fmt(d10):<8}")

    return "\n".join(lines) + "\n"


# ============================================================
# 全景模块 (可能含内部API调用)
# ============================================================

def market_deep(scanner=None):
    """大盘深度：PE分位/北向/宏观/概念/量价/涨停质量/市场快讯
    内部调 AKShare/Tushare API ~70s
    """
    lines = [_sep(), "  大盘深度", f"{'─'*58}"]

    try:
        from market import MarketAnalyzer
    except ImportError:
        lines.append("  MarketAnalyzer 不可用")
        return "\n".join(lines) + "\n"

    # 宏观速览
    try:
        macro = MarketAnalyzer.get_macro_context()
        if macro:
            parts = []
            if macro.get("lpr_1y") is not None: parts.append(f"LPR1Y {macro['lpr_1y']}%")
            if macro.get("lpr_5y") is not None: parts.append(f"LPR5Y {macro['lpr_5y']}%")
            if macro.get("cpi") is not None: parts.append(f"CPI {macro['cpi']}%")
            if macro.get("pmi") is not None: parts.append(f"PMI {macro['pmi']}%")
            if macro.get("m2") is not None: parts.append(f"M2 {macro['m2']}%")
            if parts: lines.append(f"  宏观: {'  '.join(parts)}")
    except Exception:
        pass

    # PE分位
    try:
        pe_info, pe_cap = MarketAnalyzer.get_pe_percentile()
        if pe_info:
            lines.append(f"  上证PE(TTM): {pe_info['pe']}  近5年分位: {pe_info['percentile']}%  {pe_info['level']}")
    except Exception:
        pass

    # 北向资金
    try:
        nb = MarketAnalyzer.get_northbound_context()
        if nb and nb.get("合计", 0) != 0:
            sh_a = "↑" if nb.get("沪股通", 0) > 0 else "↓"
            sz_a = "↑" if nb.get("深股通", 0) > 0 else "↓"
            tot_a = "↑" if nb.get("合计", 0) > 0 else "↓"
            lines.append(f"  北向: 沪 {sh_a}{abs(nb['沪股通']):.1f}亿  深 {sz_a}{abs(nb['深股通']):.1f}亿  合计 {tot_a}{abs(nb['合计']):.1f}亿")
    except Exception:
        pass

    # 热门概念 Top5
    try:
        concepts = MarketAnalyzer.get_concept_boards_top(5)
        if concepts:
            strs = [f"{'↑' if p>0 else '↓'}{n} {abs(p):+.1f}%" for n, p in concepts]
            lines.append(f"  概念: {'  '.join(strs[:3])}")
    except Exception:
        pass

    # 量价对比
    if scanner and hasattr(scanner, 'snapshot'):
        try:
            exch = MarketAnalyzer.get_exchange_amounts(scanner.snapshot)
            if exch:
                lines.append(f"  成交: 沪 {exch['sh']:.0f}亿  深 {exch['sz']:.0f}亿  合计 {exch['total']:.0f}亿")
                sh_trend = MarketAnalyzer.get_index_trend("sh000001", "上证指数")
                if sh_trend:
                    vcp = sh_trend["vol_change_pct"]
                    arr = "↑" if vcp > 0 else "↓"
                    lines.append(f"  上证量: 今日vs昨日 {arr} {abs(vcp):.1f}%")
        except Exception:
            pass

        # 涨停质量
        try:
            limitup_q = MarketAnalyzer.analyze_limitup_quality(scanner.snapshot)
            if limitup_q:
                lines.append(f"  涨停质量: 共{limitup_q['total']}只  一字{limitup_q['one_word']}({limitup_q['one_word_pct']:.0f}%)  高开{limitup_q['gap_up']}({limitup_q['gap_up_pct']:.0f}%)  低开拉起{limitup_q['pulled']}({limitup_q['pulled_pct']:.0f}%)")
        except Exception:
            pass

    # 市场快讯
    try:
        from news_service import get_market_news, format_market_news_block
        news = get_market_news(5)
        if news:
            block = format_market_news_block(news)
            # 取前3行简化为一行
            brief = block.replace("\n", " ").strip()[:120]
            if brief:
                lines.append(f"  快讯: {brief}")
    except Exception:
        pass

    text = "\n".join(lines) + "\n"
    return text


def holdings_deep(holdings_data, stock_analyses=None, scanner=None):
    """个股全景分析 —— 每个持仓的8因子评分 + 操盘面板
    stock_analyses: dict {code: {trade_data, op_panel, name, price}}
    """
    lines = [_sep(), "  个股全景分析", f"{'─'*58}"]

    if not stock_analyses:
        lines.append("  无深度分析数据（需要先运行 analyze_stock）")
        return "\n".join(lines) + "\n"

    for h in holdings_data:
        code = h.get("code", "")
        name = h.get("name", "")
        sa = stock_analyses.get(code)
        if not sa:
            continue

        td = sa.get("trade_data", {})
        op_panel = sa.get("op_panel", "")
        sr = td.get("scoring_result")
        trailing = td.get("trailing_stop", {})

        lines.append(f"\n  ▼ {name}（{code}）")

        # 操盘面板
        if op_panel:
            for line in op_panel.split("\n"):
                lines.append(f"  {line}")

        # 8因子评分明细
        if sr and sr.get("factors"):
            lines.append(_sub("8因子评分"))
            for f in sr["factors"]:
                fname = f.get("name", f.get("key", ""))
                fscore = f.get("score", 0)
                fdetail = f.get("detail", "")
                # 信号颜色: 绿=高分, 黄=中, 红=低
                indicator = "+" if fscore >= 7 else "~" if fscore >= 4 else "-"
                d = f"  {indicator} {fname:<16} {fscore}/10"
                if fdetail:
                    d += f"  ({fdetail[:30]})"
                lines.append(d)
            lines.append(f"  {'='*42}")
            lines.append(f"  总分: {sr.get('composite_score','?')}/100  {sr.get('action','')}")
            lines.append(f"  建议: {sr.get('level','')}")

        # 跟踪止损
        if trailing.get("active"):
            lines.append(f"  跟踪止损: {trailing.get('message', '')}")
            if trailing.get("stop_price"):
                lines.append(f"  当前跟踪价: {trailing['stop_price']:.2f}")
        elif td.get("stop_price", 0) > 0:
            stop = td["stop_price"]
            price = td.get("price", 0)
            dist = (price - stop) / price * 100 if price > 0 else 0
            lines.append(f"  止损: {stop:.2f}  (距现价 {dist:.1f}%)")

        # 退出规则状态
        exit_sigs = td.get("exit_signals", {})
        if exit_sigs.get("triggered"):
            lines.append(f"  [!] 退出规则触发:")
            for msg in exit_sigs.get("messages", []):
                lines.append(f"      {msg}")

    return "\n".join(lines) + "\n"


def wyckoff_signals_full(scanner, prev_track_results=None, top_n=15):
    """威科夫信号精选 —— 全部信号明细 + 支撑阻力 + 昨日回顾"""
    lines = [_sep(), "  威科夫信号精选", f"{'─'*58}"]

    if not hasattr(scanner, 'results') or not scanner.results:
        lines.append("  无数据")
        return "\n".join(lines) + "\n"

    valid = [r for r in scanner.results if r.get("wyckoff_sig") not in ("-", "无信号", "数据不足", "无数据")]
    if not valid:
        lines.append("  今日无显著信号")
        return "\n".join(lines) + "\n"

    # 用 enrich_with_financials 补充支撑阻力
    try:
        enriched = scanner.enrich_with_financials(valid, top_n)
    except Exception:
        enriched = valid[:top_n]

    lines.append(f"  {'信号':<10} {'代码':<8} {'名称':<6} {'得分':<4} {'现价':<7} {'支撑':<7} {'阻力':<7} {'阶段':<22}")
    lines.append(f"  {'-'*72}")
    for r in enriched[:top_n]:
        sym = r["code"].split(".")[1]
        sig = r["wyckoff_sig"]
        ph = (r.get("phase", "") or "")[:20]
        sup = f"{r.get('chip_support', ''):<7}" if r.get('chip_support') else "-"+" "*6
        res = f"{r.get('chip_resistance', ''):<7}" if r.get('chip_resistance') else "-"+" "*6
        lines.append(f"  {sig:<10} {sym:<8} {r['name']:<6} {r['wyckoff_score']:<4} {r['price']:<7.2f} {sup:<7} {res:<7} {ph:<22}")

    # 推荐（买入信号+多头）
    buy_signals = {"SOS", "Spring", "LPS", "EVR", "Compression"}
    buys = [r for r in enriched if r["wyckoff_sig"] in buy_signals and r.get("trend") == "多头"]
    if buys:
        lines.append(f"\n  推荐关注（买入信号+多头）:")
        for r in buys[:8]:
            sym = r["code"].split(".")[1]
            sig = r["wyckoff_sig"]
            amt = f"{r.get('amount', 0)/1e8:.1f}亿"
            lines.append(f"  {r['name']}({sym}) {sig} 得分{r['wyckoff_score']} 成交{amt}")

    # 昨日选股回顾
    if prev_track_results:
        prev_date, track_results = prev_track_results
        if track_results:
            lines.append(f"\n  昨日选股回顾 ({prev_date}):")
            lines.append(f"  {'#':<3} {'名称':<8} {'信号':<10} {'昨收':<8} {'现价':<8} {'涨幅':<8}")
            lines.append(f"  {'-'*55}")
            for i, r in enumerate(track_results[:8]):
                if r["change_pct"] is not None:
                    arr = "↑" if r["change_pct"] >= 0 else "↓"
                    lines.append(f"  {i+1:<3} {r['name']:<8} {r['signal']:<10} {r['prev_price']:<8.2f} {r['current_price']:<8.2f} {arr}{abs(r['change_pct']):<7.2f}%")
                else:
                    lines.append(f"  {i+1:<3} {r['name']:<8} {r['signal']:<10} {r['prev_price']:<8.2f} {'N/A':<8} {'数据失败':<8}")

    return "\n".join(lines) + "\n"


def exit_rules_monitor(exit_map: dict):
    """退出规则监控 —— 当前所有持仓的退出信号状态"""
    lines = [_sep(), "  退出规则监控", f"{'─'*58}"]

    if not exit_map:
        lines.append("  无持仓数据")
        return "\n".join(lines) + "\n"

    any_triggered = False
    for code, es in exit_map.items():
        if not es.get("triggered"):
            continue
        any_triggered = True
        name = es.get("name", code)
        lines.append(f"\n  [!] {name} ({code}) — 退出信号")
        for msg in es.get("messages", []):
            lines.append(f"      {msg}")

    if not any_triggered:
        lines.append("  所有持仓退出信号正常 (未触发)")

    # 显示各持仓的关键位和状态摘要
    lines.append("")
    lines.append(f"  {'代码':<8} {'名称':<8} {'关键位':<8} {'逻辑破':<8} {'K线衰竭':<8}")
    lines.append(f"  {'-'*46}")
    for code, es in exit_map.items():
        name = es.get("name", code)
        key_price = es.get("entry_key_price", 0)
        lb = "Y" if es.get("logic_broken") else "N"
        te = "Y" if es.get("trend_exhaust") else "N"
        lines.append(f"  {code:<8} {name:<8} {key_price:<8.2f} {lb:<8} {te:<8}")

    return "\n".join(lines) + "\n"


def stock_news_section(picks_list, top_n=8):
    """选股新闻背景 —— 推荐个股相关的新闻"""
    if not picks_list:
        return ""

    try:
        from news_service import get_top_signal_news, format_news_block
    except ImportError:
        return ""

    name_map = {r["code"]: r["name"] for r in picks_list[:top_n]}
    news_dict = get_top_signal_news(picks_list[:top_n], limit_per_stock=2)
    if not news_dict:
        return ""

    lines = [_sep(), "  选股新闻背景", f"{'─'*58}"]
    block = format_news_block(news_dict, name_map)
    for line in block.split("\n"):
        lines.append(f"  {line}")

    return "\n".join(lines) + "\n"


def watchlist_analysis(scanner, watchlist_results=None, health=None, pe_cap=None):
    """股票池评分"""
    lines = [_sep(), "  股票池", f"{'─'*58}"]

    POOL_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")
    pool_stocks = []
    try:
        with open(POOL_FILE, encoding="utf-8") as f:
            pool_data = json.load(f)
            pool_stocks = [s["code"] for s in pool_data.get("stocks", [])]
    except Exception:
        pass

    # 排除持仓
    holdings_codes = {"002050", "600038", "600416"}
    pool_stocks = [s for s in pool_stocks if s not in holdings_codes]

    if not pool_stocks:
        lines.append("  股票池为空")
        return "\n".join(lines) + "\n"

    # 如果有预计算的分析结果，直接渲染
    if watchlist_results:
        for wlr in watchlist_results:
            name = wlr.get("name", "?")
            sym = wlr.get("code", "?")
            price = wlr.get("price", 0)
            score = wlr.get("score", 0)
            trend = wlr.get("trend", "未知")
            phase = wlr.get("phase", "")
            sig = wlr.get("signal", "")
            sug = wlr.get("suggestion", "观望")

            lines.append(f"\n  {name}（{sym}）")
            lines.append(f"    现价: {price:<8.2f}  评分: {score}/100")
            lines.append(f"    趋势: {trend}  信号: {sig}  阶段: {phase}")
            lines.append(f"    建议: {sug}")
    else:
        # 现场跑 scanner 已有数据
        for sym in pool_stocks:
            bs = f"sh.{sym}" if sym.startswith("6") else f"sz.{sym}"
            found = [r for r in scanner.results if r["code"] == bs]
            if found:
                r = found[0]
                sig = r.get("wyckoff_sig", "")
                score = r.get("wyckoff_score", 0)
                trend = r.get("trend", "")
                phase = r.get("phase", "")

                if score >= 65 and trend == "多头": sug = "逢低建仓"
                elif score >= 50 and trend == "多头": sug = "可关注，等回调"
                elif score >= 50: sug = "等趋势转多"
                elif score < 50 and trend == "多头": sug = "观望，趋势偏弱"
                elif score < 50: sug = "回避"
                else: sug = "观望"

                lines.append(f"\n  {r['name']}（{sym}）")
                lines.append(f"    现价: {r['price']:<8.2f}  评分: {score}/100")
                lines.append(f"    趋势: {trend}  信号: {sig}  阶段: {phase}")
                lines.append(f"    建议: {sug}")

    return "\n".join(lines) + "\n"


def fundamental_section(holdings_data, fund_results=None):
    """基本面规则评分 (四本书60条规则)"""
    lines = [_sep(), "  基本面规则评分", f"{'─'*58}"]

    if fund_results:
        for fr in fund_results:
            label = fr.get("label", "?")
            score = fr.get("score", 0)
            verdict = fr.get("verdict", "?")
            fails = fr.get("fails", 0)
            details = fr.get("details", "")
            lines.append(f"  {label}: {score}分 [{verdict}] 违反{fails}条")
            if details:
                lines.append(f"    {details}")
        return "\n".join(lines) + "\n"

    try:
        from fundamental_rules_engine import FundamentalRulesEngine, TushareDataFetcher
    except ImportError:
        lines.append("  基本面规则引擎不可用")
        return "\n".join(lines) + "\n"

    try:
        engine = FundamentalRulesEngine()
        fetcher = TushareDataFetcher()
    except Exception:
        lines.append("  引擎初始化失败")
        return "\n".join(lines) + "\n"

    code_map = {"002050": "三花智控", "600038": "中直股份", "600416": "湘电股份"}
    for h in holdings_data:
        sym = h.get("code", "")
        label = code_map.get(sym, sym)
        ts_code = sym + ".SH" if sym.startswith("6") else sym + ".SZ"
        try:
            fund_data = fetcher.fetch_full_data(ts_code)
            if not fund_data or not fund_data.get("pe_ttm"):
                lines.append(f"  {label}: 无基本面数据")
                continue
            result = engine.analyze_stock(sym, fund_data)
            score = result["score"]
            flags = result["flags"]
            fails = [f for f in flags if f["severity"] in ("fail", "veto")]
            warns = [f for f in flags if f["severity"] == "warn"]
            verdict = "通过" if result["verdict"] == "PASS" else "警告" if result["verdict"] == "WARN" else "否决"
            detail_str = "; ".join(f['detail'][:40] for f in fails[:2])
            lines.append(f"  {label}: {score}分 [{verdict}] 违反{len(fails)}条 警告{len(warns)}条")
            if detail_str:
                lines.append(f"    {detail_str}")
        except Exception:
            lines.append(f"  {label}: 分析失败")

    return "\n".join(lines) + "\n"


# ============================================================
# 控制层
# ============================================================

def quick_report(scanner, holdings_data, indices=None, breadth=None, health=None,
                 analysis_map=None, exit_map=None):
    """快速输出: 5个核心模块, ~18s"""
    out = header()
    out += market_status(indices, breadth, health)
    out += holdings_status(holdings_data, scanner, analysis_map, exit_map)
    out += stock_picks(scanner)
    out += sector_analysis(scanner)
    out += picks_performance()
    out += f"\n{_div()}\n  python run.py --full  全景深度分析\n{_div()}\n"
    return out


def full_report(scanner, holdings_data, analysis_map=None,
                stock_analyses=None, prev_track_results=None,
                watchlist_results=None, fund_results=None,
                picks_list=None,
                market_data=None):
    """全景输出: 快速 + 全部深度模块
    market_data: {indices, breadth, health, pe_cap} 可选，预计算可避免重复API
    """
    if market_data:
        indices = market_data.get("indices")
        breadth = market_data.get("breadth")
        health = market_data.get("health")
        pe_cap = market_data.get("pe_cap")
    else:
        from market import MarketAnalyzer
        indices = MarketAnalyzer.fetch_indices()
        breadth = MarketAnalyzer.analyze_breadth(scanner.snapshot) if hasattr(scanner, 'snapshot') else None
        health = MarketAnalyzer.judge(indices, breadth, None) if indices and breadth else None
        _, pe_cap = MarketAnalyzer.get_pe_percentile()

    out = header()
    # 一、大盘全景
    out += market_status(indices, breadth, health)
    out += market_deep(scanner)

    # 构建退出规则监控数据
    exit_map = {}
    code_map = {"002050": "三花智控", "600038": "中直股份", "600416": "湘电股份"}
    for h in holdings_data:
        code = h.get("code", "")
        if stock_analyses and code in stock_analyses:
            td = stock_analyses[code].get("trade_data", {})
            es = td.get("exit_signals", {})
            if es:
                es["name"] = code_map.get(code, td.get("name", code))
                exit_map[code] = es

    # 二、我的持仓
    out += holdings_status(holdings_data, scanner, analysis_map, exit_map)
    out += holdings_deep(holdings_data, stock_analyses, scanner)
    out += exit_rules_monitor(exit_map)
    out += fundamental_section(holdings_data, fund_results)

    # 三、选股系统
    out += stock_picks(scanner)
    out += picks_performance()
    out += wyckoff_signals_full(scanner, prev_track_results)
    out += stock_news_section(picks_list)

    # 四、板块分析
    out += sector_analysis(scanner)

    # 五、股票池
    out += watchlist_analysis(scanner, watchlist_results, health, pe_cap)

    out += f"\n{_div()}\n  报告完毕\n{_div()}\n"
    return out


def holdings_status_only(holdings_data):
    """极简模式: 只看持仓"""
    return holdings_status(holdings_data)
