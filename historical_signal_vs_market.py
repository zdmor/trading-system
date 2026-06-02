"""过去N个交易日 每日全市场信号 vs 大盘对比

策略：一次性缓存所有候选股的K线，再逐日切片分析，避免反复抓取。
用法:
  py historical_signal_vs_market.py --days 20
  py historical_signal_vs_market.py --days 20 --push
"""
import json, sys, time, datetime, requests, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scanner import get_scanner_config

TENCENT_KLINE_URL_TMPL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"


# ==================== 工具函数 ====================

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def get_trading_days(n=20):
    """获取近 N 个交易日"""
    today = datetime.date.today().strftime("%Y%m%d")
    try:
        import tushare as ts
        cfg = get_scanner_config()
        token = cfg.get("tushare_token", "")
        if token:
            ts.set_token(token)
        pro = ts.pro_api()
        df = pro.trade_cal(start_date="20250101", end_date=today)
        if df is None or len(df) == 0:
            raise ValueError("empty trade_cal")
        trading = df[df["is_open"] == 1]["cal_date"].tolist()
        trading.sort()
        return trading[-n:]
    except Exception as e:
        log(f"trade_cal 失败: {e}，用日期回退")
        dates = []
        d = datetime.date.today()
        while len(dates) < n:
            if d.weekday() < 5:
                dates.append(d.strftime("%Y%m%d"))
            d -= datetime.timedelta(days=1)
        dates.sort()
        return dates


def fetch_market_data(trading_days):
    """获取上证指数涨跌幅"""
    start, end = trading_days[0], trading_days[-1]
    try:
        import tushare as ts
        cfg = get_scanner_config()
        token = cfg.get("tushare_token", "")
        if token:
            ts.set_token(token)
        pro = ts.pro_api()
        df = pro.index_daily(ts_code="000001.SH", start_date=start, end_date=end)
        if df is not None and len(df) > 0:
            return {row["trade_date"]: row["pct_chg"] for _, row in df.iterrows()}
    except Exception as e:
        log(f"指数数据失败: {e}")
    return {}


def get_candidates(min_amount=3e8):
    """获取候选股票池（流动性过滤）"""
    try:
        from scanner import Scanner
        sc = Scanner()
        stocks, snapshot = sc.fetch_all_stocks(max_pages=10)
        if not snapshot:
            log("  snapshot 为空，返回空池")
            return []
        candidates = []
        for s in stocks:
            code = s.get("code", "")
            snap = snapshot.get(code, {})
            amount = snap.get("amount", 0)
            if amount >= min_amount:
                candidates.append({**s, "amount": amount})
        candidates.sort(key=lambda x: -x.get("amount", 0))
        log(f"  候选股: {len(candidates)} 只 (amount>={min_amount/1e8:.0f}亿)")
        return candidates[:80]
    except Exception as e:
        log(f"获取候选股失败: {e}")
        import traceback
        log(traceback.format_exc())
        return []


def fetch_all_klines(candidates):
    """一次性获取所有候选股的K线，返回 {code: klines}"""
    cache = {}
    total = len(candidates)
    for i, s in enumerate(candidates):
        code = s.get("code", "")
        if not code:
            continue
        key = code.replace(".", "")
        url = TENCENT_KLINE_URL_TMPL.format(code=key)
        try:
            r = requests.get(url, timeout=15)
            data = r.json()
            if data.get("code") != 0:
                continue
            klines = data.get("data", {}).get(key, {}).get("qfqday") or \
                     data.get("data", {}).get(key, {}).get("day") or []
            if klines:
                cache[code] = klines
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            log(f"  下载K线: {i+1}/{total}")
        time.sleep(0.05)
    log(f"  K线下载完成: {len(cache)}/{total} 只有效")
    return cache


def slice_klines(klines, target_date):
    """截取到 target_date(YYYYMMDD) 的K线"""
    result = []
    for k in klines:
        kd = k[0].replace("-", "")
        if kd <= target_date:
            result.append(k)
        else:
            break
    return result


def analyze_at_date(klines_slice):
    """在某日期截面运行 Wyckoff 分析，返回 (signal, score, phase)"""
    closes, highs, lows, volumes, opens = [], [], [], [], []
    for k in klines_slice:
        try:
            opens.append(float(k[1]))
            closes.append(float(k[2]))
            highs.append(float(k[3]))
            lows.append(float(k[4]))
            volumes.append(float(k[5]) * 100)
        except Exception:
            pass
    if len(closes) < 30:
        return None, 0, "数据不足"

    from scanner import WyckoffAnalyzer
    signals, _ = WyckoffAnalyzer.analyze_all(closes, highs, lows, opens, volumes)
    if len(closes) >= 60:
        phase_label, _, _ = WyckoffAnalyzer.classify_phase(closes, highs, lows, opens, volumes, signals)
    else:
        phase_label = "数据不足"

    PHASE_SIGNAL_MAP = {
        "Accum_A": ["Accum_Stage"],
        "Accum_B": ["Spring", "Compression", "Accum_Stage"],
        "Accum_C": ["Spring", "SOS", "LPS", "Accum_Stage"],
        "Markup": ["SOS", "LPS", "Markup"],
        "Distribute_A": ["EVR", "Upthrust"],
        "Distribute_B": ["Upthrust", "EVR"],
        "Markdown": ["Spring"],
        "Range": ["Compression"],
    }
    allowed = PHASE_SIGNAL_MAP.get(phase_label, [])
    signals = [s for s in signals if s[0] in allowed]
    if signals and signals[0][1] > 0:
        return signals[0][0], signals[0][1], phase_label
    return None, 0, phase_label

SIGNAL_WEIGHTS = {
    # 高信噪 → 权重 1.0
    "Spring": 1.0,
    "SOS": 1.0,
    "LPS": 1.0,
    "Accum_Stage": 1.0,
    # 低信噪 → 权重 0.5
    "EVR": 0.5,
    "Upthrust": 0.5,
    "Markup": 0.5,
    "Compression": 0.5,
}

def get_signal_weight(signal, score):
    """返回信号权重（带评分加权），噪声返回0"""
    if not signal:
        return 0
    if signal.startswith("弱") or signal in ("-", "无信号", "数据不足"):
        return 0
    base = SIGNAL_WEIGHTS.get(signal, 0.5)  # 未知信号默认0.5
    score_factor = min(score / 100.0, 1.0)  # 评分加权，上限1.0
    return base * score_factor


# ==================== 主流程 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    n = args.days
    log(f"=== 历史信号 vs 大盘对比（回看 {n} 交易日）===")

    # 1. 交易日历
    log("[1/4] 交易日历...")
    trading_days = get_trading_days(n)
    log(f"  {len(trading_days)} 天: {trading_days[0]} ~ {trading_days[-1]}")

    # 2. 大盘数据
    log("[2/4] 大盘指数...")
    market_data = fetch_market_data(trading_days)
    log(f"  {len(market_data)} 天有数据")

    # 3. 股票池
    log("[3/4] 下载K线...")
    candidates = get_candidates()
    log(f"  候选股: {len(candidates)} 只")
    kline_cache = fetch_all_klines(candidates)

    # 4. 逐日回算
    log(f"[4/5] 逐日分析 {len(trading_days)} 天 × {len(kline_cache)} 只...\n")
    results = {}
    total = len(trading_days) * len(kline_cache)
    done = 0

    for date_str in trading_days:
        display = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        buys, sells, with_data = [], [], 0
        weighted_buy = 0.0
        weighted_sell = 0.0

        for s in candidates:
            code = s.get("code", "")
            klines = kline_cache.get(code)
            if not klines:
                done += 1
                continue
            sliced = slice_klines(klines, date_str)
            if len(sliced) < 30:
                done += 1
                continue
            signal, score, phase = analyze_at_date(sliced)
            with_data += 1
            if signal and score > 0:
                w = get_signal_weight(signal, score)
                if signal in ("Spring", "二次Spring", "强势Spring", "震仓", "SOS", "LPS", "Accum_Stage"):
                    buys.append((s.get("name", code), signal, score))
                    weighted_buy += w
                else:
                    sells.append((s.get("name", code), signal, score))
                    weighted_sell += w
            done += 1

        results[date_str] = {
            "market_change": market_data.get(date_str),
            "total": with_data,
            "buy": len(buys),
            "sell": len(sells),
            "weighted_buy": round(weighted_buy, 2),
            "weighted_sell": round(weighted_sell, 2),
            "top_buys": buys[:5],
            "top_sells": sells[:5],
        }
        sh = market_data.get(date_str)
        sh_s = f"{sh:+.2f}%" if sh is not None else "N/A"
        log(f"  {display} | 上证{sh_s} | {with_data:3d}只 | 买{weighted_buy:.1f} 卖{weighted_sell:.1f} (加权)" +
            (f" | eg: {buys[0][0]}({buys[0][1]})" if buys else ""))

    # 5. 跨天平滑 (3日EMA)
    log("\n[5/5] 跨天平滑...")
    trading_days_sorted = sorted(trading_days)
    for i, d in enumerate(trading_days_sorted):
        wb = results[d].get("weighted_buy", 0)
        ws = results[d].get("weighted_sell", 0)
        raw_ratio = wb / ws if ws > 0 else (wb if wb > 0 else 0)
        results[d]["raw_ratio"] = round(raw_ratio, 2)
        if i == 0:
            smoothed = raw_ratio
        elif i == 1:
            smoothed = 0.5 * raw_ratio + 0.5 * results[trading_days_sorted[i-1]]["raw_ratio"]
        else:
            smoothed = 0.5 * raw_ratio + 0.3 * results[trading_days_sorted[i-1]]["raw_ratio"] + 0.2 * results[trading_days_sorted[i-2]]["raw_ratio"]
        results[d]["signal_ratio"] = round(smoothed, 2)

    # 6. 输出表格（灯色基于平滑信号比）
    print()
    print("=" * 85)
    print(f"  {'日期':<12} {'上证':<8} {'只数':<5} {'加权买':<7} {'加权卖':<7} {'平滑比':<8} {'灯色':<6} {'买入信号'}")
    print("=" * 85)
    for d in trading_days:
        r = results.get(d, {})
        sh = r.get("market_change")
        sh_s = f"{sh:+.2f}%" if sh is not None else "N/A"
        wb = r.get("weighted_buy", 0)
        ws = r.get("weighted_sell", 0)
        sr = r.get("signal_ratio", 0)
        dd = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        # 灯色判断
        if sr < 1.0:
            light = "红灯"
        elif sr >= 2.0:
            light = "绿灯"
        else:
            light = "黄灯"
        flag = ""
        if sr >= 2.0:
            flag = "  <<< 买入窗口"
        elif sr >= 1.1:
            flag = "  <-- 偏多"
        top_buy_names = ",".join(x[0][:6] for x in r.get("top_buys", [])[:2])
        print(f"  {dd} | {sh_s:>7} | {r.get('total',0):3d} | {wb:5.1f} | {ws:5.1f} | {sr:<8.2f} {light:<6} {top_buy_names}{flag}")

    # 7. 汇总
    green_days = sum(1 for d in trading_days if results.get(d, {}).get("signal_ratio", 0) >= 2.0)
    yellow_days = sum(1 for d in trading_days if 1.0 <= results.get(d, {}).get("signal_ratio", 0) < 2.0)
    red_days = sum(1 for d in trading_days if results.get(d, {}).get("signal_ratio", 0) < 1.0)
    total_wb = sum(results.get(d, {}).get("weighted_buy", 0) for d in trading_days)
    total_ws = sum(results.get(d, {}).get("weighted_sell", 0) for d in trading_days)
    print()
    print(f"  汇总: {n} 天 | 绿灯{green_days} 黄灯{yellow_days} 红灯{red_days}")
    print(f"  加权总信号: 买入 {total_wb:.1f}  vs  卖出 {total_ws:.1f}")
    if total_ws > 0:
        print(f"  加权买卖比: {total_wb/total_ws:.2f}")
    print()

    # 8. 保存
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "signal_vs_market_history.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "trading_days": trading_days,
            "results": {k: {"market_change": v["market_change"], "total": v["total"],
                           "weighted_buy": v["weighted_buy"], "weighted_sell": v["weighted_sell"],
                           "signal_ratio": v.get("signal_ratio", 0), "raw_ratio": v.get("raw_ratio", 0),
                           "top_buy": v["top_buys"][:3], "top_sell": v["top_sells"][:3]}
                       for k, v in results.items()}
        }, f, ensure_ascii=False, indent=2)
    log(f"结果已保存: {out_path}")

    # 9. 飞书推送
    if args.push:
        _push_feishu(trading_days, results, market_data)


def _push_feishu(trading_days, results, market_data):
    cfg = get_scanner_config()
    webhook = cfg.get("notify", {}).get("webhook_url", "")
    if not webhook:
        log("  [WARN] 未配置 webhook_url")
        return
    lines = ["【历史信号 vs 大盘】", ""]
    lines.append(f"{'日期':<12} {'大盘':<8} {'加权买':<7} {'加权卖':<7} {'灯色':<6}")
    lines.append("-" * 42)
    for d in trading_days:
        r = results.get(d, {})
        sh = r.get("market_change")
        sh_s = f"{sh:+.2f}%" if sh is not None else "N/A"
        wb = r.get("weighted_buy", 0)
        ws = r.get("weighted_sell", 0)
        sr = r.get("signal_ratio", 0)
        dd = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        if sr < 1.0:
            light = "红灯"
        elif sr >= 2.0:
            light = "绿灯"
        else:
            light = "黄灯"
        lines.append(f"{dd:<12} {sh_s:<8} {wb:<7.1f} {ws:<7.1f} {light:<6}")
    total_wb = sum(results.get(d, {}).get("weighted_buy", 0) for d in trading_days)
    total_ws = sum(results.get(d, {}).get("weighted_sell", 0) for d in trading_days)
    lines.append("")
    lines.append(f"加权总买卖: {total_wb:.1f}/{total_ws:.1f} = {total_wb/total_ws:.2f}")
    try:
        requests.post(webhook, json={"msg_type": "text", "content": {"text": "\n".join(lines)}}, timeout=10)
        log("  飞书推送完成")
    except Exception as e:
        log(f"  飞书推送失败: {e}")


if __name__ == "__main__":
    main()
