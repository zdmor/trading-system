"""
股票池长期跟踪 — 信号+趋势变化检测
每日运行: 读取 scanner.yaml 股票池 → 分析每只 → 缓存对比 → 报告变化

用法:
  python watch_pool.py              # 完整分析+变化检测
  python watch_pool.py --push       # 分析+推送变化到飞书
  python watch_pool.py --full       # 全量输出（含无变化股票）

缓存: watch_pool_cache.json (自动维护)
"""

import sys, os, time, json
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from scanner import WyckoffAnalyzer
from pattern_detector import PatternDetector

# 凯利仓位计算
try:
    from factor_7_validate import compute_factors_for_stock
    from kelly_position import compute_kelly_position
    from factor_weights import get_weights
    _KELLY_OK = True
except Exception:
    _KELLY_OK = False

# 因子->权重名映射 (与 scanner._F2W_MAP 一致)
_F2W = {
    "risk_reward": "risk_reward",
    "volume": "volume",
    "candlestick": "candlestick",
    "trend_momentum": "tech_strength",
    "relative_strength": "relative_strength",
    "volatility": "volatility",
    "orbit_compression": "orbit_compression",
    "lyapunov": "lyapunov",
    "hurst": "hurst",
    "fractal_dim": "fractal_dim",
    "attractor_shape": "attractor_shape",
}

# 盈亏比分值->赔率映射
_RR_SCORE_TO_RATIO = {95: 3.5, 80: 2.5, 65: 2.0, 50: 1.5, 35: 1.0, 20: 0.5, 10: 0.3}


_WEIGHTS_CACHE = None

def _load_weights():
    global _WEIGHTS_CACHE
    if _WEIGHTS_CACHE is None:
        _WEIGHTS_CACHE = get_weights()
    return _WEIGHTS_CACHE

def _compute_kelly(factor_scores, tf_score):
    """从因子评分计算凯利仓位系数"""
    wcache = _load_weights()
    try:
        total_w = 0
        weighted_sum = 0
        kelly_factors = []
        for fname, wname in _F2W.items():
            w = wcache.get(wname, 0.01)
            fv = factor_scores.get(fname, 50)
            f_score = fv.get("score", fv) if isinstance(fv, dict) else (fv if isinstance(fv, (int, float)) else 50)
            weighted_sum += f_score * w
            total_w += w
            item = {"key": wname, "weight": w, "score": f_score}
            if fname == "risk_reward":
                rr_ratio = _RR_SCORE_TO_RATIO.get(round(f_score / 10) * 10, 1.5)
                item["ratio"] = rr_ratio
            kelly_factors.append(item)

        composite = weighted_sum / total_w if total_w > 0 else 50
        tf_coeffs = {3: 1.15, 2: 1.00, 1: 0.85, 0: 0.60}
        coeff = tf_coeffs.get(tf_score, 1.0)
        composite = min(100, composite * coeff) if tf_score == 3 else composite * coeff

        kr = compute_kelly_position(composite, kelly_factors, market_score=50, sector_score=50)
        return round(kr["position_factor"], 2)
    except Exception:
        return 0.0


# --- 路径 ---
BASE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(BASE_DIR, "config")
CACHE_PATH = os.path.join(BASE_DIR, "watch_pool_cache.json")

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,350,qfq"
MIN_AMOUNT = 500_000_000
POOL_PAGE_SIZE = 30

# ─── 工具 ───

def load_pool():
    """从 scanner.yaml 读取股票池列表，补充股票名称"""
    from scanner import get_scanner_config
    cfg = get_scanner_config()
    wl = cfg.get("watchlist", {})
    pool = wl.get("pool", [])
    holdings = wl.get("holdings", [])

    # 名称查询：用 AKShare 一次性拉取全市场名称
    name_map = {}
    try:
        import akshare as ak
        df = ak.stock_info_a_code_name()
        for _, r in df.iterrows():
            name_map[r["code"]] = r["name"]
    except Exception:
        pass

    for item in pool:
        if not item.get("name") and item["code"] in name_map:
            item["name"] = name_map[item["code"]]
    for item in holdings:
        if not item.get("name") and item["code"] in name_map:
            item["name"] = name_map[item["code"]]

    return pool, holdings


def load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def tencent_code(code):
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_kline(tc_code, retries=2):
    url = TENCENT_KLINE_URL.format(code=tc_code)
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=10)
            data = r.json()
            if data.get("code") != 0:
                return None
            klines = (data.get("data", {}).get(tc_code, {}).get("qfqday") or
                      data.get("data", {}).get(tc_code, {}).get("day") or [])
            return klines
        except Exception:
            time.sleep(1)
    return None


def parse_klines(klines):
    closes, highs, lows, volumes, opens = [], [], [], [], []
    for k in klines:
        try:
            opens.append(float(k[1]))
            closes.append(float(k[2]))
            highs.append(float(k[3]))
            lows.append(float(k[4]))
            volumes.append(float(k[5]) * 100)
        except (ValueError, IndexError):
            pass
    return closes, highs, lows, volumes, opens


def calc_ma(arr, n):
    return np.mean(arr[-n:]) if len(arr) >= n else None


def analyze_stock(code, name=""):
    """完整分析单只股票，返回状态字典"""
    result = {
        "code": code, "name": name,
        "trend": "错误", "strength": 0,
        "wyckoff_sig": "-", "wyckoff_score": 0, "wyckoff_detail": "",
        "phase": "-", "tf_score": 0,
        "tf_monthly": "-", "tf_weekly": "-", "tf_daily": "-",
        "price": 0, "amount": 0, "date": datetime.now().strftime("%Y-%m-%d"),
        "error": None,
    }

    tc = tencent_code(code)
    klines = fetch_kline(tc)
    if not klines:
        result["error"] = "K线获取失败"
        return result

    closes, highs, lows, volumes, opens = parse_klines(klines)
    if len(closes) < 50:
        result["error"] = f"数据不足({len(closes)}天)"
        return result

    # 价格+成交额
    price = closes[-1]
    result["price"] = round(price, 2)

    # 流动性（当日成交额）
    result["amount"] = round(price * volumes[-1], 2)

    # 趋势 + 三重滤网
    if len(closes) >= 200:
        ma50 = calc_ma(closes, 50)
        ma200 = calc_ma(closes, 200)
        ma20 = calc_ma(closes, 20)
        result["trend"] = "多头" if ma50 > ma200 else "空头"
        result["strength"] = round((ma50 / ma200 - 1) * 100, 1)
        result["tf_monthly"] = "多" if price > ma200 else "空"
        result["tf_weekly"] = "多" if price > ma50 else "空"
        result["tf_daily"] = "多" if price > ma20 else "空"
        result["tf_score"] = int(price > ma200) + int(price > ma50) + int(price > ma20)
    elif len(closes) >= 50:
        result["trend"] = "数据不足"
    else:
        result["trend"] = "新股"

    # 威科夫形态
    if len(closes) >= 30:
        try:
            signals, extra_info = WyckoffAnalyzer.analyze_all(
                closes, highs, lows, opens, volumes
            )
            if signals and signals[0][1] > 0:
                # 过滤弱信号（弱Spring/弱LPS 不参与选股）
                strong = [s for s in signals if not s[0].startswith("弱")]
                if strong:
                    result["wyckoff_sig"] = strong[0][0]
                    result["wyckoff_score"] = strong[0][1]
                    result["wyckoff_detail"] = strong[0][2][:40] if len(strong[0]) > 2 else ""
        except Exception:
            pass

    # 威科夫阶段
    if len(closes) >= 50:
        try:
            phase, phase_detail, _ = WyckoffAnalyzer.detect_phase(
                closes, highs, lows, volumes, result["trend"],
                [(result["wyckoff_sig"], result["wyckoff_score"], result["wyckoff_detail"])] if result["wyckoff_sig"] not in ("-",) else [],
                {}
            )
            result["phase"] = phase
        except Exception:
            pass

    # 凯利仓位
    result["kelly_pct"] = 0.0
    if _KELLY_OK and len(closes) >= 50:
        try:
            data_dict = {"close": np.array(closes), "high": np.array(highs),
                         "low": np.array(lows), "vol": np.array(volumes),
                         "open": np.array(opens)}
            factor_scores = compute_factors_for_stock(data_dict, len(closes) - 1, None)
            if factor_scores:
                result["kelly_pct"] = _compute_kelly(factor_scores, result.get("tf_score", 2))
        except Exception:
            pass

    return result


def detect_changes(current, cached):
    """对比当前 vs 缓存，返回有变化的字段"""
    changes = {}
    fields = ["trend", "wyckoff_sig", "wyckoff_score", "tf_score", "phase", "price"]
    for f in fields:
        old = cached.get(f)
        new = current.get(f)
        if old is not None and new is not None and old != new:
            changes[f] = (old, new)

    # 只标记有意义的变化（忽略纯价格波动）
    meaningful = ["trend", "wyckoff_sig", "tf_score", "phase"]

    # tf_score 变化 → 展开每层
    if "tf_score" in changes:
        for layer in ["tf_monthly", "tf_weekly", "tf_daily"]:
            ol = cached.get(layer)
            nl = current.get(layer)
            if ol and nl and ol != nl:
                changes[layer] = (ol, nl)

    # 过滤：只保留有意义的字段
    changes = {k: v for k, v in changes.items() if k in meaningful or k.startswith("tf_")}
    return changes


def calc_suggestion(state, cost=None, pnl_pct=None):
    """持仓操作建议"""
    trend = state.get("trend", "")
    sig = state.get("wyckoff_sig", "")
    if trend == "空头":
        return "▼ 警惕止损"
    if sig in ("Upthrust",) and pnl_pct is not None and pnl_pct > 5:
        return "▼ 减仓"
    if sig in ("Spring", "SOS", "LPS"):
        return "▲ 加仓/持有"
    if pnl_pct is not None and pnl_pct < -15:
        return "!! 止损"
    if pnl_pct is not None and pnl_pct > 20:
        return "◆ 考虑止盈"
    return "▶ 持有"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="股票池长期跟踪")
    parser.add_argument("--push", action="store_true", help="推送变化到飞书")
    parser.add_argument("--full", action="store_true", help="全量输出（含无变化）")
    parser.add_argument("--no-pool", action="store_true", help="只输出持仓，跳过股票池")
    parser.add_argument("--preclose", action="store_true", help="收盘前模式：等 14:54 再拉数据推送")
    args = parser.parse_args()

    pool, holdings = load_pool()
    all_items = [{"code": s["code"], "name": s.get("name", ""), "type": "pool"} for s in pool]
    for h in holdings:
        all_items.append({"code": h["code"], "name": h.get("name", ""), "type": "holding",
                          "cost": h.get("cost"), "shares": h.get("shares")})

    # 收盘前模式：等到 14:54 再拉数据
    if args.preclose:
        now = datetime.now()
        target = now.replace(hour=14, minute=54, second=0, microsecond=0)
        if now < target:
            wait = (target - now).total_seconds()
            print(f"  等待 {wait:.0f} 秒到 14:54 拉数据...")
            time.sleep(wait)
        print()

    print(f"股票池跟踪 — {len(all_items)} 只")
    print(f"  池 {len(pool)} 只 | 仓 {len(holdings)} 只")
    print()

    # 并发分析
    current_state = {}
    errors = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_map = {ex.submit(analyze_stock, it["code"], it["name"]): it for it in all_items}
        done = 0
        for fut in as_completed(fut_map):
            it = fut_map[fut]
            done += 1
            try:
                result = fut.result()
                current_state[it["code"]] = result
                # 补充持仓信息
                if it["type"] == "holding":
                    result["_cost"] = it.get("cost")
                    result["_shares"] = it.get("shares")
                    if it.get("cost") and result.get("price"):
                        result["_pnl_pct"] = round((result["price"] / it["cost"] - 1) * 100, 1)
                if result.get("error"):
                    errors.append((it["code"], it["name"], result["error"]))
                if done % 10 == 0:
                    print(f"  分析进度: {done}/{len(all_items)}")
            except Exception as e:
                errors.append((it["code"], it["name"], str(e)))
                current_state[it["code"]] = {"code": it["code"], "error": str(e)}
            time.sleep(0.03)

    # 加载缓存
    cache = load_cache()

    # ─── 输出：股票池总表 ───
    has_changes = False
    updates = []
    no_changes = []
    holdings_data = []

    if not args.no_pool:
        print(f"\n  {'代码':<8} {'名称':<8} {'趋势':<10} {'信号':<16} {'阶段':<12} {'三滤':<4} {'Kelly':<6} {'价格':<9} {'变化':<30}")
        print("  " + "-" * 108)

        for it in all_items:
            c = it["code"]
            if it["type"] == "holding":
                continue  # 持仓单独输出
            state = current_state.get(c, {})
            if state.get("error"):
                print(f"  {c:<8} {it['name']:<8} {'❌ ' + state['error']:<70}")
                continue

            cached = cache.get(c, {})
            changes = detect_changes(state, cached) if cached else {}

            trend_str = f"{state['trend']}({state['strength']:+.1f}%)"
            sig_str = f"{state['wyckoff_sig']}({state['wyckoff_score']})" if state['wyckoff_sig'] not in ("-", "无信号") else "-"
            tf_str = f"{state['tf_monthly']}{state['tf_weekly']}{state['tf_daily']}"
            kp = state.get("kelly_pct", 0)
            kelly_str = f"{kp:.0%}" if isinstance(kp, (int, float)) and kp > 0 else "-"
            price_str = f"{state['price']:.2f}"

            change_parts = []
            if changes:
                has_changes = True
                for k, label in [("trend", "趋势"), ("wyckoff_sig", "信号"),
                                  ("tf_monthly", "月线"), ("tf_weekly", "周线"),
                                  ("tf_daily", "日线"), ("phase", "阶段")]:
                    if k in changes:
                        change_parts.append(f"{label}:{changes[k][0]}→{changes[k][1]}")
            change_str = " | ".join(change_parts) if change_parts else ("-" if cached else "首次")
            marker = ">>" if changes else "  "
            print(f"  {marker}{c:<8} {state.get('name', it['name']):<8} {trend_str:<10} {sig_str:<16} {state['phase']:<12} {tf_str:<4} {kelly_str:<6} {price_str:<9} {change_str:<30}")

            if changes:
                updates.append({"code": c, "state": state, "changes": changes})
            else:
                no_changes.append(c)

    # ─── 输出：持仓操作指南 ───
    holdings_data = []
    for it in all_items:
        if it["type"] != "holding":
            continue
        c = it["code"]
        state = current_state.get(c, {})
        if state.get("error"):
            continue
        pnl = state.get("_pnl_pct")
        sug = calc_suggestion(state, it.get("cost"), pnl)
        holdings_data.append({"code": c, "name": state.get("name", ""),
                              "state": state, "cost": it.get("cost"),
                              "shares": it.get("shares"), "pnl": pnl, "sug": sug})

    if holdings_data:
        print(f"\n  【持仓操作指南】")
        print(f"  {'代码':<8} {'名称':<8} {'趋势':<10} {'信号':<16} {'三滤':<4} {'Kelly':<6} {'现价':<8} {'成本':<8} {'浮盈':<8} {'建议':<14}")
        print("  " + "-" * 105)
        for h in holdings_data:
            st = h["state"]
            trend_str = f"{st['trend']}({st['strength']:+.1f}%)"
            sig_str = f"{st['wyckoff_sig']}({st['wyckoff_score']})" if st['wyckoff_sig'] not in ("-", "无信号") else "-"
            tf_str = f"{st['tf_monthly']}{st['tf_weekly']}{st['tf_daily']}"
            kp = st.get("kelly_pct", 0)
            kelly_str = f"{kp:.0%}" if isinstance(kp, (int, float)) and kp > 0 else "-"
            price_str = f"{st['price']:.2f}" if st.get('price') else "-"
            cost_str = f"{h['cost']:.2f}" if h['cost'] else "-"
            pnl_str = f"{h['pnl']:+.1f}%" if h['pnl'] is not None else "-"
            print(f"  {h['code']:<8} {h['name']:<8} {trend_str:<10} {sig_str:<16} {tf_str:<4} {kelly_str:<6} {price_str:<8} {cost_str:<8} {pnl_str:<8} {h['sug']:<14}")

    # ─── 错误 ───
    if errors:
        print(f"\n  [错误] {len(errors)} 只分析失败:")
        for c, n, e in errors:
            print(f"    {c} {n}: {e}")

    # ─── 变化汇总 ───
    print(f"\n  总 {len(all_items)} 只 | 有变化 {len(updates)} 只 | 无变化 {len(no_changes)} 只 | 失败 {len(errors)} 只")

    if updates:
        print(f"\n  【变化汇总】")
        for u in updates:
            c = u["code"]
            st = u["state"]
            ch = u["changes"]
            items = [f"{k}:{v[0]}->{v[1]}" for k, v in ch.items()]
            print(f"  >> {c} {st.get('name', '')}: {' | '.join(items)}")

    if not args.full and not has_changes and not args.no_pool:
        print(f"\n  [无变化] — 全部股票状态与上次一致")

    # 保存缓存
    cache_data = {}
    for c, state in current_state.items():
        cache_data[c] = {k: state[k] for k in ["code", "name", "trend", "strength",
                          "wyckoff_sig", "wyckoff_score", "tf_score",
                          "tf_monthly", "tf_weekly", "tf_daily",
                          "phase", "price", "amount", "kelly_pct", "date"]
                         if k in state}
    save_cache(cache_data)
    print(f"\n  缓存已保存 ({len(cache_data)} 只)")

    # ─── 飞书推送 ───
    if args.push and (updates or holdings_data):
        try:
            from notifier import send_card, make_div, make_hr, make_note
            divs = []

            # 股票池变化
            if updates:
                for u in updates:
                    st = u["state"]
                    ch = u["changes"]
                    lines = [f"**{st.get('name', '')}({u['code']})**  {st['trend']} | {st['wyckoff_sig']}({st['wyckoff_score']})"]
                    for k, (old, new) in ch.items():
                        lines.append(f"  {k}: {old} → {new}")
                    divs.append(make_div("\n".join(lines)))

            # 持仓状态
            # 有信号的池股（用于推送）
            signal_stocks = []
            for it in all_items:
                if it["type"] == "holding":
                    continue
                st = current_state.get(it["code"], {})
                if st.get("wyckoff_sig") in ("Spring", "SOS", "LPS", "Upthrust"):
                    kp = st.get("kelly_pct", 0)
                    k_str = f" Kelly:{kp:.0%}" if isinstance(kp, (int, float)) and kp > 0 else ""
                    signal_stocks.append(f"{st.get('name', '')}({it['code']}) {st['wyckoff_sig']}({st['wyckoff_score']}){k_str}")

            if signal_stocks:
                if divs:
                    divs.append(make_hr())
                s_lines = ["**有信号池股**"]
                s_lines.extend(signal_stocks)
                divs.append(make_div("\n".join(s_lines)))

            if holdings_data:
                if divs:
                    divs.append(make_hr())
                h_lines = ["**持仓操作指南**"]
                for h in holdings_data:
                    kp = h["state"].get("kelly_pct", 0)
                    k_str = f" Kelly:{kp:.0%}" if isinstance(kp, (int, float)) and kp > 0 else ""
                    h_lines.append(f"{h['name']}({h['code']}) {h['pnl']:+.1f}%{k_str} | {h['sug']}")
                divs.append(make_div("\n".join(h_lines)))

            # 取 webhook 地址
            from scanner import get_scanner_config
            _cfg = get_scanner_config()
            _hook = _cfg.get("notify", {}).get("webhook_url", "")

            send_card(
                _hook,
                f"股票池跟踪 ({datetime.now().strftime('%m-%d')})",
                [make_hr(), *divs]
            )
            print("  推送成功")
        except Exception as e:
            print(f"  推送失败: {e}")

    print()


if __name__ == "__main__":
    main()
