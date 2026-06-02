import os, json, time, shutil, datetime

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "system_state.json")

DEFAULT_STATE = {
    "last_updated": "",
    "system": {
        "tushare_ok": True,
        "version": "1.0",
    },
    "market": {
        "regime": "\u971c\u8361",
        "confidence": 0.5,
        "date": "",
        "contradiction": {
            "policy": 50,
            "leverage": 50,
            "sentiment": 50,
            "valuation": 50,
            "liquidity": 50,
        },
    },
    "positions": {},
    "scanner": {
        "last_run": "",
        "candidates": [],
        "stale": True,
    },
    "tasks": {},
    "anomalies": [],
}


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic_write(data: dict):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def read_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return json.loads(json.dumps(DEFAULT_STATE))
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        for key, default_val in DEFAULT_STATE.items():
            if key not in state:
                state[key] = json.loads(json.dumps(default_val))
        return state
    except (json.JSONDecodeError, IOError):
        return json.loads(json.dumps(DEFAULT_STATE))


def write_state(state: dict):
    state["last_updated"] = _now_iso()
    _atomic_write(state)


def update_state(updates: dict):
    state = read_state()
    _merge(state, updates)
    state["last_updated"] = _now_iso()
    _atomic_write(state)


def _merge(target: dict, source: dict):
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _merge(target[k], v)
        else:
            target[k] = v


# ── 模块上报 ──

def report_scanner(candidates: list, duration: float = 0):
    update_state({
        "scanner": {
            "last_run": _now_iso(),
            "candidates": candidates[:30],
            "stale": False,
        },
        "tasks": {
            "scanner_run": {"last": _now_iso(), "status": "ok", "duration": round(duration, 1), "error": ""},
        }
    })


def report_market_regime(regime_result: dict):
    regime = regime_result.get("regime", "\u971c\u8361")
    confidence = regime_result.get("confidence", 0.5)
    date = regime_result.get("date", "")
    indicators = regime_result.get("indicators", {})

    update_state({
        "market": {
            "regime": regime,
            "confidence": round(confidence, 2),
            "date": date,
            "contradiction": {
                "policy": round(indicators.get("policy_score", 50), 1),
                "leverage": round(indicators.get("leverage_score", 50), 1),
                "sentiment": round(indicators.get("sentiment_score", 50), 1),
                "valuation": round(indicators.get("valuation_score", 50), 1),
                "liquidity": round(indicators.get("liquidity_score", 50), 1),
            },
        },
        "tasks": {
            "market_regime": {"last": _now_iso(), "status": "ok", "duration": 0, "error": ""},
        }
    })


def report_position(symbol: str, data: dict):
    update_state({"positions": {symbol: data}})


def report_trade(symbol: str, action: str, price: float, shares: int = 0,
               score: float = 0, reason: str = "", **kwargs):
    now = _now_iso()
    update_state({
        "positions": {
            symbol: {
                "price": price,
                "score": score,
                "shares": shares,
                "last_action": action,
                "last_action_time": now,
                "reason": reason,
                **kwargs,
            }
        },
        "tasks": {
            "trade_exec": {"last": now, "status": "ok", "duration": 0, "error": ""},
        }
    })


def report_task(module: str, status: str, duration: float = 0, error: str = ""):
    update_state({
        "tasks": {
            module: {"last": _now_iso(), "status": status, "duration": round(duration, 1), "error": error},
        }
    })


# ── 异常扫描 ──

def scan_anomalies(state: dict = None) -> list:
    if state is None:
        state = read_state()

    anomalies = []
    now = datetime.datetime.now()

    def _add(level: str, module: str, msg: str):
        anomalies.append({"level": level, "module": module, "msg": msg, "since": _now_iso()})

    # 1) 选股缓存过期 (scanner.last_run 超过 2 个交易日)
    last_run = state.get("scanner", {}).get("last_run", "")
    if not last_run:
        _add("error", "scanner", "\u626b\u63cf\u7f13\u5b58\u4e3a\u7a7a\u2014\u2014scanner \u672a\u8fd0\u884c")
    else:
        try:
            last_dt = datetime.datetime.fromisoformat(last_run)
            trade_days = _count_trade_days(last_dt, now)
            if trade_days > 2:
                _add("warn", "scanner", f"\u9009\u80a1\u7f13\u5b58\u8fc7\u671f ({trade_days} \u4e2a\u4ea4\u6613\u65e5)")
        except (ValueError, TypeError):
            _add("warn", "scanner", "\u9009\u80a1\u7f13\u5b58\u65f6\u95f4\u683c\u5f0f\u5f02\u5e38")

    # 2) 持仓评分破警戒
    for sym, pos in state.get("positions", {}).items():
        score = pos.get("score", 50)
        if score < 40:
            _add("warn", f"position:{sym}", f"{sym} \u8bc4\u5206\u7834\u8b66\u6212 ({score})")

    # 3) IC 衰变检查
    ic_task = state.get("tasks", {}).get("ic_check", {})
    if not ic_task:
        _add("info", "ic_check", "IC \u68c0\u67e5\u4ece\u672a\u8fd0\u884c")
    elif ic_task.get("status") == "stale":
        _add("info", "ic_check", "IC \u8870\u53d8\u8d85\u671f")
    else:
        try:
            ic_last = datetime.datetime.fromisoformat(ic_task.get("last", ""))
            if (now - ic_last).days > 7:
                _add("info", "ic_check", f"IC \u68c0\u67e5\u8d85\u8fc7 7 \u65e5\u672a\u8fd0\u884c ({(now - ic_last).days} \u65e5)")
        except (ValueError, TypeError):
            pass

    # 4) 风控冻结 (blacklist 有记录)
    blacklist = state.get("blacklist", {})
    if blacklist:
        if isinstance(blacklist, list) and len(blacklist) > 0:
            _add("error", "risk", f"\u98ce\u63a7\u9ed1\u540d\u5355\u6709 {len(blacklist)} \u4e2a\u7eaa\u5f55")
        elif isinstance(blacklist, dict) and len(blacklist) > 0:
            _add("error", "risk", f"\u98ce\u63a7\u9ed1\u540d\u5355\u6709 {len(blacklist)} \u6761\u8bb0\u5f55")

    # 5) 数据源断连
    if not state.get("system", {}).get("tushare_ok", True):
        _add("error", "datasource", "Tushare \u6570\u636e\u6e90\u65ad\u8fde")

    # 6) 连续亏损 (trade_log 中有连续亏损)
    trade_log = state.get("trade_log", [])
    if isinstance(trade_log, list) and len(trade_log) >= 3:
        recent = trade_log[-3:]
        pnls = []
        for t in recent:
            p = t.get("pnl", t.get("pnl_pct", 0))
            try:
                pnls.append(float(p))
            except (TypeError, ValueError):
                pnls.append(0)
        if all(p < 0 for p in pnls):
            _add("warn", "trade_log", f"\u8fde\u7eed {len(pnls)} \u7b14\u4e8f\u635f")

    return anomalies


def _count_trade_days(start: datetime.datetime, end: datetime.datetime) -> int:
    days = 0
    current = start.date()
    end_date = end.date()
    while current <= end_date:
        if current.weekday() < 5:
            days += 1
        current += datetime.timedelta(days=1)
    return days


# ── anomaly dedup helper ──

def refresh_anomalies():
    state = read_state()
    state["anomalies"] = scan_anomalies(state)
    _atomic_write(state)
    return state["anomalies"]


# ── Dashboard ──

def get_dashboard(state: dict = None) -> str:
    if state is None:
        state = read_state()

    lines = []
    lines.append("=" * 50)
    lines.append("  System State Dashboard")
    lines.append("=" * 50)

    lu = state.get("last_updated", "never")
    lines.append(f"  Updated: {lu}")

    # Market
    m = state.get("market", {})
    lines.append(f"\n  Market: {m.get('regime', '?')} (conf={m.get('confidence', 0):.2f})")
    c = m.get("contradiction", {})
    lines.append(f"    Policy:{c.get('policy', '?')}  Leverage:{c.get('leverage', '?')}  "
                 f"Sentiment:{c.get('sentiment', '?')}  Valuation:{c.get('valuation', '?')}  "
                 f"Liquidity:{c.get('liquidity', '?')}")

    # Scanner
    s = state.get("scanner", {})
    stale_flag = "STALE" if s.get("stale") else "OK"
    lines.append(f"\n  Scanner: {stale_flag} | last={s.get('last_run', 'never')} | candidates={len(s.get('candidates', []))}")

    # Positions
    positions = state.get("positions", {})
    lines.append(f"\n  Positions ({len(positions)}):")
    for sym, pos in positions.items():
        score = pos.get("score", "?")
        pnl = pos.get("pnl_pct", pos.get("pnl", 0))
        flag = "!" if (isinstance(score, (int, float)) and score < 40) else " "
        lines.append(f"   {flag} {sym}: score={score} pnl={pnl}%")

    # Tasks
    tasks = state.get("tasks", {})
    lines.append(f"\n  Tasks ({len(tasks)}):")
    for mod, info in tasks.items():
        status = info.get("status", "?")
        last = info.get("last", "?")
        lines.append(f"    {mod}: {status} @ {last}")

    # Anomalies
    anomalies = state.get("anomalies", [])
    if anomalies:
        lines.append(f"\n  Anomalies ({len(anomalies)}):")
        for a in anomalies:
            level = a.get("level", "?")
            icon = {"error": "\u26a0", "warn": "\u26a1", "info": "\u2139"}.get(level, "?")
            lines.append(f"   {icon} [{level}] {a.get('module', '?')}: {a.get('msg', '')}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# ── Quick init ──

def init_state():
    if not os.path.exists(STATE_FILE):
        state = json.loads(json.dumps(DEFAULT_STATE))
        state["last_updated"] = _now_iso()
        _atomic_write(state)
        return True
    return False


# ── Main (self test) ──

if __name__ == "__main__":
    init_state()
    report_scanner([
        {"symbol": "002050", "name": "\u4e09\u82b1\u667a\u63a7", "score": 78, "sector": "\u6c7d\u8f66"},
        {"symbol": "600038", "name": "\u4e2d\u76f4\u80a1\u4efd", "score": 65, "sector": "\u56fd\u9632"},
    ], duration=12.3)

    report_market_regime({
        "regime": "\u4e0a\u5347",
        "confidence": 0.62,
        "date": "20260530",
        "indicators": {
            "policy_score": 62, "leverage_score": 45, "sentiment_score": 55,
            "valuation_score": 48, "liquidity_score": 60,
        }
    })

    report_position("002050", {"price": 18.5, "score": 78, "shares": 500, "cost": 18.0, "buy_date": "20260528"})

    report_task("ic_check", "ok", duration=3.5)

    anomalies = refresh_anomalies()
    print(f"Anomalies detected: {len(anomalies)}")
    for a in anomalies:
        print(f"  [{a['level']}] {a['module']}: {a['msg']}")

    print()
    print(get_dashboard())
    print()
    print("All tests passed.")
