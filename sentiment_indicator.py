"""
市场情绪指标模块

三层结构:
- 日频: 两融净买入/涨停强度/全A成交额/涨跌比/尾盘占比/北向资金/换手率
- 周频: 日频5日均值 + 趋势动量
- 月频: A股新开户/月均量能/月均两融/新股首日涨幅/基金发行

数据源: Tushare Pro + AKShare + SSE API
"""

import json, os, time, re
import pandas as pd
import numpy as np

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_CACHE = {
    "daily": {"data": None, "expire": 0.0},
    "weekly": {"data": None, "expire": 0.0},
    "monthly": {"data": None, "expire": 0.0},
    "hist_margin": {"data": None, "expire": 0.0},
    "hist_volume": {"data": None, "expire": 0.0},
    "hist_accounts": {"data": None, "expire": 0.0},
    "hist_daily_batch": {"data": None, "expire": 0.0},
    "hist_limit": {"data": None, "expire": 0.0},
    "hist_northbound": {"data": None, "expire": 0.0},
    "hist_turnover": {"data": None, "expire": 0.0},
}
_CACHE_TTL = {"daily": 300, "weekly": 600, "monthly": 3600, "hist": 600}


def _get_token():
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("_tushare", {}).get("token", "") or cfg.get("tushare_token", "")


def _get_pro():
    import tushare as ts
    return ts.pro_api(_get_token())


def _pct_to_level(pct):
    return "红" if pct >= 80 else ("黄" if pct >= 50 else "绿")


def _level_score(level):
    return {"红": 90, "黄": 60, "绿": 30}[level]


def _trading_dates(n=10):
    """最近n个交易日"""
    pro = _get_pro()
    cal = pro.trade_cal(exchange="SSE", start_date="20250101",
                        end_date=time.strftime("%Y%m%d"))
    if cal is None or not len(cal):
        return []
    open_days = cal[cal["is_open"] == 1]["cal_date"].tolist()
    # trade_cal 降序返回
    return open_days[:n]


# ══════════════════════════════════════════
# 日频情绪指标（支持 trade_date 参数）
# ══════════════════════════════════════════

def _get_margin_net(trade_date=None):
    """两融净买入额: 正=热, 负=冷"""
    now = time.time()
    key = "hist_margin"
    if _CACHE[key]["data"] is None or now >= _CACHE[key]["expire"]:
        pro = _get_pro()
        end = time.strftime("%Y%m%d")
        df = pro.margin(start_date="20210101", end_date=end)
        result = {}
        if df is not None and len(df):
            for _, row in df.iterrows():
                net = float(row["rzmre"]) - float(row["rzche"])
                result[row["trade_date"]] = net
        _CACHE[key]["data"] = result
        _CACHE[key]["expire"] = now + _CACHE_TTL["hist"]

    data = _CACHE[key]["data"]
    if trade_date and trade_date in data:
        cur = data[trade_date]
    else:
        dates = sorted(data.keys())
        cur = data[dates[-1]] if dates else 0

    vals = list(data.values())[-60:]
    pct = (sum(1 for v in vals if v < cur) / len(vals)) * 100
    return cur, _pct_to_level(pct), f"净买入{cur/1e8:+.0f}亿, 近60日{pct:.0f}%分位"


def _get_fengban_rate(trade_date=None):
    """封板率: 涨停中未炸板比例"""
    import akshare as ak
    dt = (trade_date or _last_trading_day())
    try:
        key = f"zt_{dt}"
        if key in _CACHE:
            return _CACHE[key]
        df = ak.stock_zt_pool_em(date=dt)
        if df is None or not len(df):
            return 0, "绿", "非交易日"
        total = len(df)
        zhaban = int((df["炸板次数"] > 0).sum()) if "炸板次数" in df.columns else 0
        rate = (total - zhaban) / total * 100
        lv = "红" if rate >= 65 else ("黄" if rate >= 45 else "绿")
        r = (rate, lv, f"涨停{total}只, 炸板{zhaban}只, 封板率{rate:.0f}%")
        _CACHE[key] = r
        return r
    except Exception as e:
        return 0, "绿", str(e)


def _get_limit_intensity(trade_date=None):
    """涨停强度: 涨停数/跌停数 + 沸点/冰点(历史90%分位)

    用 pro.daily 按 pct_chg >= 9.5 判定涨停, <= -9.5 判定跌停
    替代已失效的 pro.limit_list（数据只到2023年）
    """
    key = "hist_limit"
    now = time.time()

    dt = trade_date or _last_trading_day()
    data = _CACHE[key]["data"]

    # 首次调用: 预填近10个交易日
    if data is None or now >= _CACHE[key]["expire"]:
        data = {}
        pro = _get_pro()
        for d in _trading_dates(10):
            df = pro.daily(trade_date=d)
            if df is not None and len(df):
                data[d] = {
                    "up": int((df["pct_chg"] >= 9.5).sum()),
                    "down": int((df["pct_chg"] <= -9.5).sum()),
                }
        _CACHE[key]["data"] = data
        _CACHE[key]["expire"] = now + _CACHE_TTL["hist"]

    # 如果 dt 不在缓存中，补抓
    if dt not in data:
        df = _get_pro().daily(trade_date=dt)
        if df is not None and len(df):
            data[dt] = {
                "up": int((df["pct_chg"] >= 9.5).sum()),
                "down": int((df["pct_chg"] <= -9.5).sum()),
            }

    cur = data.get(dt, {"up": 0, "down": 0})
    up, down = cur["up"], cur["down"]
    ratio = up / down if down > 0 else 99.0

    up_vals = sorted([v["up"] for v in data.values()])
    down_vals = sorted([v["down"] for v in data.values()])
    up_90 = np.percentile(up_vals, 90) if len(up_vals) >= 10 else 999
    down_90 = np.percentile(down_vals, 90) if len(down_vals) >= 10 else 999

    flags = []
    if up >= up_90:
        flags.append("沸点")
    if down >= down_90:
        flags.append("冰点")
    flag_str = ",".join(flags) if flags else "正常"

    lv = "红" if ratio >= 5 else ("黄" if ratio >= 1 else "绿")
    return ratio, lv, f"涨停{up}/跌停{down}, 比={ratio:.1f}, {flag_str}"


def _get_volume_heat(trade_date=None):
    """全A成交额（中证全指 000985.CSI）"""
    key = "hist_volume"
    now = time.time()
    if _CACHE[key]["data"] is None or now >= _CACHE[key]["expire"]:
        pro = _get_pro()
        end = time.strftime("%Y%m%d")
        df = pro.index_daily(ts_code="000985.CSI", start_date="20210101", end_date=end)
        result = {}
        if df is not None and len(df):
            for _, row in df.iterrows():
                result[row["trade_date"]] = float(row["amount"]) * 1000  # 千元→元
        _CACHE[key]["data"] = result
        _CACHE[key]["expire"] = now + _CACHE_TTL["hist"]

    data = _CACHE[key]["data"]
    if trade_date and trade_date in data:
        cur = data[trade_date]
    else:
        dates = sorted(data.keys())
        cur = data[dates[-1]] if dates else 0

    vals = list(data.values())[-60:]
    pct = (sum(1 for v in vals if v < cur) / len(vals)) * 100
    return cur, _pct_to_level(pct), f"全A成交额{cur/1e8:.0f}亿, 近60日{pct:.0f}%分位"


def _last_trading_day():
    """最近一个交易日"""
    cal = _get_pro().trade_cal(exchange="SSE", start_date="20250101",
                                end_date=time.strftime("%Y%m%d"))
    if cal is not None and len(cal):
        open_days = cal[cal["is_open"] == 1]["cal_date"].tolist()
        # trade_cal 按降序返回（最新在前），取第一个
        if open_days:
            return open_days[0]
    return time.strftime("%Y%m%d")


def _get_advance_decline(trade_date=None):
    """涨跌比: 全市场上涨/下跌家数比"""
    pro = _get_pro()
    dt = (trade_date or _last_trading_day())
    try:
        df = pro.daily(trade_date=dt)
        if df is None or not len(df):
            return 0, "绿", "无数据"
        up = (df["pct_chg"] > 0).sum()
        down = (df["pct_chg"] < 0).sum()
        total = up + down
        ratio = up / down if down > 0 else 99
        # 涨跌比 >2 红, 1-2 黄, <1 绿
        lv = "红" if ratio >= 2 else ("黄" if ratio >= 1 else "绿")
        return ratio, lv, f"涨{up}/跌{down}, 比={ratio:.2f}"
    except Exception as e:
        return 0, "绿", str(e)


def _get_tail_ratio(trade_date=None):
    """尾盘30分钟成交占比"""
    import akshare as ak
    try:
        # AKShare 分钟线只返回当日，忽略trade_date参数
        df = ak.stock_zh_a_minute(symbol="sh000001", period="5", adjust="")
        if df is None or not len(df):
            return 0, "绿", "无分钟数据"
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        total = df["volume"].sum()
        tail = df[df["day"].str[-8:] >= "14:30"]
        tail_v = tail["volume"].sum()
        if total == 0:
            return 0, "绿", "零成交"
        ratio = tail_v / total * 100
        lv = "红" if ratio >= 15 else ("黄" if ratio >= 11 else "绿")
        return ratio, lv, f"尾盘占比{ratio:.1f}% ({total:,.0f}手)"
    except Exception as e:
        return 0, "绿", str(e)


def _get_buyback_ratio(trade_date=None):
    """回购/解禁比"""
    pro = _get_pro()
    end = time.strftime("%Y%m%d")
    try:
        rep = pro.repurchase(start_date="20250101", end_date=end)
        sf = pro.share_float(start_date="20250101", end_date=end)
        rep_amt = rep["amount"].sum() if rep is not None and len(rep) else 0
        sf_amt = sf["float_share"].sum() if sf is not None and len(sf) else 1
        ratio = (rep_amt / sf_amt) / 1e6  # 亿/万股
        lv = "红" if ratio >= 1.0 else ("黄" if ratio >= 0.3 else "绿")
        return ratio, lv, f"回购{rep_amt/1e8:.0f}亿 / 解禁{sf_amt:.0f}万股, {ratio:.2f}亿/万股"
    except Exception as e:
        return 0, "绿", str(e)


def _get_northbound_flow(trade_date=None):
    """北向资金净流入: 正=外资看多, 负=外资跑路"""
    key = "hist_northbound"
    now = time.time()
    if _CACHE[key]["data"] is None or now >= _CACHE[key]["expire"]:
        pro = _get_pro()
        end = time.strftime("%Y%m%d")
        df = pro.moneyflow_hsgt(start_date="20210101", end_date=end)
        result = {}
        if df is not None and len(df):
            for _, row in df.iterrows():
                result[row["trade_date"]] = float(row["north_money"])
        _CACHE[key]["data"] = result
        _CACHE[key]["expire"] = now + _CACHE_TTL["hist"]

    data = _CACHE[key]["data"]
    if trade_date and trade_date in data:
        cur = data[trade_date]
    else:
        dates = sorted(data.keys())
        cur = data[dates[-1]] if dates else 0

    vals = list(data.values())[-60:]
    pct = (sum(1 for v in vals if v < cur) / len(vals)) * 100
    lv = "红" if pct >= 80 else ("黄" if pct >= 50 else "绿")
    return cur, lv, f"北向净{cur/1e4:+.1f}亿, 近60日{pct:.0f}%分位"


def _get_turnover_rate(trade_date=None):
    """全市场换手率: 上证指数自由换手率, 高=活跃亢奋, 低=冷清冰点"""
    key = "hist_turnover"
    now = time.time()
    if _CACHE[key]["data"] is None or now >= _CACHE[key]["expire"]:
        pro = _get_pro()
        end = time.strftime("%Y%m%d")
        df = pro.index_dailybasic(ts_code="000001.SH",
                                   start_date="20250101", end_date=end)
        result = {}
        if df is not None and len(df):
            for _, row in df.iterrows():
                result[row["trade_date"]] = float(row["turnover_rate_f"])
        _CACHE[key]["data"] = result
        _CACHE[key]["expire"] = now + _CACHE_TTL["hist"]

    data = _CACHE[key]["data"]
    if trade_date and trade_date in data:
        cur = data[trade_date]
    else:
        dates = sorted(data.keys())
        cur = data[dates[-1]] if dates else 0

    vals = list(data.values())[-60:]
    pct = (sum(1 for v in vals if v < cur) / len(vals)) * 100
    lv = "红" if pct >= 80 else ("黄" if pct >= 50 else "绿")
    return cur, lv, f"换手率{cur:.2f}%, 近60日{pct:.0f}%分位"


# ══════════════════════════════════════════
# 月频情绪指标
# ══════════════════════════════════════════

def _get_new_accounts_hist():
    """A股新开户历史数据"""
    now = time.time()
    if _CACHE["hist_accounts"]["data"] and now < _CACHE["hist_accounts"]["expire"]:
        return _CACHE["hist_accounts"]["data"]

    import requests as req
    sse_url = "https://query.sse.com.cn/commonQuery.do"
    sse_headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"}
    accounts = []
    cy = time.localtime().tm_year
    for year in range(2021, cy + 1):
        params = {"jsonCallBack": "jsonpCallback", "sqlId": "COMMON_SSE_TZZ_M_ALL_ACCT_C", "isPagination": "false"}
        if year < cy:
            params["MDATE"] = f"{year}12"
        try:
            r = req.get(sse_url, params=params, headers=sse_headers, timeout=15)
            m = re.search(r"jsonpCallback\((\{.*\})\)", r.text, re.DOTALL)
            if not m:
                continue
            data = json.loads(m.group(1))
            for row in data.get("result", []):
                term = row.get("TERM", "")
                a = row.get("A_ACCT", "0")
                if "合计" in term or "累计" in term or float(a) == 0:
                    continue
                accounts.append(float(a))
        except Exception:
            continue
    _CACHE["hist_accounts"]["data"] = accounts
    _CACHE["hist_accounts"]["expire"] = now + _CACHE_TTL["hist"]
    return accounts


def _get_new_accounts(trade_date=None):
    """A股新开户"""
    hist = _get_new_accounts_hist()
    if len(hist) < 6:
        return 0, "绿", "数据不足"
    cur = hist[-1]
    pct = (sum(1 for v in hist if v < cur) / len(hist)) * 100
    return cur, _pct_to_level(pct), f"本月{cur:.1f}万户, 历史{pct:.0f}%分位 ({len(hist)}个月)"


def _get_monthly_volume(trade_date=None):
    """月均量能"""
    hist = _get_volume_heat.__wrapped__ if hasattr(_get_volume_heat, "__wrapped__") else None
    _ = _get_volume_heat()  # ensure cache
    data = _CACHE["hist_volume"]["data"]
    if not data:
        return 0, "绿", "数据不足"
    monthly = {}
    for d, v in data.items():
        m = d[:6]
        monthly.setdefault(m, []).append(v)
    months = sorted(monthly.keys())
    cur_avg = np.mean(monthly[months[-1]])
    avgs = [np.mean(monthly[m]) for m in months]
    pct = (sum(1 for a in avgs if a < cur_avg) / len(avgs)) * 100
    return cur_avg / 1e8, _pct_to_level(pct), f"月均{cur_avg/1e8:.0f}亿, 历史{pct:.0f}%分位 ({len(months)}个月)"


def _get_monthly_margin(trade_date=None):
    """月均两融余额"""
    pro = _get_pro()
    end = time.strftime("%Y%m%d")
    df = pro.margin(start_date="20210101", end_date=end)
    if df is None or not len(df):
        return 0, "绿", "数据不足"
    df["month"] = df["trade_date"].str[:6]
    monthly = df.groupby("month")["rzye"].mean()
    vals = monthly.values
    cur = vals[-1]
    pct = (sum(1 for v in vals if v < cur) / len(vals)) * 100
    return cur / 1e8, _pct_to_level(pct), f"月均{cur/1e8:.0f}亿, 历史{pct:.0f}%分位 ({len(vals)}个月)"


def _get_ipo_first_day(trade_date=None):
    """新股首日涨幅月均值"""
    pro = _get_pro()
    end = time.strftime("%Y%m%d")
    try:
        ipos = pro.new_share(start_date="20250101", end_date=end)
        if ipos is None or not len(ipos):
            return 0, "绿", "无新股"
        gains = []
        for _, row in ipos.iterrows():
            issue_date = str(int(row["issue_date"])) if pd.notna(row["issue_date"]) and row["issue_date"] else ""
            price = float(row["price"]) if pd.notna(row["price"]) else 0
            if not issue_date or price <= 0:
                continue
            d = pro.daily(ts_code=row["ts_code"], start_date=issue_date, end_date=issue_date)
            if d is not None and len(d):
                gains.append((issue_date[:6], (float(d["close"].iloc[0]) / price - 1) * 100))
        if not gains:
            return 0, "绿", "无首日数据"
        monthly = {}
        for m, g in gains:
            monthly.setdefault(m, []).append(g)
        all_avgs = [np.mean(v) for v in monthly.values()]
        cur_avg = all_avgs[-1]
        pct = (sum(1 for a in all_avgs if a < cur_avg) / len(all_avgs)) * 100
        return cur_avg, _pct_to_level(pct), f"本月均值{cur_avg:.0f}%, 历史{pct:.0f}%分位 ({len(gains)}只)"
    except Exception as e:
        return 0, "绿", str(e)


def _get_fund_heat(trade_date=None):
    """基金发行热度"""
    pro = _get_pro()
    import akshare as ak
    try:
        df = ak.fund_scale_change_em()
        if df is None or not len(df):
            return 0, "绿", "无数据"
        buy_col, sell_col = df.columns[3], df.columns[4]
        df_sorted = df.sort_values(by=df.columns[1])
        last = df_sorted.iloc[-1]
        prev = df_sorted.iloc[-2] if len(df_sorted) > 1 else last
        last_buy = float(last[buy_col])
        ratio = last_buy / float(prev[buy_col]) if float(prev[buy_col]) > 0 else 1.0
        net = last_buy - float(last[sell_col])
        lv = "红" if (ratio > 1.3 and net > 0) else ("黄" if (ratio > 1.1 or net > 0) else "绿")
        etf_trend = ""
        try:
            shares = []
            for code in ["510300.SH", "510050.SH", "159919.SZ", "510880.SH"]:
                etf = pro.fund_share(ts_code=code, start_date="20250101", end_date=time.strftime("%Y%m%d"))
                if etf is not None and len(etf) > 20:
                    s = etf.sort_values("trade_date")["fd_share"].astype(float)
                    shares.append((s.iloc[-1] - s.iloc[-21]) / s.iloc[-21] * 100)
            if shares:
                etf_trend = f", ETF{np.mean(shares):+.1f}%/20日"
        except Exception:
            pass
        return last_buy, lv, f"季度{last.iloc[1]}, 申购{ratio:.2f}倍, 净{net/1e4:+.0f}亿{etf_trend}"
    except Exception as e:
        return 0, "绿", str(e)


# ══════════════════════════════════════════
# 公共API
# ══════════════════════════════════════════

def _detect_overheat(items, n_days=60):
    """3σ 过热/过冷检测
    对每个有历史分比数据的指标，判断当前是否超出均值±3σ
    返回标记列表 [{name, status, detail}]
    """
    signals = []
    # 拿到最近60个交易日的各指标打分序列
    dates = _trading_dates(n_days + 5)
    if len(dates) < 20:
        return signals

    for item in items:
        name = item["name"]
        hist_scores = []
        for d in dates:
            try:
                d_items = _compute_items(_DAILY_INDICATORS, trade_date=d)
                match = [x for x in d_items if x["name"] == name]
                if match:
                    hist_scores.append(_level_score(match[0]["level"]))
            except Exception:
                continue
        if len(hist_scores) < 10:
            continue

        mu = np.mean(hist_scores)
        sigma = np.std(hist_scores)
        cur = _level_score(item["level"])

        if sigma < 5:
            continue  # 波动太小不判
        if cur > mu + 3 * sigma:
            signals.append({"name": name, "status": "过热",
                            "detail": f"当前{cur:.0f}, 均值{mu:.0f}±{3*sigma:.0f}(3σ)"})
        elif cur < mu - 3 * sigma:
            signals.append({"name": name, "status": "过冷",
                            "detail": f"当前{cur:.0f}, 均值{mu:.0f}±{3*sigma:.0f}(3σ)"})
    return signals


_DAILY_INDICATORS = [
    ("两融净买入", _get_margin_net, 20),
    ("涨停强度", _get_limit_intensity, 15),
    ("全A成交额", _get_volume_heat, 15),
    ("涨跌比", _get_advance_decline, 15),
    ("尾盘占比", _get_tail_ratio, 5),
    ("北向资金", _get_northbound_flow, 15),
    ("换手率", _get_turnover_rate, 15),
]

_MONTHLY_INDICATORS = [
    ("A股新开户", _get_new_accounts, 25),
    ("月均量能", _get_monthly_volume, 25),
    ("月均两融", _get_monthly_margin, 20),
    ("新股首日涨幅", _get_ipo_first_day, 15),
    ("基金发行", _get_fund_heat, 15),
]


def _build_result(items, label=""):
    scores = [_level_score(it["level"]) for it in items if it["value"] != "err"]
    avg = np.mean(scores) if scores else 30
    if avg >= 70:
        level, summary = "红", f"{label}情绪过热"
    elif avg >= 40:
        level, summary = "黄", f"{label}情绪偏热"
    else:
        level, summary = "绿", f"{label}情绪正常"
    return {"score": round(avg), "level": level, "summary": summary.strip(), "items": items}


def _compute_items(indicators, trade_date=None):
    """批量计算一组指标"""
    items = []
    for name, func, weight in indicators:
        v, lv, d = func(trade_date)
        items.append({"name": name, "value": _fmt_val(name, v), "level": lv, "detail": d})
    return items


def _fmt_val(name, v):
    if name in ("两融净买入",):
        return f"{v/1e8:+.0f}亿"
    elif name in ("全A成交额",):
        return f"{v/1e8:.0f}亿"
    elif name in ("封板率",):
        return f"{v:.0f}%"
    elif name in ("涨跌比",):
        return f"{v:.2f}"
    elif name in ("涨停强度",):
        return f"{v:.1f}"
    elif name in ("尾盘占比",):
        return f"{v:.1f}%"
    elif name in ("北向资金",):
        return f"{v/1e4:+.1f}亿"
    elif name in ("换手率",):
        return f"{v:.2f}%"
    elif name in ("A股新开户",):
        return f"{v:.1f}万户"
    elif name in ("月均量能", "月均两融"):
        return f"{v:.0f}亿"
    elif name in ("新股首日涨幅",):
        return f"{v:.0f}%"
    elif name in ("基金发行",):
        return f"{v/1e4:.0f}亿"
    return str(v)


def get_daily_sentiment(force_refresh=False):
    """日频情绪指数（含3σ过热检测）"""
    now = time.time()
    if not force_refresh and _CACHE["daily"]["data"] and now < _CACHE["daily"]["expire"]:
        return _CACHE["daily"]["data"]
    items = _compute_items(_DAILY_INDICATORS)
    result = _build_result(items, "日频")
    # 3σ 过热检测
    result["overheat"] = _detect_overheat(items)
    _CACHE["daily"]["data"] = result
    _CACHE["daily"]["expire"] = now + _CACHE_TTL["daily"]
    return result


def get_weekly_sentiment(force_refresh=False):
    """周频情绪指数: 5日均值 + 趋势动量

    Returns dict with:
      - score/level/summary: 5日平滑评分
      - momentum: "升温"/"降温"/"持平"
      - delta: 较前5日的分数变化
      - items: 各指标5日均值详情
    """
    now = time.time()
    if not force_refresh and _CACHE["weekly"]["data"] and now < _CACHE["weekly"]["expire"]:
        return _CACHE["weekly"]["data"]

    dates = _trading_dates(12)  # 拿12天确保有10个交易日
    if len(dates) < 10:
        return {"score": 50, "level": "黄", "summary": "周频数据不足",
                "momentum": "持平", "delta": 0, "items": []}

    recent_5 = dates[-5:]
    prev_5 = dates[-10:-5]

    def avg_score(dates_list):
        """一组日期上的平均分，跳过无数据的日期"""
        scores = []
        for d in dates_list:
            items = _compute_items(_DAILY_INDICATORS, trade_date=d)
            valid = [it for it in items if it["value"] != "err" and "非交易日" not in it["detail"]]
            if len(valid) >= 3:  # 至少3个有效指标
                scores.append(np.mean([_level_score(it["level"]) for it in valid]))
        return np.mean(scores) if scores else None

    cur_avg = avg_score(recent_5)
    prev_avg = avg_score(prev_5)

    if cur_avg is None:
        cur_avg = 50
    if prev_avg is None:
        delta = 0
    else:
        delta = cur_avg - prev_avg

    if delta >= 8:
        momentum = "升温"
    elif delta <= -8:
        momentum = "降温"
    else:
        momentum = "持平"

    # 使用最新的有效日频数据做指标详情
    items = get_daily_sentiment()["items"]

    if cur_avg >= 70:
        level, summary = "红", f"周频情绪过热, 5日均{cur_avg:.0f}分"
    elif cur_avg >= 40:
        level, summary = "黄", f"周频情绪偏热, 5日均{cur_avg:.0f}分"
    else:
        level, summary = "绿", f"周频情绪正常, 5日均{cur_avg:.0f}分"

    result = {
        "score": round(cur_avg),
        "level": level,
        "summary": summary,
        "momentum": momentum,
        "delta": round(delta),
        "items": items,
        "overheat": _detect_overheat(items),
    }
    _CACHE["weekly"]["data"] = result
    _CACHE["weekly"]["expire"] = now + _CACHE_TTL["weekly"]
    return result


def get_monthly_sentiment(force_refresh=False):
    """月频情绪指数（长线）"""
    now = time.time()
    if not force_refresh and _CACHE["monthly"]["data"] and now < _CACHE["monthly"]["expire"]:
        return _CACHE["monthly"]["data"]
    items = _compute_items(_MONTHLY_INDICATORS)
    result = _build_result(items, "月频")
    _CACHE["monthly"]["data"] = result
    _CACHE["monthly"]["expire"] = now + _CACHE_TTL["monthly"]
    return result


def get_sentiment(force_refresh=False):
    """综合情绪（日:周:月 = 40:30:30, 向后兼容）"""
    daily = get_daily_sentiment(force_refresh)
    weekly = get_weekly_sentiment(force_refresh)
    monthly = get_monthly_sentiment(force_refresh)
    combined = round(daily["score"] * 0.4 + weekly["score"] * 0.3 + monthly["score"] * 0.3)
    all_items = daily["items"] + weekly["items"] + monthly["items"]
    scores = [_level_score(it["level"]) for it in all_items if it["value"] != "err"]
    avg = np.mean(scores) if scores else 30
    level = "红" if avg >= 70 else ("黄" if avg >= 40 else "绿")
    return {
        "score": combined, "level": level,
        "summary": f"综合情绪{['正常','偏热','过热'][['绿','黄','红'].index(level)]}",
        "items": all_items, "daily": daily, "weekly": weekly, "monthly": monthly,
        "overheat": daily.get("overheat", []) + weekly.get("overheat", []),
    }


if __name__ == "__main__":
    import sys

    print("═══ 日频 ═══")
    d = get_daily_sentiment(True)
    print(f"{d['score']} [{d['level']}] {d['summary']}")
    for it in d["items"]:
        print(f"  {it['name']}: {it['value']} [{it['level']}] {it['detail']}")

    print("\n═══ 周频 ═══")
    w = get_weekly_sentiment(True)
    print(f"{w['score']} [{w['level']}] {w['summary']}")
    print(f"  动量: {w['momentum']} (Δ{w['delta']:+d})")
    for it in w["items"]:
        print(f"  {it['name']}: {it['value']} [{it['level']}]")

    print("\n═══ 月频 ═══")
    m = get_monthly_sentiment(True)
    print(f"{m['score']} [{m['level']}] {m['summary']}")
    for it in m["items"]:
        print(f"  {it['name']}: {it['value']} [{it['level']}] {it['detail']}")

    print("\n═══ 综合 ═══")
    c = get_sentiment()
    print(f"{c['score']} [{c['level']}] {c['summary']}")
    if c.get("overheat"):
        print("\n⚠ 3σ 异常信号:")
        for s in c["overheat"]:
            print(f"  {s['name']}: {s['status']} — {s['detail']}")
