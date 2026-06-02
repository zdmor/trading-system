"""
豆包选股规则 — 完整扫描（缓存版）
===============================
全部5条可量化条件：
  1. 资金异动 — 月内涨停或倍量
  2. 放量长上影线 — 上涨初期/横盘,碰前高需放量
  3. 一进二标的 — 市值<100亿,25天涨停,未破涨停底价
  4. 强势股回调 — 涨停后确认底部,大盘跌它不跌
  5. 量价背离 — 连跌2天+缩量跌更多

用法: python doubao_rules_scan.py
"""
import sys, os, time, json
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

from scanner import Scanner
import requests

CACHE_DIR = r"D:\ClaudeWorkspace\trading_system\data_cache\daily"
LOOKBACK = 25


def get_candidates(top_n=500):
    s = Scanner()
    s.fetch_all_stocks()
    candidates = s.filter_candidates()
    codes = []
    seen = set()
    for c in candidates:
        code = c["code"]
        name = c.get("name", "")
        sym = code.split(".")[-1]
        if sym.startswith(("8", "4", "9")) or "ST" in name or code in seen:
            continue
        seen.add(code)
        codes.append((code, name, sym))
        if len(codes) >= top_n:
            break
    return codes


def fetch_market_index():
    """上证指数日线"""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000001,day,,,500,qfq"
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        klines = data.get("data", {}).get("sh000001", {}).get("qfqday") or \
                 data.get("data", {}).get("sh000001", {}).get("day", [])
        if klines and len(klines) >= 100:
            rows = [{"date": pd.Timestamp(k[0]), "close": float(k[2])} for k in klines]
            return pd.DataFrame(rows).reset_index(drop=True)
    except:
        return None


def get_cached_df(sym):
    """从本地缓存读取日线数据"""
    # 检查 sh_ 和 sz_ 前缀
    for prefix in ("sh_", "sz_"):
        path = os.path.join(CACHE_DIR, f"{prefix}{sym}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            if len(df) >= 60:
                return df.reset_index(drop=True)
    return None


def get_market_cap_tencent(sym):
    """腾讯行情接口获取市值(亿)"""
    prefix = "sh" if sym.startswith("6") else "sz"
    url = f"http://qt.gtimg.cn/q={prefix}{sym}"
    try:
        r = requests.get(url, timeout=5)
        parts = r.text.split("~")
        if len(parts) > 45:
            cap_str = parts[45]
            if cap_str:
                cap = float(cap_str) / 1e8
                return cap
    except:
        pass
    return None


def check_condition1_anomaly(df):
    """条件1: 资金异动 — 月内涨停或倍量"""
    if df is None or len(df) < LOOKBACK:
        return False, ""
    recent = df.tail(LOOKBACK)
    avg_vol = recent["volume"].mean()
    reasons = []

    limit_up = recent[recent["pct_chg"] >= 9.5]
    if len(limit_up) > 0:
        reasons.append(f"涨停{len(limit_up)}次")

    max_vol_ratio = recent["volume"].max() / avg_vol if avg_vol > 0 else 0
    if max_vol_ratio >= 2.0:
        reasons.append(f"倍量{max_vol_ratio:.1f}x")

    return len(reasons) > 0, " | ".join(reasons) if reasons else ""


def check_condition2_long_shadow(df):
    """条件2: 放量长上影线 — 上涨初期/横盘,碰前高需放量"""
    if df is None or len(df) < 60:
        return False, ""
    recent = df.tail(10)
    full = df.tail(120)  # 更长的历史判断前高和横盘
    avg_vol = recent["volume"].mean()

    for i in range(len(recent) - 1, -1, -1):
        row = recent.iloc[i]
        body = abs(row["close"] - row["open"])
        upper = row["high"] - max(row["close"], row["open"])
        vol_ratio = row["volume"] / avg_vol if avg_vol > 0 else 0

        if body <= 0:
            continue

        if upper >= body * 2.0 and vol_ratio >= 1.5:
            date_str = row["date"].strftime("%m-%d")

            # 上涨初期判断：当前位置在60日均线附近（偏离不超过20%）
            ma60 = full["close"].tail(60).mean()
            current = row["close"]
            deviation = abs(current / ma60 - 1)

            # 判断是否前期高点附近
            lookback_highs = full["high"].values
            recent_high = np.max(lookback_highs[-120:-5])  # 排除最近5天
            at_high = current >= recent_high * 0.95
            vol_vs_high = vol_ratio

            details = []
            if deviation <= 0.15:
                details.append("上涨初期")
            else:
                details.append("偏离较大")

            if at_high:
                details.append(f"近前高({current:.0f}≈{recent_high:.0f})")
                if vol_vs_high >= 2.0:
                    details.append("放量过前高")
                else:
                    details.append("量不足")

            return True, f"{date_str} 上影{upper:.2f}/实体{body:.2f} 量比{vol_ratio:.1f}x | {' '.join(details)}"

    return False, ""


def check_condition3_yijiner(df, market_cap):
    """条件3: 一进二标的 — 市值<100亿,25天涨停,未破涨停底价,横盘突破"""
    if df is None or len(df) < LOOKBACK:
        return False, ""
    if market_cap is not None and market_cap > 100:
        return False, "市值超"

    recent = df.tail(LOOKBACK)

    # 找涨停日
    limit_up_days = recent[recent["pct_chg"] >= 9.5]
    if len(limit_up_days) == 0:
        return False, "无涨停"

    # 最新涨停日
    last_limit = limit_up_days.iloc[-1]
    last_limit_idx = limit_up_days.index[-1]
    limit_low = last_limit["low"]

    # 没跌破涨停底价
    after_limit = df.loc[last_limit_idx:]
    lowest_after = after_limit["low"].min()
    if lowest_after < limit_low * 0.98:
        return False, f"跌破涨停底价({lowest_after:.2f}<{limit_low:.2f})"

    # 横盘突破判断：涨停前后量能对比
    before_vol = df.loc[max(0, last_limit_idx - 10):last_limit_idx, "volume"].mean()
    after_vol = after_limit["volume"].mean() if len(after_limit) > 1 else 0

    details = []
    if before_vol > 0 and after_vol > 0 and before_vol > after_vol * 1.2:
        details.append("缩量横盘")
    else:
        details.append("量正常")

    price_loc = f"现价{float(last_limit['close']):.0f}"
    is_breakout = float(after_limit["high"].max()) > float(recent["high"].iloc[:-len(after_limit)].max()) if len(after_limit) < len(recent) else False
    if is_breakout:
        details.append("突破前高")

    return True, f"近{len(limit_up_days)}次涨停 | {price_loc} | {' | '.join(details)}"


def check_condition4_strong_pullback(df, mkt_df):
    """条件4: 强势股回调 — 涨停后回调确认底部,大盘跌它不跌"""
    if df is None or len(df) < 40 or mkt_df is None:
        return False, ""

    recent = df.tail(20)

    # 找涨停日
    limit_days = recent[recent["pct_chg"] >= 9.5]
    if len(limit_days) == 0:
        return False, "无涨停"

    for idx in reversed(limit_days.index):
        limit_idx = df.index.get_loc(idx)
        after = df.iloc[limit_idx + 1:]
        if len(after) < 5:
            continue

        close = after["close"].values
        low = after["low"].values

        # 回调确认底部: 最低点后反弹
        min_idx = np.argmin(low)
        if min_idx < len(close) - 2:
            if close[-1] > low[min_idx] * 1.02:
                pullback_pct = (low[min_idx] / df.iloc[limit_idx]["close"] - 1) * 100

                # 对比大盘
                limit_date = df.iloc[limit_idx]["date"]
                mkt_match = mkt_df[mkt_df["date"] == limit_date]
                if len(mkt_match) > 0:
                    mkt_idx = mkt_df.index.get_loc(mkt_match.index[0])
                    mkt_after = mkt_df.iloc[mkt_idx + 1: mkt_idx + 1 + len(after)]
                    if len(mkt_after) > 0:
                        mkt_change = (mkt_after["close"].iloc[-1] / mkt_after["close"].iloc[0] - 1) * 100
                        if mkt_change < -2:
                            return True, f"涨停后回调{pullback_pct:.1f}% | 大盘跌{mkt_change:.1f}% 它不跌"

        break

    return False, ""


def check_condition5_divergence(df):
    """条件5: 量价背离 — 连跌2天+缩量跌更多"""
    if df is None or len(df) < 10:
        return False, ""

    recent = df.tail(5)
    if len(recent) < 5:
        return False, ""

    closes = recent["close"].values
    volumes = recent["volume"].values

    # 最近2天连续下跌
    if closes[-1] < closes[-2] and closes[-2] < closes[-3]:
        vol_change = volumes[-1] / max(volumes[-3], 0.001)
        price_change = (closes[-1] / closes[-3] - 1) * 100
        if vol_change < 0.8 and price_change < -2:
            return True, f"连跌{price_change:.1f}% 缩量{vol_change:.1f}x"
    return False, ""


def run():
    t0 = time.time()
    print("=" * 60)
    print("  豆包规则 — 完整选股扫描")
    print("=" * 60)

    print("\n[1/4] 获取候选股 + 大盘数据...")
    candidates = get_candidates(500)
    mkt_df = fetch_market_index()
    print(f"  候选: {len(candidates)}只 | 大盘: {'有' if mkt_df is not None else '无'}")

    print("\n[2/4] 缓存读取 + 逐只检查...")
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    print(f"  缓存: {len(cache_files)}个CSV文件")

    results = []
    for i, (code, name, sym) in enumerate(candidates):
        df = get_cached_df(sym)
        if df is None:
            continue
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df["close"].pct_change() * 100

        price = float(df["close"].iloc[-1])

        c1_ok, c1_reason = check_condition1_anomaly(df)
        c2_ok, c2_reason = check_condition2_long_shadow(df)
        mc = get_market_cap_tencent(sym)
        c3_ok, c3_reason = check_condition3_yijiner(df, mc)
        c4_ok, c4_reason = check_condition4_strong_pullback(df, mkt_df)
        c5_ok, c5_reason = check_condition5_divergence(df)

        if any([c1_ok, c2_ok, c3_ok, c4_ok, c5_ok]):
            results.append({
                "code": sym, "name": name, "price": price,
                "mc": mc,
                "c1": c1_reason, "c2": c2_reason, "c3": c3_reason,
                "c4": c4_reason, "c5": c5_reason,
            })

        if (i + 1) % 100 == 0 or i == len(candidates) - 1:
            print(f"  进度: {i+1}/{len(candidates)} 命中{len(results)}", end="\r")

    print(f"\n  命中: {len(results)}只 ({time.time()-t0:.0f}s)")

    if not results:
        print("\n无符合条件个股")
        return

    print("\n[3/4] 选股结果")
    print("=" * 60)

    # 多条件命中优先
    def hit_score(r):
        return sum([bool(r["c1"]), bool(r["c2"]), bool(r["c3"]), bool(r["c4"]), bool(r["c5"])])

    results.sort(key=lambda r: (-hit_score(r), r["code"]))

    # A: 多条件命中
    multi = [r for r in results if hit_score(r) >= 3]
    if multi:
        print(f"\n  ▎A. 多条件共振（3+条同时命中）")
        print(f"  {'代码':<8} {'名称':<8} {'现价':<8} {'条件':<30}")
        print(f"  {'-'*60}")
        for r in multi[:15]:
            conds = []
            if r["c1"]: conds.append("资金异动")
            if r["c2"]: conds.append("长上影")
            if r["c3"]: conds.append("一进二")
            if r["c4"]: conds.append("强势回调")
            if r["c5"]: conds.append("量价背离")
            print(f"  {r['code']:<8} {r['name']:<8} {r['price']:<8.2f} {','.join(conds)}")

    # B: 一进二标的
    c3_list = [r for r in results if r["c3"]]
    if c3_list:
        print(f"\n  ▎B. 一进二标的（市值<100亿+涨停+未破底）")
        print(f"  {'代码':<8} {'名称':<8} {'现价':<8} {'市值':<8} {'详情'}")
        print(f"  {'-'*60}")
        for r in c3_list[:15]:
            mc_str = f"{r['mc']:.0f}亿" if r["mc"] else "?"
            print(f"  {r['code']:<8} {r['name']:<8} {r['price']:<8.2f} {mc_str:<8} {r['c3']}")

    # C: 强势股回调
    c4_list = [r for r in results if r["c4"]]
    if c4_list:
        print(f"\n  ▎C. 强势股回调（涨停后回调+大盘跌它不跌）")
        print(f"  {'代码':<8} {'名称':<8} {'现价':<8} {'详情'}")
        print(f"  {'-'*50}")
        for r in c4_list[:10]:
            print(f"  {r['code']:<8} {r['name']:<8} {r['price']:<8.2f} {r['c4']}")

    # D: 量价背离
    c5_list = [r for r in results if r["c5"]]
    if c5_list:
        print(f"\n  ▎D. 量价背离（连跌缩量跌更多）")
        print(f"  {'代码':<8} {'名称':<8} {'现价':<8} {'详情'}")
        print(f"  {'-'*50}")
        for r in c5_list[:10]:
            print(f"  {r['code']:<8} {r['name']:<8} {r['price']:<8.2f} {r['c5']}")

    # E: 放量长上影线
    c2_list = [r for r in results if r["c2"]]
    if c2_list:
        print(f"\n  ▎E. 放量长上影线")
        print(f"  {'代码':<8} {'名称':<8} {'现价':<8} {'详情'}")
        print(f"  {'-'*50}")
        for r in c2_list[:10]:
            print(f"  {r['code']:<8} {r['name']:<8} {r['price']:<8.2f} {r['c2']}")

    print(f"\n{'='*60}")
    print(f"  总耗时: {time.time()-t0:.0f}s | 候选{len(candidates)} | 命中{len(results)}")
    print()


if __name__ == "__main__":
    run()
