"""
因子截面相关性矩阵 — 7 因子 Pearson 相关系数

流程:
  1. Tushare daily_basic → 取流通市值前 N 只股票
  2. 批量下载日线数据 + 大盘数据
  3. 每只股票运行 WyckoffAnalyzer + StockScorer 完整因子评分
  4. 计算 7 因子截面 Pearson 相关矩阵
  5. 输出 JSON + 终端报告

用法:
  python factor_corr_matrix.py [--n-stocks 50]

输出:
  output_v2/factor_correlation.json
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from data_providers.tushare_provider import _get_pro
from scoring import StockScorer, WEIGHTS

# 懒加载 WyckoffAnalyzer (依赖 scanner, 只在此脚本需要)
_WYCOFF_OK = False
try:
    from scanner import WyckoffAnalyzer, PatternDetector
    _WYCOFF_OK = True
except ImportError:
    pass

_FACTOR_KEYS = list(WEIGHTS.keys())
_OUTPUT_DIR = os.path.join(_BASE, "output_v2")
_OUTPUT_FILE = os.path.join(_OUTPUT_DIR, "factor_correlation.json")


# ─────────────────────── 数据获取 ───────────────────────


def fetch_top_stocks(n: int = 50):
    """用 Tushare daily_basic 取流通市值前 N 只."""
    pro = _get_pro()
    today = datetime.now().strftime("%Y%m%d")
    df = pro.daily_basic(trade_date=today)
    if df is None or df.empty:
        cal = pro.trade_cal(
            start_date=(datetime.now() - timedelta(days=10)).strftime("%Y%m%d"),
            end_date=today,
        )
        if cal is not None and not cal.empty:
            for d in reversed(cal[cal["is_open"] == 1]["cal_date"].tolist()):
                df = pro.daily_basic(trade_date=d)
                if df is not None and not df.empty:
                    break

    if df is None or df.empty:
        print("  ERROR: daily_basic 为空")
        return []

    df = df[~df["ts_code"].str.startswith("8")]
    df = df[~df["ts_code"].str.startswith("4")]
    df = df.sort_values("circ_mv", ascending=False).head(n)

    rows = []
    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        code = ts_code.split(".")[0]
        market = "sh" if code.startswith("6") else "sz"
        rows.append({
            "code": f"{market}.{code}",
            "ts_code": ts_code,
            "price": float(row["close"]),
        })
    return rows


def fetch_kline_batch(stocks, days=400):
    """批量日线 {bs_code: DataFrame}."""
    pro = _get_pro()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    result = {}
    for i in range(0, len(stocks), 30):
        group = stocks[i:i + 30]
        ts_codes = [s["ts_code"] for s in group]
        try:
            df = pro.daily(adj='qfq', ts_code=",".join(ts_codes),
                           start_date=start, end_date=end)
            if df is None or df.empty:
                continue
            for s in group:
                sd = df[df["ts_code"] == s["ts_code"]].sort_values("trade_date")
                if len(sd) < 60:
                    continue
                result[s["code"]] = sd
        except Exception:
            continue
        time.sleep(0.15)
    return result


def fetch_market_kline():
    """大盘指数（上证综指）日线."""
    pro = _get_pro()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
    df = pro.index_daily(ts_code="000001.SH",
                         start_date=start, end_date=end)
    if df is None or df.empty:
        return None
    return df.sort_values("trade_date")


# ─────────────────────── build_scorer ───────────────────────


def build_scorer(sd: pd.DataFrame, price: float, stock_info: dict,
                 market_df=None):
    """从 K 线数据构建 StockScorer，含 Wyckoff 信号 + 大盘."""
    closes = sd["close"].astype(float).values
    highs = sd["high"].astype(float).values
    lows = sd["low"].astype(float).values
    volumes = (sd["vol"].astype(float) * 100).values
    opens = sd["open"].astype(float).values
    n = len(closes)

    # 趋势
    if n >= 200:
        ma50 = np.mean(closes[-50:])
        ma200 = np.mean(closes)
        trend_dir = "多头" if ma50 > ma200 else "空头"
        strength = round((ma50 / ma200 - 1) * 100, 1)
    else:
        trend_dir = "数据不足"
        strength = 0
    trend = {"direction": trend_dir, "strength": strength}

    # ATR
    atr_pct = 0.0
    if n >= 21:
        trs = [max(highs[-j] - lows[-j],
                   abs(highs[-j] - closes[-j - 1]),
                   abs(lows[-j] - closes[-j - 1]))
               for j in range(1, 21)]
        atr = np.mean(trs)
        atr_pct = round(atr / price * 100, 2) if price > 0 else 0
    vol = {"atr_pct": atr_pct}

    # 支撑/阻力
    supports, resistances = [], []
    if n >= 120:
        rl, rh = lows[-120:], highs[-120:]
        for j in range(10, 120, 5):
            seg_l = rl[max(0, j - 5):j + 6]
            if rl[j] == min(seg_l):
                supports.append(float(rl[j]))
            seg_h = rh[max(0, j - 5):j + 6]
            if rh[j] == max(seg_h):
                resistances.append(float(rh[j]))
    supports = sorted(set(supports))[-5:] if supports else []
    resistances = sorted(set(resistances))[:5] if resistances else []
    levels = {"supports": supports, "resistances": resistances, "current": price}
    stop_price = supports[-1] if supports else price * 0.93
    exit_prices = resistances[:3] if resistances else [
        price * 1.05, price * 1.10, price * 1.15]

    # Wyckoff 信号
    wyckoff_signals, wyckoff_phase = [], ""
    if _WYCOFF_OK and n >= 30:
        try:
            wyckoff_signals, extra = WyckoffAnalyzer.analyze_all(
                closes, highs, lows, opens, volumes)
            if n >= 50:
                wyckoff_phase, _, _ = WyckoffAnalyzer.detect_phase(
                    closes, highs, lows, volumes, trend_dir,
                    wyckoff_signals, extra)
        except Exception:
            pass

    # DataFrame
    df_dict = {"open": opens, "high": highs, "low": lows,
               "close": closes, "volume": volumes}
    for period, label in [(20, "ma20"), (50, "ma50"), (200, "ma200")]:
        if n >= period:
            ma = np.convolve(closes, np.ones(period) / period, mode="valid")
            df_dict[label] = np.pad(ma, (period - 1, 0),
                                    constant_values=np.nan)
        else:
            df_dict[label] = np.full(n, np.nan)
    df_stock = pd.DataFrame(df_dict)

    # 大盘（index_daily 列名为 vol → volume）
    mdf = None
    if market_df is not None:
        mdf = market_df[["close", "vol"]].copy()
        mdf.rename(columns={"vol": "volume"}, inplace=True)

    return StockScorer(
        df=df_stock, price=price, trend=trend, vol=vol,
        levels=levels, stop_price=stop_price, exit_prices=exit_prices,
        wyckoff_signals=wyckoff_signals, wyckoff_phase=wyckoff_phase,
        symbol=stock_info["code"], market_df=mdf,
    )


def collect_factor_scores(scorer) -> dict:
    """7 因子原始分."""
    scores = {}
    call_map = [
        ("wyckoff", scorer.score_wyckoff),
        ("trend_momentum", scorer.score_momentum),
        ("risk_reward", scorer.score_risk_reward),
        ("volume", scorer.score_volume),
        ("candlestick", scorer.score_candlestick),
        ("relative_strength", scorer.score_relative_strength),
        ("volatility", scorer.score_volatility),
    ]
    for key, method in call_map:
        try:
            scores[key] = method().get("score", 50)
        except Exception:
            scores[key] = 50
    return scores


# ─────────────────────── 相关矩阵 ───────────────────────


def compute_correlation_matrix(scores_list: list) -> dict:
    """7×7 Pearson + 摘要."""
    n = len(scores_list)
    if n < 10:
        return {"error": f"样本不足: {n}"}

    arr = np.array([[s[k] for k in _FACTOR_KEYS] for s in scores_list])
    corr = np.corrcoef(arr.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    matrix = {}
    for i, ki in enumerate(_FACTOR_KEYS):
        matrix[ki] = {kj: round(corr[i][j], 4)
                      for j, kj in enumerate(_FACTOR_KEYS)}

    high_pairs = []
    for i in range(len(_FACTOR_KEYS)):
        for j in range(i + 1, len(_FACTOR_KEYS)):
            r = corr[i][j]
            if abs(r) >= 0.6:
                high_pairs.append({
                    "factor_1": _FACTOR_KEYS[i],
                    "factor_2": _FACTOR_KEYS[j],
                    "pearson_r": round(r, 4),
                })

    triu = [abs(corr[i][j])
            for i in range(len(_FACTOR_KEYS))
            for j in range(i + 1, len(_FACTOR_KEYS))]
    mean_abs_r = round(float(np.mean(triu)), 4) if triu else 0

    return {
        "n_stocks": n,
        "update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "factor_keys": _FACTOR_KEYS,
        "matrix": matrix,
        "summary": {
            "mean_abs_correlation": mean_abs_r,
            "high_correlation_pairs": high_pairs,
        },
        "raw_scores": [
            {k: round(s[k], 1) for k in _FACTOR_KEYS}
            for s in scores_list
        ],
    }


# ─────────────────────── 主流程 ───────────────────────


def run(n_stocks: int = 50):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    print(f"\n  因子相关性矩阵 — {n_stocks} 只截面")
    print("  " + "=" * 40)

    # 1. 股票列表
    print("  [1/4] 取流通市值前 {} 只...".format(n_stocks))
    stocks = fetch_top_stocks(n_stocks)
    if not stocks:
        print("  ERROR: 无股票数据")
        return
    print(f"        取到 {len(stocks)} 只")

    # 2. K 线
    print("  [2/4] 下载日线 + 大盘...")
    kline_map = fetch_kline_batch(stocks)
    print(f"        个股: {len(kline_map)} 只")
    if len(kline_map) < 10:
        print("  ERROR: K 线不足")
        return

    # 大盘
    market_df = fetch_market_kline()
    if market_df is not None:
        print(f"        大盘: {len(market_df)} 条")
    else:
        print("        大盘: 无数据 (relative_strength 将恒为 50)")

    # 3. 因子分
    print("  [3/4] 计算 7 因子评分...")
    all_scores = []
    for s in stocks:
        sd = kline_map.get(s["code"])
        if sd is None:
            continue
        try:
            price = float(sd["close"].iloc[-1])
            scorer = build_scorer(sd, price, s, market_df=market_df)
            all_scores.append(collect_factor_scores(scorer))
        except Exception as e:
            print(f"    {s['code']} 评分失败: {e}")
            continue

    print(f"        有效样本: {len(all_scores)}")
    if len(all_scores) < 10:
        print("  ERROR: 样本不足")
        return

    # 4. 相关矩阵
    print("  [4/4] 计算 Pearson 相关矩阵...")
    result = compute_correlation_matrix(all_scores)

    with open(_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  输出: {_OUTPUT_FILE}")

    _print_report(result)
    return result


def _print_report(result: dict):
    print(f"\n  {'=' * 50}")
    print(f"  7 因子相关性矩阵  (n={result['n_stocks']})")
    print(f"  平均绝对相关: {result['summary']['mean_abs_correlation']}")
    print(f"  {'=' * 50}")

    keys = _FACTOR_KEYS
    matrix = result["matrix"]

    # 表头
    header = "  " + "".join(f"{k[:5]:>8}" for k in keys)
    print(header)
    for ki in keys:
        row = f"  {ki[:12]:<12}"
        for kj in keys:
            v = matrix[ki][kj]
            row += f"{v:>8.2f}"
        print(row)

    pairs = result["summary"].get("high_correlation_pairs", [])
    if pairs:
        print(f"\n  高相关对 (|r| >= 0.6):")
        for p in pairs:
            print(f"    {p['factor_1']:>20}  vs  {p['factor_2']:<20}  r={p['pearson_r']:+.4f}")
    else:
        print(f"\n  无高相关对 — 冗余度低")

    # 低方差检测
    print(f"\n  解读:")
    print(f"    |r| < 0.3 → 低共线")
    print(f"    |r| > 0.6 → 高共线, 考虑合并/降权")
    print(f"    平均 |r| < 0.3 → 多因子结构健康")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stocks", type=int, default=50, help="截面样本数")
    args = parser.parse_args()
    run(n_stocks=args.n_stocks)
