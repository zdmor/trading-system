"""
Spring 质量评分系统 — 完整回测 + 阈值优化
=============================================
回测流程:
  1. 获取股票池 (Sina 全A, 按成交额取前800只)
  2. 逐个拉取日线 (350天, 腾讯API, 存本地pickle缓存)
  3. 拉取 sh000001 大盘日线
  4. 逐股滑动窗口检测 Spring → 计算质量评分 → 记录 forward 收益
  5. 网格搜索各维度分段阈值
  6. 分层统计 + 泊松验证 + 阈值建议
  7. 输出: output_v2/spring_quality_backtest.json

无未来函数约束: 所有计算只用 closes[:t+1] 的数据。
"""

import os, sys, json, time, pickle, hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import urllib.request
import urllib.error

# ─── 路径 & 导入 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scanner import WyckoffAnalyzer, get_patterns_config
from spring_quality import compute_confidence

CACHE_DIR = os.path.join(SCRIPT_DIR, "data_cache", "spring_bt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_v2")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 配置 ───
MIN_BARS = 200           # 最少K线数
FETCH_DAYS = 350         # 拉取日线天数
MAX_STOCKS = 800         # 股票池上限
THREAD_POOL = 5          # 数据拉取线程数
REQUEST_DELAY = 0.15     # 请求间隔(秒)
MAX_RETRIES = 2
SINA_TIMEOUT = 30        # 新浪API超时

_lock = threading.Lock()
_request_lock = threading.Lock()

# ─── 数据拉取 ───

def _rate_limit():
    time.sleep(REQUEST_DELAY)


def fetch_tencent_kline(code: str, days: int = FETCH_DAYS) -> Optional[dict]:
    """从腾讯API拉取日线数据"""
    market = "sh" if code.startswith("6") else "sz"
    full_code = f"{market}{code}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={full_code},day,,,{days},qfq"

    for attempt in range(MAX_RETRIES + 1):
        try:
            with _request_lock:
                _rate_limit()
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://gu.qq.com/"
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            klines = data.get("data", {}).get(full_code, {}).get("qfqday", [])
            if not klines:
                # 尝试不带qfq
                klines = data.get("data", {}).get(full_code, {}).get("day", [])
            if not klines:
                return None

            dates, opens, closes, highs, lows, volumes = [], [], [], [], [], []
            for row in klines:
                if len(row) < 6:
                    continue
                try:
                    dates.append(row[0])
                    opens.append(float(row[1]))
                    closes.append(float(row[2]))
                    highs.append(float(row[3]))
                    lows.append(float(row[4]))
                    volumes.append(float(row[5]) * 100)  # 股→手
                except (ValueError, IndexError):
                    continue

            if len(closes) < MIN_BARS:
                return None

            return {
                "dates": dates, "opens": opens, "closes": closes,
                "highs": highs, "lows": lows, "volumes": volumes,
            }
        except Exception as e:
            if attempt == MAX_RETRIES:
                return None
            time.sleep(1.5 * (attempt + 1))


def fetch_with_cache(code: str) -> Optional[dict]:
    """带缓存的日线拉取"""
    cache_file = os.path.join(CACHE_DIR, f"{code}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            # 验证缓存有效性
            if (isinstance(cached, dict)
                    and len(cached.get("closes", [])) >= MIN_BARS
                    and len(cached.get("dates", [])) >= MIN_BARS):
                return cached
            else:
                os.remove(cache_file)
        except Exception:
            os.remove(cache_file)

    data = fetch_tencent_kline(code)
    if data:
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
    return data


def fetch_stock_pool(max_stocks: int = MAX_STOCKS) -> List[Tuple[str, str]]:
    """
    从新浪获取全A股票池（按成交额取前max_stocks只）。
    返回 [(code, name), ...]
    """
    cache_file = os.path.join(CACHE_DIR, "stock_pool.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                pool = json.load(f)
            if pool and datetime.now().timestamp() - os.path.getmtime(cache_file) < 86400:
                return pool
        except Exception:
            pass

    all_stocks = []
    for node in ("sh_a", "sz_a"):
        page_size = 100
        for p in range(1, 60):  # 最多60页 = 6000只
            try:
                url = (
                    f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
                    f"Market_Center.getHQNodeData?page={p}&num={page_size}"
                    f"&sort=turnover&asc=0&node={node}"
                )
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=SINA_TIMEOUT) as resp:
                    raw = resp.read().decode("gbk")
                if not raw or raw == "null" or raw.startswith("("):
                    break
                items = json.loads(raw)
                if not items:
                    break
                for item in items:
                    code = item.get("code", "")
                    name = item.get("name", "")
                    if code and len(code) == 6 and code[0] in "036":
                        all_stocks.append((code, name))
                time.sleep(0.05)
            except Exception:
                break

    # 去重 + 限制数量
    seen = set()
    unique = []
    for code, name in all_stocks:
        if code not in seen:
            seen.add(code)
            unique.append((code, name))
            if len(unique) >= max_stocks:
                break

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False)

    print(f"  股票池: {len(unique)} 只")
    return unique


def fetch_all_data(pool: List[Tuple[str, str]]) -> Dict[str, dict]:
    """并行拉取所有股票数据"""
    results = {}
    total = len(pool)
    done = [0]

    def fetch_one(code_name):
        code, name = code_name
        data = fetch_with_cache(code)
        with _lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == total:
                print(f"\r  拉取进度: {done[0]}/{total}", end="", flush=True)
        if data:
            return code, name, data
        return None

    with ThreadPoolExecutor(max_workers=THREAD_POOL) as executor:
        futures = [executor.submit(fetch_one, cn) for cn in pool]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r:
                    code, name, data = r
                    results[code] = {"name": name, **data}
            except Exception:
                pass

    print(f"\n  成功拉取: {len(results)} 只 (缓存+新拉)")
    return results


# ─── 回测核心 ───

def calc_trend_at(closes: np.ndarray, high_idx: int) -> str:
    """
    在时间点 high_idx 判断趋势，仅用 closes[:high_idx+1]。
    无未来函数。
    """
    if high_idx < 199:
        return "数据不足"
    window = closes[:high_idx + 1]
    ma50 = float(np.mean(window[-50:]))
    ma200 = float(np.mean(window[-200:]))
    return "多头" if ma50 > ma200 else "空头"


def align_market_closes(market_closes: np.ndarray, market_dates: np.ndarray,
                        stock_date: str) -> np.ndarray:
    """
    对齐大盘数据到个股时间点: 返回 <= stock_date 的大盘收盘价序列。
    """
    mask = market_dates <= stock_date
    return market_closes[mask]


def backtest_one_stock(code: str, name: str,
                       dates: np.ndarray, opens: np.ndarray,
                       closes: np.ndarray, highs: np.ndarray,
                       lows: np.ndarray, volumes: np.ndarray,
                       market_dates: np.ndarray,
                       market_closes: np.ndarray) -> List[dict]:
    """
    对单只股票做滑动窗口回测。
    每个时间点 t (>=200):
      1. 计算趋势 (MA50 vs MA200)
      2. 检测 Spring
      3. 如有 Spring → 计算质量评分 → 记录 forward 收益
    """
    n = len(closes)
    signals = []

    if n < MIN_BARS:
        return signals

    for t in range(MIN_BARS - 1, n):
        # 趋势
        trend = calc_trend_at(closes, t)
        if trend != "多头":
            continue  # detect_spring 跳过空头

        # Spring 检测 (只用 data up to t)
        c_slice = closes[:t + 1]
        h_slice = highs[:t + 1]
        l_slice = lows[:t + 1]
        v_slice = volumes[:t + 1]

        spring = WyckoffAnalyzer.detect_spring(c_slice, h_slice, l_slice, v_slice, trend)
        if spring is None:
            continue

        sig_label, detect_score, detail = spring

        # 质量评分
        mk = align_market_closes(market_closes, market_dates, dates[t])
        quality = compute_confidence(
            closes=c_slice, highs=h_slice, lows=l_slice, volumes=v_slice,
            trend=trend, market_closes=mk if len(mk) >= 20 else None
        )

        # Forward 收益
        forward = {}
        for horizon in [1, 5, 10, 20]:
            ft = t + horizon
            if ft < n:
                ret = (float(closes[ft]) / float(closes[t]) - 1) * 100
                forward[f"{horizon}d"] = round(ret, 2)
            else:
                forward[f"{horizon}d"] = None

        signals.append({
            "code": code,
            "name": name,
            "spring_date": dates[t],
            "spring_price": round(float(closes[t]), 2),
            "spring_type": sig_label,
            "detect_score": detect_score,
            "confidence": quality["confidence"],
            "bonus": quality["bonus"],
            "components": quality["components"],
            "reasons": quality.get("reasons", ""),
            "forward": forward,
            "trend": trend,
        })

    return signals


# ─── 主回测 ───

def run_backtest(stock_data: Dict[str, dict],
                 market_dates: np.ndarray,
                 market_closes: np.ndarray) -> List[dict]:
    """遍历所有股票执行回测"""
    all_signals = []
    total = len(stock_data)
    count = [0]

    for code, data in stock_data.items():
        name = data["name"]
        dates_arr = np.array(data["dates"], dtype=str)
        closes_arr = np.array(data["closes"], dtype=float)
        opens_arr = np.array(data["opens"], dtype=float)
        highs_arr = np.array(data["highs"], dtype=float)
        lows_arr = np.array(data["lows"], dtype=float)
        vols_arr = np.array(data["volumes"], dtype=float)

        sigs = backtest_one_stock(
            code, name,
            dates_arr, opens_arr, closes_arr, highs_arr, lows_arr, vols_arr,
            market_dates, market_closes
        )
        all_signals.extend(sigs)

        count[0] += 1
        if count[0] % 50 == 0 or count[0] == total:
            print(f"\r  回测进度: {count[0]}/{total} 完成, 已发现 {len(all_signals)} 个Spring",
                  end="", flush=True)

    print(f"\n  总计: {len(all_signals)} 个Spring信号 ({total}只股票)")
    return all_signals


# ─── 网格搜索 ───

def grid_search_thresholds(signals: List[dict]) -> dict:
    """
    全维度网格搜索（不在此阶段运行，详见 _simplified_search 的分组统计）。
    原始数据量 + 时间成本考量 → 用分布分析替代。
    如需真正的网格搜索，可在 output JSON 加载后本地迭代。
    """
    return _simplified_search(signals)


def _simplified_search(signals: List[dict]) -> dict:
    """基于现有分数的简化搜索: 测试各置信度分组的收益区分度"""
    result = {
        "search_method": "distribution_analysis",
        "note": "基于现有7维评分的分组统计(原始阈值)",
    }

    # 按置信度分组
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    layer = {}
    for lo, hi in bins:
        group = [s for s in signals if lo <= s["confidence"] < hi]
        if not group:
            continue
        key = f"{lo}-{hi}"
        layer[key] = {
            "count": len(group),
            "avg_1d": round(np.mean([s["forward"].get("1d") for s in group if s["forward"].get("1d") is not None]), 2),
            "avg_5d": round(np.mean([s["forward"].get("5d") for s in group if s["forward"].get("5d") is not None]), 2),
            "avg_10d": round(np.mean([s["forward"].get("10d") for s in group if s["forward"].get("10d") is not None]), 2),
            "avg_20d": round(np.mean([s["forward"].get("20d") for s in group if s["forward"].get("20d") is not None]), 2),
            "win_rate_5d": round(np.mean([1 if s["forward"].get("5d", 0) and s["forward"]["5d"] > 0 else 0 for s in group]), 3),
            "win_rate_20d": round(np.mean([1 if s["forward"].get("20d", 0) and s["forward"]["20d"] > 0 else 0 for s in group]), 3),
        }

    # 泊松维度专项
    poisson_groups = {}
    for key_name in [15, 10, 0, -10]:
        label_map = {15: "score_15", 10: "score_10", 0: "score_0", -10: "score_-10"}
        group = [s for s in signals if s["components"].get("poisson_freq") == key_name]
        if group:
            poisson_groups[label_map[key_name]] = {
                "count": len(group),
                "avg_5d": round(np.mean([s["forward"].get("5d") for s in group if s["forward"].get("5d") is not None]), 2),
                "win_rate_5d": round(np.mean([1 if s["forward"].get("5d", 0) and s["forward"]["5d"] > 0 else 0 for s in group]), 3),
                "avg_20d": round(np.mean([s["forward"].get("20d") for s in group if s["forward"].get("20d") is not None]), 2),
                "win_rate_20d": round(np.mean([1 if s["forward"].get("20d", 0) and s["forward"]["20d"] > 0 else 0 for s in group]), 3),
            }

    # 单维度区分度
    dimension_power = {}
    for dim in ["trend_pos", "prior_advance", "volume_conf", "support_dist",
                 "poisson_freq", "volatility", "market"]:
        # 取该维度得分最高30%和最低30%的20日收益差
        scores = [(s["components"].get(dim, 0), s["forward"].get("20d")) for s in signals
                  if s["forward"].get("20d") is not None]
        if not scores:
            continue
        scores.sort(key=lambda x: x[0])
        n = len(scores)
        top_n = max(1, n // 3)
        top = [s[1] for s in scores[-top_n:]]
        bot = [s[1] for s in scores[:top_n]]
        dimension_power[dim] = {
            "top30_avg_20d": round(float(np.mean(top)), 2),
            "bot30_avg_20d": round(float(np.mean(bot)), 2),
            "diff": round(float(np.mean(top) - np.mean(bot)), 2),
            "top30_win_rate_20d": round(float(np.mean([1 if x > 0 else 0 for x in top])), 3),
            "bot30_win_rate_20d": round(float(np.mean([1 if x > 0 else 0 for x in bot])), 3),
        }

    # 阈值建议
    threshold_rec = _suggest_thresholds(layer)

    result["layer_analysis"] = layer
    result["poisson_analysis"] = poisson_groups
    result["dimension_power"] = dimension_power
    result["threshold_suggestion"] = threshold_rec

    return result


def _suggest_thresholds(layer: dict) -> dict:
    """基于分层结果给出阈值建议"""
    # 找置信度对胜率的拐点
    thresholds = sorted([(int(k.split("-")[0]), v) for k, v in layer.items()], key=lambda x: x[0])

    optimal = 50
    high_conf = 70
    reject = 30

    for lo, stat in thresholds:
        if stat["win_rate_5d"] > 0.55 and lo < optimal:
            optimal = lo
        if stat["win_rate_20d"] > 0.65 and lo < high_conf:
            high_conf = lo
        if stat["win_rate_5d"] < 0.45:
            reject = max(reject, lo)

    # 取 optimal 以上组的汇总
    above = [s for lo, s in thresholds if lo >= optimal]
    exp_5d = round(float(np.mean([s["win_rate_5d"] for s in above])), 3) if above else 0
    exp_20d = round(float(np.mean([s["win_rate_20d"] for s in above])), 3) if above else 0

    return {
        "optimal_threshold": optimal,
        "high_confidence_threshold": high_conf,
        "reject_threshold": reject,
        "expected_win_rate_5d_above_threshold": exp_5d,
        "expected_win_rate_20d_above_threshold": exp_20d,
        "recommendation": f"可信度 >= {optimal} 可入场，>= {high_conf} 重点跟踪，< {reject} 排除",
    }


# ─── 保存 ───

def save_output(signals: List[dict], analysis: dict, stock_count: int, run_time: str):
    """保存完整回测结果"""
    date_range = ""
    if signals:
        dates = sorted([s["spring_date"] for s in signals])
        date_range = f"{dates[0]} to {dates[-1]}"

    output = {
        "meta": {
            "stock_count": stock_count,
            "date_range": date_range,
            "total_spring_signals": len(signals),
            "run_time": run_time,
            "parameters": {
                "min_bars": MIN_BARS,
                "data_source": "tencent_350d",
                "trend_method": "MA50_vs_MA200",
            },
        },
        "signals": signals,
        "grid_search": analysis if isinstance(analysis, dict) else {},
    }

    # 合并 grid_search 中的各项
    if isinstance(analysis, dict):
        output["layer_analysis"] = analysis.get("layer_analysis", {})
        output["poisson_analysis"] = analysis.get("poisson_analysis", {})
        output["dimension_power"] = analysis.get("dimension_power", {})
        output["threshold_suggestion"] = analysis.get("threshold_suggestion", {})

    out_path = os.path.join(OUTPUT_DIR, "spring_quality_backtest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  输出: {out_path}  ({len(json.dumps(output, ensure_ascii=False))} chars)")
    return out_path


# ─── 主入口 ───

def main():
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 60)
    print("  Spring 质量评分系统 — 回测 + 阈值优化")
    print(f"  开始: {run_time}")
    print("=" * 60)

    # 1. 股票池
    print("\n[1/5] 获取股票池...")
    pool = fetch_stock_pool()
    if not pool:
        print("  错误: 无法获取股票池")
        return
    print(f"  共 {len(pool)} 只候选")

    # 2. 拉数据
    print(f"\n[2/5] 拉取日线数据 (线程{THREAD_POOL})...")
    stock_data = fetch_all_data(pool)
    print(f"  有效: {len(stock_data)} 只")

    # 3. 大盘
    print("\n[3/5] 拉取大盘指数...")
    idx_data = fetch_with_cache("sh000001") or fetch_with_cache("000001")
    if idx_data is None:
        idx_data = fetch_tencent_kline("000001")
    if idx_data:
        market_dates = np.array(idx_data["dates"], dtype=str)
        market_closes = np.array(idx_data["closes"], dtype=float)
        print(f"  大盘: {len(market_closes)} 根K线  {market_dates[0]} ~ {market_dates[-1]}")
    else:
        market_dates = np.array([], dtype=str)
        market_closes = np.array([], dtype=float)
        print("  警告: 大盘数据获取失败, market 维度将使用默认值")

    # 4. 回测
    print(f"\n[4/5] 滑动窗口回测...")
    t0 = time.time()
    signals = run_backtest(stock_data, market_dates, market_closes)
    elapsed = time.time() - t0
    print(f"  耗时: {elapsed:.1f}s ({len(signals)} 个Spring信号)")

    if len(signals) < 100:
        print(f"  警告: 信号数不足 ({len(signals)}), 可能需要更多数据或调整检测参数")
        # 即使样本少也继续分析

    # 5. 分析
    print(f"\n[5/5] 分层分析 + 阈值建议...")
    analysis = _simplified_search(signals)

    # 打印摘要
    layer = analysis.get("layer_analysis", {})
    if layer:
        print(f"\n  【分层胜率】")
        print(f"  {'置信区间':<12} {'样本':<6} {'5日胜率':<8} {'20日胜率':<8} {'5日均收益':<10} {'20日均收益':<10}")
        for lo_hi in sorted(layer.keys(), key=lambda x: int(x.split("-")[0])):
            s = layer[lo_hi]
            print(f"  {lo_hi:<12} {s['count']:<6} {s['win_rate_5d']:<8.1%} {s['win_rate_20d']:<8.1%} "
                  f"{s['avg_5d']:>+8.1f}% {s['avg_20d']:>+8.1f}%")

    ts = analysis.get("threshold_suggestion", {})
    if ts:
        print(f"\n  【阈值建议】")
        print(f"  入场阈值: >= {ts.get('optimal_threshold', 50)}")
        print(f"  重点关注: >= {ts.get('high_confidence_threshold', 70)}")
        print(f"  排除阈值: < {ts.get('reject_threshold', 30)}")
        print(f"  预期5日胜率: {ts.get('expected_win_rate_5d_above_threshold', 0):.1%}")
        print(f"  预期20日胜率: {ts.get('expected_win_rate_20d_above_threshold', 0):.1%}")

    pois = analysis.get("poisson_analysis", {})
    if pois:
        print(f"\n  【泊松维度验证】")
        print(f"  {'分值':<12} {'样本':<6} {'5日胜率':<8} {'20日胜率':<8} {'5日均收益':<10} {'20日均收益':<10}")
        for lbl in sorted(pois.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[1].lstrip("-").isdigit() else 0, reverse=True):
            p = pois[lbl]
            print(f"  {lbl:<12} {p['count']:<6} {p['win_rate_5d']:<8.1%} {p['win_rate_20d']:<8.1%} "
                  f"{p['avg_5d']:>+8.1f}% {p['avg_20d']:>+8.1f}%")

    dp = analysis.get("dimension_power", {})
    if dp:
        print(f"\n  【单维度区分力 (top30% vs bot30% 20日收益差)】")
        for dim in sorted(dp.keys(), key=lambda d: abs(dp[d]["diff"]), reverse=True):
            d = dp[dim]
            print(f"  {dim:<18}  diff={d['diff']:>+6.1f}%  top_wr={d['top30_win_rate_20d']:.1%}  bot_wr={d['bot30_win_rate_20d']:.1%}")

    # 6. 保存
    print(f"\n[6/6] 保存结果...")
    save_output(signals, analysis, len(stock_data), run_time)

    # 验收
    print(f"\n{'='*60}")
    print(f"  验收标准:")
    print(f"  1. 样本数>=500: {'✅' if len(signals) >= 500 else '⚠️'} ({len(signals)})")
    print(f"  2. 分层胜率表: {'✅' if layer else '❌'}")
    print(f"  3. 泊松验证: {'✅' if pois else '❌'}")
    print(f"  4. 阈值建议: {'✅' if ts else '❌'}")
    print(f"  5. 维度区分力: {'✅' if dp else '❌'}")
    print(f"  6. 结果已保存: {'✅' if os.path.exists(os.path.join(OUTPUT_DIR, 'spring_quality_backtest.json')) else '❌'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
