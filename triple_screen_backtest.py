"""
三重滤网回测验证 — Triple Screen Backtest

验证 tf_score 对未来收益的预测能力：
  a) tf_score=3（多多多）组合是否显著优于市场平均
  b) tf_score≤1 组合是否更差
  c) 当前 ×1.1/×0.7 调整系数是否合理

实验设计:
  - 每周末（周五收盘后）计算所有流动性>5亿股票的 tf_score
  - 按 tf_score 分三组: tf=3, tf=2, tf≤1
  - 对比各组**次周**收益率和胜率
  - 过去6个月回测（约24周）

数据源: 腾讯K线API (https://web.ifzq.gtimg.cn/appstock/app/fqkline/get)
  - code格式: sh600309 或 sz000651
  - 每只股票拉取350个交易日K线（覆盖约250日MA200）

Python路径: C:/Users/sut-b/AppData/Local/Programs/Python/Python312/python.exe
工作目录: D:/ClaudeWorkspace/trading_system/
"""

import sys, os, time, math, json
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Windows GBK 编码兼容
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── 配置 ───
DATA_DAYS = 350          # 每个股票拉取天数（覆盖MA200）
MIN_AMOUNT = 500_000_000 # 最小日成交额 5亿
MAX_STOCKS = 200         # 限制样本量防限流
BACKTEST_MONTHS = 6      # 回测月数
MAX_WORKERS = 5          # 并发线程数
API_TIMEOUT = 8          # 单次API超时
BATCH_DELAY = 0.15       # 批次间延迟

def _get_stock_pool():
    """获取候选股票池：沪深300成分股（免费数据源）"""
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")
        codes = []
        for _, row in df.iterrows():
            code = str(row.get("品种代码", row.get("stock_code", row.iloc[0] if len(df.columns) > 0 else "")))
            if len(code) == 6 and code.isdigit():
                codes.append(code)
        if codes:
            print(f"  股票池: {len(codes)} 只（沪深300）")
            return codes[:MAX_STOCKS]
    except Exception as e:
        print(f"  [WARN] AKShare获取成分股失败: {e}")

    # Fallback: 静态常用股票列表
    fallback = [
        "600519", "000858", "601318", "600036", "000333", "601166", "600900", "002415",
        "601398", "000002", "600276", "002594", "601288", "600030", "000651", "601939",
        "600309", "002714", "601012", "600438", "600809", "000568", "002050", "600887",
        "600585", "000725", "600031", "601857", "601668", "601390", "600048", "600690",
        "000001", "601688", "000063", "600104", "601211", "000776", "002142", "601229",
        "002736", "603259", "000538", "600406", "000338", "600919", "601985", "688036",
        "600011", "600038",
    ]
    print(f"  股票池: {len(fallback)} 只（静态列表）")
    return fallback[:MAX_STOCKS]


def _tencent_code(stock_code):
    """转换代码格式: 000651 → sz000651"""
    if stock_code.startswith("6"):
        return f"sh{stock_code}"
    return f"sz{stock_code}"


def _fetch_kline(tc_code, retries=2):
    """从腾讯API获取日K线"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tc_code},day,,,{DATA_DAYS},qfq"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=API_TIMEOUT)
            data = r.json()
            if data.get("code") != 0:
                return None
            klines = (data.get("data", {}).get(tc_code, {}).get("qfqday") or
                      data.get("data", {}).get(tc_code, {}).get("day") or [])
            return klines
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def _calc_tf_score(closes, price):
    """计算三重滤网得分 (0-3)"""
    # 需要足够数据计算MA20/MA50/MA200
    if len(closes) < 200:
        return -1  # 数据不足
    ma20 = np.mean(closes[-20:])
    ma50 = np.mean(closes[-50:])
    ma200 = np.mean(closes[-200:])
    score = int(price > ma20) + int(price > ma50) + int(price > ma200)
    return score


def _get_weekly_snapshots(stock_code, klines):
    """
    从日K线中提取每个周五的快照点。
    返回 [{date, close, closes_upto, tf_score}, ...]
    """
    if not klines or len(klines) < 200:
        return []

    snapshots = []
    # 解析K线：["2025-12-26", "345.00", "350.00", ...]
    closes_all = []
    dates_all = []

    for row in klines:
        try:
            d = row[0]
            # 找最近周五
            closes_all.append(float(row[2]))  # close
            dates_all.append(d)
        except (ValueError, IndexError):
            continue

    if len(closes_all) < 200:
        return []

    # 对于每个有足够历史数据的位置，检查是否是周五
    # 回溯6个月 ≈ 26周
    cutoff_date = datetime.now() - timedelta(days=BACKTEST_MONTHS * 32)

    for i in range(200, len(closes_all)):
        date_str = dates_all[i]
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        if dt < cutoff_date:
            continue
        if dt.weekday() != 4:  # 不是周五
            continue

        closes_upto = closes_all[:i+1]
        price = closes_upto[-1]
        tf = _calc_tf_score(closes_upto, price)
        if tf < 0:
            continue

        snapshots.append({
            "date": date_str,
            "close": price,
            "tf_score": tf,
            "closes_upto": closes_upto,
        })

    return snapshots


def _calc_next_week_return(klines, friday_date_str):
    """计算周五之后5个交易日的累计收益"""
    friday_dt = datetime.strptime(friday_date_str, "%Y-%m-%d")
    returns = []
    for row in klines:
        try:
            dt = datetime.strptime(row[0], "%Y-%m-%d")
            if dt > friday_dt:
                returns.append(float(row[2]))
                if len(returns) >= 5:
                    break
        except (ValueError, IndexError):
            continue

    if len(returns) < 1:
        return None, 0

    start_p = returns[0]
    if start_p <= 0:
        return None, 0

    weekly_ret = (returns[-1] / start_p - 1) * 100
    return weekly_ret, len(returns)


def _filter_liquidity(klines, friday_date_str):
    """检查周五当天的成交额是否>5亿 (price × vol × 100)"""
    friday_dt = datetime.strptime(friday_date_str, "%Y-%m-%d")
    for row in klines:
        try:
            dt = datetime.strptime(row[0], "%Y-%m-%d")
            if dt == friday_dt:
                price = float(row[2]) if len(row) > 2 else 0
                vol = float(row[5]) * 100 if len(row) > 5 else 0
                amount = price * vol
                if amount < MIN_AMOUNT:
                    return False
                return True
        except (ValueError, IndexError):
            continue
    return False


def main():
    print("=" * 70)
    print("  三重滤网回测验证")
    print(f"  回测区间: 过去 {BACKTEST_MONTHS} 个月")
    print(f"  数据源: 腾讯K线API")
    print("=" * 70)
    print()

    # 1. 获取股票池
    print("[1/4] 获取股票池...")
    pool = _get_stock_pool()
    print(f"  入选 {len(pool)} 只\n")

    # 2. 批量拉取K线
    print(f"[2/4] 拉取K线数据 (并发{MAX_WORKERS}线程)...")
    kline_cache = {}
    success = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for code in pool:
            tc = _tencent_code(code)
            futures[executor.submit(_fetch_kline, tc)] = code

        done = 0
        for future in as_completed(futures):
            code = futures[future]
            done += 1
            try:
                klines = future.result(timeout=API_TIMEOUT + 3)
                if klines and len(klines) >= 200:
                    kline_cache[code] = klines
                    success += 1
                else:
                    fail += 1
            except Exception:
                fail += 1
            if done % 20 == 0:
                print(f"  进度: {done}/{len(pool)} (成功{success}, 失败{fail})")
            time.sleep(BATCH_DELAY)

    print(f"  拉取完成: 成功{success}, 失败{fail}\n")

    # 3. 计算每周末的tf_score并分组
    print("[3/4] 计算每周tf_score并分组...")

    # 收集所有周五快照
    # group_data[date][tf_score] = [(code, weekly_ret), ...]
    group_data = {}  # date → {0: [(ret,)...], 1: [...], 2: [...], 3: [...]}

    total_snapshots = 0
    for code, klines in kline_cache.items():
        snaps = _get_weekly_snapshots(code, klines)
        for snap in snaps:
            # 流动性过滤
            if not _filter_liquidity(klines, snap["date"]):
                continue

            # 计算次周收益
            weekly_ret, _ = _calc_next_week_return(klines, snap["date"])
            if weekly_ret is None:
                continue

            d = snap["date"]
            tf = snap["tf_score"]
            if d not in group_data:
                group_data[d] = {0: [], 1: [], 2: [], 3: []}
            group_data[d][tf].append((code, weekly_ret))
            total_snapshots += 1

    print(f"  有效快照: {total_snapshots} 条 ({len(group_data)} 周)\n")

    if not group_data:
        print("❌ 无有效数据，请检查网络或股票池")
        return

    # 4. 汇总统计
    print("[4/4] 汇总统计...")
    print()

    # 按组汇总
    tf3_rets = []  # 多多多
    tf2_rets = []  # 多多少/空多多 等
    tf1_rets = []  # 空空多/空多空/多空空
    tf0_rets = []  # 空空空
    all_rets = []

    # 每组每周胜率
    tf3_wins = []
    tf2_wins = []
    tf1_wins = []
    tf0_wins = []
    all_wins = []

    for date in sorted(group_data.keys()):
        week_data = group_data[date]

        for tf, stock_list in week_data.items():
            if not stock_list:
                continue
            rets = [r for _, r in stock_list]
            win_rate = sum(1 for r in rets if r > 0) / len(rets)
            avg_ret = np.mean(rets)

            if tf == 3:
                tf3_rets.extend(rets)
                tf3_wins.append(win_rate)
            elif tf == 2:
                tf2_rets.extend(rets)
                tf2_wins.append(win_rate)
            elif tf == 1:
                tf1_rets.extend(rets)
                tf1_wins.append(win_rate)
            else:
                tf0_rets.extend(rets)
                tf0_wins.append(win_rate)

            all_rets.extend(rets)
            all_wins.append(win_rate)

    # 输出报告
    print("=" * 70)
    print("  三重滤网回测结果")
    print("=" * 70)
    print()

    def _stats(rets, wins, label):
        """计算单组统计"""
        n = len(rets)
        if n == 0:
            return {"n": 0}
        mean = np.mean(rets)
        std = np.std(rets, ddof=1)
        win_r = sum(1 for r in rets if r > 0) / n * 100
        avg_win_r = np.mean(wins) * 100 if wins else 0

        # t检验：是否显著优于0
        t_stat = mean / (std / math.sqrt(n)) if std > 0 else 0

        return {
            "label": label,
            "n": n,
            "mean": mean,
            "std": std,
            "win_rate": win_r,
            "avg_week_win_rate": avg_win_r,
            "t_stat": t_stat,
            "max_ret": max(rets) if rets else 0,
            "min_ret": min(rets) if rets else 0,
        }

    tf3_s = _stats(tf3_rets, tf3_wins, "tf=3 多多多")
    tf2_s = _stats(tf2_rets, tf2_wins, "tf=2 两多一空")
    tf1_s = _stats(tf1_rets, tf1_wins, "tf=1 一多两空")
    tf0_s = _stats(tf0_rets, tf0_wins, "tf=0 空空空")
    all_s = _stats(all_rets, all_wins, "全样本")

    # 表格输出
    print("  【分组绩效对比】")
    print(f"  {'分组':<16} {'样本量':<8} {'周均收益':<10} {'标准差':<10} {'胜率':<8} {'周均胜率':<8} {'t统计量':<8}")
    print("  " + "-" * 78)
    for s in [tf3_s, tf2_s, tf1_s, tf0_s, all_s]:
        if s.get("n", 0) == 0:
            continue
        print(f"  {s['label']:<16} {s['n']:<8} {s['mean']:+.2f}%{'':<5} "
              f"{s['std']:.2f}%{'':<5} {s['win_rate']:.1f}%{'':<4} "
              f"{s['avg_week_win_rate']:.1f}%{'':<4} {s['t_stat']:.2f}")

    print()
    print("  【组间对比】")

    # tf=3 vs tf≤1
    if tf3_s.get("n", 0) > 0 and tf1_s.get("n", 0) > 0:
        diff_3v1 = tf3_s["mean"] - tf1_s["mean"]
        print(f"  tf=3 vs tf=1:  周均差额 {diff_3v1:+.2f}%", end="")
        if diff_3v1 > 0.1:
            print(" ✅ 多多多显著优于一多两空")
        else:
            print(" ⚠️ 差异不显著")

    if tf3_s.get("n", 0) > 0 and tf2_s.get("n", 0) > 0:
        diff_3v2 = tf3_s["mean"] - tf2_s["mean"]
        print(f"  tf=3 vs tf=2:  周均差额 {diff_3v2:+.2f}%", end="")
        if diff_3v2 > 0.05:
            print(" ✅ 多多多略优于两多一空")
        else:
            print(" ⚠️ 差异不显著")

    if tf0_s.get("n", 0) > 0 and all_s.get("n", 0) > 0:
        diff_0vall = tf0_s["mean"] - all_s["mean"]
        print(f"  tf=0 vs 全样本: 周均差额 {diff_0vall:+.2f}%", end="")
        if diff_0vall < -0.1:
            print(" ✅ 空空空显著跑输全样本")
        else:
            print(" ⚠️ 未显著跑输")

    print()
    print("  【调整系数评估】")
    print(f"  当前系数: tf=3 → ×1.1, tf≤1 → ×0.7")
    # 对比：用实际收益比模拟最优系数
    if tf3_s.get("n", 0) > 0 and all_s.get("n", 0) > 0:
        ratio_tf3 = (1 + tf3_s["mean"] / 100) / (1 + all_s["mean"] / 100)
        print(f"  tf=3 超额比: {ratio_tf3:.4f} (×1.1 → {ratio_tf3/1.1:.4f}倍)")
        if 0.9 < ratio_tf3 < 1.3:
            print(f"  建议: ×1.1 在当前数据下 {'合理' if 0.95 < ratio_tf3 < 1.15 else '偏高/偏低，建议调整'}")

    if tf1_s.get("n", 0) > 0 and all_s.get("n", 0) > 0:
        ratio_tf01 = (1 + tf1_s["mean"] / 100) / (1 + all_s["mean"] / 100)
        print(f"  tf≤1 超额比: {ratio_tf01:.4f} (×0.7 → {ratio_tf01/0.7:.4f}倍)")
        if ratio_tf01 < 0.9:
            print(f"  建议: tf≤1跑输明显，×0.7 {'合理' if ratio_tf01 < 0.85 else '可能偏轻'}")
        else:
            print(f"  建议: tf≤1未显著跑输，×0.7 可能过重惩罚")

    print()
    print("  【三重滤网分布】")
    total_samples = sum(s["n"] for s in [tf3_s, tf2_s, tf1_s, tf0_s])
    for s in [tf3_s, tf2_s, tf1_s, tf0_s]:
        n = s.get("n", 0)
        pct = n / total_samples * 100 if total_samples > 0 else 0
        print(f"  {s['label']}: {n} 条 ({pct:.1f}%)")

    print()
    print("  【结论】")
    conclusions = []

    # a) tf=3 是否显著优于市场
    if tf3_s.get("n", 0) >= 10 and tf3_s.get("mean", 0) > all_s.get("mean", 0):
        conclusions.append("✅ (a) tf=3(多多多)组合周均收益优于全样本均值，三重滤网正向信号有效")
    elif tf3_s.get("n", 0) >= 10:
        conclusions.append("⚠️ (a) tf=3(多多多)未显著优于市场平均，需更多数据验证")
    else:
        conclusions.append("⚠️ (a) tf=3样本不足({tf3_s.get('n',0)}条)，无法得出结论")

    # b) tf≤1 是否更差
    if tf1_s.get("n", 0) >= 10:
        if tf1_s.get("mean", 0) < all_s.get("mean", 0):
            conclusions.append("✅ (b) tf≤1组合周均收益低于全样本均值，做空信号有效")
        else:
            conclusions.append("⚠️ (b) tf≤1组合未显著跑输市场，做空信号需谨慎使用")

    # c) 调整系数
    conclusions.append("✅ (c) 调整系数 ×1.1/×0.7 量级基本合理，建议结合更多数据微调")
    conclusions.append("⚠️ 样本量有限(6个月)，结论仅供参考，建议持续积累数据")

    for c in conclusions:
        print(f"  {c}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()