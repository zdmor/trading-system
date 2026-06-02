"""
三重滤网系数定期重校
========================

目标: 基于近期市场数据，重新校准 tf_score 各组别的前瞻收益系数，
     避免硬编码系数随市场风格漂移而失效。

方法（简化版）:
  1. 获取上证指数历史日线数据（含未来5日收益）
  2. 滚动计算每日的 tf_score（基于 MA20/MA50/MA200 位置）
  3. 按 tf_score 分组，取各组平均 forward_5d_return
  4. 以 tf=2 组收益为基准归一化（tf=2 → 1.00），输出新系数

用法:
  python triple_screen_recalib.py                    # 全流程运行
  python triple_screen_recalib.py --months 6         # 用更长时间窗口
  python triple_screen_recalib.py --check-only       # 只检查漂移不更新
"""

import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# ──────────────────── 硬编码基准系数 ────────────────────
# 回测周期: 2026-06-01, t=3.34 显著
CURRENT_COEFFS = {3: 1.15, 2: 1.00, 1: 0.85, 0: 0.60}

# 持久化路径（相对 trading_system 根目录）
COEFF_FILE = "data_cache/tf_coeffs.json"


# ──────────────────── 数据层 ────────────────────

def _get_tushare_pro():
    """懒加载 tushare pro 实例"""
    import tushare as ts
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    token = ""
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        token = cfg.get("_tushare", {}).get("token", "")
    except Exception:
        pass
    if not token:
        raise RuntimeError("Tushare token not found in config.json[_tushare.token]")
    ts.set_token(token)
    return ts.pro_api()


def _fetch_index_bars(pro, ts_code="000001.SH", months=3):
    """
    获取指数日线 + 前复权 close, 确保足够计算 MA200。
    months 表示往前拉多少个月作为分析窗口。
    """
    end_date = datetime.now().strftime("%Y%m%d")
    # 多拉一些（months + 12 个月确保有 MA200 数据 + 前视5日）
    start_dt = datetime.now() - timedelta(days=(months + 12) * 31)
    start_date = start_dt.strftime("%Y%m%d")

    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date,
                         fields="trade_date,close,high,low,vol")
    if df is None or df.empty:
        raise RuntimeError(f"No index data for {ts_code} in range {start_date}~{end_date}")

    # 升序排列（旧 -> 新）
    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def _compute_tf_score_and_forward(df, forward_days=5):
    """
    对 DataFrame（已升序）滚动计算:
      - MA20, MA50, MA200
      - tf_score = int(close > MA200) + int(close > MA50) + int(close > MA20)
      - forward_{d}d_return: 第 i 天收盘到 i+forward_days 天的收益率
    返回新的 DataFrame（仅含有效窗口内的行）。
    """
    closes = df["close"].values
    n = len(closes)

    ma20 = np.full(n, np.nan)
    ma50 = np.full(n, np.nan)
    ma200 = np.full(n, np.nan)

    for i in range(n):
        if i >= 19:
            ma20[i] = np.mean(closes[i - 19:i + 1])
        if i >= 49:
            ma50[i] = np.mean(closes[i - 49:i + 1])
        if i >= 199:
            ma200[i] = np.mean(closes[i - 199:i + 1])

    df = df.copy()
    df["ma20"] = ma20
    df["ma50"] = ma50
    df["ma200"] = ma200

    # tf_score
    def _tf(row):
        c = row["close"]
        score = 0
        if c > row["ma200"]:
            score += 1
        if c > row["ma50"]:
            score += 1
        if c > row["ma20"]:
            score += 1
        return score

    df["tf_score"] = df.apply(_tf, axis=1)

    # 前视收益
    fwd_col = f"forward_{forward_days}d_return"
    fwd_vals = np.full(n, np.nan)
    for i in range(n - forward_days):
        fwd_vals[i] = (closes[i + forward_days] / closes[i]) - 1.0
    df[fwd_col] = fwd_vals

    # 只保留有完整 MA200 + 前视收益的行
    valid = df.dropna(subset=["ma200", fwd_col]).reset_index(drop=True)
    return valid


def _normalize_coeffs(group_means):
    """
    group_means: {tf: avg_forward_return}
    以 tf=2 为基准归一化，返回 {tf: coeff}
    """
    base = group_means.get(2, None)
    if base is None or base == 0:
        # 基准组无数据或收益为0，返回原始系数
        return dict(CURRENT_COEFFS)

    new_coeffs = {}
    for tf in range(4):
        raw = group_means.get(tf, 0)
        new_coeffs[tf] = round(raw / base, 4)

    # tf=2 自身应精确为 1.0
    new_coeffs[2] = 1.00
    return new_coeffs


# ──────────────────── 主 API ────────────────────

def recalibrate(months=3, forward_days=5):
    """
    基于过去 months 个月的指数数据，重算三重滤网系数。

    Parameters
    ----------
    months : int
        分析窗口（月数），含前导 MA200 预热数据。
    forward_days : int
        前视收益窗口（交易日数），默认 5 日 ≈ 1 周。

    Returns
    -------
    dict
        {tf_score: coeff} 归一化后的新系数。
    """
    pro = _get_tushare_pro()
    df = _fetch_index_bars(pro, ts_code="000001.SH", months=months)
    df = _compute_tf_score_and_forward(df, forward_days=forward_days)

    if df.empty:
        print("[WARN] 有效数据为空，返回 CURRENT_COEFFS")
        return dict(CURRENT_COEFFS)

    # 按 tf_score 分组聚合
    grouped = df.groupby("tf_score")[f"forward_{forward_days}d_return"].mean()
    group_means = {int(k): round(v * 100, 4) for k, v in grouped.items()}
    for tf in range(4):
        if tf not in group_means:
            group_means[tf] = 0.0

    print(f"\n  [TF分组平均 {forward_days}日收益 (%%)]")
    for tf in sorted(group_means):
        print(f"    tf={tf}: {group_means[tf]:+.4f}%")

    # 归一化
    new_coeffs = _normalize_coeffs(group_means)
    return new_coeffs


def check_coeff_drift(new_coeffs, threshold=0.20):
    """
    检查新系数相对 CURRENT_COEFFS 的漂移是否超过 threshold。

    Parameters
    ----------
    new_coeffs : dict
        新校准的系数 {tf: coeff}
    threshold : float
        允许的最大相对漂移（默认 20%）。

    Returns
    -------
    list of str
        漂移警告信息列表，无漂移时返回空列表。
    """
    warnings = []
    for tf in range(4):
        old = CURRENT_COEFFS.get(tf, 1.0)
        new = new_coeffs.get(tf, 1.0)
        if old == 0:
            continue
        drift = abs(new - old) / old
        if drift > threshold:
            warnings.append(
                f"[DRIFT] tf={tf}: {old:.4f} → {new:.4f} "
                f"(漂移 {drift*100:.1f}% > {threshold*100:.0f}%)"
            )
    return warnings


def save_coeffs(coeffs, filepath=None):
    """持久化系数到 JSON 文件。"""
    path = filepath or os.path.join(os.path.dirname(__file__), COEFF_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "coeffs": coeffs,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": "三重滤网系数，校准自 SH000001 指数滚动分析",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  [已保存] {path}")
    return path


def load_coeffs(filepath=None):
    """从 JSON 加载持久化的系数，文件不存在时返回 CURRENT_COEFFS。"""
    path = filepath or os.path.join(os.path.dirname(__file__), COEFF_FILE)
    if not os.path.exists(path):
        return dict(CURRENT_COEFFS)
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    coeffs = payload.get("coeffs", CURRENT_COEFFS)
    # 确保 int 键
    return {int(k): v for k, v in coeffs.items()}


# ──────────────────── CLI ────────────────────

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="三重滤网系数重校")
    parser.add_argument("--months", type=int, default=3,
                        help="分析窗口月数 (default: 3)")
    parser.add_argument("--forward-days", type=int, default=5,
                        help="前视收益窗口交易日数 (default: 5)")
    parser.add_argument("--check-only", action="store_true",
                        help="只检查漂移，不保存")
    parser.add_argument("--show-current", action="store_true",
                        help="显示当前系数并退出")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.show_current:
        print(f"\n  当前系数: {CURRENT_COEFFS}")
        print(f"  持久化系数: {load_coeffs()}")
        sys.exit(0)

    print(f"  [三重滤网系数重校] 窗口={args.months}月, 前视={args.forward_days}日")
    print(f"  当前基准系数: {CURRENT_COEFFS}")

    try:
        new_coeffs = recalibrate(months=args.months, forward_days=args.forward_days)
    except Exception as e:
        print(f"\n  [ERROR] 重校失败: {e}")
        sys.exit(1)

    print(f"\n  新系数: {new_coeffs}")

    # 漂移检查
    drift_warnings = check_coeff_drift(new_coeffs)
    if drift_warnings:
        print("\n  [漂移警告]")
        for w in drift_warnings:
            print(f"    {w}")
    else:
        print("\n  漂移检查通过 (全部 < 20%)")

    if not args.check_only:
        save_coeffs(new_coeffs)
        print("  [完成] 系数已持久化")
    else:
        print("  [跳过保存] --check-only 模式")
