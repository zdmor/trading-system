"""
动态因子权重系统

基于 IC 回测的 ICIR 动态调整各因子权重。
当某因子近期预测能力增强时自动提高权重，减弱时降低。

思路源自华创固收"准确度系数加权":
  w_eff = w_base x (1 + k x icir_norm)

ICIR = mean_ic / std_ic — 衡量因子稳定性与预测能力的综合指标
"""
import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

# 基础权重（P1-4 六因子架构，合计 1.0）
BASE_WEIGHTS = {
    "tech_strength": 0.28,
    "risk_reward": 0.18,
    "volume": 0.15,
    "candlestick": 0.05,
    "sector": 0.16,
    "relative_strength": 0.18,
}

# 调节参数
K = 0.3           # ICIR 影响强度（0=完全固定，0.5=最大调整幅度~±15%）
MIN_WEIGHT = 0.02  # 单因子最小权重
MAX_MULTIPLIER = 2.0  # 单因子最大为基础权重的倍数

# 文件路径
_IC_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_ic_cache.json")


def _load_ic_cache() -> dict:
    if os.path.exists(_IC_CACHE_FILE):
        try:
            with open(_IC_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_ic_cache(cache: dict):
    with open(_IC_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def update_ic_cache(ic_results: dict) -> dict:
    """更新 IC 缓存并返回调整后的权重

    Args:
        ic_results: {
            "tech_strength": {"ic": 0.05, "icir": 0.8, "win_rate": 0.55},
            ...
        }
    """
    cache = _load_ic_cache()
    cache["data"] = ic_results
    cache["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _save_ic_cache(cache)
    return compute_weights(ic_results)


def compute_weights(ic_data: Optional[dict] = None) -> dict:
    """
    计算有效权重:
      1) icir 归一化到 [-1, 1]
      2) w_eff = w_base * (1 + K * icir_norm)
      3) 钳制 + 归一化合计 = 1.0
    """
    if ic_data is None:
        cache = _load_ic_cache()
        ic_data = cache.get("data", {})

    if not ic_data:
        return dict(BASE_WEIGHTS)

    # 提取 ICIR，缺失的因子取 0
    icirs = {}
    for key in BASE_WEIGHTS:
        d = ic_data.get(key, {})
        icirs[key] = d.get("icir", 0) or 0

    # ICIR 归一化到 [-1, 1]
    vals = list(icirs.values())
    max_abs = max(abs(v) for v in vals) if vals else 1.0
    if max_abs < 0.01:
        max_abs = 1.0

    raw = {}
    for key in BASE_WEIGHTS:
        icir_norm = max(-1.0, min(1.0, icirs[key] / max_abs))
        raw[key] = BASE_WEIGHTS[key] * (1.0 + K * icir_norm)

    # 钳制
    for key in raw:
        hi = BASE_WEIGHTS[key] * MAX_MULTIPLIER
        raw[key] = max(MIN_WEIGHT, min(hi, raw[key]))

    # 归一化
    total = sum(raw.values())
    weights = {k: round(v / total, 4) for k, v in raw.items()} if total > 0 else dict(BASE_WEIGHTS)

    # 修正四舍五入误差
    diff = round(1.0 - sum(weights.values()), 4)
    if diff:
        key_max = max(weights, key=weights.get)
        weights[key_max] = round(weights[key_max] + diff, 4)

    return weights


def get_weights() -> dict:
    """对外接口：获取当前有效权重"""
    return compute_weights()


def show_weight_report():
    """打印权重报告"""
    base = BASE_WEIGHTS
    eff = get_weights()
    cache = _load_ic_cache()
    ic_data = cache.get("data", {})
    updated = cache.get("updated", "从未更新")

    print(f"\n{'='*55}")
    print(f"  因子权重报告 (更新: {updated})")
    print(f"{'='*55}")
    print(f"  {'因子':<18} {'基础':>6} {'有效':>6} {'变动':>6}  {'ICIR':>6} {'IC':>8}")
    print(f"  {'-'*55}")
    for key in base:
        b = base[key]
        e = eff.get(key, 0)
        d = e - b
        d_str = f"{d:+.0%}" if abs(d) >= 0.005 else ""
        ic = ic_data.get(key, {})
        icir = ic.get("icir", 0)
        ic_val = ic.get("ic", 0)
        icir_str = f"{icir:+.1f}" if abs(icir) >= 0.05 else ""
        ic_str = f"{ic_val:+.4f}" if abs(ic_val) >= 0.0001 else ""
        print(f"  {key:<18} {b:>6.0%} {e:>6.0%} {d_str:>6}  {icir_str:>6} {ic_str:>8}")
    print(f"  {'-'*55}")
    print(f"  {'合计':<18} {sum(base.values()):>6.0%} {sum(eff.values()):>6.0%}")
    print()


# ─── 滚动 IC 数据库 + 时间衰减 ───

_ROLLING_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ic_rolling_db.json")
_FACTOR_KEYS = ["tech_strength", "risk_reward", "volume", "candlestick", "sector", "relative_strength"]


def _spearman_rank(x, y):
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    d = rx - ry
    return float(1 - 6 * np.sum(d**2) / (n * (n**2 - 1)))


def _load_rolling_db() -> dict:
    if os.path.exists(_ROLLING_DB_FILE):
        try:
            with open(_ROLLING_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"pairs": [], "pending": []}
    return {"pairs": [], "pending": []}


def _save_rolling_db(db: dict):
    with open(_ROLLING_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def append_snapshot(code: str, date: str, price: float,
                    factors: dict, db: dict = None):
    """存储因子快照，5 日后自动验证收益。"""
    was_provided = db is not None
    if db is None:
        db = _load_rolling_db()
    existing = {(p["code"], p["date"]) for p in db.get("pending", [])}
    key = (code, date)
    if key not in existing:
        db["pending"].append({
            "date": date,
            "code": code,
            "factors": {k: factors.get(k, 50) for k in _FACTOR_KEYS},
            "price": price,
        })
    if not was_provided:
        _save_rolling_db(db)
    return db


def check_pending(outcome_prices: dict) -> int:
    """检查到期 pending，计算 5 日收益，移入 pairs。

    每个 run.py 执行时自动调用。
    需要 outcome_prices: {code: 当前价格}
    """
    db = _load_rolling_db()
    today = datetime.now()
    new_pending = []
    completed = 0

    for p in db.get("pending", []):
        try:
            dt = datetime.strptime(p["date"], "%Y-%m-%d")
        except ValueError:
            continue
        days_passed = (today - dt).days
        code = p["code"]

        if days_passed >= 5:
            cur_price = outcome_prices.get(code)
            if cur_price and p["price"] > 0:
                fwd = (cur_price / p["price"] - 1) * 100
                db.setdefault("pairs", []).append({
                    "date": p["date"],
                    "code": code,
                    "factors": p["factors"],
                    "fwd_5d": round(fwd, 2),
                })
                completed += 1
        else:
            new_pending.append(p)

    db["pending"] = new_pending
    _save_rolling_db(db)
    return completed


def recompute_ic_from_rolling(halflife: int = 60) -> Optional[dict]:
    """从滚动 pairs 计算带时间衰减的 IC/ICIR，更新缓存。

    按日期分组 → 每截面 Spearman IC → 指数衰减加权均值/标准差

    Args:
        halflife: 衰减半衰期（天），默认 60 天

    Returns:
        {因子key: {ic, icir, win_rate}} 或 None（样本不足 50）
    """
    db = _load_rolling_db()
    pairs = db.get("pairs", [])
    if len(pairs) < 50:
        return None  # 样本不足以计算

    today = datetime.now()
    date_groups = defaultdict(list)
    for p in pairs:
        date_groups[p["date"]].append(p)

    ic_by_factor = {k: {"ics": [], "ws": []} for k in _FACTOR_KEYS}

    for date_str, group in date_groups.items():
        if len(group) < 10:
            continue
        try:
            days_ago = (today - datetime.strptime(date_str, "%Y-%m-%d")).days
        except ValueError:
            continue
        w = 2 ** (-days_ago / halflife)

        for k in _FACTOR_KEYS:
            scores = np.array([p["factors"].get(k, 50) for p in group])
            fwds = np.array([p["fwd_5d"] for p in group])
            ic = _spearman_rank(scores, fwds)
            ic_by_factor[k]["ics"].append(ic)
            ic_by_factor[k]["ws"].append(w)

    ic_data = {}
    for k in _FACTOR_KEYS:
        ics = np.array(ic_by_factor[k]["ics"])
        ws = np.array(ic_by_factor[k]["ws"])
        if len(ics) < 3:
            continue
        w_mean = float(np.average(ics, weights=ws))
        variance = float(np.average((ics - w_mean) ** 2, weights=ws)) if len(ics) > 1 else 0.001
        w_std = np.sqrt(variance) if variance > 0 else 0.001
        icir = w_mean / max(w_std, 0.001)
        win_rate = float(np.sum((ics > 0) * ws) / max(np.sum(ws), 0.001))
        ic_data[k] = {"ic": round(w_mean, 4), "icir": round(icir, 4), "win_rate": round(win_rate, 4)}

    if ic_data:
        update_ic_cache(ic_data)
        print(f"  [自动进化] IC 已从 {len(pairs)} 对滚动数据重算，半衰期 {halflife}d")
    return ic_data


# ─── 因子分歧度 ───

DIVERGENCE_LOW = 0.15   # 分歧度 < 0.15 -> 高度一致，警惕拥挤
DIVERGENCE_HIGH = 0.40  # 分歧度 > 0.40 -> 高度分歧，因子有效区分


def compute_factor_divergence(scan_results: list) -> dict:
    """
    计算因子分歧度 (std/mean)

    思路源自华创固收"久期分歧度 = 标准差 / 均值"。
    分歧度低 -> 因子在该时点区分度不足，市场对这只票的评分高度一致。
    分歧度高 -> 因子有效区分优劣。

    Args:
        scan_results: [{"code": str, "factors": [{"key": str, "score": float}, ...]}, ...]

    Returns:
        {"tech_strength": {"mean": 65.0, "std": 15.0, "divergence": 0.23, "signal": "正常"}, ...}
    """
    # 按因子名收集分数
    factor_scores = {}
    for stock in scan_results:
        for f in stock.get("factors", []):
            key = f.get("key", "")
            score = f.get("score", 50)
            factor_scores.setdefault(key, []).append(score)

    result = {}
    for key, scores in factor_scores.items():
        if len(scores) < 5:
            continue
        arr = np.array(scores)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        divergence = std / mean if mean > 0 else 0.0

        if divergence < DIVERGENCE_LOW:
            signal = "拥挤预警"
        elif divergence < 0.25:
            signal = "偏一致"
        elif divergence > DIVERGENCE_HIGH:
            signal = "高度分歧"
        else:
            signal = "正常"

        result[key] = {
            "mean": round(mean, 1),
            "std": round(std, 1),
            "divergence": round(divergence, 3),
            "signal": signal,
            "n": len(scores),
        }

    return result


def print_divergence_report(divergence: dict):
    """打印分歧度报告"""
    if not divergence:
        print("  无分歧度数据")
        return

    print(f"\n{'='*55}")
    print(f"  因子分歧度报告")
    print(f"{'='*55}")
    print(f"  {'因子':<18} {'均值':>6} {'标准差':>6} {'分歧度':>7} {'信号':<10}")
    print(f"  {'-'*55}")
    for key in divergence:
        d = divergence[key]
        print(f"  {key:<18} {d['mean']:>6.1f} {d['std']:>6.1f} {d['divergence']:>7.3f} {d['signal']:<10}")
    print(f"  {'-'*55}")
    print(f"  分歧度 < 0.15 -> 高度一致, 警惕拥挤交易")
    print()


if __name__ == "__main__":
    show_weight_report()
