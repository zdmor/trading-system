"""
贝叶斯后验置信度更新模块

将"策略是否失效"的二元判断转化为概率更新。
核心思想：没有"有效/失效"的开关，只有概率的连续变化。

核心公式:
    posterior = (prior_alpha + wins) / (prior_alpha + prior_beta + total_trades)

先验: Beta(11, 11) → 均值50%, 等效20次观测(10W/10L)

用法:
    from bayesian_confidence import BayesianConfidence
    bc = BayesianConfidence()
    bc.update_overall(won=True)         # 记录一次正确预测
    bc.update_factor("wyckoff", True)   # 记录某因子的一次正确
    status = bc.report()                # 获取置信度报告
    factor = bc.factor_multiplier()     # 获取仓位调整系数
"""
import os, json
import numpy as np
from datetime import datetime

_CONF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bayesian_confidence.json")

# 先验参数: Beta(alpha, beta)
# 默认 Beta(11,11) → 均值0.5, 等效20次观测
PRIOR_ALPHA = 11
PRIOR_BETA = 11

FACTOR_KEYS = [
    "overall",
    "risk_reward", "trend_momentum", "relative_strength",
    "wyckoff", "volatility", "volume", "candlestick",
]


def _default_state():
    return {
        "alpha": PRIOR_ALPHA,
        "beta": PRIOR_BETA,
        "wins": 0,
        "losses": 0,
        "total": 0,
        "last_update": None,
    }


def _load_db():
    if os.path.exists(_CONF_FILE):
        try:
            with open(_CONF_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_db(db):
    with open(_CONF_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _beta_mean(alpha, beta):
    return alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5


def _beta_std(alpha, beta):
    s = alpha + beta
    if s <= 0:
        return 0.5
    return np.sqrt(alpha * beta / (s * s * (s + 1)))


def _beta_ci(alpha, beta, prob=0.9, n_samples=50000):
    """90% 置信区间 — 通过 Monte Carlo 采样估计"""
    if alpha <= 0 or beta <= 0:
        return (0.45, 0.55)
    samples = np.random.beta(alpha, beta, n_samples)
    tail = (1 - prob) / 2
    return (float(np.quantile(samples, tail)),
            float(np.quantile(samples, 1 - tail)))


class BayesianConfidence:
    """贝叶斯置信度跟踪器"""

    def __init__(self, db=None):
        self.db = db if db is not None else _load_db()
        self._ensure_keys()

    def _ensure_keys(self):
        for key in FACTOR_KEYS:
            if key not in self.db:
                self.db[key] = dict(_default_state())

    def _save(self):
        self.db["_meta"] = {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "schema": "v1",
        }
        _save_db(self.db)

    # ─── 更新方法 ───

    def update(self, key: str, won: bool, count: int = 1):
        """
        记录一次因子预测结果。

        Args:
            key: 因子名 (FACTOR_KEYS 之一)
            won: True=预测正确(上涨/信号兑现), False=错误
            count: 批量更新条数 (默认1)
        """
        if key not in self.db:
            return
        state = self.db[key]
        if won:
            state["alpha"] += count
            state["wins"] += count
        else:
            state["beta"] += count
            state["losses"] += count
        state["total"] += count
        state["last_update"] = datetime.now().strftime("%Y-%m-%d")
        self._save()

    def update_overall(self, won: bool, count: int = 1):
        """更新整体策略置信度"""
        self.update("overall", won, count)

    def update_factor(self, factor_key: str, won: bool, count: int = 1):
        """更新单因子置信度"""
        if factor_key not in FACTOR_KEYS or factor_key == "overall":
            return
        self.update(factor_key, won, count)

    # ─── 查询方法 ───

    def get_confidence(self, key: str) -> dict:
        """获取某因子的后验置信度"""
        if key not in self.db:
            return {}
        state = self.db[key]
        a, b = state["alpha"], state["beta"]
        mean = _beta_mean(a, b)
        std = _beta_std(a, b)
        ci_lo, ci_hi = _beta_ci(a, b)

        base_prior = (PRIOR_ALPHA + PRIOR_BETA) / 2  # 50%
        signal = "置信提升" if mean > base_prior + 0.05 else \
                 "置信衰减" if mean < base_prior - 0.05 else "中性"
        # 宽置信区间 = 不确定
        if std > 0.12:  # < 100次观测时标准差偏大
            signal += " (不确定)"
        elif std < 0.05:  # > 200次观测
            pass

        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_90": (round(ci_lo, 4), round(ci_hi, 4)),
            "wins": state["wins"],
            "losses": state["losses"],
            "total": state["total"],
            "signal": signal,
        }

    def get_overall(self) -> dict:
        """获取整体策略置信度"""
        return self.get_confidence("overall")

    def factor_multiplier(self, key: str = "overall") -> float:
        """
        仓位调整系数 [0.5, 1.5]

        置信度<40% → 0.5 (减半仓)
        置信度50% → 1.0 (不变)
        置信度>60% → 1.5 (加五成)

        线性插值，钳制在 [0.5, 1.5]
        """
        conf = self.get_confidence(key)
        mean = conf.get("mean", 0.5)
        mult = 0.5 + (mean - 0.3) / 0.4  # 0.3→0.5, 0.5→1.0, 0.7→1.5
        return max(0.5, min(1.5, mult))

    def factor_weight_adjustment(self) -> dict:
        """
        返回各因子的权重调整乘数。

        基于因子置信度与整体置信度的对比:
        - 因子置信度 > 整体置信度 + 5% → 权重上浮 (+10%)
        - 因子置信度 < 整体置信度 - 5% → 权重下浮 (-10%)
        - 其余不变
        """
        overall = self.get_overall().get("mean", 0.5)
        adj = {}
        for key in FACTOR_KEYS:
            if key == "overall":
                continue
            f_conf = self.get_confidence(key).get("mean", 0.5)
            if f_conf > overall + 0.05:
                adj[key] = 1.10
            elif f_conf < overall - 0.05:
                adj[key] = 0.90
            else:
                adj[key] = 1.0
        return adj

    # ─── 数据摄入 ───

    def ingest_score_log(self, score_log_path: str = None):
        """
        从 score_log.json 摄入已确认的评分→涨跌结果，更新整体置信度。
        """
        path = score_log_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "score_log.json")
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return 0

        ingested = 0
        for rec in records:
            outcome = rec.get("outcome")
            if outcome not in ("up", "down"):
                continue
            won = outcome == "up"
            # 检查是否已摄入 (通过去重)
            key = f"{rec['date']}_{rec['symbol']}"
            seen = set(self.db.get("_ingested", []))
            if key in seen:
                continue
            self.update_overall(won)
            self.db.setdefault("_ingested", []).append(key)
            ingested += 1

        if ingested > 0:
            self._save()
        return ingested

    def ingest_picks_history(self, picks_history_path: str = None):
        """
        从 picks_history.json 摄入选股结果。

        每只选股在 check_days 后的涨跌视为一次预测验证。
        wyckoff 因子的置信度按信号类型映射:
        - Spring/SOS/LPS → 上涨=正确
        - Upthrust/EVR  → 下跌=正确
        """
        path = picks_history_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "picks_history.json")
        if not os.path.exists(path):
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return 0

        from main import DataFetcher
        import io, contextlib
        fetcher = DataFetcher()
        ingested = 0
        today = datetime.now()

        buy_sigs = {"SOS", "Spring", "LPS", "Compression", "Markup"}
        sell_sigs = {"Upthrust", "EVR"}

        for rec in records:
            # 已摄入跳过
            key = f"pick_{rec['date']}_{rec['code']}"
            seen = set(self.db.get("_ingested_picks", []))
            if key in seen:
                continue

            entry_date_str = rec["date"]
            try:
                entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d")
            except ValueError:
                continue
            days_passed = (today - entry_dt).days
            check_days = 5
            if days_passed < check_days:
                continue

            signal = rec.get("signal", "")
            if signal not in buy_sigs and signal not in sell_sigs:
                continue

            # 获取当前价格（抑制 DataFetcher 的 login/logout 输出）
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    daily = fetcher.get_daily(rec["code"], days=check_days + 5)
                if daily is None or len(daily) < 2:
                    continue
            except Exception:
                continue

            dates = daily["date"].values.astype(str)
            closes = daily["close"].values.astype(float)
            entry_idx = None
            for i, d in enumerate(dates):
                if d.startswith(entry_date_str):
                    entry_idx = i
                    break
            if entry_idx is None:
                continue

            check_idx = entry_idx + check_days
            if check_idx >= len(closes):
                continue

            entry_price = closes[entry_idx]
            check_price = closes[check_idx]
            ret = (check_price / entry_price - 1) * 100

            if signal in buy_sigs:
                won = ret > 0
            else:
                won = ret < 0

            self.update_overall(won)
            self.update_factor("wyckoff", won)
            self.db.setdefault("_ingested_picks", []).append(key)
            ingested += 1

        if ingested > 0:
            self._save()
        return ingested

    # ─── 报告 ───

    def report(self) -> str:
        """生成置信度报告"""
        overall = self.get_overall()

        lines = []
        lines.append(f"\n{'=' * 55}")
        lines.append(f"  贝叶斯策略置信度")
        lines.append(f"{'=' * 55}")

        # 整体
        o_mean = overall.get("mean", 0.5)
        o_ci = overall.get("ci_90", (0, 0))
        o_total = overall.get("total", 0)
        o_signal = overall.get("signal", "中性")
        lines.append(f"\n  整体策略  {o_mean:.1%}  90%CI:[{o_ci[0]:.0%},{o_ci[1]:.0%}]")
        lines.append(f"  已验证: {o_total}次  判定: {o_signal}")

        # 建议
        mult = self.factor_multiplier()
        if mult > 1.2:
            lines.append(f"  建议: 正常仓位 (信心充足, 乘数{mult:.2f})")
        elif mult < 0.8:
            lines.append(f"  建议: 减仓 (信心不足, 乘数{mult:.2f})")
        else:
            lines.append(f"  建议: 正常仓位 (乘数{mult:.2f})")

        # 各因子
        lines.append(f"\n  {'因子':<18} {'置信度':>7} {'90%CI':>14} {'验证':>5} {'判定':<14}")
        lines.append(f"  {'-' * 60}")
        for key in FACTOR_KEYS:
            if key == "overall":
                continue
            conf = self.get_confidence(key)
            if conf.get("total", 0) == 0:
                continue
            mean = conf.get("mean", 0.5)
            ci = conf.get("ci_90", (0, 0))
            t = conf.get("total", 0)
            sig = conf.get("signal", "")
            lines.append(
                f"  {key:<18} {mean:>6.1%}  [{ci[0]:.0%},{ci[1]:.0%}]"
                f"  {t:>4}次 {sig:<14}"
            )

        lines.append(f"\n  {'=' * 55}")
        lines.append(f"  贝叶斯更新公式: posterior = (prior + wins) / (prior_n + total)")
        lines.append(f"  先验: Beta({PRIOR_ALPHA},{PRIOR_BETA}) = 等效20次观测(10W/10L)")
        lines.append(f"  更新: {_CONF_FILE}")

        return "\n".join(lines)


def run_bayesian_update():
    """独立运行入口：摄入历史数据 + 输出报告"""
    bc = BayesianConfidence()

    n1 = bc.ingest_score_log()
    n2 = bc.ingest_picks_history()

    print(f"  [贝叶斯] 摄入 {n1} 条评分记录 + {n2} 条选股记录")

    o = bc.get_overall()
    print(f"  [贝叶斯] 整体置信度: {o.get('mean', 0.5):.1%} ({o.get('total', 0)}次验证)")
    print(bc.report())

    return bc


if __name__ == "__main__":
    run_bayesian_update()
