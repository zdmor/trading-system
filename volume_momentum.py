"""
量比动量系统（Volume Momentum System）
核心逻辑: 量比均线 + 斜率 + 综合评分

公式:
  量比 = 当日成交量 / 近N日均量
  量比均线 = 量比序列的移动平均
  量比斜率 = 量比均线的线性回归斜率
  综合评分 = 量比均线 × (1 + 斜率)

参数可调:
  lookback_days: 回溯天数（默认20）
  vol_ratio_period: 量比计算窗口（默认5）
  slope_period: 斜率计算窗口（默认3）
"""
import numpy as np
from typing import Optional


class VolumeMomentum:
    """量比动量分析器"""

    DEFAULTS = {
        "lookback_days": 20,     # 回溯天数
        "vol_ratio_period": 5,   # 量比均线窗口
        "slope_period": 3,       # 斜率周期
    }

    # 各市态参数
    REGIME_PARAMS = {
        "bull": {
            "vol_ratio_threshold": 1.2,
            "ma_threshold": 1.2,
            "slope_threshold": 0.03,
            "score_threshold": 2.5,
        },
        "neutral": {
            "vol_ratio_threshold": 1.5,
            "ma_threshold": 1.5,
            "slope_threshold": 0.05,
            "score_threshold": 3.0,
        },
        "bear": {
            "vol_ratio_threshold": 2.0,
            "ma_threshold": 2.0,
            "slope_threshold": 0.08,
            "score_threshold": 3.5,
        },
    }

    SCORE_BANDS = [
        (3.5, float("inf"), "强信号", "量能充沛且加速，重点关注"),
        (2.0, 3.5, "中等信号", "量能温和放大，结合走势确认"),
        (1.3, 2.0, "一般信号", "量能正常，观望或持仓"),
        (0.5, 1.3, "偏弱", "量能不足，不急于进场"),
        (float("-inf"), 0.5, "弱势信号", "量能极度萎缩，谨慎"),
    ]

    def __init__(self, volumes: list, regime: str = "neutral", closes: list = None,
                 vol_history: list = None):
        """
        Args:
            volumes: 成交量序列（最近N天，最新在最后）
            regime: 市态 "bull" / "neutral" / "bear"
            closes: 收盘价序列（用于判断涨跌方向）
            vol_history: 该股票历史量比序列（不含当日），用于百分位归一化
        """
        self.volumes = np.array(volumes, dtype=float)
        self.closes = np.array(closes, dtype=float) if closes else None
        self.regime = regime if regime in self.REGIME_PARAMS else "neutral"
        self.params = {**self.DEFAULTS, **self.REGIME_PARAMS[self.regime]}
        self.vol_history = np.array(vol_history, dtype=float) if vol_history else None

    def calc_vol_ratio(self) -> np.ndarray:
        """量比序列：每个位置 / 前N日均量"""
        vs = self.volumes
        n = self.params["lookback_days"]
        if len(vs) < n + 2:
            return np.array([1.0])
        ratios = np.zeros(len(vs))
        for i in range(len(vs)):
            if i < n:
                ratios[i] = 1.0
            else:
                avg = np.mean(vs[i - n:i])
                ratios[i] = vs[i] / avg if avg > 0 else 1.0
        return ratios

    def calc_ma(self, ratios: np.ndarray) -> np.ndarray:
        """量比均线"""
        period = self.params["vol_ratio_period"]
        if len(ratios) < period:
            return np.array([1.0])
        return np.convolve(ratios, np.ones(period) / period, mode="valid")

    def calc_slope(self, ma: np.ndarray) -> float:
        """量比斜率：最近 slope_period 天的线性回归斜率"""
        period = self.params["slope_period"]
        if len(ma) < period:
            return 0.0
        segment = ma[-period:]
        x = np.arange(period)
        # 线性回归: slope = cov(x,y) / var(x)
        if np.std(segment) == 0:
            return 0.0
        slope = np.polyfit(x, segment, 1)[0]
        # 量比均线斜率天然在 [-2, 2] 范围，不做硬截断
        return float(slope)

    def analyze(self) -> dict:
        """完整分析
        返回:
            vol_ratio: 最新量比
            vol_ratio_ma: 量比均线值
            slope: 量比斜率
            composite_score: 综合评分
            signal: 信号强度
            direction: 量价方向（放量上涨/放量下跌/缩量/正常）
            suggestion: 操作建议
        """
        ratios = self.calc_vol_ratio()
        ma = self.calc_ma(ratios)

        current_ratio = float(ratios[-1]) if len(ratios) > 0 else 1.0
        current_ma = float(ma[-1]) if len(ma) > 0 else 1.0
        slope = self.calc_slope(ma)

        # 综合评分 = 量比均线 × (1 + 斜率)
        composite = current_ma * (1 + slope)

        # 价格方向
        pct_chg = 0.0
        if self.closes is not None and len(self.closes) >= 2:
            pct_chg = (self.closes[-1] / self.closes[-2] - 1) * 100

        # 信号判定
        percentile = None
        if self.vol_history is not None and len(self.vol_history) >= 20:
            percentile = self._calc_percentile(current_ratio, self.vol_history)
        signal, suggestion = self._classify(composite, percentile)

        # 量价方向判定
        direction = self._judge_direction(current_ratio, pct_chg)

        # 过滤检查
        checks = self._check_filters(current_ratio, current_ma, slope)

        return {
            "vol_ratio": round(current_ratio, 2),
            "vol_ratio_ma": round(current_ma, 2),
            "slope": round(slope, 4),
            "composite_score": round(composite, 2),
            "signal": signal,
            "direction": direction,
            "suggestion": suggestion,
            "pct_chg": round(pct_chg, 2),
            "checks_passed": all(checks.values()),
            "checks": checks,
            "percentile": percentile,
        }

    def _judge_direction(self, vol_ratio: float, pct_chg: float) -> str:
        """量价方向判断"""
        if vol_ratio >= 1.5 and pct_chg >= 2:
            return "放量上涨"
        elif vol_ratio >= 1.5 and pct_chg <= -2:
            return "放量下跌"
        elif vol_ratio < 0.7:
            return "缩量"
        elif vol_ratio >= 1.5:
            return "放量震荡"
        return "正常"

    def _check_filters(self, ratio: float, ma: float, slope: float) -> dict:
        """三重过滤"""
        p = self.params
        return {
            "量比达标": ratio >= p["vol_ratio_threshold"],
            "均线达标": ma >= p["ma_threshold"],
            "斜率达标": slope >= p["slope_threshold"],
        }

    def _calc_percentile(self, value: float, history: np.ndarray) -> float:
        """计算量比在自身历史中的百分位"""
        if history is None or len(history) < 20:
            return 50.0
        rank = np.sum(history < value) / len(history) * 100
        return round(min(99.0, max(1.0, rank)), 1)

    def _classify(self, score: float, percentile: float = None) -> tuple:
        """评分 → 信号等级。优先百分位，回退绝对阈值"""
        if percentile is not None:
            if percentile >= 90:
                return "强信号", "量能处于自身历史高位，重点关注"
            elif percentile >= 75:
                return "中等信号", "量能高于常态，结合走势确认"
            elif percentile >= 50:
                return "一般信号", "量能正常，观望或持仓"
            elif percentile >= 25:
                return "偏弱", "量能偏低，不急于进场"
            else:
                return "弱势信号", "量能极度萎缩，谨慎对待"
        # 回退：绝对阈值
        for lo, hi, signal, suggestion in self.SCORE_BANDS:
            if lo <= score < hi:
                return signal, suggestion
        if score >= 5.0:
            return "强信号", "重点关注，可小仓位试错"
        return "弱势信号", "量能萎缩，谨慎对待"

    @staticmethod
    def recommend_regime(index_closes: list) -> str:
        """根据指数走势推荐市态参数"""
        if len(index_closes) < 20:
            return "neutral"
        arr = np.array(index_closes)
        ma20 = np.mean(arr[-20:])
        ma60 = np.mean(arr[-60:]) if len(arr) >= 60 else ma20
        current = arr[-1]

        if current > ma20 > ma60:
            return "bull"
        elif current < ma20 < ma60:
            return "bear"
        return "neutral"


if __name__ == "__main__":
    import sys

    # 模拟数据测试
    np.random.seed(42)
    base = 100
    vols = []

    # 阶段1：缩量（前30天）
    for i in range(30):
        vols.append(base + np.random.normal(0, 5))

    # 阶段2：放量（后20天）
    for i in range(20):
        base += np.random.normal(2, 3) * 5
        vols.append(max(1, base + np.random.normal(0, 10)))

    vm = VolumeMomentum(vols, regime="neutral")
    r = vm.analyze()

    print("═══ 量比动量分析 ═══")
    print(f"  市态参数: {vm.regime}")
    print(f"  量比: {r['vol_ratio']:.2f}")
    print(f"  量比均线: {r['vol_ratio_ma']:.2f}")
    print(f"  量比斜率: {r['slope']:.4f}")
    print(f"  综合评分: {r['composite_score']:.2f}")
    print(f"  信号: {r['signal']}")
    print(f"  建议: {r['suggestion']}")
    print(f"  三重过滤: {r['checks_passed']}")
    for k, v in r['checks'].items():
        print(f"    {k}: {'通过' if v else '未通过'}")

    # 各市态对比
    print("\n═══ 各市态参数对比 ═══")
    for regime in ["bull", "neutral", "bear"]:
        vm2 = VolumeMomentum(vols, regime=regime)
        r2 = vm2.analyze()
        print(f"  {regime}: 阈值{r2['composite_score']:.1f} → {r2['signal']}")
