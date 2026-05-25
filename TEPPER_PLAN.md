# 泰珀方法论 × A股评分系统 -- 改进清单

> 基于 investment-masters skill 的 Tepper 模块，对标当前 trading-system 架构。

## Tepper 三原则 → 系统映射

| Tepper 原则 | 当前系统状态 | 差距 | 改进 |
|---|---|---|---|
| 恐慌中买入 | `sentiment_indicator.py` 有情绪值，但**没跟仓位联动** | 情绪极值不触发任何加仓 | Kelly 加"情绪放大器" |
| 赔率不对称下重注 | `score_risk_reward()` 算 R/R，但**只影响评分，不影响仓位** | R/R 3:1 和 1:1 仓位一样 | R/R 直接进 Kelly 的 b 参数 |
| 错了立刻砍 | 有止损价配置，但**无硬止损执行** | 止损靠人手动 | 增加止损触发提醒/日志 |

---

## 改进 1: 情绪放大器（2 行改 kelly_position.py）

**逻辑：** 市场越恐慌，edge 放大越激进。泰珀的核心是在别人恐惧时贪婪。

```python
# kelly_position.py 中 compute_kelly_position() 里，加在"大盘熊市打折扣"之后：

# 泰珀修正：情绪极端时放大 edge
try:
    from sentiment_indicator import get_sentiment_snapshot
    snap = get_sentiment_snapshot()
    sentiment = snap.get("composite", 50)
    if sentiment < 30:
        position *= 1.5   # 极端恐慌 → 加仓 50%
    elif sentiment < 40:
        position *= 1.25  # 偏恐慌 → 加仓 25%
    elif sentiment > 75:
        position *= 0.5   # 贪婪 → 减半
except Exception:
    pass  # 情绪模块不可用时静默跳过
```

## 改进 2: R/R 直连 Kelly 的 b 参数

**当前问题：** `kelly_position.py` 的 `FACTOR_STATS` 用固定 `avg_win`/`avg_loss`，不看每只票的实际 R/R。

**改进：** 从 `breakdown_lines` 里取出 `risk_reward` 因子的 ratio，直接替换 b：

```python
# kelly_position.py compute_kelly_position() 里，找到 risk_reward 因子：

for f in factors:
    if f["key"] == "risk_reward":
        rr_ratio = f.get("ratio", None)
        if rr_ratio and rr_ratio > 0:
            avg_b = rr_ratio  # 直接用该票实际赔率
            avg_win = rr_ratio * avg_loss  # 反推
```

## 改进 3: 硬止损提醒

**逻辑：** 不做自动卖出（风险太大），但每次都显式输出止损线：

```python
# scoring.py compute() 返回字典里加一行：

"stop_price": self.stop_price,
"stop_pct": round((self.price - self.stop_price) / self.price * 100, 1),
```

这样每份报告底部都有一行："止损: ¥XX.XX (-X.X%)"，泰珀风格——知道最坏情况再下注。

---

## 优先级

| 改进 | 改动量 | 影响 | 建议 |
|---|---|---|---|
| 1. 情绪放大器 | 5 行 | 大（极端行情仓位翻倍） | **立即做** |
| 2. R/R 进 Kelly b | 3 行 | 中（个股区分度提升） | 跟着做 |
| 3. 止损提醒 | 2 行 | 小（纯展示） | 顺手做 |

---

## 一句话总结

> 泰珀不是"赌徒"——他是在 edge 极端时按凯利公式下重注的人。你的系统已经有所有原料（情绪、R/R、Kelly），只差把绳子系紧。
