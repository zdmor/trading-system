# 三花交易系统

A 股威科夫交易系统。倾向派架构：市场状态识别驱动决策，频率统计做风险约束，自适应校准做参数优化。

**顶层设计：** [系统哲学白皮书](docs/system_philosophy.md)

## 核心架构

```
大盘数据 → Wyckoff 阶段分类 → 2D 门控(阶段×信号比) → scanner 选股 → pre_buy_checklist → 执行
```

### 三层概率架构

| 层 | 职责 | 核心组件 |
|---|------|---------|
| 倾向派（核心） | 市场状态识别 | `market_regime._classify_wyckoff_phase()`、PHASE_SIGNAL_MAP |
| 频率派（工具） | 统计基准与风险约束 | 信号比红黄绿灯、因子 IC |
| 自适应层（优化） | 窗口式参数校准 | 因子权重月更、信号质量季更 |

### 8 因子评分

| 因子 | 权重 | 说明 |
|------|------|------|
| 威科夫信号 | 20% | Spring/SOS/Upthrust 类型+阶段 |
| 盈亏比 | 16% | 支撑/阻力位计算的风险报酬比 |
| 量比动量 | 9% | 量比均线×斜率 综合评分 |
| K线形态 | 5% | 吞没/锤子/十字星/射击之星 |
| 板块强度 | 11% | 行业排名+景气度五维 |
| 趋势动量 | 15% | MA排列/RSI/波动率 |
| 大盘趋势 | 12% | 指数MA排列/量能确认 |
| 相对强度 | 12% | 个股vs大盘超额收益 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 全市场扫描
python scanner.py

# 个股分析
python main.py <股票代码>

# 大盘状态
python market_regime.py

# 买入检查清单
python pre_buy_checklist.py --ticker <代码> --price <价格>
```

需要 Tushare Pro token 配置在 `config.json`。

## 项目路线图

- **Phase 1**（本周）：偏离上限 25% + 止损集成 + RPS 排名
- **Phase 2**（6 月）：信号反馈闭环 + 通道分类 + 动态调参
- **Phase 3**（7 月）：分钟线分析 + 板块轮动 + 缠论三买

详见 `docs/system_philosophy.md` 第 6 节。

## 边界

专注 A 股中长线（持仓 20-60 日）。不做短线/期权/量化打板/纯 ML 预测/跨市场。

## 数据源

- 实时行情：新浪财经 HTTP API
- K 线：腾讯财经 HTTP API
- 基本面/龙虎榜：Tushare Pro
- 板块/舆情：AKShare
