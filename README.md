# A股交易分析系统

多因子评分 + 威科夫形态识别的 A 股分析工具。

数据源：新浪财经（实时行情）、腾讯财经（K线）、Tushare Pro（龙虎榜/个股质地）、AKShare（板块/舆情/关键词热度）。

## 安装

```bash
pip install -r requirements.txt
```

需要 Tushare Pro token（5000积分）请配置 `config.json`：
```json
{"tushare_token": "你的token"}
```

## 系统架构

### 8 因子评分系统 (`scoring.py`)

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

### 扫描系统 (`scanner.py`)

- 流动性过滤（成交额门槛）
- 威科夫形态检测（Spring/SOS/LPS/Upthrust）
- 板块轮动分组
- 多重过滤（量比动量+均线+斜率）
- Feishu 推送通知

### 模块文件

| 文件 | 功能 |
|------|------|
| `volume_momentum.py` | 量比动量系统：量比均线 + 斜率 + 综合评分 |
| `buzz_monitor.py` | 热度监控：股吧评分 + 关键词热度 + 过热预警 |
| `sentiment_indicator.py` | 情绪指标：每日/每周市场情绪 + 3σ 过热检测 |
| `lhb_analyzer.py` | 龙虎榜分析：席位分类 + 溢价追踪 |
| `pattern_detector.py` | 形态检测：双顶/双底/头肩顶/头肩底/V转 |
| `factor_ic_rolling.py` | 因子 IC 滚动回测 |
| `sector_heat.py` | 板块景气度五维评分 |
| `recommendation_tracker.py` | 推荐记录追踪 |

## 用法

### 全市场扫描
```bash
python scanner.py
```

### 个股分析
```bash
python main.py <股票代码>
python main.py 002050 80000 --position 100,51.44
```

### 量比动量扫描
```bash
python vol_mom_scan.py --top 200 --regime neutral
```

### 龙虎榜分析
```bash
python lhb_analyzer.py
```

### 情绪指标
```bash
python sentiment_indicator.py
```

## 数据来源

- 实时行情：新浪财经 HTTP API
- K 线：腾讯财经 HTTP API
- 龙虎榜/个股质地：Tushare Pro
- 板块/舆情/关键词：AKShare
- 行业分类：baostock
