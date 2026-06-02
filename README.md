# 超我交易系统

A 股威科夫交易系统。倾向派架构：状态识别驱动，频率约束，自适应校准。

**训练框架：** [Alpha Trader 方法论](docs/alpha_trader.md)
**操作手册：** [交易体系手册](docs/trading_manual.md)
**系统原理：** [架构白皮书](docs/philosophy.md)

## 快速开始

```bash
pip install -r requirements.txt
python scanner.py       # 全市场扫描
python market_regime.py # 大盘状态
python main.py <代码>   # 个股分析
```

## 项目进度

```
Phase 1 (风控底线)   ░░░░░░░░░░ 本周目标
Phase 2 (反馈闭环)   ░░░░░░░░░░ 6月目标
Phase 3 (精度提升)   ░░░░░░░░░░ 7月目标
```

数据源: Tushare Pro / AKShare / 腾讯财经 / 新浪财经
边界: A股中长线，不做短线/期权/打板/纯ML
