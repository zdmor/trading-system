<!-- README_ZH_START -->
# 超我交易系统｜历史威科夫研究参考

> 当前跨库定位：`LEGACY_WYCKOFF_RESEARCH_SYSTEM / REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED`

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

## 项目进度（历史 README 记录）

```text
Phase 1 (风控底线)   ░░░░░░░░░░ 本周目标
Phase 2 (反馈闭环)   ░░░░░░░░░░ 6月目标
Phase 3 (精度提升)   ░░░░░░░░░░ 7月目标
```

数据源：Tushare Pro / AKShare / 腾讯财经 / 新浪财经。  
原设计边界：A股中长线，不做短线/期权/打板/纯ML。

**当前治理边界：** 本仓库是 Research / Reference，不是当前 WTOS Runtime 或 Live Authority。当前用户侧 WTOS 主线是 `zdmor/Stock-Analysis`；任何 WTOS 任务仍先从 `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml` 解析。

README 只是人类导航；跨库角色以 `zdmor/Meta-System@main:REPOSITORY_REGISTRY.yaml` 为准。

<!-- README_EN_START -->
# Super-Ego Trading System | Legacy Wyckoff Research Reference

> Current portfolio role: `LEGACY_WYCKOFF_RESEARCH_SYSTEM / REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED`

This repository contains an A-share Wyckoff/state-recognition trading-system prototype built around regime identification, frequency constraints, and adaptive calibration.

Main references:

- `docs/alpha_trader.md` — Alpha Trader methodology;
- `docs/trading_manual.md` — trading-system manual;
- `docs/philosophy.md` — architecture/philosophy.

Historical data sources include Tushare Pro, AKShare, Tencent Finance, and Sina Finance. The original scope focused on medium/long-horizon A-share research rather than short-term limit-up trading, options, or pure-ML strategies.

**Current governance boundary:** this repository is Research / Reference only. It is not the current WTOS Runtime or Live Authority. The current user-facing WTOS mainline is `zdmor/Stock-Analysis`, and every WTOS task still starts from `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`.

README is human orientation only. Cross-repository role classification is owned by `zdmor/Meta-System@main:REPOSITORY_REGISTRY.yaml`.
