<!-- README_ZH_START -->
# trading-system｜历史威科夫 / 市场状态交易研究系统

> 状态：`REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED`  
> 默认分支：`master`  
> 角色：`LEGACY_WYCKOFF_RESEARCH_SYSTEM`  
> README 角色：`HUMAN_ORIENTATION_ONLY / NOT_WTOS_AUTHORITY`

## 1. 仓库概述

本仓库保存一套历史 A 股交易系统研究实现，核心围绕 **Wyckoff / 市场状态识别、风险约束、回测、置信度校准、情绪/市场辅助信号** 等展开。仓库包含大量 Python 研究脚本、方法文档、历史计划和变更记录。

它现在是 Research / Reference，不是当前 WTOS Runtime、Live Authority 或生产交易系统。

## 2. 当前状态

根据 `zdmor/Meta-System@main:REPOSITORY_REGISTRY.yaml`：

```text
class  = RESEARCH_REFERENCE
role   = LEGACY_WYCKOFF_RESEARCH_SYSTEM
status = REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED
```

因此仓库内历史“Phase 1/2/3”、旧计划或旧脚本状态不等于今天的 Current。

## 3. 历史核心思路

历史设计重点包括：

- Wyckoff / 市场结构与状态识别；
- 中长线 A 股研究；
- 风控底线、反馈闭环、自适应校准；
- 多种 backtest / calibrated backtest；
- Bayesian confidence / fusion；
- bull-bear debate、buzz/market 辅助观察；
- 将研究结果用于方法验证，而非直接产生当前生产 Authority。

历史数据源包括 Tushare Pro、AKShare、腾讯财经、新浪财经等；是否仍可直接运行必须按当前代码/依赖重新验证。

## 4. 主要结构与资料

仓库根目录包含大量研究脚本和文档。重要入口包括：

| 路径 | 作用 |
|---|---|
| `README.md` | 当前人类导航 |
| `CHANGELOG.md` | 历史变更记录 |
| `P1_TASKS.md` | 历史阶段任务 |
| `TEPPER_PLAN.md` | 历史研究计划 |
| `SZFY_WORKBUDDY_TASK.md` | 历史 SZFY/WorkBuddy 任务材料；不表示当前 SZFY owner |
| `backtest.py` | 历史回测入口之一 |
| `backtest_spring_quality.py` | Spring quality 相关历史回测 |
| `batch_backtest.py` | 批量回测辅助 |
| `calibrated_backtest.py` | 校准型回测研究 |
| `bayesian_confidence.py` / `bayesian_fusion.py` | Bayesian 置信度/融合研究 |
| `bull_bear_debate.py` | 多视角/多空研究逻辑 |
| `buzz_monitor.py` | 热度/辅助监测研究 |
| `backup_claude.sh` | 历史本地备份辅助脚本；不属于当前 GitHub 全局备份标准 |
| `docs/` | Alpha Trader、trading manual、philosophy 等历史方法文档（按需读取） |

实际文件比上表更多；表格列的是理解仓库所需的关键入口，不是第二份完整 inventory。

## 5. 关键文档

历史方法入口包括：

- `docs/alpha_trader.md` — Alpha Trader 方法论；
- `docs/trading_manual.md` — 历史交易体系手册；
- `docs/philosophy.md` — 系统架构/哲学说明。

这些文档可用于研究，但任何旧规则若想进入 Current 系统，都必须重新验证并由真正 owner 明确批准。

## 6. 历史运行方式

历史 README 曾给出：

```bash
pip install -r requirements.txt
python scanner.py
python market_regime.py
python main.py <代码>
```

这些命令仅代表历史设计入口。当前能否执行、依赖是否完整、数据源是否仍有效，需要在明确研究任务中重新验证；README 不把历史命令声明为当前生产保证。

## 7. 使用方式

推荐流程：

```text
当前业务问题
-> 先解析真正 Current owner
-> 若需要 Wyckoff/状态识别历史研究
-> 进入 trading-system exact file/module
-> 提取候选方法
-> 独立验证
-> 再决定是否引用/迁移到 owner repository
```

禁止直接把整个仓库重新接入 WTOS。

## 8. Current / Canonical / Authority

- 跨仓库 role/status：`zdmor/Meta-System@main:REPOSITORY_REGISTRY.yaml`
- WTOS startup：`zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`
- 当前 WTOS 用户侧 owner：当前 resolver 指向 `zdmor/Stock-Analysis`
- 本仓 Live Authority：`NONE`

## 9. 与其他仓库关系

本仓主要作为 Wyckoff / 市场状态研究参考。SZFY、WTOS、A-Share Intelligence 等当前工作应回到各自 owner；本仓文件名中出现这些项目名称，不改变跨仓库 owner 关系。

## 10. 最近重要调整

### 2026-09-10｜跨库身份明确

Meta-System 明确本仓为 legacy Wyckoff research reference，除非显式 reactivation，不作为 Current。

### 2026-09-10｜README Governance 对齐

README 升级为详细双语仓库说明，补齐真实研究资产、关键脚本、使用边界和 Current pointer。

## 11. 生命周期 / 已知限制

- Repository：Reference-only unless explicitly reactivated；
- 历史代码：可研究，不保证当前依赖可运行；
- 历史回测：不能直接当成当前策略证据；
- 任何晋级：必须重新验证并进入真正 owner。

## 12. README 同步状态

```text
README_STANDARD = zdmor/Meta-System@main:standards/REPOSITORY_README_STANDARD.md
README_LANGUAGE_ORDER = CHINESE_THEN_ENGLISH
README_DETAIL_BASELINE = PASS
ROLE_STATUS_ALIGNED = PASS
README_SYNC = PASS_AT_THIS_COMMIT
```

<!-- README_EN_START -->
# trading-system | Legacy Wyckoff / Market-State Trading Research System

> Status: `REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED`  
> Default branch: `master`  
> Role: `LEGACY_WYCKOFF_RESEARCH_SYSTEM`  
> README role: `HUMAN_ORIENTATION_ONLY / NOT_WTOS_AUTHORITY`

## 1. Repository overview

This repository preserves a historical A-share trading-system research implementation centered on Wyckoff/market-state recognition, risk constraints, backtesting, confidence calibration, and supporting sentiment/market signals. It contains numerous Python research scripts, methodology documents, plans, and change records.

It is now Research / Reference, not current WTOS Runtime, Live Authority, or production trading system.

## 2. Current status

Meta-System classifies it as `RESEARCH_REFERENCE / LEGACY_WYCKOFF_RESEARCH_SYSTEM / REFERENCE_ONLY_UNLESS_EXPLICITLY_REACTIVATED`. Historical phases, plans, and script states are therefore not current by default.

## 3. Historical research scope

Key themes include Wyckoff/market structure, medium/long-horizon A-share research, risk/feedback/calibration loops, multiple backtest variants, Bayesian confidence/fusion, bull-bear debate, and market/buzz observation. Historical data sources included Tushare Pro, AKShare, Tencent Finance, and Sina Finance; current runnability must be revalidated.

## 4. Directory structure and assets

Important surfaces include `CHANGELOG.md`, `P1_TASKS.md`, `TEPPER_PLAN.md`, `SZFY_WORKBUDDY_TASK.md`, multiple backtest scripts, Bayesian research modules, `bull_bear_debate.py`, `buzz_monitor.py`, `backup_claude.sh`, and the `docs/` methodology area. The list highlights orientation points rather than duplicating the complete file inventory.

## 5. Key documents

Historical methodology references include `docs/alpha_trader.md`, `docs/trading_manual.md`, and `docs/philosophy.md`. They remain research material and require new validation before any promotion into a current owner system.

## 6. Historical execution

Older README instructions referenced scanner, market-regime, and main analysis commands. They are historical entry points only; dependencies and data access must be validated again for any current research run.

## 7. Usage workflow

Resolve the true Current owner first. Enter this repository only when Wyckoff/market-state historical research is relevant, load exact files, extract a candidate method, independently validate it, and only then reference or migrate it into the real owner repository.

## 8. Current / Canonical / Authority

- Cross-repository role/status: `zdmor/Meta-System@main:REPOSITORY_REGISTRY.yaml`
- WTOS startup: `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`
- Current user-facing WTOS owner: currently routed to `zdmor/Stock-Analysis`
- Live Authority here: `NONE`

## 9. Repository relationships

This repository is primarily a Wyckoff/market-state research reference. File names mentioning SZFY, WTOS, or other projects do not transfer ownership away from their current owner repositories.

## 10. Major recent changes

On 2026-09-10 Meta-System explicitly classified this repository as legacy research reference. The README was expanded to the global detailed bilingual standard.

## 11. Lifecycle / limitations

The repository stays reference-only unless explicitly reactivated. Historical code is not guaranteed to run under current dependencies, and old backtests are not current strategy evidence. Promotion requires revalidation and explicit owner adoption.

## 12. README sync status

```text
README_STANDARD = zdmor/Meta-System@main:standards/REPOSITORY_README_STANDARD.md
README_LANGUAGE_ORDER = CHINESE_THEN_ENGLISH
README_DETAIL_BASELINE = PASS
ROLE_STATUS_ALIGNED = PASS
README_SYNC = PASS_AT_THIS_COMMIT
```
