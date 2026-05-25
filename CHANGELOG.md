# CHANGELOG

> 与 D:\DiskMigration\MySecondBrain\references\CHANGELOG.md 同步
> 每次 git commit + tag 后更新

---

## v1.6.2-sectorfix (2026-05-26)
- 重构: sector 从评分因子降级为仓位约束（IC=-0.032 负预测能力）
- 修改: WEIGHTS 重分配 tech_strength 33% / risk_reward 22% / volume 18% / candlestick 5% / relative_strength 22%
- 新增: sector 约束 — 板块评分<40 仓位×0.75, <30 仓位×0.5
- 同步: BASE_WEIGHTS / FACTOR_STATS 一致更新

---

## v1.6.1-factorstats (2026-05-26)
- 修改: kelly_position.py FACTOR_STATS 基于 IC 回测数据更新
- 更新: tech_strength 0.55→0.56, risk_reward 0.58→0.39, volume 0.53→0.41
- 更新: candlestick 0.51→0.51(不变), sector 0.54→0.37, relative_strength 0.56→0.51
- 修复: sector 因子板块缓存 + sector_heat fallback

---

## v1.6.0-evolution (2026-05-26)
- 新增: 滚动 IC 数据库（ic_rolling_db.json），每日扫描自动积累因子-收益对
- 新增: 时间衰减 IC（60天半衰期），近期数据权重高于历史
- 新增: 校准反馈闭环（calibration.json → 评分纠偏），系统性乐观/悲观自动修正
- 修改: run.py 集成自主进化钩子，扫描即学习
- 修改: scoring.py 集成校准纠偏

---

## v1.5.0-icbridge (2026-05-25)
- 新增: p1_8_ic_backtest.py 全因子IC回测（新6因子 + 市态分组）
- 新增: IC缓存桥接 — p1_8_ic_backtest.py 自动写入 factor_ic_cache.json
- 运行: 41截面 IC 回测，动态权重系统已激活

---

## v1.4.0-tepper (2026-05-25)
- 新增: R/R 直连 Kelly b 参数（score_risk_reward 实际赔率替代统计值）
- 新增: 情绪放大器（恐慌<30 仓位×1.5，贪婪>75 减半）
- 新增: compute() 返回 stop_price + stop_pct 止损信息
- 新增: TEPPER_PLAN.md 泰珀方法论映射文档

---

## v1.3.0-debate (2026-05-25)
- 新增: 多因子多空辩论模块 bull_bear_debate.py
- 新增: 因子分歧度指标（std/mean，拥挤预警信号）
- 修改: scoring.py 集成辩论裁决，否决时强制降仓位
- 修改: scoring.py 集成动态权重（ICIR调整），替代固定 WEIGHTS

## v1.2.0-fundamental (2026-05-25)
- 新增: 基本面分析系统 fundamental_analyzer.py
- 新增: 市值风云种子池 32 只股票
- 新增: fundamental_pool.json 持久化 PASS/FAIL/WATCH 评级
- 新增: 导出 Markdown 报告到 vault

## v1.1.0-weights (2026-05-25)
- 新增: 动态因子权重系统 factor_weights.py
- 新增: 基于 ICIR 的权重自动调整
- 新增: factor_ic_cache.json 持久化
- 修改: factor_ic_rolling.py 运行后自动写入 IC 缓存
- 修复: tushare_provider.py config.json 编码问题（指定 utf-8 打开）

## v1.0.0-p1 (2026-05-25)
- 重构: 8 因子→6 因子（wyckoff+动量合并为 tech_strength, 大盘移出权重）
- 新增: Kelly 仓位模块 kelly_position.py
- 新增: 仓位决策矩阵 position_matrix.py
- 修改: scoring.py WEIGHTS 更新为 6 因子架构
- 修改: volume_momentum.py 百分位量比替换绝对阈值（P0遗留）

## v0.3.0-volume (2026-05-24)
- 新增: 量比动量系统 volume_momentum.py
- 新增: vol_mom_scan.py 全市场量比扫描
- 新增: 百分位量比归一化（120日历史窗口）
- 删除: 追高惩罚模块（P0-1 证明 A 股有动量效应，惩罚有害）

## v0.2.0-scanner (2026-05)
- 新增: 全市场扫描选股 scanner.py
- 新增: 舆情监控 buzz_monitor.py
- 新增: 龙虎榜席位分析 lhb_analyzer.py
- 新增: 情绪 3sigma 过热检测 sentiment_indicator.py

## v0.1.0-wyckoff (2025)
- 新增: 威科夫形态检测引擎 v2.2
- 新增: 多因子评分系统 scoring.py（8 因子初版）
- 新增: 回测系统 backtest.py
- 新增: 推送模块（飞书通知）
