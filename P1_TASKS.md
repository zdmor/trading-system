# Workbuddy P1 任务指令

## P1-8: 全因子IC回测适配新架构（关键，先做）

### 目标
创建一个新的IC回测脚本，用 `StockScorer.compute()` 的6因子评分计算滚动IC/ICIR，将结果写入 `factor_ic_cache.json`，使动态权重系统生效。

### 文件
- **新建**: `factor_ic_scoring.py`（参考现有 `factor_ic_rolling.py` 结构）
- **输出**: `factor_ic_cache.json`（用 `factor_weights.update_ic_cache()` 写入）
- **不修改**: `scoring.py`、`factor_weights.py`

### 实现要点

1. **数据获取**
   - 沿用 Tencent K线 API（已有 `get_klines()` 函数），取500日K线
   - 候选取前200只（参考 `scanner.fetch_all_stocks()` + `filter_candidates()`）

2. **时间截面**
   - 取所有股票的共有日期，跳过前120天
   - 截面间隔 shift=5日（参考现有 rolling 逻辑）
   - 每个截面保留5日前向收益空间

3. **因子计算 —— 关键差异**
   - 对每只股票在每个截面，构造完整的 `df` DataFrame（含 ma20/ma50/ma200/atr/atr_pct 列）
   - 在新线程中调用 `StockScorer(df, ..., market_df=None)` 的 `compute()` 方法
   - 提取返回的 `factors` 列表中的 score × weight 原始值
   - 共6个因子key: `tech_strength`, `risk_reward`, `volume`, `candlestick`, `sector`, `relative_strength`

4. **IC 计算**
   - 跨截面用 Spearman 秩相关（已有 `spearman_rank()` 函数）
   - 每个截面计算6个因子的截面 IC
   - 累加后计算各因子的 mean_ic, std_ic, icir, win_rate

5. **结果保存**
   - 调用 `factor_weights.update_ic_cache()` 写入：
   ```python
   from factor_weights import update_ic_cache
   update_ic_cache({
       "tech_strength": {"ic": mean_ic, "icir": icir, "win_rate": win_rate},
       ...
   })
   ```
   - 打印汇总表格（同现有 rolling 风格）

### 验证
- 运行 `python factor_ic_scoring.py --top 200`
- 完成后运行 `python factor_weights.py`，应看到6个因子都有 IC/ICIR 数据
- 运行 `python scoring.py` 的简单导入测试，确认 compute() 中动态权重生效

---

## P1-7: 资金流向因子（次优先）

### 目标
在 `scoring.py` 中新增 `score_fund_flow()` 方法，将主力资金净流入/流出量化为 0-100 分，权重 8%。

### 修改文件
- `scoring.py`

### 实现要点

1. **新增方法 `score_fund_flow(self)`**
   ```python
   def score_fund_flow(self):
       """
       资金流向评分 (权重8%)
       基于 data_providers.get_stock_moneyflow() 的净流入数据
       """
   ```
   - 调用 `from data_providers import get_stock_moneyflow`
   - 入参：`self.symbol`（股票代码）
   - 获取当日主力净流入额（万元）
   - 归一化到 0-100：
     - 净流入 > 500万 → 85-100
     - 净流入 100-500万 → 65-85
     - 净流入 -100~100万 → 45-65
     - 净流出 100-500万 → 25-45
     - 净流出 > 500万 → 0-25
   - 数据获取失败 → 返回 50 分（中性）
   - 返回格式: `{"score": int, "label": str, "detail": str, "net_flow": float}`

2. **修改 `WEIGHTS` 字典**
   - 从各因子匀出8%（tech_strength -3%, risk_reward -1%, volume -1%, sector -1%, relative_strength -2%）
   - 新权重: tech_strength=25, risk_reward=17, volume=14, candlestick=5, sector=15, relative_strength=16, **fund_flow=8**
   - 合计仍为 100

3. **修改 `compute()` 方法**
   - 在因子循环中添加 `("fund_flow", self.score_fund_flow)`
   - 在 `compute_kelly_position` 调用中传入 fund_flow 因子

### 验证
- 用 mock 或真实 symbol 测试，确认返回格式正确
- 确认 WEIGHTS 合计 = 1.0
- 确认 compute() 返回值中包含 fund_flow 因子得分
