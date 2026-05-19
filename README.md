# A股交易分析系统

基于趋势跟踪 + ATR 仓位管理 + 威科夫形态识别的 A 股分析工具。

数据源全部来自免费公开接口（腾讯财经 HTTP API、新浪行情），无需注册、无需 token。

## 安装

```bash
pip install -r requirements.txt
```

## 用法

### 个股分析

```bash
python main.py <股票代码> [账户总资金] [--position 持仓股数,持仓均价]
```

示例：

```bash
# 分析三花智控，默认资金8万
python main.py 002050

# 指定资金，带持仓
python main.py 002050 80000 --position 100,51.44
```

### 全市场扫描

```bash
python main.py scan [最低成交额(亿)]
```

示例：

```bash
# 成交额≥5亿
python main.py scan 5

# 快速模式（跳过趋势检查）
python main.py scan 5 --quick
```

## 输出说明

### 个股报告
- 趋势方向（MA50/MA200）
- 波动率（ATR14）
- 关键价位（支撑/阻力）
- 仓位管理（基于固定比例风险）
- 入场信号检查
- 操作建议（持仓/加仓/止损/止盈）

### 扫描报告
- 板块分组分布
- 威科夫形态检测：
  - **Spring（弹簧）**：跌破支撑后快速收回，放量确认
  - **SOS（强势信号）**：放量阳线高位收盘
  - **LPS（最后支撑点）**：缩量回调至支撑，最佳入场点
  - **Upthrust（上冲回落）**：突破阻力后收回，派发信号
- 信号精选排名

## 策略配置

复制 `config.example.json` 为 `config.json`，可自定义每只股票的止损止盈参数。

## 数据来源

- 实时行情：新浪财经 HTTP API
- K 线数据：腾讯财经 HTTP API
- 行业分类：baostock

## 风险提示

本工具基于技术分析，仅供参考，不构成投资建议。股市有风险，投资需谨慎。
