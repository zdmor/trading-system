# WorkBuddy 5-Hour Continuous Research Work Order — ETF M5 Validation

**Date:** 2026-09-12  
**Status:** RESEARCH_ONLY / PUBLIC TASK SPEC / NOT WTOS LIVE AUTHORITY  
**Owner:** User  
**Executor:** local WorkBuddy  
**Time budget:** about 5 hours continuous execution; do not pause for questions.

## 0. Critical operating rule

This is a **continuous autonomous research job**. Once started:

- DO NOT ask the user or ChatGPT for clarification.
- DO NOT stop because one package/API/source fails.
- DO NOT wait for approval between phases.
- If one path is blocked for >15 minutes, record the blocker and immediately switch to the next fallback path.
- Use the full time budget productively. If core work finishes early, spend remaining time on robustness tests, code cleanup, data-quality checks, sensitivity analysis, and report improvement.
- Never place trades, never connect to a broker for write actions, never modify WTOS Live Authority, private GitHub repositories, account state, risk policy, or automation configuration.
- Never expose passwords, tokens, cookies, private keys, or other secrets in reports/logs/email.

The job must end with concrete local artifacts and one final email report. No intermediate question is allowed.

---

# 1. Research question

Determine whether a **mainline + relative-strength + platform breakout** timing method can be made practical for exchange-traded funds (ETFs), and whether the current equity-style rule can be reused directly or needs ETF-specific parameters.

The stock-side reference idea is only a research reference here, not an instruction to change production:

1. strong mainline/context;
2. strong vehicle relative strength;
3. uptrend;
4. 15-session consolidation/platform;
5. close breaks above prior platform high;
6. volume/turnover expands;
7. avoid chasing far above the breakout;
8. define structural invalidation.

The key ETF question is:

> Does the same structure work on ETFs, and if so, what parameter ranges are robust enough to justify a future formal proposal?

---

# 2. Phase 0 — self-capability audit, maximum 10 minutes

Do this first and continue automatically.

Record to `D:\WorkSpace\WorkBuddy\reports\ETF_M5_5H\00_CAPABILITY_SNAPSHOT.md`:

- Windows version / CPU / RAM / free disk space;
- Python version(s);
- whether pandas/numpy/scipy are usable;
- whether Git CLI is usable;
- whether public GitHub HTTPS/raw URLs can be read;
- whether AKShare can be imported/installed;
- whether a Tushare token already exists in the local environment **without printing the token**;
- whether local TongDaXin/TDX data directories contain usable ETF daily history;
- whether internet HTTP requests work;
- approximate download speed / obvious rate limits;
- candidate working directory.

Choose a working directory under:

`D:\WorkSpace\WorkBuddy\research\ETF_M5_5H_2026-09-12\`

Do not use Desktop, Downloads, D:\Data lifecycle folders, or application install directories for temporary research files.

## Mandatory fallback ladder

Use the highest-quality available source, but never stop for a missing source:

1. existing local daily ETF history if sufficiently complete;
2. existing local Tushare access if already configured;
3. AKShare/public Eastmoney-compatible data;
4. other reliable public daily ETF source;
5. if the full ETF universe cannot be obtained, use the largest defensible representative universe available and label the result `PARTIAL_UNIVERSE`.

Do not ask for credentials.

---

# 3. Universe and data-quality requirements

Target: mainland China exchange-traded funds with daily OHLCV/amount history.

Preferred scope:

- Shanghai + Shenzhen exchange-traded ETFs;
- exclude obvious money-market/cash management funds if they create meaningless low-volatility signals;
- keep broad index, sector/theme, industry, commodity, cross-border, bond ETFs as separate categories where identifiable;
- where classification is uncertain, retain the ETF but mark category `UNKNOWN` rather than guessing.

For each ETF, try to obtain:

- code;
- name;
- list date / delist date if available;
- daily date;
- open/high/low/close;
- volume;
- amount/turnover value if available;
- adjusted-price information if needed/available;
- category or tracked index/theme where reliably available.

Data rules:

- No look-ahead.
- At a signal date, use only information available on or before that date.
- If list/delist dates are available, enforce point-in-time eligibility.
- Do not silently fill missing prices with 0.
- `UNKNOWN != 0`.
- Detect duplicate rows, date gaps, impossible prices, nonpositive prices, and extreme bad ticks.
- Prefer completed daily bars only.

Produce `01_DATA_QUALITY.md` and machine-readable `data_quality.json`.

---

# 4. Baseline strategy A — exact equity-style translation

First test the naive hypothesis: **use the equity-style parameters almost unchanged on ETFs**.

For each ETF/date with enough history:

### A1. ETF relative strength

Because an ETF is already a basket, do not invent a fake "stock inside industry" relationship.

Compute ETF 20-session return percentile against the eligible ETF universe, and separately within its category if a reliable category exists.

Test two context variants:

- `GLOBAL_ETF_RS`: ETF 20d return percentile >= 80%;
- `CATEGORY_ETF_RS`: category median 20d return is in top 20% across categories AND ETF is top 20% within its category.

If category mapping is incomplete, keep GLOBAL_ETF_RS as the minimum comparable baseline.

### A2. Trend

All required:

- SMA20 > SMA60;
- SMA20_today > SMA20_5_sessions_ago.

### A3. Platform breakout

Use prior 15 sessions excluding the signal bar:

- prior15_high;
- prior15_low;
- platform range = prior15_high / prior15_low - 1;
- platform range <= 20%;
- signal close > prior15_high;
- signal volume >= prior15 average volume * 1.50.

Also compute an amount-based version:

- signal amount >= prior15 average amount * 1.50.

### A4. Entry/invalidation research fields

For comparison only:

- entry_trigger = prior15_high;
- max_buy candidate = signal close * 1.01;
- structural invalidation candidate = prior15_low * 0.995;
- signal expiry candidate = 3 sessions.

Do not call these production rules. They are research fields.

---

# 5. Strategy B — ETF-adapted parameter grid

ETFs generally have lower idiosyncratic volatility than individual equities, so test whether tighter platform and lighter volume/amount expansion are more practical.

Run a systematic grid. Minimum dimensions:

### Lookback/platform window

- 10 sessions
- 15 sessions
- 20 sessions

### Maximum platform range

- 8%
- 12%
- 15%
- 20%

### ETF 20d relative-strength percentile

- 70%
- 80%
- 90%

### Breakout confirmation

- close > prior-window high
- optional stricter variant: close >= prior-window high * 1.002

### Volume ratio

- 1.10
- 1.20
- 1.30
- 1.50

### Amount ratio, if amount is available

- 1.10
- 1.20
- 1.30
- 1.50

### Trend filter

Compare:

- T0: no MA trend filter;
- T1: SMA20 > SMA60;
- T2: SMA20 > SMA60 AND SMA20 rising over 5 sessions.

### Liquidity filter

If amount is available, compare at least:

- no minimum;
- 20d median amount >= CNY 20m;
- >= CNY 50m;
- >= CNY 100m.

Do not optimize endlessly. The purpose is to identify broad stable regions, not the single best-looking parameter tuple.

---

# 6. Outcome measurement

For every signal compute, where future data exists:

- forward close-to-close return: 1d, 3d, 5d, 10d, 20d;
- maximum favorable excursion (MFE) over 3/5/10/20d;
- maximum adverse excursion (MAE) over 3/5/10/20d;
- win rate;
- median return;
- mean return;
- 5% trimmed mean;
- signal count;
- unique ETF count;
- unique signal-date count;
- signals per ETF and per date;
- simple turnover estimate.

If feasible, subtract simple round-trip cost scenarios:

- 0.05%
- 0.10%
- 0.20%

Label these assumptions clearly; do not claim precise execution cost.

---

# 7. Robustness tests — mandatory

A promising result is not enough. For all finalists, run:

1. **Remove top 1% winners**.
2. **Remove top 5% winners**.
3. **Winsorize/trim both tails**.
4. **Date-equal weighting** so one hot day cannot dominate.
5. **ETF-equal weighting** so one fund cannot dominate.
6. **Category-equal weighting** where categories are available.
7. Split history into chronological segments, preferably:
   - first 60%;
   - next 20%;
   - final 20% pseudo-out-of-sample.
8. If enough history exists, run walk-forward or rolling validation.
9. Compare bull / sideways / weak-market periods using a simple broad-market regime proxy if readily available. If not available, skip with explicit reason instead of inventing one.
10. Check sensitivity: neighboring parameter values should not collapse immediately.

A candidate is stronger if it stays useful after tail removal and across adjacent parameter values.

---

# 8. Comparison questions that must be answered

The final report must give direct answers to all of these:

1. Does the **exact stock-style 15d / 20% / 1.5x** rule work on ETFs?
2. Is volume or transaction amount more useful for ETF confirmation?
3. Is ETF-wide RS enough, or does category/theme context materially improve it?
4. Which platform width is more appropriate for ETFs: 8/12/15/20%?
5. Which volume/amount expansion region is stable: 1.1/1.2/1.3/1.5x?
6. Is a 15-session window still reasonable?
7. Does MA20>MA60 + rising MA20 add value or merely reduce samples?
8. What is the trade-off between sample count and expected return?
9. Do profits survive after top winners are removed?
10. Is there enough evidence to propose a formal ETF M5 candidate, or should status remain RESEARCH_ONLY?

---

# 9. Candidate selection rule

At the end, choose at most **three** candidates:

- `ETF_M5_CANDIDATE_A` — best robustness / recommended research candidate;
- `ETF_M5_CANDIDATE_B` — conservative alternative;
- `ETF_M5_CANDIDATE_C` — only if genuinely differentiated.

For each candidate report exact parameters and:

- sample count;
- 5d/10d/20d mean;
- medians;
- win rates;
- trimmed mean;
- top-5%-winner-removed result;
- date-equal result;
- ETF-equal result;
- pseudo-OOS result;
- MAE/MFE;
- liquidity characteristics;
- failure modes;
- where it should NOT be used.

If no candidate passes robustness checks, say `NO_ROBUST_ETF_M5_CANDIDATE`.

Do not force a positive conclusion.

---

# 10. Engineering deliverables

Create under:

`D:\WorkSpace\WorkBuddy\research\ETF_M5_5H_2026-09-12\`

Minimum outputs:

- `README.md` — reproduction instructions and result summary;
- `00_CAPABILITY_SNAPSHOT.md`;
- `01_DATA_QUALITY.md`;
- `02_METHOD.md`;
- `03_RESULTS_SUMMARY.md`;
- `04_ROBUSTNESS.md`;
- `05_FINAL_RECOMMENDATION.md`;
- `data_quality.json`;
- `parameter_grid_results.csv` or parquet;
- `signal_level_results.csv` or parquet, if size is reasonable;
- `final_candidates.json`;
- reusable Python scripts under `src\`;
- `run_manifest.json` with timestamps, data sources, package versions, command line, row counts, hashes of key output files, and known limitations.

The scripts must be rerunnable. Avoid notebook-only logic.

Every numeric conclusion in the Markdown final report should be traceable to a machine-readable result.

---

# 11. Continuous-execution policy

Target elapsed work: **about 5 hours**.

Suggested time allocation, but adapt automatically:

- 0:00–0:10 capability audit;
- 0:10–1:10 acquire/normalize ETF data;
- 1:10–2:10 implement baseline and event engine;
- 2:10–3:20 parameter grid;
- 3:20–4:20 robustness / pseudo-OOS;
- 4:20–5:00 analysis, reruns, code cleanup, final report.

If data acquisition takes longer, reduce grid breadth before sacrificing robustness.

If computation finishes early, use remaining time for:

- independent rerun from clean intermediate data;
- parameter-neighbor stability;
- data leakage audit;
- survivorship-bias audit;
- category mapping improvement;
- cost/slippage sensitivity;
- code tests;
- report verification.

Do not idle.

Write a local checkpoint file at least every 45 minutes. Do not email checkpoints unless there is a safety-critical issue. The final email is enough.

---

# 12. Hard safety boundaries

PROHIBITED:

- broker write/order submission;
- modifying private WTOS repositories;
- changing any live trading rule;
- changing account/risk data;
- deleting or moving user files unrelated to this research;
- modifying the ongoing D:\Data file-management system except reading free-space information;
- exposing secrets;
- installing intrusive system software;
- disabling security software;
- rebooting the PC;
- asking the user for confirmation mid-run.

Package installs are allowed only inside an isolated Python environment or user-space environment when needed. If installation is blocked, use another method and continue.

---

# 13. Final email

When complete, send **one** email to the user with subject:

`[ETF M5][5H RESULT] WorkBuddy 独立验证完成`

The body must contain only:

1. `STATUS = COMPLETE / PARTIAL / FAILED_WITH_EVIDENCE`;
2. elapsed time;
3. data source and universe size;
4. strongest conclusion in plain Chinese;
5. baseline stock-style rule result;
6. best ETF-adapted candidate parameters and key numbers;
7. robustness verdict;
8. major limitations/blockers;
9. local output directory;
10. whether you recommend `PROMOTE_FOR_CHATGPT_REVIEW`, `KEEP_RESEARCH_ONLY`, or `REJECT`.

Do not claim any production or live-trading authorization.

---

# 14. Definition of done

The task is DONE only when:

- a real ETF historical dataset was analyzed, not merely discussed;
- the exact equity-style rule was tested;
- an ETF-adapted grid was tested;
- robustness checks were completed;
- outputs are reproducible;
- final candidate(s) or a clear rejection were produced;
- one final email was sent;
- no private WTOS authority/runtime was modified.

**Keep moving forward until the time budget is exhausted or the above definition of done is satisfied with robust evidence.**
