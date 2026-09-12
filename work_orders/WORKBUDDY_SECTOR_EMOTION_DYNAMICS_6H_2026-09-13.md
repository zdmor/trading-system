# WorkBuddy Work Order — Sector Emotion Dynamics 6H

Date: 2026-09-13
Owner: ChatGPT / user-directed
Executor: WorkBuddy local PC
Mode: RESEARCH_ONLY
Public coordination repo: `zdmor/trading-system`

## 0. Hard boundary

This task is **not WTOS Live Authority** and must not modify any private WTOS repository, automation, broker setting, account state, risk rule, or execution permission.

Preserve:

- `RESEARCH != LIVE_AUTHORITY`
- `SECTOR_EMOTION != FORMAL_M3`
- `SIGNAL != TRADE`
- `UNKNOWN != 0`
- no Broker Write

Do not ask the user questions during execution. Do not wait for approval between phases. When a source or package fails, record it and move to the next route.

This is a **continuous deep-work task**. Do not stop after obtaining one attractive result. Use the available time for independent replication, robustness checks, failure analysis, and audit.

---

# 1. Mandatory capability alignment — FIRST, then continue automatically

Before starting research, spend no more than ~20 minutes producing:

`00_CAPABILITY_ALIGNMENT.md`

For each item below mark `AVAILABLE / PARTIAL / UNAVAILABLE`, show one short proof, then choose the best route. **Do not email and wait. Continue immediately.**

1. Python execution and version
2. ability to create isolated venv / install packages
3. pandas / numpy availability or pure-Python fallback
4. PowerShell / bash / git CLI
5. public GitHub read
6. public GitHub write to `zdmor/trading-system`
7. long-running local process
8. local disk space for several GB of research data
9. web search / normal webpage read
10. browser automation / JavaScript page interaction
11. direct HTTP download capability
12. zip/csv/json/parquet handling
13. ability to hash files and create manifests
14. email send/reply and attachment support
15. scheduler / restart-safe checkpoint capability
16. accessible China-market data routes, tested individually:
   - Sina
   - Eastmoney
   - Tencent
   - NetEase or other public endpoints
   - AKShare if installable
   - Tushare only if an actually available token exists; do not assume one
   - any public industry-index source you can verify
17. whether historical industry/sector index OHLCV/amount is available
18. whether historical constituent membership is available
19. whether historical broad-market benchmark data is available

At the end of the capability file write:

- `PRIMARY_DATA_ROUTE`
- `SECONDARY_REPLICATION_ROUTE`
- `EXPECTED_LIMITATIONS`
- `NO_USER_INPUT_REQUIRED = TRUE`

Then **immediately start Phase 2**.

---

# 2. Research question

We already know a static 0–100 sector-emotion score is useful as a **current heat description**, but a first short-history study did **not** show that “higher static score = higher future return”.

The new research question is:

> Is the **change of sector emotion** more useful than the static level?

Specifically test whether states such as:

- `HEATING` — warming rapidly from medium level
- `STRENGTHENING` — already strong and still improving
- `HOT_PERSISTENT` — high level, stable, participation still healthy
- `OVERHEATED_DIVERGENCE` — high score but participation / candle / relative-strength deterioration
- `COOLING` — still elevated but falling quickly
- `RETREAT` — weak and deteriorating
- `RECOVERY` — rebound from weak/cooling state
- `NEUTRAL`

have different **future 5/10/20/40-session relative returns**.

The task is not to make a pretty heat score. The task is to determine whether a dynamic state has **incremental predictive / decision-support value** beyond the static score.

---

# 3. Data requirement

## 3.1 Target history

Target:

- minimum acceptable: 3 years daily data
- preferred: >=5 years
- ideal: 8–10 years if a trustworthy source is available

Do not shorten history merely because a fast endpoint returns less.

## 3.2 Sector universe

Preferred order:

1. Shenwan L1 sector indices with stable identity/history
2. another well-defined first-level China sector taxonomy if SW data is not publicly obtainable
3. a reproducible sector proxy universe constructed from liquid sector ETFs only as a fallback, clearly labeled proxy

Do not silently mix incompatible taxonomies.

## 3.3 Benchmark

Use a broad A-share benchmark when available. If not, build and document a reproducible broad-market proxy.

## 3.4 Point-in-time discipline

If constituent-level reconstruction is used, do not apply current constituents to historical dates unless historical membership is unavailable and the result is explicitly labeled `SURVIVORSHIP_RISK`.

Static sector-index history is preferred because it avoids a large part of membership reconstruction complexity.

## 3.5 Data fields

Try to obtain at least:

- date
- open
- high
- low
- close
- volume
- amount / turnover amount if available

If true amount is unavailable, keep `amount = UNKNOWN`; do not call `volume * price` true amount. A proxy may be separately named `turnover_proxy`.

Persist raw downloaded data locally. Do not push large raw datasets to GitHub. Record hashes and source URLs/identifiers.

---

# 4. Self-contained V0 six-dimension definition

This is a research specification, not a live rule. Implement the following reproducibly and keep each raw component before percentile normalization.

## 4.1 Relative Strength

Use sector return minus benchmark return over:

- 1d
- 3d
- 5d
- 10d
- 20d

Start with equal weights. Also test one recency-weighted variant as a robustness comparison, but do not optimize dozens of weight sets.

## 4.2 Trend Momentum

Combine only information available at date t:

- sector 5d return
- sector 20d return
- close > MA5
- close > MA20
- MA5 slope
- MA20 slope

## 4.3 Volume / Price Confirmation

Preferred with real amount:

- amount / trailing 20d median amount
- volume / trailing 20d median volume
- signed by price direction and relative-performance direction

Desired interpretation:

- rising + outperforming + participation expansion = positive
- falling + participation expansion = negative
- weak price with contracting participation = less negative than panic expansion

If amount unavailable, compute a separate volume-only version and label it.

## 4.4 Candlestick Emotion

From same-day OHLC only:

- close-location value within high-low range
- real body / range
- upper-wick / range
- lower-wick / range
- direction of body

Guard zero-range days explicitly.

## 4.5 Directional Volatility

Measure realized range / ATR-style volatility relative to trailing normal level, but directionally sign it:

- expanding volatility with positive return / outperformance = positive attack
- expanding volatility with negative return / underperformance = negative panic/distribution
- low volatility is not automatically good or bad

## 4.6 Persistence

Use only trailing information:

- number of days outperforming benchmark in last 5 and 10 sessions
- number of days close > MA5 in last 5 and 10
- distance to 20d high
- optionally consecutive outperformance streak, capped to avoid domination

## 4.7 Normalization

Primary V0:

- same-date cross-sector percentile, 0–100
- static total = equal-weight average of six dimensions

Do not optimize weights in the full sample.

---

# 5. Dynamic variables

For each sector-date compute:

- `S_t` static six-dimension score
- `D1 = S_t - S_t-1`
- `D3 = S_t - S_t-3`
- `D5 = S_t - S_t-5`
- 5-day score slope
- score acceleration: recent slope minus prior slope
- drawdown from trailing 10d maximum score
- change in relative-strength dimension over 3/5d
- change in volume-price dimension over 3/5d
- change in persistence dimension over 3/5d
- divergence flags, e.g. score high but volume-price falling, score high but RS falling

Do not use future observations.

---

# 6. Candidate state families

Do **not** hard-code one magic state definition on the full sample.

Use a bounded candidate family, then freeze definitions on training data.

Suggested state concepts:

### HEATING
Medium score, strong positive D3/D5, positive RS change.

### STRENGTHENING
Score already above median, still rising, persistence non-negative, participation not deteriorating.

### HOT_PERSISTENT
High score, neither collapsing nor accelerating excessively, volume-price healthy, persistence high.

### OVERHEATED_DIVERGENCE
High score but one or more of:

- D3/D5 negative
- volume-price deterioration
- RS deterioration
- large drawdown from recent score high
- strong upper-wick / weak close-location deterioration if available

### COOLING
Previously high/strong score, now falling quickly.

### RETREAT
Below-median score plus negative slope and weak relative strength.

### RECOVERY
Prior COOLING/RETREAT state followed by strong positive D3/D5 and restored RS/participation.

### NEUTRAL
None of the above.

Threshold candidates should use a **small interpretable grid**, for example score levels around 40/50/60/70/80 and changes around 5/10/15 points. Do not brute-force hundreds of arbitrary thresholds.

---

# 7. Train / validation / test discipline

This is mandatory.

Chronologically split history:

- first 50% = TRAIN
- next 25% = VALIDATION
- last 25% = TEST

Alternative rolling walk-forward is encouraged as an additional check.

Rules:

1. Candidate state thresholds may be compared on TRAIN.
2. Choose at most 2–3 definitions per state family.
3. Freeze before TEST.
4. Do not retune after seeing TEST.
5. If you discover a bug, fix it and rerun all splits; document the bug.

Output a frozen-state definition file before final test results are generated if practical.

---

# 8. Main evaluation

For every static score bucket and every dynamic state, calculate future sector **relative return vs benchmark** at:

- 5 sessions
- 10 sessions
- 20 sessions
- 40 sessions

Report:

- sample count
- mean
- median
- win rate
- standard deviation
- 25/75 percentile
- max adverse future relative return where feasible
- max favorable future relative return where feasible
- bootstrap 95% CI for mean and median where computationally practical

Also calculate:

- state-to-state transition matrix
- probability that HEATING -> STRENGTHENING / HOT_PERSISTENT
- probability that HOT_PERSISTENT -> COOLING
- probability that OVERHEATED_DIVERGENCE precedes underperformance

---

# 9. Critical incremental-value tests

These are more important than finding the highest raw return.

## 9.1 Dynamic vs static

Compare:

- static score only
- score change only
- static + dynamic state

Question: does dynamic information improve separation beyond static heat level?

## 9.2 Matched-level test

Within similar static-score bands (e.g. 60–70, 70–80, 80–90):

- compare rising score vs flat score vs falling score

This directly tests whether `70 rising` is different from `70 falling`.

## 9.3 50->70 versus 80+

Explicitly test the user's practical intuition:

- medium-to-high rapid heating
- already very hot

Which has better next 10/20d relative return?

## 9.4 Overheat divergence

Within score >=75 or >=80, compare:

- participation/RS healthy
- participation/RS deteriorating

Question: can divergence distinguish healthy leadership from exhaustion?

## 9.5 Ablation

Test removing:

- candlestick emotion
- directional volatility
- volume-price

We particularly want to know whether candlestick emotion contributes anything after other features are known.

---

# 10. Robustness requirements

Do not stop at one aggregate table.

Run as many of the following as available time/data permits:

1. yearly breakdown
2. bull / range / weak market regimes
3. high-vol / low-vol regimes
4. early history vs recent history
5. alternate broad benchmark
6. secondary data source replication for a subset
7. leave-one-sector-out sensitivity
8. winsorized and median-based results
9. remove the best 1% and 5% future-return observations
10. date-equal-weight results so one period cannot dominate
11. sector-equal-weight results
12. bootstrap resampling by date block, not only individual rows
13. 20–30 failure-case review: states that looked strongest but failed badly

---

# 11. Time / work-depth rule

This task should use approximately **6 hours of useful continuous work**, not 10 minutes of calculation followed by an early stop.

Do not intentionally sleep or idle merely to consume wall-clock time. Instead:

- if core research finishes early, run secondary-source replication;
- extend history if possible;
- add walk-forward checks;
- inspect data anomalies;
- review failure cases;
- independently rerun the pipeline from a clean directory;
- compare pure-Python and pandas path if both exist;
- audit future-data leakage;
- test state-threshold neighborhood stability;
- improve reproducibility and documentation.

Do not ask questions mid-task. If one avenue is blocked, keep moving.

Checkpoint locally every ~45–60 minutes with a timestamped file, but **do not stop for our response**.

---

# 12. Local output package

Create a local directory such as:

`D:\WorkSpace\WorkBuddy\research\SECTOR_EMOTION_DYNAMICS_6H_2026-09-13\`

Required outputs:

- `00_CAPABILITY_ALIGNMENT.md`
- `01_DATA_SOURCE_AND_UNIVERSE.md`
- `02_DATA_QUALITY.json`
- `03_STATIC_SCORE_SPEC.md`
- `04_DYNAMIC_STATE_SPEC.md`
- `05_FROZEN_STATE_DEFINITIONS.json`
- `06_MAIN_RESULTS.csv`
- `07_TRANSITION_MATRIX.csv`
- `08_MATCHED_LEVEL_TEST.csv`
- `09_ABLATION_RESULTS.csv`
- `10_WALK_FORWARD_OR_TIME_SPLIT.md`
- `11_ROBUSTNESS.md`
- `12_FAILURE_CASES.md`
- `13_FINAL_RECOMMENDATION.md`
- `run_manifest.json`
- source code / scripts
- file hashes / data-source references

Raw large datasets may stay local; include hashes and row/date counts.

---

# 13. Public GitHub evidence package

You have public write access to `zdmor/trading-system`.

After completion, upload the **reviewable evidence package** under a path such as:

`research/workbuddy/sector_emotion_dynamics_2026-09-13/`

Upload:

- code
- specs
- small/medium result CSV/JSON/MD
- manifest
- hashes
- final recommendation

Do not upload secrets, tokens, private-user data, massive raw market datasets, or local-machine confidential files.

If GitHub write fails, keep the full package locally and send the exact local path + hashes in the final email.

---

# 14. Final decision format

The final recommendation must choose exactly one primary verdict:

- `PROMOTE_FOR_CHATGPT_REVIEW`
- `OBSERVE_MORE`
- `REJECT_DYNAMIC_SECTOR_STATE`
- `PARTIAL_DATA_INSUFFICIENT`

And answer directly:

1. Is dynamic sector emotion better than static sector heat?
2. Which state has the strongest evidence?
3. Does HEATING outperform already-HOT sectors?
4. Can OVERHEATED_DIVERGENCE identify exhaustion?
5. Does candlestick emotion add value?
6. Does directional volatility add value?
7. Does volume-price confirmation remain useful?
8. What is the independent TEST-period result?
9. What are the three largest data/method limitations?
10. What, if anything, is good enough to propose for WTOS M3 review?

No claim may exceed the tested evidence.

---

# 15. Final email

When the task is complete, send one final email to the user/ChatGPT with subject:

`[M3 SECTOR DYNAMICS][6H RESULT] WorkBuddy 独立验证完成`

Include:

- capability alignment summary
- elapsed time and major phases
- data source/universe/history
- main test-period results
- dynamic vs static conclusion
- robustness conclusion
- limitations
- local output path
- public GitHub evidence path/commit if successful
- exact recommended next step

Do not modify private WTOS / Live Authority. End of work order.
