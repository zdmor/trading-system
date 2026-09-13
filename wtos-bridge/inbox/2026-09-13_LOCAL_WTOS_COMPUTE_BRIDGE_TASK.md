# TASK — Prepare WTOS Local Compute Bridge

Status: REQUESTED_BY_USER
Date: 2026-09-13
Target: user's always-on Windows PC / local AI agent
Mode: READ_FIRST / DESIGN_AND_PREPARE_ONLY / NO_PRODUCTION_CUTOVER

## Objective

Prepare a local-compute path that can eventually replace high-cost GitHub Actions computation while preserving uninterrupted WTOS-1 / WTOS-2 production.

Preferred architecture:

`LOCAL WINDOWS PC -> WTOS deterministic compute -> GitHub state/result files -> ChatGPT WTOS consumer`

Do not try to directly control ChatGPT through an unofficial private API. GitHub is the communication/state bridge.

## 1. Resolve Current first

Before doing any work, read Current exactly in this order:

1. `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`
2. Follow the resolver to `zdmor/Stock-Analysis@main:SYSTEM_MANIFEST.yaml`
3. Read only task-required Current files.

At minimum inspect:

- `runtime/chatgpt/COMPUTE_ORCHESTRATION.yaml`
- `.github/workflows/wtos-compute.yml`
- `.github/workflows/wtos-history-cache.yml`
- `.github/workflows/wtos-full-market-technical-scan.yml`
- WTOS-2 runtime/workflow only if needed for local bridge design

Do not infer Current from old chats, memory, frozen material, old handoffs, or filename recency.

## 2. Hard boundaries

Preserve all Current boundaries:

- `PROJECT != RUNTIME != LIVE_AUTHORITY`
- `RESEARCH != LIVE_AUTHORITY`
- `UNKNOWN != 0`
- `Signal != Trade`
- `PROPOSED_RESULT != BROKER_ORDER != BROKER_FILL`
- `broker_write = false`

Do NOT modify:

- Live Authority
- formal M5 method
- risk policy
- Broker Write
- ChatGPT Scheduled Tasks
- WTOS-1 / WTOS-2 trading logic
- current GitHub production triggers

Do NOT cut over production yet.

## 3. Local feasibility audit

On the Windows PC, determine whether the existing Stock-Analysis code can run locally with equivalent semantics to GitHub Actions.

Check:

- local repository checkout/update path
- Python version/environment
- dependency installation
- secure local Tushare token availability
- market collection
- Data Quality Gate
- rolling 150-session history refresh
- full-market technical scan
- WTOS compute bundle
- M4/M5 deterministic feature generation
- WTOS-2 intraday producer feasibility if applicable
- local validation of produced JSON
- ability to publish results to GitHub without unintentionally triggering expensive Actions

Do not expose any credential or private runtime payload in this PUBLIC repository.

## 4. Bridge contract

Prefer reusing Current WTOS schemas/paths so ChatGPT does not care whether the producer is GitHub Actions or local Windows.

Relevant production surfaces include:

- `wtos-runtime:latest/wtos_compute.json`
- `wtos-runtime:latest/market_risk_panel.json`
- `wtos-runtime:latest/full_market_technical_scan.json`
- `wtos-runtime:latest/m5_mainline_breakout.json`
- `wtos-runtime:latest/wtos2.json`

For the first stage, do NOT overwrite production outputs. If local execution is feasible, use a shadow namespace first, e.g.:

- `local-shadow/latest/wtos_compute.json`
- `local-shadow/latest/full_market_technical_scan.json`
- `local-shadow/latest/m5_mainline_breakout.json`
- `local-shadow/latest/wtos2.json`

Every local result should expose at least:

- producer identity
- generated_at
- market_fact_as_of / trade date
- Stock-Analysis source commit SHA
- data-quality status
- validation status
- `broker_write=false`
- explicit statement that local result is NOT Live Authority by itself

## 5. Parallel validation plan

Prepare for this sequence only; do not cut over without explicit user approval:

1. Keep GitHub Actions production unchanged.
2. Run local producer in parallel for 3–5 trading days.
3. Compare local vs GitHub output for the same decision windows.
4. Record schema parity, numeric/value parity, freshness, timing and failures.
5. Propose cutover only after parity evidence is sufficient.

`LOCAL_RESULT != PRODUCTION_RESULT` until formally accepted.

## 6. Public mailbox reply

This public repository is communication only, not WTOS Source of Truth.

When finished with this audit/preparation, create or update:

`wtos-bridge/status/LOCAL_WTOS_BRIDGE_STATUS.md`

Use exactly this compact structure:

```text
STATUS: PASS | PARTIAL | BLOCKED
MACHINE_READY: YES | NO | PARTIAL
REPO_READY: YES | NO | PARTIAL
PYTHON_READY: YES | NO | PARTIAL
DEPENDENCIES_READY: YES | NO | PARTIAL
DATA_SOURCE_READY: YES | NO | PARTIAL
LOCAL_COMPUTE_READY: YES | NO | PARTIAL
GITHUB_PUBLISH_READY: YES | NO | PARTIAL
SECRET_HANDLING_SAFE: YES | NO | PARTIAL
UNINTENDED_ACTION_TRIGGER_RISK: LOW | MEDIUM | HIGH | UNKNOWN
PARITY_TEST_READY: YES | NO | PARTIAL
BLOCKERS:
- ...
NEXT_ACTION:
- ...
```

Do not include secrets, private account data, or private WTOS runtime contents.

## 7. User priorities

1. Daily WTOS-1 / WTOS-2 production must not be interrupted.
2. This month Actions usage should be controlled, but production takes priority over zero cost.
3. Next month, the 2000 included Actions minutes must be explicitly budgeted with reserve.
4. High-cost deterministic computation is a candidate to move local only after parity validation.

## Immediate action

Start the local feasibility audit and bridge preparation now.

Do NOT disable current GitHub production.
Do NOT change Live Authority or Runtime.
Do NOT place secrets in this public mailbox.
