# WTOS Local Bridge Bootstrap — Supabase Listener + Read-Only Production Repo Access

Status: USER_APPROVED
Date: 2026-09-14
Mode: BOOTSTRAP_ONLY / NO_PRODUCTION_CUTOVER / NO_BROKER_WRITE

## Current truth

Resolve Current first:
1. `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`
2. Follow redirect to `zdmor/Stock-Analysis@main:SYSTEM_MANIFEST.yaml`

Current user-facing WTOS remains `WTOS_CHATGPT` in `zdmor/Stock-Analysis`.
Do not modify production runtime, Live Authority, risk policy, formal M5, ChatGPT automations, GitHub production triggers, or Broker Write.

## Objective

Bootstrap the preferred bridge architecture:

`ChatGPT <-> Supabase <-> local Windows agent`

and separately prepare read-only access to:

`zdmor/Stock-Analysis`

for local shadow execution.

Supabase is communication only. GitHub remains Current / durable WTOS state.

## A. Restore local Supabase listener

Locate the existing local listener / CatPaw worker previously used with Supabase project `chatgpt-workbuddy-queue` and table `public.wtos_ai_messages`.

Do not recreate infrastructure unless the existing implementation is genuinely unavailable.

Target worker identity previously used: `wtos-newpc-01`.

Restore the listener so that it can:
- poll/claim messages addressed `recipient=catpaw`;
- emit ACK and RESULT rows back to `recipient=chatgpt`;
- preserve existing auth/secrets locally;
- never print or commit Supabase credentials;
- run without modifying WTOS production.

A queued test already exists:

`TEST_ID = SUPABASE_BRIDGE_TEST_20260914_001`

When the listener is restored, process that message normally. Expected RESULT payload includes:
- `TEST_ID=SUPABASE_BRIDGE_TEST_20260914_001`
- `reply_text=SUPABASE_BRIDGE_PASS`
- `semantic_processed=true`
- `HOSTNAME`
- `WORKER_ID`
- `MANUAL_INTERVENTION=false`

Do not execute trading or WTOS mutation as part of this test.

## B. Stock-Analysis credential boundary

The current local fine-grained PAT does not have access to the private production repository.

Do NOT request broad token permissions.
Do NOT add write access yet.
Do NOT add Actions, Administration, Secrets, Variables, Webhooks, or organization-wide permissions.

Desired first-stage credential scope is only:
- repository: `zdmor/Stock-Analysis`
- Metadata: Read-only
- Contents: Read-only

If this read-only credential has not yet been granted by the user, report `PRIVATE_REPO_READ_ACCESS=BLOCKED_WAITING_USER` and stop that branch of work cleanly.

Once available, verify without exposing the token:
- `gh repo view zdmor/Stock-Analysis`
- authenticated `git ls-remote`
- clone/fetch to a separate local shadow checkout
- read `SYSTEM_MANIFEST.yaml`

Do not push anything to Stock-Analysis in this phase.

## C. No public-repo write dependency

`zdmor/trading-system` is now bootstrap/backup communication only.
Do not require local write permission to this public repository.
Do not treat inability to push here as a production blocker.

Primary live communication target is Supabase `wtos_ai_messages`.

## D. Report result through Supabase

Once listener recovery is complete, return a sanitized RESULT to ChatGPT through Supabase with:

- `TEST_ID=LOCAL_BRIDGE_BOOTSTRAP_20260914_001`
- `SUPABASE_LISTENER=PASS|PARTIAL|BLOCKED`
- `SUPABASE_TEST_MESSAGE_PROCESSED=YES|NO`
- `WORKER_ID`
- `HOSTNAME`
- `PRIVATE_REPO_READ_ACCESS=PASS|BLOCKED_WAITING_USER|BLOCKED_OTHER`
- `LOCAL_STOCK_ANALYSIS_CHECKOUT=YES|NO`
- `PRODUCTION_CHANGED=NO`
- `BROKER_WRITE=false`
- `BLOCKERS`
- `NEXT_ACTION`

No secrets, tokens, account state, or private runtime payloads in the RESULT.

## Safety invariants

- `LOCAL_RESULT != PRODUCTION_RESULT`
- `SUPABASE != LIVE_AUTHORITY`
- `PROJECT != RUNTIME != LIVE_AUTHORITY`
- `UNKNOWN != 0`
- `Signal != Trade`
- `PROPOSED_RESULT != BROKER_ORDER != BROKER_FILL`
- `broker_write=false`

Do not cut over production.