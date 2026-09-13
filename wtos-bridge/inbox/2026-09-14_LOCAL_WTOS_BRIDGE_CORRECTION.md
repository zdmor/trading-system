# CORRECTION — Local WTOS Bridge Audit

Status: REQUIRED_CORRECTION
Date: 2026-09-14
Mode: READ_ONLY / NO_PRODUCTION_CHANGE

Your prior audit contains false blockers caused by resolving the wrong repository / local checkout.

## Authoritative correction

1. `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml` explicitly redirects Current WTOS-CHATGPT to:
   `zdmor/Stock-Analysis@main:SYSTEM_MANIFEST.yaml`.

2. `zdmor/Stock-Analysis` DOES exist. It is a private repository. If your local tooling reports it as nonexistent, treat that as an authentication/visibility/access problem, not repository absence.

3. The Current production workflows are intentionally NOT under `zdmor/WTOS@main`. They are under `zdmor/Stock-Analysis@main`, including at least:
   - `.github/workflows/wtos-compute.yml`
   - `.github/workflows/wtos-history-cache.yml`
   - `.github/workflows/wtos-full-market-technical-scan.yml`

4. Current deterministic compute code also exists under `zdmor/Stock-Analysis`, including:
   - `wtos_compute/runner.py`
   - other files referenced by the workflows

Do not infer missing Current implementation from the paused WTOS_LOCAL repository.

## Required next actions

A. Verify authenticated GitHub access locally:
- `gh auth status`
- `gh repo view zdmor/Stock-Analysis`

B. If accessible, clone or fetch the private Current repository into a separate local path. Do not overwrite the paused WTOS_LOCAL checkout.

C. Restart the audit from:
1. `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml`
2. follow redirect to `zdmor/Stock-Analysis@main:SYSTEM_MANIFEST.yaml`
3. inspect only task-required Current files

D. Re-evaluate local feasibility against the actual Stock-Analysis implementation.

E. Do not change production, Actions triggers, Live Authority, M5, risk policy, ChatGPT automations, or Broker Write.

F. Publish the sanitized final result to:
`wtos-bridge/status/LOCAL_WTOS_BRIDGE_STATUS.md`

The status file must distinguish:
- REPOSITORY_ABSENT
- LOCAL_CHECKOUT_ABSENT
- AUTHENTICATION_BLOCKED
- PRIVATE_REPO_ACCESS_BLOCKED

Do not collapse these into one blocker.

## Acceptance rule

The bridge audit is not complete until the Current private repository has either:
- been successfully accessed and audited locally, or
- been shown inaccessible with explicit authenticated access evidence.

Do not report `Stock-Analysis does not exist` unless GitHub itself returns that conclusion under valid authenticated access.

No secrets or private runtime payloads may be committed to this public relay.
