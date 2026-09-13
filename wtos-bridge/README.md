# WTOS Public Bridge Mailbox

Purpose: low-sensitivity communication relay between ChatGPT and the user's local Windows agent.

- `inbox/` — sanitized tasks from ChatGPT to the local agent.
- `status/` — sanitized replies from the local agent.

## Security boundary

This repository is PUBLIC. Never commit:

- Tushare tokens
- GitHub PATs/tokens
- account positions, cash, cost basis, or broker data
- private WTOS runtime bundles
- personal identifiers
- secrets of any kind

Formal WTOS durable state and Live Authority do NOT move here. Current WTOS remains resolved from `zdmor/WTOS@main:SYSTEM_MANIFEST.yaml` and then `zdmor/Stock-Analysis@main:SYSTEM_MANIFEST.yaml`.

This mailbox is communication only, not Source of Truth and not Live Authority.
