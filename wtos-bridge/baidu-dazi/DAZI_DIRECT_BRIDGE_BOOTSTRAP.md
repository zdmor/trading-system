# Baidu Dazi Direct Bridge Bootstrap

Status: USER_APPROVED_BOOTSTRAP
Mode: COMMUNICATION_INFRASTRUCTURE_ONLY / NO_PRODUCTION_CHANGE
Worker ID: `baidu-dazi-01`

## Objective

Establish a persistent direct communication bridge:

`ChatGPT <-> Supabase <-> Baidu Dazi local agent`

After bootstrap, the user must not need to relay prompts or results manually.

This bridge is communication infrastructure only. It does not change WTOS Runtime, Live Authority, M5, Risk Policy, Automation, or Broker Write.

## Existing local prerequisites

Reuse the existing local environment variables already used by the WTOS/CatPaw Supabase channel. Do not print or commit their values:

- `WTOS_SUPABASE_URL`
- `WTOS_SUPABASE_KEY`
- `WTOS_WORKER_TOKEN`

Do not create a new plaintext credential if these are present and valid.

## Supabase project

Existing project: `chatgpt-workbuddy-queue`

Dedicated Dazi transport has already been created server-side.

Worker identity:

`baidu-dazi-01`

Dedicated RPCs:

- `claim_dazi_bridge_message_worker_v1`
- `post_dazi_bridge_message_worker_v1`

Existing generic heartbeat RPC may be reused:

- `heartbeat_workbuddy_worker_v1`

The dedicated queue table is server-managed:

- `dazi_bridge_messages`

Do not require direct table SELECT/UPDATE access. Use RPCs.

## Required behavior

Create a persistent local bridge owned by Baidu Dazi, not CatPaw and not WorkBuddy.

The bridge must:

1. Start automatically or remain persistently running on the always-on Windows machine.
2. Poll at a reasonable cadence, approximately every 30-60 seconds.
3. Call `claim_dazi_bridge_message_worker_v1` with:
   - `p_worker_id = baidu-dazi-01`
   - `p_worker_token = value from WTOS_WORKER_TOKEN`
   - a reasonable lease such as 300 seconds.
4. When a message is claimed, send an ACK through `post_dazi_bridge_message_worker_v1`.
5. Pass the claimed task to the real Baidu Dazi agent execution context, not a lightweight fixed semantic responder.
6. Execute within the task's stated permissions/boundaries.
7. Return RESULT / QUESTION / STATUS through `post_dazi_bridge_message_worker_v1`.
8. Preserve correlation_id and parent_message_id.
9. Never expose secrets in logs, prompts, Git commits, result payloads, screenshots, or status files.
10. Maintain fail-closed behavior when a task asks for permissions outside its allowed boundary.

## Critical implementation question

The transport loop alone is not enough. It must connect to the actual Baidu Dazi agent execution mechanism.

Use whatever officially supported/local mechanism is available in the current Baidu Dazi environment, for example a persistent agent task, supported local automation interface, workspace watcher, command interface, or other agent-native mechanism.

Do not fake autonomous execution with a fixed rule-based responder.

If Baidu Dazi has no supported way to accept a task programmatically after the user leaves the conversation, return:

`DAZI_AUTONOMOUS_EXECUTION_INTERFACE = BLOCKED`

and explain the exact limitation. Do not claim the direct bridge is complete.

## First queued acceptance test

A server-side test message is already waiting:

`DAZI_BRIDGE_BOOTSTRAP_20260914_001`

After the persistent bridge is really operational, it should automatically claim that message without the user copying its content.

Return RESULT containing at least:

- `TEST_ID = DAZI_BRIDGE_BOOTSTRAP_20260914_001`
- `reply_text = DAZI_BRIDGE_PASS`
- `WORKER_ID = baidu-dazi-01`
- `autonomous_bridge = true`
- `manual_intervention = false`
- `persistent_listener = true`
- `agent_executor = BAIDU_DAZI`
- `broker_write = false`

## WTOS safety boundaries

For bootstrap and acceptance testing:

- Do not modify `zdmor/Stock-Analysis` production files.
- Do not modify GitHub Actions production schedules/triggers.
- Do not modify ChatGPT automations.
- Do not modify Live Authority.
- Do not modify formal M5 method.
- Do not modify Risk Policy.
- Do not enable Broker Write.
- Do not submit any broker order.

`COMMUNICATION_PASS != WTOS_PRODUCTION_CUTOVER`

## Completion condition

Bootstrap is complete only when ChatGPT can independently observe the queued test message changing to completed and can read a Dazi-generated RESULT through Supabase, with no user relay of task content or result content.
