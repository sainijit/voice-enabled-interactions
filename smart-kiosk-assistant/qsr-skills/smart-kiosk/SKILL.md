---
name: smart-kiosk
description: "Answer Smart Kiosk queue-depth, queue-trend, order-activity/revenue, and menu-board-state questions, and safely simplify or restore the on-screen menu board through the smart_kiosk MCP service."
version: 1.0.0
platforms: [linux]
metadata:
  hermes:
    tags: [QSR, Kiosk, Queue, Menu-Board, Ordering]
---

# Smart Kiosk

Use the `smart_kiosk` MCP service as the only source of live kiosk queue,
order-activity, and menu-board facts. Never answer a kiosk operations question
from memory or an earlier turn when a fresh tool call is available.

## Contract Discovery

Before the first `smart_kiosk` tool call in a conversation, call `describe`.
Use the returned contract to identify the current read tools, action tools,
input schemas, event types, and action gates — tool availability can change as
the service evolves. Do not use `describe` as evidence for a live fact.

Map each request to the required capability below, then choose the compatible
tool from `describe`. If no compatible tool is available, explain that the
service contract does not support the request rather than guessing.

## Tool Routing

| User intent or question | Tool | Fields to use | Answer rule |
|---|---|---|---|
| How deep is the queue right now / how many people are waiting | `get_queue_depth` | `available`, `count`, `nearby`, `status`, `timestamp` | State `count` (and `nearby` if non-zero) and `status` (LOW/MEDIUM/HIGH). If `available` is false, say the queue sensor is unreachable rather than guessing a number. |
| Is the queue always deep at this time / queue trend for today, yesterday, or a date | `get_queue_history` | `period`, `samples`, `avg_count`, `max_count` | State the period used, the sample count, and both the average and max depth. A single call answers one period; call again for a second period rather than inferring a comparison. |
| Orders today/yesterday/on a date, revenue, best sellers | `get_order_stats` | `period`, `orders`, `revenue`, `top_items` | Report `orders` and `revenue` for the stated `period`; list `top_items` in the returned order (already ranked by quantity) and do not re-sort or invent items. |
| What's on the menu board right now / is the board simplified | `get_board_state` | `mode`, `reason`, `updated_at` | State `mode` (`full` or `simplified`); include `reason` and `updated_at` only if present. |
| Any other kiosk queue, order, or board-state analysis | Best compatible read tool from `describe` | All relevant returned fields | Call once, reason over the result, show brief counts/arithmetic, and state missing evidence instead of guessing. |

## Complex Questions

For trends, comparisons, or open-ended questions, call the narrowest read tool
that returns the needed data, then reason over that single result. Use only
fields present in the response and distinguish observed facts from
recommendations. A single `get_queue_depth` snapshot does not prove a trend —
use `get_queue_history` for trend/frequency questions instead of guessing from
one sample.

## Board Actions

`set_board_mode` is the only actuator this service exposes. It simplifies or
restores the kiosk's on-screen menu board (qsr-design action #6: deep queue ->
simplify board) and is policy-gated `automatic` (rate-limited to 20 calls per
60 seconds) — it executes immediately once called, with no separate human
approval step. A diagnostic question alone never authorizes a call.

1. Call `get_queue_depth` and/or `get_board_state` to confirm the current
   queue status and board mode before acting.
2. Call `set_board_mode` only when the user explicitly asks to simplify or
   restore the board, or when acting on a clear, stated trigger (for example,
   "the queue is deep, simplify the board") that the user asked you to act on.
3. Pass `mode` as exactly `full` or `simplified`, and include a short `reason`
   describing why (for example, the queue depth/status that triggered it).
4. Report the tool result's `mode`, `reason`, and `updated_at` back to the
   user as confirmation of what changed.

Never infer authorization to change the board from a diagnostic question or
from urgency alone.

## Owner Extension

Service owners should add specialized question mappings to the Tool Routing
table as new read/act tools are added to
[`kiosk_core/qsr_mcp/mcp_service.py`](../../kiosk_core/qsr_mcp/mcp_service.py).
See [`docs/qsr-mcp.md`](../../docs/qsr-mcp.md) for the full tool/event
contract and keep live values out of this file.
