# Smart Kiosk — QSR MCP Surface

Exposes kiosk-core to the Central QSR Agent (retail-use-cases#100/#103,
`qsr-design`) via `mcp_service_sdk`, following the same pattern as Order
Accuracy's dine-in/take-away MCP services. This is a **separate** MCP
surface from `kiosk_core/ordering/mcp_server.py`'s `/mcp` mount, which is
only for the internal rag-service ordering agent.

## Module layout

| File | Responsibility |
|---|---|
| `kiosk_core/qsr_mcp/mcp_service.py` | `ServiceServer` config, event types, read/act tool implementations |
| `kiosk_core/qsr_mcp/mcp_server.py` | Runs the MCP app + queue-depth sampling poller on background threads |
| `kiosk_core/qsr_mcp/board_state.py` | Durable (JSON-file) kiosk menu-board mode: `full` / `simplified` |
| `kiosk_core/queue_client.py` | HTTP client to the standalone `queue-service` |

Started from `main.py`'s lifespan when `KIOSK_CORE_QSR_MCP_ENABLED=true`
(default). Listens on `KIOSK_CORE_QSR_MCP_PORT` (default `8014`,
streamable-http transport) — distinct from the ordering feature's `/mcp`.

## Contract (`describe`)

**Event types**
- `queue_depth_sample` — `{count, nearby, status}`, emitted every
  `KIOSK_CORE_QSR_MCP_QUEUE_POLL_SECONDS` (default 30s) by polling
  queue-service.
- `board_mode_changed` — `{mode, reason}`, emitted whenever `set_board_mode`
  runs.

**Read tools**
- `get_queue_depth()` — current count/status from queue-service.
- `get_queue_history(period)` — sample count/avg/max for
  `today|yesterday|all|YYYY-MM-DD`.
- `get_order_stats(period, top_n)` — confirmed order count, revenue, top
  items, from the existing ordering SQLite DB.
- `get_board_state()` — current board mode and when/why it last changed.

**Act tool** (gate: `AUTOMATIC`, rate-limited to 20 calls/60s)
- `set_board_mode(mode, reason=None)` — `mode` is `"full"` or `"simplified"`.
  Implements qsr-design's autonomous action #6: "Kiosk queue deep/short ->
  Kiosk board simplify".

## Configuration

See `kiosk_core/config.py` (`QSR_MCP_*` constants) / `.env.example` for all
`KIOSK_CORE_QSR_MCP_*` environment variables.

## Notes / follow-ups

- Tool names/schemas were derived from `qsr-design` and the Order Accuracy
  reference implementation, since retail-use-cases#100/#103 were not
  reachable from this environment (org SAML SSO). Revisit against the ticket
  text once accessible.
- `set_board_mode` only flips a backend flag today; wiring kiosk-ui to
  visibly react to `board_mode` is a follow-up.
