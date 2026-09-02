# Smart AI Kiosk — ADK + MCP Technical Demo Walkthrough

**Purpose:** Reference document for the live technical demo covering the ordering
conversation flow between `kiosk-core` and `rag-service`, the ADK agent, MCP tool
calling, and anticipated Q&A.

**Captured from a live run:** conversation session `5c896e45-0ff4-43e9-a535-80aff972f217`,
order `ORD-3`, 2026-07-02.

---

## 1. Architecture Recap

```
User (voice) → kiosk-ui → kiosk-core (Session Manager)
                              ├── audio-analyzer (Whisper ASR + Pyannote diarization)
                              ├── rag-service (RAG Q&A  +  ADK Ordering Agent)
                              │        └── MCP client → kiosk-core MCP server (/mcp/mcp)
                              │                              └── OrderingService → SQLite
                              └── text-to-speech (TTS reply audio)
```

- **kiosk-core** hosts the MCP **server** (`fastmcp`, mounted at `/mcp`) exposing ordering
  tools, and also calls **into** rag-service's `/api/v1/agent/chat` as an HTTP client.
- **rag-service** hosts the ADK **agent** (`OrderingAgent`) which is an MCP **client** to
  kiosk-core, plus a native `knowledge_lookup` tool for RAG-backed Q&A.
- LLM: `OpenVINO/Qwen3-4B-int8-ov` served by `ovms-llm` (OpenVINO Model Server), accessed
  via LiteLLM's OpenAI-compatible client (`base_url=http://ovms-llm:8000/v3`).

---

## 2. Conversation Trace (live captured)

| # | User said | Prompt to rag-service (`/api/v1/agent/chat`) | MCP tool called | Tool args | Tool result | LLM latency |
|---|---|---|---|---|---|---|
| 1 | "What are the items do you serve?" | `transcription="What are the items do you serve in your restaurant?"` | *none* (Rule 0 — canned reply, grounded in system prompt) | — | — | 3805 ms |
| 2 | "I would like to explore on burgers." | `transcription="I would like to explore on burgers."` | `list_products` | `{'category': 'burgers'}` | 7 items (Aloo Tikki ₹119 … Spicy Chicken Crunch ₹179) | 6940 ms |
| 3 | "I would like to order classic chicken burger." | `transcription="I would like to order classic chicken burger."` | `place_order` | `{'user_id':'kiosk-user','items':[{'product_id':'Classic Chicken Burger','quantity':1}]}` | `order_id=3, status=draft, total=169.0` + upsell (Fries/Peri Peri) | 6846 ms |
| 4 | "I would like to add classic french fries." | `transcription="I would like to add classic french fries."` | `update_order` | `{'order_id':3,'items':[{'product_id':'Classic French Fries (Regular)','quantity':1}]}` | `total=258.0` + upsell (Peri Peri/Pepsi) | 6732 ms |
| 5 | "I would like to confirm my order please." | `transcription="I would like to confirm my order please."` | `confirm_order` | `{'order_id': 3}` | `status=confirmed, total=258.0` → **ORD-3** | 3195 ms |

### Full HTTP request body (kiosk-core → rag-service), built in `kiosk_core/agent_client.py`
```json
{
  "transcription": "<user's spoken text, transcribed by Whisper>",
  "session_id": "5c896e45-0ff4-43e9-a535-80aff972f217",
  "user_id": "kiosk-user",
  "history": [ { "role": "user|assistant", "content": "..." } ]   // optional
}
```

---

## 3a. kiosk-core ↔ rag-service — Protocol, Request/Response Format & Streaming

**Protocol:** Plain **HTTP/1.1 REST**, JSON over TCP — a single synchronous
request/response call (`httpx.Client(...).post(...)`, `trust_env=False`),
**not** SSE, **not** streaming, **not** MCP/JSON-RPC. This is the one hop in the
whole pipeline that is a plain FastAPI request/response — MCP/streaming only
happens one layer deeper, between rag-service and kiosk-core's MCP server
(see §4).

**Endpoint:** `POST http://rag-service:8020/api/v1/agent/chat`
Content-Type: `application/json` both ways. Defined with Pydantic models in
`rag-service/api/agent_endpoints.py`; called from `kiosk_core/agent_client.py`.

**Request** (`AgentChatRequest`):
```json
{
  "transcription": "I would like to order classic chicken burger.",
  "user_id": "kiosk-user",
  "session_id": "5c896e45-0ff4-43e9-a535-80aff972f217",
  "history": []
}
```
| Field | Type | Notes |
|---|---|---|
| `transcription` | `str` | required — ASR output for this turn |
| `user_id` | `str` | default `"anonymous"` |
| `session_id` | `str` | required — maps 1:1 to the ADK session |
| `history` | `list[{role, content}]` | optional — only sent if non-empty; used to reseed ADK session state after a rag-service restart |

**Response** (`AgentChatResponse`):
```json
{
  "reply": "Great! I've added a Classic Chicken Burger (169.0) to your order. Would you like to add Classic French Fries (Regular) (₹89) or Peri Peri Fries (Regular) (₹99)? Say 'confirm' to place your order.",
  "tool_calls": ["place_order"]
}
```
| Field | Type | Notes |
|---|---|---|
| `reply` | `str` | final natural-language reply, ready for TTS |
| `tool_calls` | `list[str]` | names of MCP/native tools invoked this turn (empty if none) |

**Why not streaming here:** kiosk-core needs the **complete** reply text before
it can hand it to the TTS service (partial/mid-sentence text would produce
broken audio) and before it can know the full list of `tool_calls` for its own
`[PIPELINE]` latency logging — so the whole ADK tool-calling loop (both LLM
passes + MCP round trip) completes server-side inside rag-service, and only
the final `{reply, tool_calls}` JSON is returned in one shot. On the kiosk-core
side, `agent_client.py`'s `chat()` is written as a Python generator (`yield
reply`, then `yield {"_tool_calls": ...}`) purely to fit the existing streaming
interface shared with other clients (e.g. TTS) — but it still waits for the
single blocking HTTP response before yielding; there is no chunked/streamed
transfer over the wire in this hop.

**Contrast with the MCP hop:** rag-service ↔ kiosk-core's `/mcp/mcp` endpoint
(§4) uses the `streamable_http` MCP transport — an HTTP POST to submit the
JSON-RPC call plus a persistent `GET` SSE stream to receive the async result —
which is genuinely a different protocol from this plain REST hop.

---

## 3. The Real LLM Prompt (system instruction, `rag-service/agentic/ordering_agent.py`)

Every turn, OVMS actually receives: **system instruction + tool function-declarations
(6 MCP tools + knowledge_lookup) + conversation history + this turn's transcription**.
The system instruction (`_AGENT_INSTRUCTION`) is fixed and never contains product data —
it only encodes *behavior rules*:

```
You are the ordering assistant for QuickBite Express, a QSR voice kiosk.

## GROUNDING (most important)
Never state a product name, id, or price that did not come from a tool result in
THIS conversation. Don't guess, invent, or recall prices.

## Rules (check in order)
0. GENERAL "what do you serve" (no food type named) → no tool, canned reply.
1. ORDER ("I want X", "add X") → call place_order/update_order directly
   (no list_products first). On success, reply with item name + price + upsell,
   then ask to confirm.
2. INFO question (ingredients, allergens, hours) → call knowledge_lookup.
3. BROWSE a named category → call list_products(category), list EVERY item
   returned verbatim (name + price), then ask which one.
4. MANAGE ("show my order" / "confirm") → get_order / confirm_order.

## Style
Concise, 2-3 sentences. Never invent ids/names/prices.
/no_think
```

**Key demo point:** No product names, prices, or tool schemas are hardcoded in this
prompt — it is 100% behavioral/grounding rules. All factual data comes from tool
results at runtime.

---

## 4. MCP Wire Protocol — Call Format

```
rag-service (MCP client)                    kiosk-core (MCP server, fastmcp @ /mcp)
──────────────────────                      ──────────────────────────────────────
── One-time handshake (agent bootstrap) ──
POST /mcp/mcp        → 307 redirect
POST /mcp/mcp/       → 200 OK   (session opened)
   Received session ID: a27dd788-9d6a-41d3-...
   Negotiated protocol version: 2025-11-25
GET  /mcp/mcp/       → 200 OK   (SSE stream kept open for async responses)
session.list_tools() → discovers 6 tools + JSON-schema for each

── Per tool call (repeated every turn that needs a tool) ──
POST /mcp/mcp  → 307 → POST /mcp/mcp/  → 200 OK
   JSON-RPC: CallToolRequest{ name: "place_order", arguments: {...} }
   kiosk-core log: "Processing request of type CallToolRequest"
   kiosk-core log: "[MCP-SERVER] place_order user=kiosk-user order_id=3 total=169.00"
   ← JSON result streamed back over the persistent session
```

- **Transport:** `streamable_http` (official `mcp` Python SDK), via
  `mcp.client.streamable_http.streamablehttp_client`.
- **Session lifecycle:** opened **once** per rag-service process lifetime at agent
  bootstrap and **reused** for every subsequent tool call (not re-negotiated per turn) —
  see `_sessions` dict cache in `mcp_client.py`.
- **Config:** `rag-service/agentic/resources/mcp_servers.json`
  ```json
  { "servers": [ { "name": "kiosk-core", "transport": "http",
                    "url": "http://kiosk-core:8012/mcp/mcp", "timeout": 30.0 } ] }
  ```

---

## 5. How ADK Loads MCP Tools Without Hardcoding Them

No tool name, parameter, or schema is written anywhere in the prompt. The whole
chain is discovered and synthesized at runtime:

1. **Bootstrap** — `bootstrap_mcp_tools()` reads `mcp_servers.json`, opens a
   persistent `ClientSession` to kiosk-core, calls `session.initialize()`
   (MCP handshake) then `session.list_tools()`.
2. **Discovery** — kiosk-core's fastmcp server introspects its own
   `@mcp.tool()`-decorated Python functions and returns each tool's `name`,
   `description`, and JSON-schema `inputSchema` (auto-derived from Python type
   hints) — e.g. `list_products(category: str)`. This schema is the single
   source of truth; rag-service never defines it independently.
3. **Dynamic Python function synthesis** — `_make_mcp_callable()` in
   `ordering_agent.py` takes that JSON schema and builds a **real Python
   function at runtime** using `inspect.Signature` + `__annotations__`
   (e.g. gives it a genuine `category: str` keyword parameter). This is
   necessary because a naive `**kwargs` wrapper introspects as a zero-arg
   function — ADK would then always call the tool with empty arguments.
4. **Wrap as ADK `FunctionTool`** — `FunctionTool(synthesized_fn)`. ADK inspects
   the function's signature/docstring and auto-generates the LLM-facing
   function-calling declaration (JSON schema) from it.
5. **Attach to `LlmAgent`** — `LlmAgent(tools=adk_tools, instruction=_AGENT_INSTRUCTION)`.
   At inference time, ADK serializes the instruction + all tool declarations +
   history + this turn's message into the request sent to OVMS via LiteLLM.
   The LLM (function-calling trained) decides whether/which tool to invoke.
6. **Execution loop** — ADK's runner intercepts the LLM's tool-call event,
   invokes the synthesized function → `call_tool()` → real MCP `CallToolRequest`
   to kiosk-core → JSON result → `_compress_tool_result()` strips ~70-80% of
   unneeded fields (cuts prompt tokens for next turn) → fed back to the LLM for
   a **second pass** to compose the final natural-language reply.
7. **Resilience** — `_refresh_mcp_tools()` re-discovers tools on every `chat()`
   call if the registry is empty, handling kiosk-core restarts gracefully.

**One-liner for the demo:** *"kiosk-core publishes its own tool schemas over the
wire via MCP; ADK dynamically builds native Python callables from those schemas
at runtime and exposes them to the LLM as function-calling tools — adding a new
MCP tool in kiosk-core requires zero prompt or code changes in rag-service."*

---

## 6. Latency Breakdown (this run)

| # | User message | LLM latency (OVMS, `AGENT→OVMS`→`AGENT←OVMS`) | Tool called | End-to-end pipeline (kiosk-core `wall`) | ttft | tts |
|---|---|---|---|---|---|---|
| 1 | "What do you serve?" | 3805 ms | none | 6891 ms | 4880 ms | 2011 ms |
| 2 | "Explore burgers" | 6940 ms | list_products | 10756 ms | 6949 ms | 3806 ms |
| 3 | "Order classic chicken burger" | 6846 ms | place_order | 10268 ms | 6863 ms | 3405 ms |
| 4 | "Add classic french fries" | 6732 ms | update_order | 9207 ms | 6742 ms | 2465 ms |
| 5 | "Confirm order" | 3195 ms | confirm_order | 4413 ms | 3204 ms | 1208 ms |

**Why tool-calling turns are ~2x slower:** a tool-call turn requires **two LLM
passes** — (1) decide to call the tool + emit arguments, (2) compose the final
reply after the tool result is injected — plus the MCP round-trip (~15-90ms,
negligible). Non-tool turns (1 and 5's confirm) need only one LLM pass.

`asr=0ms` in all rows because ASR happens earlier in the streaming audio chunk
pipeline, before the turn is dispatched to the agent (not part of the LLM call).

---

## 7. ADK vs LangChain — Why ADK Was Chosen

- **Native MCP tool consumption** — ADK's `FunctionTool` + dynamic signature
  synthesis directly consumes MCP-discovered tools with zero adapter layer.
  LangChain needs the separate `langchain-mcp-adapters` package.
- **Model-agnostic via LiteLLM** — `LiteLlm(model="openai/OpenVINO/Qwen3-4B-int8-ov",
  base_url="http://ovms-llm:8000/v3")` — swapping inference backends needs no
  chain rewiring.
- **Built-in session/runner** — ADK's `Runner` + `SessionService` manage
  per-`session_id` conversation state out of the box; LangChain requires
  explicit `RunnableWithMessageHistory` wiring.
- **Typed tool-call events** — ADK streams `event.tool_call.name` events,
  giving clean observability/logging (`[AGENT] Tool invoked: ...`) without
  custom callback plumbing.
- **Simpler for single-agent tool orchestration** — this app has one agent
  routing between `knowledge_lookup` (RAG) and MCP ordering tools; ADK's
  `LlmAgent` model fits better than LangChain's heavier chain/graph
  abstractions for this scope.

---

## 8. Anticipated Q&A

**Q: Is the product catalogue hardcoded in the prompt?**
A: No. The system prompt only has behavior rules. Product names/prices always
come from live `list_products` / `place_order` tool results in the current
conversation — the grounding rule explicitly forbids the LLM from stating
anything not returned by a tool this turn.

**Q: What happens if kiosk-core is down when rag-service starts?**
A: `bootstrap_mcp_tools()` fails discovery silently at startup; `chat()` calls
`_refresh_mcp_tools()` on every turn to retry discovery until kiosk-core is
reachable, then rebuilds the agent with the discovered tools.

**Q: Why two LLM passes per tool-calling turn?**
A: Standard function-calling flow — pass 1: LLM emits a tool call with
arguments; ADK executes it; pass 2: LLM sees the tool result in context and
produces the natural-language reply grounded in that result.

**Q: How is the MCP session kept efficient (not reopened every call)?**
A: `mcp_client.py` caches a `ClientSession` per server in a module-level dict
(`_sessions`), opened once at bootstrap via `streamablehttp_client` +
`session.initialize()`, and reused for every `call_tool()` — only closed on
error/timeout, in which case it self-heals by reopening on next call.

**Q: How are large tool results kept from blowing up context size?**
A: `_compress_tool_result()` strips ~70-80% of fields (product_id, category,
raw nested objects) before storing the result in ADK session history — keeping
only what the LLM needs to phrase the reply (name, price, order_id, total,
upsell display strings).

**Q: Where is speaker diarization / voice authentication in this flow?**
A: Handled upstream in `audio-analyzer` (Pyannote) and `identity-service`
before the transcript reaches rag-service — this demo trace starts after ASR
has already produced clean, speaker-filtered text.

**Q: What's the actual database change on `confirm_order`?**
A: `OrderingService.confirm_order()` (kiosk-core) flips the order's `status`
from `draft` → `confirmed` in SQLite (`orders` table) and returns the final
total; no new row is created — same `order_id=3` throughout.

---

## 9. Key File References

| File | Role |
|---|---|
| `rag-service/agentic/ordering_agent.py` | ADK `LlmAgent`, system instruction, MCP→FunctionTool synthesis, tool-result compression |
| `rag-service/agentic/mcp_client.py` | MCP client — session management, tool discovery, `call_tool()` |
| `rag-service/agentic/resources/mcp_servers.json` | MCP server registry config |
| `rag-service/agentic/adk_runtime.py` | LiteLLM model wrapper, ADK runner/session service factory |
| `kiosk_core/ordering/mcp_server.py` | fastmcp server — exposes ordering tools (`list_products`, `place_order`, etc.) |
| `kiosk_core/agent_client.py` | HTTP client, kiosk-core → rag-service `/api/v1/agent/chat` |
| `kiosk_core/audio_session.py` | Per-turn pipeline orchestration, latency logging (`[PIPELINE]`) |
