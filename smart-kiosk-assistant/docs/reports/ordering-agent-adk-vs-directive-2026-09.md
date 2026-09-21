# ADK vs. Non-ADK (Lab-Style Directive) Ordering Agent — Comparison Report

**Date:** September 2026
**Scope:** `rag-service` ordering agent — root-causing production LLM TTFT
(~604–740 ms reported for tool-calling turns) and validating whether the
gap is architectural (fixable in software) or a genuine hardware/model
limit, per the external team's pushback that the "No" targets weren't a
hardware ceiling.

## Background

Production reported LLM TTFT of ~600–740 ms for tool-calling turns, cited
as the largest single contributor to voice-to-voice latency and attributed
to "the model must emit several preamble tokens before the tool-parser can
emit a usable delta — inherent to the model/hardware combination." A
second team's prototype (`kiosk-voice-lab-main`), running on the *same*
hardware class (Panther Lake, Core Ultra X7 358H) and the *same* model
(Qwen3-4B INT4 on OVMS), measured LLM first-token at 88 ms cold / 25–30 ms
warm — a ~20x difference from production's number on ostensibly comparable
silicon. This report documents the investigation into that gap and a
working proof that the difference is architectural, not silicon.

## Phase 0 — raw OVMS TTFT decomposition

Before touching any application code, LLM TTFT was measured directly
against OVMS (bypassing ADK, LiteLLM, and the FastAPI layer entirely) using
the real production system instruction and real MCP tool schemas, on the
verified-Panther-Lake host this stack runs on. 5 arms, 8 runs each after 2
warmups:

| Arm | Config | TTFT (median) |
|---|---|---|
| A | No instruction, no tools | 32.6 ms |
| B | Real instruction + 12 native `tools=[]`, Q&A | 44.5 ms |
| B2 | Same, forced cart-mutation tool call | 42.0 ms |
| C | Lab-style `<act>` directive instruction, no `tools=`, Q&A | 41.0 ms |
| C2 | Same, cart mutation | 40.2 ms |

**Conclusion:** raw TTFT on this hardware is 32–45 ms regardless of
whether native `tools=[]` schemas are present — indistinguishable from the
lab's 25–88 ms range, and nowhere near production's reported 600–740 ms.
This overturns the stated root cause: 13 (12) tools forcing preamble
tokens is not a hardware/model limit on this silicon. This corroborates
the external team's position that the gap is a design/implementation
choice, not something the "No" targets attribute to hardware.

While investigating, `plugins/kiosk/directive_mode.py` was found already
implementing the lab's mechanism (single call, `<act>` directive grammar,
no `tools=`, pooled HTTP client, sentence-level TTS release) — but shipped
**disabled by default** (`AGENT_DIRECTIVE_MODE=false`) and scoped only to
cart-mutation turns, falling back to the full ADK/tool-calling pipeline for
Q&A/browse turns.

## Phase 1 — standalone non-ADK agent + head-to-head A/B

To test the hypothesis end-to-end (not just at the raw-OVMS level), a new
standalone agent, `plugins/kiosk/ordering_agent_without_adk.py`
(`OrderingAgentWithoutADK`), was built matching the production `chat()`
contract exactly:

- Root-fact turns (hours/location/etc.) → 0 LLM calls (reused unchanged
  from production's `reply_templates`/`knowledge_tool.root_facts()` path).
- Cart-mutation turns → routed through the existing, unmodified
  `directive_mode.run_turn()` (1 LLM call, no `tools=`).
- Q&A / browse / order-state-read turns → **new** single-call path
  (`_run_generic_turn`): deterministically pre-fetches `knowledge_lookup()`
  and/or `get_current_order` context (no LLM decision step), then makes
  exactly one streamed completion with no `tools=` field — the turn type
  production's directive mode explicitly does *not* cover today.

Wired in behind a benchmark-only route, `POST
/api/v1/agent/chat-no-adk`, alongside the existing `/chat` (ADK). Both
endpoints run inside the same live `rag-service` container against the
same OVMS backend, so results are directly comparable — no separate
process, model, or hardware.

### A/B results (warm, N=10/arm, same prompts, same live stack)

| Turn type | Agent | LLM calls | **TTFT (median)** | llm_ms (median) | Wall (median) |
|---|---|---|---|---|---|
| Cart mutation ("add a burger") | ADK `/chat` | 1 | 128 ms | 681 ms | 682 ms |
| Cart mutation | non-ADK `/chat-no-adk` | 1 | 140 ms | 691 ms | 693 ms |
| Q&A ("price of X") | ADK `/chat` | 1 | **749 ms** | 835 ms | 844 ms |
| Q&A | non-ADK `/chat-no-adk` | 1 | **105 ms** | 770 ms | 772 ms |

**Cart mutation shows no delta** — both paths already run through the same
`directive_mode.run_turn()` code, since `AGENT_DIRECTIVE_MODE=true` in this
environment already routes mutations away from ADK. This is expected and
confirms the two endpoints are otherwise apples-to-apples (no confound
from a different model, container, or GPU state).

**Q&A/browse is where the gap lives:** ADK's tool-calling pipeline (schema
injection + tool-selection decode before an answer can begin) costs **~644
ms of extra TTFT** versus the lab-style single no-`tools=` call, even
though total generation time (`llm_ms`) is similar (835 ms vs. 770 ms).
TTFT — not total decode time — gates when TTS/speech can start, so this
644 ms is customer-felt latency, not just an internal metric.

## Correctness regression sweep

`tests/benchmarks/replay_all_conversations.py` (extended with an
`--agent-url` override) was run against all 562 recorded conversation
fixtures (546 non-empty turns), once per agent, on the same live stack:

| Agent | Conversations passed | Turns passed | Guard-corrections |
|---|---|---|---|
| ADK `/chat` | 562 / 562 | 546 / 546 | 0 |
| non-ADK `/chat-no-adk` | 562 / 562 | 546 / 546 | 0 |

No hard failures, empty replies, or error-signature replies in either
agent across the full fixture set. This is a **shallow correctness check**
(no crash / no empty reply / no guard-correction), not a semantic-quality
grade — see "Known gaps" below for what it does not verify.

## Root cause, confirmed

The production ~600–740 ms is **not** OVMS/model decode time forced by
tool count (Phase 0 showed 32–45 ms raw TTFT regardless of tools). The
real production number is explained by:

1. **Two full LLM round trips per Q&A/tool-calling turn** (tool-selection
   call, then a narration call) vs. the lab's one — and `llm_metrics`
   appears to sum TTFT/latency across both calls into a single reported
   figure, inflating the apparent per-call cost.
2. **ADK/LiteLLM tool-schema overhead on the decision call** — confirmed
   directly in this A/B: the ADK Q&A path's TTFT (749 ms) is ~7x the
   non-ADK single-call path's TTFT (105 ms) on the identical prompt,
   identical container, identical OVMS instance.
3. A fresh MCP streamable-HTTP session opened per tool call adds further,
   separately-measurable overhead (~130 ms observed `mcp_ms` in the
   non-ADK path), independent of (1) and (2).

This confirms the external team's position: the "No" targets are not a
hardware ceiling on this silicon. `directive_mode.py` already proves the
fix works for cart-mutation turns in production today; it was simply never
extended to Q&A/browse turns — the same "bigger design change" the
original write-up flagged as needing explicit scoping/approval before
implementation.

## Known gaps in the new agent (by design — a benchmarking vehicle, not a production replacement)

- No customer-name memory, no dietary-preference memory.
- No cart-state staleness/confirmation guard.
- No speculative/dry-run drafting.
- Reuses the guard chain only where cheap/safe — not the full
  menu/removal/confirm truthfulness-guard chain the production ADK plugin
  runs.
- `_run_generic_turn`'s knowledge-hint and order-state-read regexes are new
  heuristics, not the production intent/guard system; their coverage
  against edge-case phrasing beyond the 562-conversation fixture set is
  unverified.

## Recommendation

Do not ship `ordering_agent_without_adk.py` / `/chat-no-adk` as a
production replacement. Instead, use this A/B as validation to **extend
`directive_mode` (already shipped, already gated by
`AGENT_DIRECTIVE_MODE`) to cover read-only Q&A/browse turns**, reusing its
existing single-call/no-`tools=`/pooled-client mechanism rather than
introducing a second, parallel agent implementation. This keeps the
production guard chain, memory features, and confirmation logic intact
while removing the ADK tool-selection round trip that this report shows
costs ~644 ms of TTFT on Q&A turns, with no measured correctness
regression across the existing 562-conversation fixture set.

## Artifacts

- `plugins/kiosk/ordering_agent_without_adk.py` — standalone non-ADK agent
  (new).
- `rag-service/api/agent_endpoints.py` — `POST
  /api/v1/agent/chat-no-adk` (benchmark-only twin of `/chat`).
- `tests/benchmarks/agent_latency_benchmark.py` — `--agent-url` /
  `$RAG_AGENT_URL` override (for the A/B).
- `tests/benchmarks/replay_all_conversations.py` — `--agent-url` /
  `$RAG_AGENT_URL` override (for the regression sweep).
- Raw results: `/tmp/phase0/adk_baseline.json`, `/tmp/phase0/no_adk.json`,
  `/tmp/phase0/ab_internal_metrics.json`, `/tmp/phase0/replay_adk.json`,
  `/tmp/phase0/replay_noadk.json` (session-local, not committed).
