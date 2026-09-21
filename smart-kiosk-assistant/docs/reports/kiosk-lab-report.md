# Kiosk Voice Lab (`kiosk-voice-lab-main`) — voice-to-voice latency report

**Prepared by:** Engineering (Smart AI Kiosk)
**Date of run:** 2026-09-16, 15:50 IST
**Status:** Live measurement. Every number below is from this run — no estimates, no carried-over figures.

---

## 1. What was run

| Item | Value |
|---|---|
| Command | `make bench` — **unmodified**, the lab's own official benchmark |
| Script | `experiments/e7_3_endpoint.py` |
| Pipeline class | `AdaptivePipeline` (the lab's current/best design) |
| Domain | **Retail lookup kiosk** — `kb/store_kb.json`, 80-question golden set |
| Audio variant | `clean` |
| Session log | `sessions/bench-20260916-155015.jsonl` |
| Machine state | VEI stack **fully stopped** (`docker compose down`), load average 1.50 |

The VEI stack was shut down for the duration of this run. This matters: OVMS holds
GPU memory and the audio-analyzer holds the NPU, and the lab wants both of those
devices. Measuring the lab while VEI was running would have produced a
contention-inflated number that unfairly flattered VEI in the comparison.

---

## 2. Headline result

```
V2V p50 1037 ms   p95 1334 ms   n=75
false cuts: 5/80
```

Consistent with the previous run on this machine (p50 1060 ms / p95 1346 ms), so
this is a stable, repeatable figure rather than a lucky pass.

---

## 3. Full stage breakdown

n = 75 (the 5 false-cut turns are excluded, as the lab's own script does).

| Stage | Median | Mean | p95 | Min | Max | What it measures |
|---|---|---|---|---|---|---|
| **`v2v_ms`** (customer-facing) | **1037.1** | 1040.2 | 1333.9 | 555.7 | 1683.3 | last real word spoken → first audio out |
| `endpoint_delay_ms` | 512.1 | 538.2 | 656.3 | 455.4 | 779.7 | last word → endpoint decides the turn is over |
| `compute_ttfa_ms` | 522.3 | 501.9 | 802.5 | 48.1 | 1083.2 | endpoint decision → first audio (pure pipeline compute) |
| `asr_visible_ms` | **−69.7** | −63.0 | 90.0 | −153.3 | 147.1 | endpoint decision → transcript ready |
| `rag_ms` | 562.4 | 557.4 | 859.2 | 22.2 | 1056.9 | transcript ready → retrieval complete |
| `llm_ttft_ms` | 95.7 | 98.2 | 115.3 | 87.0 | 125.6 | retrieval done → first LLM token |
| `first_clause_ms` | **−425.6** | −415.8 | −45.7 | −1047.2 | 147.5 | retrieval done → first speakable clause |
| `tts_first_ms` | 425.6 | 423.3 | 662.1 | 0.0 | 1047.2 | first clause ready → first audio written |
| `llm_total_ms` | 763.9 | 842.2 | 1308.6 | 296.7 | 1992.8 | full reply generated (context only) |

### Mechanism hit rates

| Mechanism | Hit rate |
|---|---|
| `prefix_cache` (LLM prefix caching active) | **100%** (75/75) |
| `spec_hit` (speculative draft matched the final transcript) | **98.7%** (74/75) |
| `endpoint_complete_decision` (sentence "read as finished") | 100% (8/8 eligible) |
| `opener_preready` (opening audio pre-synthesised in time) | **21.3%** (16/75) |

---

## 4. Reading the numbers — where the time actually goes

**The structure is a clean two-way split:**

```
v2v 1037 ms  =  endpoint_delay 512 ms  +  compute_ttfa 522 ms
                (deliberate silence wait)   (real pipeline work)
```

**Three observations that matter for our own design:**

**(a) The two negative numbers are the whole point of the architecture.**
`asr_visible_ms` is −70 ms and `first_clause_ms` is −426 ms. Negative means
"already finished before this checkpoint was reached". The transcript is ready
*before* the endpoint decides the turn is over, because ASR ticks run
continuously during speech; and the first speakable clause exists *before*
retrieval completes, because the speculative draft produced it during the
customer's own silence. This is latency hidden under time the customer was
going to spend anyway — the single most important idea in the lab's design.

**(b) `rag_ms` at 562 ms is the largest single stage — and it is off the
critical path.** It does not appear in `compute_ttfa` because, on a spec hit,
the opener is already being spoken while retrieval runs. Retrieval only becomes
visible to the customer when speculation misses (1.3% of turns here).

**(c) `tts_first_ms` at 426 ms is the real content of `compute_ttfa`.**
This is the lab's genuine remaining cost: `compute_ttfa` (522 ms) is
overwhelmingly TTS synthesis of the opening phrase on Kokoro/CPU. The lab's
mitigation — pre-synthesising the opener during the silence wait — only lands
**21.3%** of the time. On the other ~79% of turns the customer waits the full
~426 ms for the opener to be synthesised. **This is the specific weakness that
the Smart Kiosk's opener cache was built to eliminate** (see the VEI report).

**(d) The benchmark deliberately uses a patient endpoint.** `e7_3_endpoint.py`
overrides `cfg.endpoint_short_ms = 500` with the comment *"patient operating
point for the tradeoff curve"*, while `pipeline/config.py` ships a default of
**150 ms** for live use. So this 1037 ms is the lab's *cautious* configuration,
not its fastest. Even so it still records 5/80 (6.25%) false cuts; running at
150 ms would cut roughly 350 ms off `endpoint_delay` but would push the
false-cut rate higher. This trade is discussed properly in the comparison report
— quoting 1037 ms as "the lab's speed" without this caveat would overstate our
own advantage.

---

## 5. Configuration (`pipeline/config.py`)

| Component | Model | Device |
|---|---|---|
| LLM | `qwen3-4b-int4` | **GPU** |
| ASR | `distil-small-int8` | **NPU** |
| TTS | `kokoro-v1.0.onnx` (`af_heart`) | **CPU** |
| Retrieval | `kb_index` (NumPy embeddings, top-k 3) | — |

| Endpoint knob | Bench value | Shipped default |
|---|---|---|
| `endpoint_short_ms` | **500** (overridden by the bench script) | 150 |
| `endpoint_long_ms` | 1100 | 1100 |

---

## 6. Caveat on domain — read before comparing

`make bench` exercises the **retail lookup** domain (general store Q&A over
`store_kb.json`). It is a *RAG question-answering* workload.

The lab does ship a QSR ordering mode (`make demo-qsr`, `kb/qsr_kb.json`,
`pipeline/cart.py`), but **`make bench` is not wired to it** — the benchmark
script takes no `--domain` flag and the manifest points at the retail fixtures.
There is therefore no lab-published QSR latency number to quote.

This is the central fairness caveat of the whole exercise: the lab's 1037 ms is
a retrieval-heavy Q&A turn, whereas the Smart Kiosk's number is an ordering turn
with a live cart mutation. The comparison report handles this explicitly rather
than presenting the two as like-for-like.
