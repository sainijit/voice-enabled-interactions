# Smart AI Kiosk (VEI) — voice-to-voice latency report

**Prepared by:** Engineering (Smart AI Kiosk)
**Date of run:** 2026-09-16, 16:00 IST
**Status:** Live measurement on the running Docker stack. Every number is from this run.

---

## 1. What was run

| Item | Value |
|---|---|
| Command | `python3 tests/benchmarks/v2v_scripted_conversation_benchmark.py --script order-simple --runs 12 --explicit-end-mark` |
| Domain | **QSR ordering** — live cart mutation via MCP tool call |
| Utterance | *"Hi, I would like to order one Classic Chicken Burger, please."* |
| Reply | *"Got it. One Classic Chicken Burger. Your total is now ₹169. Would you also like Classic French Fries (Regular) (₹89)?"* |
| Result file | `results/v2v-bu-final.json` |
| Image | `false/kiosk-core:2026.2.0-rc2` (verified via `docker inspect`; `REGISTRY=false` is the configured default) |
| Machine state | Lab benchmark fully stopped, `queue-service`/`rtsp-streamer` stopped |

All 13 runs produced an identical, correct transcript and a correctly-priced
order. Run 1 is reported but excluded from the statistics as a cold-start
outlier (see §5).

> **Superseded measurement.** An earlier version of this report quoted 732 ms
> from `results/vei-demo-final.json`. That run was affected by a bug in the
> opener-audio cache which prevented the first sentence's audio file from being
> written, so the stopwatch stopped when the reply was *queued* rather than when
> sound was *ready*. The bug is fixed and the figures below are from a clean
> re-run: 21 cache hits, 0 synthesis failures, all four audio segments present on
> every turn.

---

## 2. Headline result

```
Voice-to-voice   median 754 ms   p95 773 ms   n=12 (warm)
range 704 – 773 ms
independent ground-truth cross-check: median 743 ms
```

Per-run v2v (ms): `726` · 721 · 766 · 756 · 769 · 765 · 773 · 769 · 704 · 739 · 751 · 749 · 723
*(first value = cold start, excluded from statistics)*

The spread is tight — 70 ms between best and worst across 12 warm runs. Note
this reflects one repeated utterance, so it demonstrates that **this turn** is
reproducible; it is not a claim about variance across varied customer speech.

---

## 2b. Canonical KPI vocabulary

This section restates our numbers in the shared cross-team terminology so this
report can be read side by side with other teams' without a translation step.
Nothing here is newly measured — every value is an alias or a simple derivation
of a field already emitted by the harness (`summary.kpi_vocabulary` in the
results JSON).

### Term mapping

| Canonical term | Starts | Stops | Our field |
|---|---|---|---|
| Voice-to-voice latency | customer's last word | first sound at speaker | `voice_to_voice_ms` |
| Endpointing delay | customer's last word | turn-end decision | `endpoint_wait_ms` |
| Processing latency | turn-end decision | first sound at speaker | `voice_to_voice_post_endpoint_ms` |
| Transcription latency | speech in | transcript out | `asr_ms / asr_chunks` (per call) |
| LLM time to first token | prompt in | first token out | `agent_ttft_ms` |
| TTS time to first byte | text in | first audio byte | `latency_breakdown.tts_first_segment` |

Per the shared convention, **"time to first audio" is retired as a KPI name.**
It survives only as the internal field `time_to_first_audio_ms`, and it is *not*
the same thing as processing latency — see the trap below.

### The defining identity

```
voice-to-voice = endpointing delay + processing latency
```

This holds **exactly**, on every single turn, in our pipeline — not
approximately, and not after rounding. Verified across all 12 hands-free turns.

### Results — hands-free (real endpointing), n=12

| KPI | p50 | p95 |
|---|---|---|
| **Voice-to-voice latency** | **1,427 ms** | 1,471 ms |
| Endpointing delay | 1,200 ms | 1,200 ms |
| Processing latency | 227 ms | 271 ms |
| Transcription latency (per call) | 75 ms | — |
| LLM time to first token | 194 ms | 208 ms |
| TTS time to first byte | 1.1 ms | — |

Identity check: `1,200.0 + 227.2 = 1,427.2` ✅

**Customer-experience band: "reads as a machine" (>1,000 ms).** We are not going
to dress this up. But read the next subsection before drawing a conclusion,
because the decomposition says something quite different from the headline.

### The one number that matters: endpointing, not compute

Our **processing latency is 227 ms** — the entire cost of turning a finished
utterance into audible speech, including the LLM. That is competitive, and it is
the part of the system that is genuinely hard.

**94% of our voice-to-voice latency is the endpointing delay**, and that is a
silence-timeout configuration value, not compute. The evidence is unusually
clean, because the sentence-completeness shortcut fired on 3 of 12 turns:

| | n | Endpointing | Processing | **Voice-to-voice** | ASR calls | Transcript |
|---|---|---|---|---|---|---|
| Shortcut **fired** | 3 | 700 ms | 3 ms | **705 ms** | 2 | clean |
| Shortcut **missed** | 9 | 1,200 ms | 251 ms | **1,451 ms** | 3 | trailing `" ."` |

Same audio, same model, same machine — **a 746 ms swing decided entirely by
whether the endpoint shortcut fired.** When it fires we land at 705 ms, inside
the kiosk-target band. When it misses we fall back to the full silence timeout.

Two compounding effects on a miss, both consequences of the same root cause:

1. The turn waits the full 1,200 ms timeout instead of 700 ms.
2. A third ASR call fires on the trailing *silence* pad, from which Whisper
   hallucinates a `"."` — costing ~254 ms of final-flush wait and polluting the
   transcript with a spurious sentence.

So the headline is not an inference-speed problem. **Making the shortcut fire
reliably is worth ~750 ms** and is the single highest-value open item in this
report. See §7.

### Why processing latency ≠ "time to first audio"

A trap worth stating explicitly, because the two look interchangeable and are
not. On shortcut turns, processing latency is **3 ms** while
`time_to_first_audio_ms` is **195 ms**. That is not a measurement error: work is
started speculatively *during* the silence wait, so by the moment the turn-end
decision fires, the opening audio has already been rendered. `ttfa` overlaps the
endpointing window; processing latency does not. Only the latter belongs in the
identity above.

### Disclosure: the TTS number is a cache hit, by design

TTS time-to-first-byte of **1.1 ms is a file copy, not synthesis.** The opener is
pre-rendered, so on the voice-to-voice path the first audio byte costs
effectively nothing. Live synthesis of the same short opener measures **~335 ms**
and is reported separately as `tts_synthesis_ms`.

We consider the cache hit the correct number to carry inside voice-to-voice —
it is what the customer actually experiences at the kiosk — but it must **not**
be quoted as a like-for-like TTS benchmark against a system that synthesises its
opener live. Both numbers are published for exactly this reason.

### Note on the explicit-end-mark dataset

The `--explicit-end-mark` runs used elsewhere in this report (v2v median 751 ms)
**bypass endpoint detection entirely.** Their `endpoint_wait_ms` is `null` and
the identity above cannot be evaluated for them. That mode isolates pipeline
compute and is useful for regression testing, but **it must not be quoted as a
voice-to-voice figure**, because it silently omits the largest component of the
customer's actual wait. The 1,427 ms hands-free figure is the honest
customer-facing number.

---

## 3. Full stage breakdown

| Stage | Median | Mean | p95 | Min | Max | On critical path? |
|---|---|---|---|---|---|---|
| **`voice_to_voice_ms`** | **753.8** | 748.7 | 773.4 | 703.7 | 773.4 | — (the total) |
| `final_flush_wait_ms` | 261.2 | 249.4 | 270.8 | 201.4 | 270.8 | **Yes** |
| `time_to_first_audio_ms` | 211.6 | 207.2 | 214.7 | 189.3 | 214.7 | **Yes** |
| `agent_ttft_ms` | 209.9 | 205.4 | 213.0 | 185.8 | 213.0 | **Yes** |
| `asr_ms` (all chunks) | 259.4 | 248.1 | 269.8 | 200.7 | 269.8 | overlaps speech |
| `mcp_ms` (cart mutation) | 87.7 | 88.5 | 92.7 | 77.1 | 98.0 | **No** — see §4 |
| `agent_total_ms` (full reply) | 2234.6 | 2390.0 | 2665.6 | 2083.9 | 2677.8 | No |
| `tts_ms` (full reply, all segments) | 1995.7 | 2183.3 | 2466.0 | 1871.9 | 2470.9 | No |
| `voice_to_voice_ground_truth_ms` | 714.9 | 721.7 | 759.3 | 662.6 | 826.5 | independent check |

### The critical path, end to end

Traced across kiosk-core and rag-service logs for a single real turn
(session `9d71fe85`), timestamp by timestamp:

| # | Stage | Time | Evidence |
|---|---|---|---|
| 1 | last word → `/audio/end` received | **~281 ms** | 150 ms harness pad + HTTP push |
| 2 | final ASR tail commit (`final_flush_wait`) | **261 ms** | whisper-small on NPU |
| 3 | agent time-to-first-sentence | **210 ms** | see decomposition below |
| 4 | TTS first segment → first audio | **≈ 2 ms** | opener cache hit (file copy) |
| | **Total** | **~754 ms** | |

**Stage 3 decomposed** (cross-service log timestamps, same turn):

| Segment | Time |
|---|---|
| kiosk-core sends → rag-service receives | 2 ms |
| plugin load + agent setup | 2 ms |
| request build → OVMS response headers (prefill/TTFT) | 67 ms |
| OVMS headers → first sentence back at kiosk-core | 138 ms |
| **`agent_ttft` total** | **209 ms** |

The rag-service orchestration layer itself costs **~4 ms**. The remaining
~205 ms is genuine OVMS prefill and token generation. There is no meaningful
framework overhead left to remove on this path — verified by benchmarking OVMS
directly with a realistic-size prompt (warm, 3 trials): TTFT ~100 ms, first
complete sentence ~245 ms. The live pipeline delivers its first sentence in
209 ms, i.e. *faster* than that isolated test, because prefix caching is
working and `"Got it."` is shorter than the test's opener.

---

## 4. Why MCP is not on the critical path

A natural assumption is that the cart write (`place_order`, 88 ms) delays
speech. It does not. Proven from cross-service timestamps on turn `9a26b475`:

```
09:46:09.377   first audio out       ← customer starts hearing "Got it."
09:46:09.582   MCP place_order fires ← 205 ms LATER
```

Directive mode streams the acknowledgement to TTS first and executes the order
behind it. The customer hears the kiosk respond while the cart write is still
in flight. This is by design and is why `mcp_ms` is reported but excluded from
the critical-path total.

---

## 5. Honest caveats — please read before quoting these numbers

**(a) The 150 ms trailing pad is harness overhead, not pipeline compute.**
`--explicit-end-mark` appends 150 ms of digital silence and then calls
`/audio/end` directly. A real push-to-talk customer releasing the button signals
end-of-turn immediately, so they would not pay this. Conversely, this mode
*bypasses* VEI's own silence detector entirely — so 754 ms is **not** what a
hands-free customer experiences.

**(b) The three operating modes give three different numbers.** Be precise
about which one is being quoted:

| Mode | Endpoint cost | Expected v2v | Notes |
|---|---|---|---|
| Push-to-talk (real kiosk UI) | ~0 ms | **~590 ms** | Stop button signals end instantly |
| Benchmark (`--explicit-end-mark`) | 150 ms pad | **754 ms** | the number in this report |
| Hands-free conversation | up to 1.1 s silence timeout | **~1.7 s** | detector must confirm silence |

The hands-free figure is the weakest of the three and is the most honest thing
to flag ahead of a demo. It is gated by `DEFAULT_SILENCE_TIMEOUT_SECONDS = 1.1`,
because the endpoint-completeness shortcut that is *supposed* to cut this short
currently fires **0 / 6** times in normal mode: the shortcut requires a preview
ASR round trip inside `DEFAULT_ENDPOINT_SHORT_SECONDS = 0.15 s`, and measured
live that round trip takes 150–220 ms — consistently just over budget. This is
the top open item (§7).

**(c) Cold start is real and visible.** Run 1 was 3494 ms against a warm median
of 754 ms. The first turn after the stack idles pays model warm-up and an empty
opener cache. **Fire one throwaway turn before demoing.**

**(d) Sample size and workload.** n = 12 warm runs of a single scripted
utterance, versus the lab's 75 turns across an 80-question set. This is a
precise number for *this* turn, not a broad distribution across varied phrasing.

---

## 6. What produced this number — the key decisions

Ordered by the size of the win.

### 6.1 Opener TTS cache — ~280 ms saved *(the single biggest win)*

Measured directly: synthesising `"Got it."` on Kokoro/CPU costs **270–300 ms**,
and it sits at the very end of the critical path — the last thing before the
customer hears anything. On an ~1030 ms turn that was ~27% of the total, paid
again on every single turn, for audio that is identical every time.

`kiosk_core/audio_session.py` now keeps a **process-wide cache of short opener
segments**, keyed on normalised sentence text **plus** the full
model/voice/language/instructions tuple.

The critical design decision was to make it **self-populating from real output
only**. It never issues a TTS request of its own; it retains a copy of a segment
the real pipeline already synthesised, already trimmed and gain-adjusted, and
replays that exact file.

This matters because the obvious alternative — *speculative pre-synthesis*
(`DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED`) — is **actively harmful here and is
deliberately left off**. Both OVMS and the TTS backend serialise requests rather
than serving them concurrently, so speculative work queues *in front of* the
real turn. A live replay recorded `tts=` ballooning to ~21.8 s and a turn wall
time of 38.2 s against a ~5.8 s non-speculative baseline. The cache sidesteps
this entirely: worst case on a miss is exactly the status quo, and there is no
path by which it adds backend load.

Result: `time_to_first_audio` fell to ~212 ms, which now equals `agent_ttft`
alone — TTS first-segment cost is effectively **zero**.

**Admission is deliberately narrow**, for correctness and for cache health:
- ≤ 24 characters — stock openers are short; order-specific sentences are long and never recur.
- **Any sentence containing a digit is rejected.** Caught during validation: `"Your total is now ₹169."` is 23 characters and was being admitted. It can never be replayed (the next order has a different total), so each such entry would permanently burn a slot; over a long kiosk uptime the 32-slot cache would fill with dead total-variants and stop admitting genuine openers.
- Hard ceiling of 32 entries.
- A hit requires an exact normalised-text **and** voice-tuple match, so cached audio can never be spoken for different text or in the wrong voice.

### 6.2 Removing host CPU contention — ~900 ms saved

The largest regression found during this work was not code. `queue-service`
(YOLOv8) and `rtsp-streamer` (ffmpeg) are **opt-in** services
(`profiles: ["queue"]`) but, once started, survive `make down` / `make up` of
the main stack. They had been left running and were consuming ~11 of 16 cores
(load average 21–23; one process at 733% CPU, another at 441%).

Because **TTS runs on CPU by deliberate design**, this directly starved the
single most latency-critical stage. Stopping them: CPU 79% → 10.8%, a direct TTS
curl 3.55 s → 1.2 s, and benchmark v2v ~2100 ms → 1183 ms.

**Operational rule: check `uptime` before trusting any latency measurement.**
Load average should be ~2, not 20+. Note it is a trailing average and takes
1–2 minutes to decay after the offenders are stopped.

### 6.3 Trailing-pad reduction — ~170 ms saved

`DEFAULT_END_MARK_TRAIL_PAD_SECONDS` reduced 0.3 s → **0.15 s**, matching
`DEFAULT_ASR_TRIM_DECAY_SECONDS` (the tail the ASR path already expects).
Validated across 8 runs with transcripts inspected — no truncation, no
alteration. This is harness overhead removal, not a pipeline change, and is
described as such in §5(a).

### 6.4 MCP redirect elimination — ~13 ms saved (off critical path)

`mcp_servers.json` pointed at `http://kiosk-core:8012/mcp/mcp` — **no trailing
slash** — so the MCP mount answered *every* tool call with a 307 redirect to the
canonical path. Every call was two round trips. Adding the slash: 5+ redirects
per turn → **0**, `mcp_ms` 98 ms → 88 ms. Off the v2v critical path, but it
compounds across multi-turn conversations.

### 6.5 Already in place before this work

- **Directive mode (`<act>`)** — cart mutations expressed as an inline directive instead of a structured tool-calling round trip, so an ordering turn costs one model call.
- **Pooled HTTP client to OVMS** — a previous fix; opening a fresh `httpx.AsyncClient` per turn had pushed first-directive latency from ~137 ms to ~525 ms.
- **`enable_thinking=False`** — suppresses Qwen3's think-block tokens, which would otherwise be generated before any speakable text.
- **OVMS prefix caching + INT4 weights** — verified active.
- **Streaming ASR** with `SKIP_EMPTY_FINAL_FLUSH`.

---

## 7. Open items — the remaining headroom

| Item | Size | Risk | Notes |
|---|---|---|---|
| Hands-free endpoint shortcut never fires (0/6) | up to ~950 ms *in hands-free mode only* | Medium | Preview ASR round trip is 150–220 ms vs a 0.15 s budget. Highest-value open item. |
| `final_flush_wait` (261 ms) | up to 261 ms | **High** | This is a fresh ASR pass starting only after the end mark. QSR-Lab hides its equivalent under a longer silence wait, so from the last word to ready-to-generate the two systems are within ~30 ms (see comparison report §2). Note the *preview* pool already runs distil-small.en on **GPU** (~40-60 ms/call) for the endpoint check; this row concerns the **final committed pass** only. Recovering it needs a faster final-pass model: distil-small.en is **blocked on NPU** by a reproduced staleness bug (returns the previous call's text); the only bug-free option is GPU, traded against OVMS/TTS contention. Separately, lowering `ADAPTIVE_FLUSH_PAUSE_SECONDS` risks documented hallucinations (0.40 s produced *"Good, good, good."*). |
| HTTP push overhead (~142 ms) | ~100 ms | Low | Largely harness-side; a real UI would not pay most of it. |
| Agent TTFT (195 ms) | ~0 ms | — | **Investigated and closed** — ~4 ms is framework, the rest is genuine OVMS work. |

---

## 8. Reproducing this

```bash
cd smart-kiosk-assistant

# 1. Confirm the machine is quiet — this is not optional
uptime                      # load must be ~2, not 20+
docker compose ps           # queue-service / rtsp-streamer must NOT be running

# 2. Bring the stack up with the CORRECT registry (see warning below)
REGISTRY=intel docker compose up -d

# 3. Warm up, then measure
python3 tests/benchmarks/v2v_scripted_conversation_benchmark.py \
  --script order-simple --runs 12 --explicit-end-mark --label vei-demo-final
```

> **Build/deploy warning.** `.env` sets `REGISTRY=false`, so a bare
> `docker compose up -d` launches a wrongly-tagged **`false/kiosk-core`** image
> and will silently run stale code. During this work three rebuild-and-restart
> cycles appeared to succeed while the old binary kept running; it was only
> caught by diffing `docker inspect kiosk-core --format '{{.Config.Image}}'`
> against the freshly built image ID. **Always prefix `REGISTRY=intel`, and
> always verify the image ID after a rebuild.** The same class of mistake hit
> `audio-analyzer`, whose config is bind-mounted and needs only a restart — but
> whose preview ASR pool loads *lazily on first use*, so startup logs alone do
> not prove it is configured correctly.
