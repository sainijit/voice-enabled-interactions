# Voice Pipeline Performance Improvement — September 2026

## Context

Latency for the voice-ordering pipeline was benchmarked using two real
customer-ordering recordings (`tests/rec1_16k.wav`, `tests/rec2_16k.wav`,
16 kHz mono, converted from the originally supplied `Recording.m4a` /
`Recording (2).m4a`). The reference point for the exercise was the
`kiosk-voice-lab-main` prototype (`kiosk-voice-handoff.pdf`), which reports
a ~700 ms voice-to-voice figure (customer's last word → first sound out of
the speaker) on comparable hardware.

Working rule for this exercise: only keep a change if it produces a
measurable improvement with no observed regression; investigate but do not
adopt anything that trades away correctness or reliability for speed.

## Baseline vs. current

| Metric | Before this work | After this work |
|---|---|---|
| Opener / turn start (`voice_to_voice_ms`) | not measured directly (TTS was silently broken — see below) | ~1.2–1.5 s |
| Full informative reply (`voice_to_voice_informative_ms`) | ~3.6–6.9 s (measured while TTS was broken, not a clean baseline) | ~4.1 s (1 LLM call) / ~8.6 s (2 LLM calls) |

The `kiosk-voice-lab-main` prototype's ~700 ms figure is **not directly
comparable** — it is achieved through a materially different, single-process
architecture (see "Why kiosk-voice-lab is faster" below). It was used as a
source of individually-adoptable techniques, not as a target this stack's
current microservice architecture can hit as-is.

## Changes made (this repo)

### 1. Fixed a production bug: 100% silent TTS failures
`docker-compose.yml` still set `KIOSK_CORE_TTS_VOICE=Ryan`, a voice name that
only existed under the old SpeechT5/Qwen TTS engine. After the TTS service
was migrated to Kokoro (see edge-ai-libraries section below), every
synthesis request 400'd silently — the kiosk produced no audio for any
reply. Fixed to `am_michael` (Kokoro's closest match to the old "Ryan"
timbre). This was found and fixed as part of setting up clean, working
latency benchmarks and was a pure bug fix independent of the rest of the
work below.

### 2. Pre-synthesized, non-committal "opener" phrase
On an ordering turn the model emits only a tool call (no prose) — the
spoken reply is templated only after the tool result returns. Measured cost
of that tool-call generation: ~1.6 s at ~48.5 ms/token for a 33-token
tool-call payload, during which the customer previously heard nothing.

A short, fixed phrase ("One moment.") is now synthesized once per process,
cached on disk, and played immediately when a turn starts while the real
agent call is still in flight. This does not shorten the turn — it fills
the dead air with real audio instead of silence, taking time-to-first-audio
from ~2 s down to a cache-hit file copy.

The opener text must stay non-committal (no item, quantity, price, or
outcome) since it is spoken before any tool has run and must never
contradict what the menu/removal/confirm guards later produce.

- Config: `KIOSK_CORE_OPENER_ENABLED` (default `true`),
  `KIOSK_CORE_OPENER_TEXT` (default `"One moment."`),
  `KIOSK_CORE_OPENER_CACHE_DIR`.
- Code: `kiosk_core/config.py`, `kiosk_core/audio_session.py`
  (`_render_opener`).

### 3. Adaptive end-of-turn endpointing (sentence-completeness shortcut)
The fixed silence timeout (`KIOSK_CORE_SILENCE_TIMEOUT_SECONDS`, 1.5 s) has
to be long enough to cover a customer hesitating mid-sentence, so every turn
— including the majority that end on an obviously finished sentence — paid
that hesitation tax.

Added a check that reads the transcript accumulated so far: if it ends on a
hesitation filler, a dangling function word ("and", "with", "I'd"), or a
trailing comma, the turn is judged mid-thought and keeps the full timeout.
Otherwise the turn commits early, at `KIOSK_CORE_ENDPOINT_SHORT_SECONDS`
(default 1.0 s) instead of 1.5 s. This only ever *shortens* the wait; if no
transcript has arrived yet the check fails closed to the original fixed
timeout, so there is no new failure mode.

- Config: `KIOSK_CORE_ENDPOINT_COMPLETE_ENABLED` (default `true`),
  `KIOSK_CORE_ENDPOINT_SHORT_SECONDS` (default `1.0`),
  `KIOSK_CORE_ENDPOINT_MIN_WORDS` (default `3`).
- Code: `kiosk_core/config.py`, `kiosk_core/audio_session.py`
  (`_looks_complete`).

### 4. Speech normalization for TTS (prices, times)
The reply text and UI keep values like "₹169" and "8 AM" as written, but the
TTS engine reads those literally as symbol-and-digit tokens rather than a
spoken price or time. Adapted from `kiosk-voice-lab-main`'s
`pipeline/speech.py`, with ₹/Rs/INR support added for this
restaurant's menu. Deterministic regex/string transforms, no model involved
— negligible latency cost, applied once inside `TtsClient.synthesize_to_file`
so every call path (streamed clauses, the opener) is covered.

- Config: `KIOSK_CORE_TTS_SPEECH_NORMALIZE_ENABLED` (default `true`).
- Code: `kiosk_core/speech_normalizer.py` (new), `kiosk_core/tts_client.py`.

### 5. Phrase-level TTS streaming + order-total truthfulness guard
Extended the existing sentence-level streaming gate to release phrases at
commas as well as sentence boundaries (not just full sentences), so audio
starts sooner within a long reply. Added a new guard (`price_guard.py`)
that tracks the authoritative order total from tool results and corrects
(never blocks) a hallucinated total if the model states one that doesn't
match, both for the early-release streaming gate and the final reply.

No latency win was observed on the two test recordings specifically
(Recording 1's reply is fully templated and bypasses the streaming gate
entirely; Recording 2's gate closes almost immediately for an unrelated,
pre-existing reason) — kept anyway as a correctness fix (price guard) and a
safe, potentially useful-elsewhere change (phrase splitting), with no
measured regression.

- Code: `plugins/kiosk/ordering_agent.py` (`_PHRASE_BREAK_RE`,
  `_SentenceGate.feed`), `plugins/kiosk/price_guard.py` (new).
- Tests: `rag-service/tests/test_price_guard.py` (17 tests),
  `tests/unit/test_sentence_gate_phrase_splitting.py` (8 tests).

### 6. ASR device: moved Whisper to NPU
Re-enabled NPU passthrough (`ACCEL_MOUNT_PATH` was silently `/dev/null`,
disabling it) and moved `whisper-small` from CPU to NPU. A controlled,
isolated A/B (same clip, same warm container, diarization forced off during
the isolation test only) showed NPU (542–557 ms) and GPU (533–584 ms)
effectively tied for `whisper-small` on this hardware. NPU was kept anyway,
matching `kiosk-voice-lab-main`'s own rationale: it frees the iGPU for
`ovms-llm` and `text-to-speech`, which contend for GPU time under load even
though the two devices tie in isolation.

`distil-whisper/distil-small.en` (the smaller model the reference prototype
actually uses on its NPU) was evaluated and **reverted**: it measured
2–3x faster in single-shot isolated benchmarks, but exhibited a confirmed
state-corruption bug on this stack's NPU + `openvino_genai` pipeline —
after processing one clip length and then a different clip length on the
same shared pipeline instance (exactly this production's request pattern:
a short warmup clip, then a real variable-length utterance), every
subsequent call returns garbled/wrong text and never recovers. Reproduced
directly against `openvino_genai.WhisperPipeline`, isolated from this
repo's own code — `whisper-small` does not exhibit the bug.
`distil-small.en` on **GPU** (not NPU) does not show the corruption either,
so it remains a candidate if GPU headroom is ever confirmed acceptable, or
if the NPU-specific bug is fixed upstream — not attempted here.

- Config: `configs/audio-analyzer/config.yaml` (`device: NPU`, model
  comments document the above).

### 7. Optional Silero VAD (default off)
Added an alternative, model-based VAD (`silero_vad.onnx`, v5, via
`onnxruntime`) as an opt-in replacement for the existing adaptive RMS-based
VAD's speech/silence classification. Bundled the ~2 MB ONNX model with the
repo rather than fetching it at runtime. Left **disabled by default** — the
RMS VAD is the well-tested, currently deployed default (see
`tests/unit/test_adaptive_vad.py`); Silero is available for future A/B
testing but was not adopted as the default in this exercise.

- Config: `KIOSK_CORE_SILERO_VAD_ENABLED` (default `false`).
- Code: `kiosk_core/silero_vad.py` (new), `kiosk_core/config.py`,
  `kiosk_core/audio_session.py`.
- Tests: `tests/unit/test_silero_vad.py` (new, does not affect default
  behavior).

### 8. Skip empty final tail-chunk ASR call (experimental, default off)
At end-of-turn, the final tail audio chunk usually holds only trailing
silence — the adaptive-pause flush earlier in the turn already sent every
real word to ASR. Transcribing pure silence still pays Whisper's fixed
per-call round trip for no benefit. Added a check that skips that final
call when no unflushed speech landed since the last flush; a genuinely
short trailing utterance still gets its final flush as before. Left
**disabled by default** pending a live A/B measurement.

- Config: `KIOSK_CORE_SKIP_EMPTY_FINAL_FLUSH_ENABLED` (default `false`).
- Code: `kiosk_core/audio_session.py`, `kiosk_core/pipeline_latency.py`
  (`AsrSpan.final_flush_skipped`).
- Tests: `tests/unit/test_skip_empty_final_flush.py` (new).

### 9. Analyzer client: explicit `language` field fix
`AnalyzerClient.transcribe()` previously omitted the `language` form field
entirely when no language was set, which let the analyzer's endpoint
default (`"en"`) apply regardless of intent — a genuinely empty string is
also swallowed as "not provided" by the multipart parser. Now always sends
an explicit value (falling back to a single space, which the analyzer's own
normalizer treats as "unset"), so English-only ASR checkpoints that reject
any language token can be genuinely configured with no language set.

- Code: `kiosk_core/analyzer_client.py`.

### 10. Latency instrumentation
Added explicit trace fields so voice-to-voice latency can be measured the
same way the reference prototype and `kiosk-voice-handoff.pdf` measure it:
customer's last word → first sound out of the speaker.

- `WallTimes.endpoint_wait_ms` — how long the endpointer waited in trailing
  silence before committing the turn.
- `WallTimes.voice_to_voice_ms` — last word → first audio (may be the
  opener).
- `WallTimes.voice_to_voice_informative_ms` — last word → first audio that
  carries the actual answer (excludes the opener).
- Code: `kiosk_core/pipeline_latency.py`.

## Investigated and explicitly NOT adopted

### Continuous rolling ASR at shorter intervals
`kiosk-voice-lab-main` re-transcribes partial audio every 150–400 ms during
speech. This repo's own `DEFAULT_CHUNK_SECONDS` history documents it being
deliberately raised from 2.5 s to 6.0 s after two production bugs: Whisper
hallucinating on mid-word splits, and short severed-tail fragments (0.5–1.5
s) being misclassified as a different speaker by diarization, silently
discarding real customer words. The reference prototype avoids this because
it runs a true streaming NPU ASR decoder with no word-boundary problem;
this stack's batch-Whisper backend would reintroduce the already-fixed
bugs if the flush interval were shortened. **Skipped.**

### OpenVINO GenAI in-process LLM pipeline (vs. current OVMS-HTTP)
Prototyped in an isolated throwaway container (never touching the
production `rag-service` container): loaded the exact same model
(`Qwen3-4B-int8-ov`) via `openvino_genai.LLMPipeline` directly on GPU, fed
the exact production system prompt and a real order request, with thinking
mode correctly disabled to match production. Result: **~799 ms** median,
essentially identical to the **~786 ms** median measured hitting OVMS over
HTTP for the same prompt/model/device — because OVMS's own `text_generation`
task is itself backed by the same OpenVINO GenAI runtime; there is no
meaningful HTTP/serving-layer tax to remove.

Switching to in-process would also cost: a ~6.5–8.3 s one-time model load
penalty per restart, loss of OVMS's continuous batching
(`max_num_seqs=4`) and prefix caching across turns
(`enable_prefix_caching`, avoids re-prefilling the ~1,300-token system
prompt every turn), loss of native `hermes3` tool-call parsing, and a full
reimplementation of ADK's tool-calling loop on raw text generation.
**Not adopted — no win, real regressions, large engineering cost.**

## Why `kiosk-voice-lab-main` is faster (~700 ms vs. this stack's 3.5–8.6 s)

The gap is architectural, not a difference in model speed:

1. **Speculative decoding while the customer is still speaking.** The
   prototype re-transcribes the partial audio buffer every 150–400 ms and
   fires a background LLM draft on every changed partial transcript,
   preempting stale drafts as newer transcripts arrive. By the time
   end-of-speech fires, a matching complete (or nearly complete) answer is
   usually already sitting ready. This repo starts ASR, the agent call, and
   TTS from zero only *after* the full silence timeout fires — there is no
   speculative pre-computation.
2. **Single in-process pipeline vs. multi-hop HTTP microservices.** The
   prototype runs ASR, LLM, and TTS in one Python process. This stack calls
   across `kiosk-core → audio-analyzer → rag-service → OVMS → text-to-speech`
   over HTTP. Per-hop overhead was confirmed small (~10–20 ms) and is not
   the dominant factor, but it forecloses any cross-service speculative
   overlap.
3. **Leaner generation config.** `max_new_tokens=80` (vs. this stack's 192),
   no conversation history resent for
   retail-mode Q&A (`context_turns=0`), and simple regex-parsed `<act>`
   directives instead of a full JSON-schema tool-calling protocol.
   The prototype also runs `Qwen3-4B` at **int4** (`scripts/setup.sh` exports
   `Qwen/Qwen3-4B-Instruct-2507` to `models/qwen3-4b-int4`), whereas this
   stack as actually deployed runs **int8**. Note the subtlety: the
   `docker-compose.yml` *default* is `OpenVINO/Qwen3-4B-int4-ov`, but both
   `.env` and `.env.example` pin `OVMS_MODEL_NAME=OpenVINO/Qwen3-4B-int8-ov`,
   and `.env` wins — `GET :8000/v3/models` on the running stack returns
   `OpenVINO/Qwen3-4B-int8-ov`. So precision *is* part of the gap. Always
   confirm the served model from `/v3/models` rather than from the compose
   default.
4. **Prefix caching shared across in-turn drafts, not just across turns**
   — per-turn-changing content (RAG chunks, cart state, question) is placed
   at the end of the prompt so every speculative draft during a turn shares
   the cached prefix.
5. **Phrase-level TTS pre-synthesis overlapped with generation** — the
   first phrase is synthesized as soon as any usable prefix of the draft
   exists, while the rest is still decoding.

Adopting the speculative-decoding architecture in full would require moving
ASR/LLM/TTS in-process and reimplementing partial-ASR ticking,
draft speculation/preemption, and prefix-shared drafting on top of ADK's
tool-calling loop — a substantially larger, higher-risk change than
anything in this document. Scoped out as a known, larger future
opportunity, not undertaken in this exercise.

## edge-ai-libraries changes (`/sainijit/edge-ai-libraries`, `microservices/`)

Committed separately in that repository (see its own commit for the diff).
Summary:

- **`microservices/text-to-speech/`**: added a Kokoro-82M TTS backend
  (`components/tts/kokoro/`, new `kokoro-onnx` runtime option) alongside
  the existing SpeechT5/Qwen backends, selectable via
  `models.tts.runtime: kokoro`. Measured at roughly 1/3 SpeechT5's
  first-phrase synthesis cost on CPU (~280 ms vs. ~2,100 ms) with 4
  intra-op threads. MIT-licensed package / Apache-2.0 model weights (same
  licensing bar the reference prototype applied when it chose Kokoro over
  Piper/MeloTTS). Added `utils/ensure_kokoro.py` (model download/cache),
  voice validation in `dto/speech_dto.py`, a runtime whitelist update in
  `utils/config_loader.py`, and a warning log (previously silent) on
  synthesis-rejection 400s in `api/openai_endpoints.py`.
- **`microservices/audio-analyzer/`**: re-tuned `config.yaml` (silence/
  no-speech/logprob thresholds relaxed to match `audio_preprocessing.denoise`
  being enabled; `beam_size=3` for real beam search instead of greedy
  decoding; `word_timestamps: true`, required for per-word diarization
  splitting). Added `benchmarks/` (isolated NPU/GPU/CPU A/B harness and
  results used to make the ASR-device decisions documented above).

## Testing performed

- 42 new unit tests added this exercise, all passing:
  `rag-service/tests/test_price_guard.py` (17),
  `tests/unit/test_sentence_gate_phrase_splitting.py` (8),
  `tests/unit/test_endpoint_completeness.py`,
  `tests/unit/test_sentence_gate_pregrounded.py`,
  `tests/unit/test_silero_vad.py`,
  `tests/unit/test_skip_empty_final_flush.py`.
- Full existing suite re-run after each change: no regressions (3
  pre-existing, unrelated failures confirmed via `git stash` A/B to be
  caused by a missing local `google.adk` dependency, not by these changes).
- Live A/B on both real recordings (`tests/rec1_16k.wav`,
  `tests/rec2_16k.wav`) after each behavior-affecting change, comparing
  against the change reverted.

## Current state / recommendation

- Fully adopted (default-on): opener pre-synthesis, adaptive endpointing,
  TTS speech normalization, phrase-level streaming + price guard, NPU ASR
  device, analyzer `language` field fix, Kokoro TTS backend.
- Available but off by default, pending further live measurement: Silero
  VAD, skip-empty-final-flush.
- Explicitly not pursued: continuous rolling ASR at shorter intervals,
  in-process OpenVINO GenAI LLM pipeline, full speculative-decoding
  architecture (scoped as a known larger future opportunity).

---

# Follow-up round: prompt-mass and endpointing (October)

Prompted by a working `kiosk-voice-lab-main` replay on the dev box, which
gave a measured reference to compare against instead of the PDF's headline
number.

## Reference measurement (the prototype, reproduced)

39 turns across 4 replay scripts (`make replay SCRIPT=demo|session2|session3|session5`),
metric = endpoint decision → first audio sample:

| | ms |
|---|---|
| min / p25 | 1 / 334 |
| **median** | **529** |
| p75 / p90 / p95 | 1032 / 1218 / 1694 |
| mean / max | 643 / 1895 |

Quality on the same runs: WER 0–4.6%, **0 false triggers, 0 forced ends**.
Adding the endpointer's own 150 ms complete-sentence wait reconstructs
≈680 ms voice-to-voice, consistent with the handoff PDF's ~700 ms.

The decisive split is *when the work happened*, not how fast it ran:

| Turn class | n | median |
|---|---|---|
| Reply audio already synthesized before endpoint | 10/39 | **159 ms** |
| Draft text ready, TTS after endpoint | 28/39 | 697 ms |
| No usable draft (cold) | 1/39 | 1895 ms |

**Draft hit rate 38/39 (97%).**

Getting the prototype to run required fixing three broken setup artifacts on
the box (recorded here because `make setup`/`make drivers` fail silently):
the Intel NPU userspace driver was never installed (`intel-level-zero-npu` /
`intel-driver-compiler-npu`, so OpenVINO enumerated CPU/GPU only), and both
`tts_models/kokoro/voices-v1.0.bin` and `models/silero_vad.onnx` were 35 KB
proxy error pages rather than the real 28 MB / 2.3 MB assets.

## Why the LLM stage costs so much more here

Measured with the actual Qwen3-4B tokenizer, not estimated:

| Component | Lab | This stack |
|---|---|---|
| Model precision | int4 | **int8** (`.env` pins it; see the correction above) |
| System prompt / agent instruction | **881** | **1,509** |
| Tool schemas | **0** (regex `<act>` directives) | **~2,697** (12 MCP tools) |
| **Static prefill per call** | **881** | **4,206** |
| **LLM calls per ordering turn** | **1** | **2** (but see below) |

The 12 MCP tool schemas alone are ~3× the lab's entire system prompt. And the
lab emits the state change and the speech from *one* generation
(`<act>add|Ranger Double Burger|2</act>Two Ranger Doubles, got it.`), where
ADK must generate a tool call, execute it, then generate the reply.

**Correction from measurement: the second LLM call is already effectively
free.** Profiling one `place_order` turn on the running stack gives:

| stage | median | share |
|---|---|---|
| LLM#1 (tool selection) | 2,334 ms | **93.1%** |
| MCP tool execution | 162 ms | 6.5% |
| LLM#2 (reply) | **11 ms** | 0.4% |

An earlier round's templated-reply shortcut means LLM#2 usually skips
generation entirely for ordering turns. So the "2 calls ≈ 2× the work" part
of the analysis above does **not** hold in practice — the cost is
overwhelmingly a *single* prefill-dominated call. That is why this round's
prefill reduction (change 11) is the only thing that moved the number, and it
is where further effort should go.

## Changes made this round

### 11. MCP tool-schema compaction (`Returns:`/`Raises:` stripped)
The MCP tool docstrings are Google-style and document their return payload in
detail for humans reading `mcp_server.py`. That prose never influences which
tool the model picks or how it fills arguments — the model sees the real
result after the call — but it was re-sent as part of the tool schema on
every LLM round-trip.

`compact_tool_description()` truncates a description at the first
`Returns:`/`Raises:`/`Yields:`/`Example:`/`Note:` heading. The summary line
and the whole `Args:` block, which *do* steer tool selection and argument
filling, are kept verbatim. Docstrings in source are untouched (project
convention requires them); only the LLM-facing copy is trimmed.

Measured through the real `MCPTool.prompt_description` code path across all
12 kiosk tools: **1,627 → 1,199 tokens, −428 per call, −856 per turn.**

**End-to-end effect (measured on the running stack, Arc iGPU, int8):**
`tests/benchmarks/agent_latency_benchmark.py --tier A --runs 5`, one
`place_order` turn, all four changes off vs. changes 11+14 on:

| | baseline | 11+14 on | Δ |
|---|---|---|---|
| agent turn, median | 2,506 ms | **2,277 ms** | **−229 ms (−9.1%)** |
| LLM#1 (tool selection), median | 2,334 ms | 2,120 ms | −214 ms |
| range (min–max) | 2,457–2,528 | 2,265–2,312 | no overlap |

The ranges do not overlap, so the win is real rather than run-to-run noise,
and it lands almost entirely on LLM#1 — exactly the prefill stage the
compaction targets.

- Config: `AGENT_COMPACT_TOOL_DESCRIPTIONS` (default `true`).
- Code: `rag-service/agentic/mcp_client.py` (`compact_tool_description`,
  `MCPTool.prompt_description`), `rag-service/agentic/config.py`,
  `plugins/kiosk/ordering_agent.py`, `rag-service/agentic/ordering_agent.py`.
- Tests: `rag-service/tests/test_mcp_client.py` (8 new).

### 12. OVMS prefill chunk budget 4096 → 8192 — TRIED, REVERTED
The hypothesis was sound: the static part of every agent prompt measured
**4,206 tokens** before history, the `[knowledge]` block or the customer's
message, so `--max_num_batched_tokens 4096` was splitting every LLM call into
multiple chunked-prefill iterations, and this is a single-customer kiosk
pinned to `PERFORMANCE_HINT=LATENCY` where chunking's throughput fairness
buys nothing.

It was measured and produced **no improvement**:

| | 4096 | 8192 |
|---|---|---|
| isolated agent turn, median | **2,277 ms** | 2,306 ms |
| 6-turn replay, `llm_ms` median | 5,951 ms | 5,869 ms |
| 6-turn replay, `llm_ms` mean | 5,682 ms | 6,112 ms |

Both deltas are inside run-to-run noise and they point in opposite
directions. The reason is change 11: compaction removes ~428 tokens per call,
which pulls the static prefill back **under** 4096, so the chunking the
larger budget was meant to avoid no longer happens. Change 11 subsumes
change 12.

Reverted to the upstream default per the project rule that a change is only
kept if it measurably improves something. Worth revisiting only if the prompt
grows materially again.

### 13. Silero VAD enabled by default
Built and unit-tested in the previous round but left off. Every turn timer —
the 0.70 s adaptive flush, the sentence-completeness shortcut, and the 1.5 s
silence endpoint — starts counting from the frame the VAD first calls
silence. An absolute RMS gate declares silence late in a room louder than the
one it was calibrated for (measured on the demo unit: silence floor RMS
~1076 against a 900 gate, so `silence_run_seconds` never accumulated and
*neither* the endpoint nor the adaptive flush could fire at all). Silero
scores speech directly, so the clock starts at the true end of the words and
every downstream timer shifts earlier with it.

**Bug found and fixed while enabling this.** Silero v5 accepts 8 kHz and
16 kHz only. `SileroVAD` documented that constraint but did not enforce it:
an unsupported rate builds a valid ONNX session and only fails much later, at
*inference* time, deep in the decoder LSTM
(`Input X must have 3 dimensions only`). The existing "fail open to RMS VAD"
guard wraps the constructor, so it never caught this — the session crashed
mid-turn instead. Turning the flag on by default converted that latent bug
into a live one for any session not at 16 kHz, which is not hypothetical:
Kokoro TTS emits **24 kHz**, so every turn of the conversation replay
benchmark failed.

The rate is now validated in `__init__`, which raises `ValueError`, which the
existing guard catches and downgrades to the RMS VAD with a concise warning
instead of a stack trace. Verified by replaying a 6-turn conversation at
24 kHz: 0/6 turns before the fix, 6/6 after.

- Config: `KIOSK_CORE_SILERO_VAD_ENABLED` (default flipped `false` → `true`).
- Code: `kiosk_core/silero_vad.py` (`SUPPORTED_SAMPLE_RATES`, constructor
  validation), `kiosk_core/audio_session.py` (`ValueError` handled distinctly
  from unexpected failures).
- Tests: `tests/unit/test_silero_vad.py` (7 existing + 10 new, run against
  the real model).

### 14. `AGENT_MAX_TOKENS` 192 → 128
Sized from measurement rather than guesswork: the longest legitimate outputs
are a full category listing (**84 tokens**) and a multi-item `update_order`
tool call (**56 tokens**), so 128 leaves ~50% headroom.

**This does not shorten a normal turn** — generation stops at EOS. It only
lowers the ceiling on the pathological case the cap exists for (a turn that
free-runs instead of calling a tool). Recorded as a worst-case bound, not a
median win.

## Investigated and explicitly NOT changed this round

### Lowering `KIOSK_CORE_ENDPOINT_SHORT_SECONDS` (1.0 s → ~0.3 s)
Proposed, then rejected on reading the code. `config.py` documents a hard
invariant: `ADAPTIVE_FLUSH_PAUSE (0.70) < ENDPOINT_SHORT (1.0) <
SILENCE_TIMEOUT (1.5)`. The completeness shortcut can only fire once the
adaptive flush's ASR round-trip has landed, measured at **~1.23–1.28 s** —
already later than the 1.0 s floor. So the binding constraint is ASR latency,
not the threshold, and lowering it would gain nothing while risking a
fail-closed check reading a stale transcript. The only real lever here is
making the transcript land sooner, which means the preview-ASR work below.

### Prompt reordering for prefix-cache reuse
Checked and found already correct: the ADK prompt is
`system + tool schemas → history → turn tail`, and the turn's second LLM call
is a strict extension of the first call's prefix, so `--enable_prefix_caching`
(already on) covers it. No change needed.

## Testing performed

- 18 new unit tests this round: 8 in `rag-service/tests/test_mcp_client.py`
  (compaction) and 10 in `tests/unit/test_silero_vad.py` (sample-rate
  validation). `tests/unit/test_silero_vad.py` is fully green (17 passed)
  against the real ONNX model.
- **Regression check by A/B against the committed baseline.** The full suite
  was run twice on the same box — once with this round's changes stashed and
  once with them applied — and produced byte-identical totals:
  `17 failed, 103 passed, 3 skipped, 32 errors` both times. The changes
  therefore introduce **no regressions**.
- Those 17 failures / 32 errors are a *host environment* fault, not a code
  fault: `.setup-venv` hits
  `ImportError: cannot load module more than once per process` from numpy
  when the functional tests do their deferred `import main`. They reproduce
  on the untouched commit. Do not read them as a baseline to accept —
  the earlier "492 passed" figure came from a healthier venv, and the venv
  should be rebuilt before the next full-suite run.
- Live stack verification: all six services healthy via `make test`; the
  compacted tool schemas confirmed reaching the agent
  (`Agent rebuilt with 12 MCP tool(s) ✓`, warmup clean on attempt 1).
- `docker compose config -q` validated after every compose edit.

### Benchmark harness fix (required to measure at all)
`tests/benchmarks/conversation_replay_benchmark.py` hard-coded
`sample_rate: "16000"` when starting the kiosk-core session, but the rate is
decided by whichever TTS backend synthesises the prompt. With Kokoro (24 kHz)
every turn was rejected outright by kiosk-core's rate check. The harness now
reads the frame rate from the synthesised WAV header, so it follows the TTS
backend instead of assuming one.

### Gotchas worth recording
- **`REGISTRY` in `.env` is a build-mode switch, not a registry name.** It is
  literally `true`. The Makefile maps `true|false` → `_ENV_REGISTRY` (`intel`)
  before calling compose; a raw `docker compose` command does not, and
  silently resolves images to `true/rag-service:...`. That is a *different,
  stale* image, so edits appear not to take effect. Always drive the stack
  through `make`, or pass `REGISTRY=intel` explicitly.
- **`make build` did not pick up source edits made after the last image
  build.** The running `rag-service` image predated the changes by ~14 h,
  while `plugins/` is bind-mounted live — so new plugin code called into old
  baked `agentic/` code and the agent failed warmup with
  `'MCPTool' object has no attribute 'prompt_description'`. Verify a change
  actually landed with
  `docker exec rag-service grep -c ... /app/rag-service/agentic/...` before
  trusting any measurement.
- **`make build` also clobbers the local Kokoro TTS image**: it builds the
  local services and then unconditionally runs
  `docker compose pull audio-analyzer text-to-speech`, overwriting the
  locally built `text-to-speech` (which has the Kokoro backend) with the
  upstream one (which does not, and rejects `runtime: kokoro`). Rebuild it
  explicitly afterwards.

## Round 2 (2026-09-08): endpoint-wait / ASR structural fix

Two changes, both shipped:

1. **Persistent `httpx.Client` in `AnalyzerClient`** (`analyzer_client.py`).
   Previously every single `transcribe_file()` call opened and closed a
   brand-new `httpx.Client()` — no keep-alive connection reuse across a
   turn's 2-5 chunk flushes. Now one client is created per `AnalyzerClient`
   instance (which is itself scoped 1:1 to an audio session / turn) and
   explicitly closed in `_finalize_run`. Free, no behaviour change.

2. **Preview-ASR flush during active speech** (`audio_session.py`,
   `config.py`: `DEFAULT_PREVIEW_FLUSH_ENABLED` / `_INTERVAL_SECONDS`, default
   1.5s). Previously a chunk was only flushed to ASR at the 6s hard cap or
   once at the 0.70s adaptive pause — so a long utterance's *entire* audio
   was on the critical path for one ASR round-trip once the customer
   stopped talking. Now a chunk is also flushed periodically **while the
   customer is still talking** (self-relaunching: never more than one
   in-flight preview call, respecting audio-analyzer's global
   `WhisperPipeline.generate()` lock), so most of a long utterance is
   transcribed off the critical path and the chunk flushed at the pause is
   much shorter.

   This reuses the exact same `_flush_chunk` path as the existing
   chunk-cap/adaptive-pause flushes (same diarization gating, same
   transcript-append code) — it only changes *when* a chunk boundary is cut.

   **Bug found and fixed while building this**: adding more non-final
   (non-diarized) flushes per turn exposed a **latent, pre-existing
   duplication bug**. audio-analyzer's `append_to_session` mode returns the
   flat `"text"` field as the FULL cumulative session transcript on every
   call (confirmed in the sibling `audio-analyzer` repo's `pipeline.py`:
   `text_parts = [session_state["text"]] + [chunk_text]`), but kiosk-core's
   dedup logic only ever applied to the `segments` array (used only when
   diarization is requested), never to the flat-text fallback used by every
   non-diarized intermediate flush. With only one intermediate flush per
   turn (the old norm) this was invisible; with 3-5 preview flushes per
   turn it produced 2-3x duplicated transcripts. Fixed by adding a second,
   string-prefix-based cursor (`_last_cumulative_flat_text`) for the
   flat-text path, and — because a later *diarized* final flush's
   segment-timestamp cursor (`_last_analyzer_segment_end`) was going stale
   during the intervening non-diarized flushes — unifying both cursors
   around each chunk's own client-measured duration, advanced after every
   flush regardless of response shape. See the code comments in
   `_flush_chunk` for the full reasoning.

### Measured impact (rec1_16k.wav / rec2_16k.wav, live stack, warm)

- No duplicated transcripts, no regressions: `2 failed / 17 errors` in
  `tests/functional/test_api.py -m tier1` before and after (pre-existing
  `beartype` circular-import fault in this venv, confirmed identical via
  git-stash A/B — unrelated to these changes).
- `endpoint_wait_ms` did **not** shrink for either test clip (still hits the
  full 1.5s / 1.3s silence timeout). Root cause: for a *short* utterance
  (a few seconds, like both test clips), the adaptive-pause flush already
  only ever covered roughly one utterance's worth of audio even in the OLD
  design — preview-flushing doesn't reduce the SIZE of that specific
  round-trip further for short clips, and per-call ASR overhead is fixed
  (not proportional to clip length), so its landing time is essentially
  unchanged (~1.25-1.3s, same as previously measured). This mechanism's
  benefit is expected to show up on LONGER utterances, where the old design
  paid one large round-trip at the pause and the new design has already
  transcribed most of it beforehand — not on these two short clips.
- `voice_to_voice_informative_ms` (last word → real answer) was
  **~5.75-5.8s for rec1, essentially unchanged from the ~5.75s baseline**,
  once measured on a genuinely successful order-completion turn (see
  gotcha below — an unrelated rag-service bug was initially making turns
  fail fast, which produced a misleadingly low number). For these two short
  clips, the dominant remaining cost is the agent/LLM + TTS pipeline
  (~4.3s from opener to first real spoken segment), not ASR or endpoint
  wait — matching the still-open "remaining opportunity" below.

### Gotcha hit again this round
Rebuilding `kiosk-core` and restarting `rag-service` reproduced the
**exact, already-documented** `'MCPTool' object has no attribute
'prompt_description'` bootstrap failure from the section above — a stale
`rag-service` image whose baked `agentic/` code didn't match the live
`plugins/` bind-mount. `docker compose build rag-service` (not just
`restart`/`--force-recreate`) resolved it. Recorded here again because it
recurred despite being documented — worth eliminating properly (e.g. bake
a build-time content hash check) rather than re-discovering it each round.

## Remaining opportunity (still the big one — confirmed by this round's measurement)

Speculative drafting during speech. This round's data reinforces it: even
after removing ASR/endpoint-wait as a major cost for short utterances, the
agent (LLM TTFT ~2.1-2.4s once warm) + TTS-of-first-segment dominate the
remaining ~4.3s. The prototype's 97% draft-hit rate is what buys its
median; nothing in this round touches *when* that work happens. The staged
path is: a throwaway preview-ASR tick (now partially built above — the
preview flush IS committed to the real transcript, not throwaway, since it
reuses the existing commit path; a true throwaway variant for driving
*speculative agent drafts* without waiting for endpoint would go further)
→ speculative agent drafts on changed previews with newest-wins preemption,
read-only turns first so no tool ever executes speculatively → first-phrase
TTS pre-synthesis from the draft prefix, reusing the existing opener cache.

## Round 3: Speculative drafting — implemented, safety-verified, defaulted OFF

All three phases from the "remaining opportunity" above were implemented:

1. **Phase 1 (cache/prefix warming)** — on every mid-speech preview flush,
   fire a speculative agent draft against the transcript-so-far, in the
   background, with `speculative=True`.
2. **Phase 2 (dry-run tool-enabled drafts)** — the draft is allowed to call
   real ordering tools (`place_order`, `update_order`, etc.) so its predicted
   reply reflects a real tool outcome, but every mutating tool call is
   forced into `dry_run=True` at the dispatch chokepoint
   (`_make_mcp_callable` in `plugins/kiosk/ordering_agent.py`), regardless of
   what the LLM requests, and the `dry_run` param itself is stripped from
   the LLM-visible tool schema so the model can never see or set it.
   `dry_run` skips `await db.commit()` in `ordering/service.py` — sqlite
   auto-rolls back the uncommitted transaction on connection close, so no
   restructuring of the repository/transaction layer was needed. Only the
   LATEST-started draft is ever kept ("newest-wins"); anything superseded by
   a newer preview is discarded outright.
3. **Phase 3 (TTS pre-synthesis)** — once a draft completes, its predicted
   reply is split into sentences and synthesised ahead of time into a
   text-keyed cache. The real turn's `_tts_worker` checks this cache before
   calling TTS — only an exact-text cache HIT is ever served, so nothing is
   spoken that the real, fully-guarded pipeline didn't independently decide
   to say.

**Safety verification (before any live test):** ran all 5 dry-run-capable
service methods against a real temp SQLite file — confirmed zero row
persistence in every dry-run case and correct persistence in the real case.
Live-replayed `rec1_16k.wav` afterward and diffed `orders`/`order_items` row
counts before/after — exactly one new (real) order, no leakage from the 3
speculative drafts that fired during that replay.

**Latency finding — this backfired and was disabled by default.**
Live-replaying `rec2_16k.wav` (a longer, multi-item utterance — the case
Phase 1-3 should help most) showed:

```
[SPEC-DRAFT] gen=4 ready ... tool_calls=['place_order']      @ 12:29:03.839
[SPEC-TTS]   pre-synth sentence (51 chars) gen=4              @ 12:29:07.376
[SPEC-TTS]   pre-synth sentence (57 chars) gen=4              @ 12:29:13.470
[SPEC-TTS]   pre-synth sentence (51 chars) gen=4              @ 12:29:18.514
[SPEC-TTS]   pre-synth sentence (37 chars) gen=4              @ 12:29:21.598
[SPEC-TTS]   pre-synth sentence (61 chars) gen=4               @ 12:29:24.960
[PIPELINE]   wall=38171ms ... tts=21764ms llm=7908ms(2) tools=['place_order']
```

The real turn's own `tts=` time (21.8s) almost exactly matches the ~21s the
5 speculative TTS calls consumed immediately before/during it. Re-running
the identical clip with both speculative flags forced off dropped wall time
from **38.2s → 19.8s** on the same clip, with zero speculative log lines.
Conclusion: the TTS backend (and very likely the LLM/OVMS backend too, given
llm=7908ms(2) is also elevated) **serialises requests** rather than serving
them concurrently — it does not have a separate worker or queue priority for
the real, user-facing turn. Firing speculative LLM/TTS calls during speech
means the real turn's own calls queue up behind them, so the "warm cache /
pre-synthesised sentence" benefit never has a chance to pay off; it's pure
added load on a single-lane backend.

**Action taken:** `DEFAULT_SPECULATIVE_DRAFT_ENABLED` and
`DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED` in `kiosk_core/config.py` were
flipped to default `false` (env-var override still available for future
testing). The code remains in place, fully safety-verified, as an opt-in —
it should not be re-enabled without first confirming the LLM/TTS backends
can serve a speculative and a real request concurrently (e.g. multiple OVMS/
TTS worker replicas, or a priority queue that lets the real turn preempt
speculative work), and re-validating via a live replay.

**Deliberately not implemented (recommended follow-up, still valid):** even
if backend concurrency is fixed, the current safe scope still runs the
real turn's own full LLM call every time — it only warms caches / pre-
synthesises audio speculatively. The larger optimisation of skipping the
LLM/guards entirely and replaying a matched draft's tool calls for real
(closing the rest of the gap to the lab's ~700ms) was not attempted here;
it requires replaying draft results through the same guard/reply-
construction pipeline the LLM path uses, not just re-dispatching raw tool
calls, and is a separately-scoped, higher-risk piece of work.

## Update: CPU contention root cause, HTTP keep-alive, and int4 vs int8 (this segment)

### 1. HTTP connection reuse (fixed, but not the bottleneck)

`TtsClient` and `AgentClient` were creating a brand-new `httpx.Client` on
every call (no connection reuse), unlike `AnalyzerClient` which already held
one persistent client per session. Both were fixed to match that pattern
(persistent client + `close()`, wired into `_finalize_run`). Benchmarked
before/after: **no measurable latency change** — connection setup on the
local Docker bridge network costs a few ms, negligible next to multi-second
model/compute costs. Kept the fix anyway (strictly correct, zero downside),
but it is **not** the explanation for the latency gap vs. the lab.

### 2. Real root cause: CPU contention from `queue-service` / `rtsp-streamer`

`docker stats` showed `queue-service` (YOLOv8 person-counting) at
**~650-750% CPU** and `rtsp-streamer` at **~350-500% CPU** — together
permanently consuming 10-12 of the host's 16 cores, **regardless of voice
activity**. `ovms-llm` and `audio-analyzer` run mostly on GPU, but
`text-to-speech` (Kokoro) is CPU-only and has to compete for whatever CPU is
left.

Controlled experiment (stop both services, re-run replay, restart them
after):
- Raw TTS call time: **1.3s → 0.45s** (~3x)
- `voice_to_voice_informative_ms`: **~5.7-6.6s → ~3.2-4.0s** (35-45% faster)

This is the single most validated, highest-confidence finding of the whole
investigation — bigger than any model/quantization change.

**Action taken:** added a `queue` compose profile (mirroring the existing
`identity` profile) to `rtsp-streamer` and `queue-service`, gated by the
existing `KIOSK_CORE_QUEUE_SERVICE_ENABLED` env var (now dual-purpose:
app-layer *and* container-startup gate). `Makefile` got a matching
`QUEUE=true|false` variable (default `true`, preserving current behaviour)
wired into `build`/`up`/`down`/`show-config`. Run `QUEUE=false make up` to
benchmark cleanly without the queue/RTSP load; default `make up` behaves
exactly as before.

### 3. int4 vs int8 quantization — deep dive

Initial full-pipeline A/B looked like a wash (~1.5-1.6s LLM time either
way). Deeper investigation found two compounding effects that were hiding
the real picture:

- **OVMS prefix caching** (`--enable_prefix_caching true`) is real and
  large: identical/shared-prefix prompts drop from full prefill cost to a
  cached-prefix floor on repeat. Within a single conversation, turn 1
  (cache miss) costs several seconds; turns 2+ (cache hit on the static
  system+tools prefix) drop 3-4x.
- **That cache has limited capacity and thrashes under many concurrent/
  historical distinct sessions.** After accumulating dozens of test
  sessions against one `ovms-llm` instance, *every* subsequent call reverted
  to full-prefill cost (~8-10s) with zero cache benefit, even for further
  turns in a brand-new session. Restarting `ovms-llm` (clearing its cache)
  immediately restored the fast, cache-hit steady state for a fresh session.
  **This is why production traffic — many different customers/sessions
  interleaved — never sees the clean cache benefit an isolated single-
  session test shows: real traffic is exactly the "many distinct prefixes"
  condition that thrashes the cache.**
- Prefix caching only ever saves **prefill** (processing input tokens); it
  cannot help **decode** (generating the output tokens one at a time),
  which is memory-bandwidth bound and where lower-precision weights
  (int4 vs int8) should matter most.

Clean, apples-to-apples test (fresh `ovms-llm` restart, single session,
`queue-service`/`rtsp-streamer` stopped to remove that noise source):

| | Turn 1 (cold) | Turns 2-4 (cache hit, steady state) |
|---|---|---|
| int8 | ~2.4s | ~1.25-1.3s |
| int4 | ~3.6s | ~1.14-1.26s |

int4's cold/prefill turn was *slower* (extra dequantization overhead during
the compute-bound prefill phase), but its steady-state/decode-heavy turns
were **~7-10% faster** than int8, consistent with int4 helping the
memory-bandwidth-bound decode phase. This is a real, if modest, benefit —
not the "wash" the earlier noisy comparison suggested.

**Gotcha (repeated twice this segment, worth calling out plainly):**
`OVMS_MODEL_NAME` is consumed by *both* `ovms-llm` (`--source_model`) and
`rag-service` (`AGENT_LLM_MODEL`). Recreating only one container after
changing it causes a model-name mismatch — OVMS returns
`404 Mediapipe graph definition not found`, which the agent's error handling
silently turns into "Sorry, I encountered an error" with a suspiciously fast
(~15-20ms) `llm_ms`. **Always recreate both `ovms-llm` and `rag-service`
together when changing `OVMS_MODEL_NAME`.**

**Net recommendation:** keep **int4** — same quality, modest but real decode
latency win, no observed downside. The bigger, unresolved lever is prefix
cache capacity/eviction policy under realistic concurrent-session load,
which is an OVMS-server-configuration question (cache size limits), not a
model-precision question, and would need further investigation with OVMS's
own cache-sizing flags if pursued further.

**Current stack state after this segment:** `OVMS_MODEL_NAME=OpenVINO/
Qwen3-4B-int4-ov` (both `ovms-llm` and `rag-service` recreated to match),
`queue-service`/`rtsp-streamer` restarted and running (default `QUEUE=true`
behaviour restored).

## Session: end-to-end voice-to-voice logging, opener removal, ASR device test, and the silence-wait fix

### 1. Accurate voice-to-voice logging added
Added explicit `[VOICE2VOICE]` log lines in `kiosk_core/audio_session.py`:
`event=last_word_spoken` (backdated to the true end-of-speech instant, not
the moment the endpoint loop happens to notice it) and
`event=first_response_audio` (the instant the first real-answer TTS segment
hits disk), with `voice_to_voice_ms` computed directly between them. Grep
pattern for monitoring: `docker logs kiosk-core --since <window> | grep
VOICE2VOICE`.

**Bug found and fixed while adding this:** the existing API-reported
`voice_to_voice_informative_ms` was silently undercounting real latency by
~700-800ms. `_finalize_run`'s `t0` is stamped only *after*
`self._flush_queue.join()` completes — i.e. after the tail chunk's ASR
round-trip — but the old formula reconstructed the last-word instant as
`t0 - endpoint_wait_seconds`, never accounting for that ASR round-trip gap.
Fixed by adding a `self._t_last_word` monotonic anchor set precisely at
endpoint-commit time, used directly by both the new logs and
`_record_turn_trace`'s `v2v_ms`/`v2v_informative_ms`. **All voice-to-voice
numbers reported earlier in this investigation (the ~5.7-8.2s range) were
understating true latency by several hundred ms** — treat those as
directionally correct but not exact.

**Known gap (not yet fixed):** `BrowserStreamSession` can end via the
browser's own client-side VAD (`/audio/end` → `signal_end()`) instead of the
silence-timeout loop that sets `_t_last_word`. On that path the anchor never
fires and the reported number falls back to a less accurate one. Turns that
exit via the silence-timeout loop (as analyzed below) get fully accurate
numbers; some real gradio-UI turns may not.

### 2. Opener removed
`KIOSK_CORE_OPENER_ENABLED=false` set in `.env` (parity with kiosk-voice-lab,
which has no opener). Clean disable — `_emit_opener()` early-returns, no
side effects. Side effect: `voice_to_voice_ms` (non-informative) now measures
time-to-first-sentence-*text*, not first-*audio* — use
`voice_to_voice_informative_ms` (or the new `[VOICE2VOICE]` logs) as the
correct metric going forward.

### 3. ASR on GPU + queue-service/rtsp-streamer stopped
Added a reversible override — `AUDIO_ANALYZER__MODELS__ASR__DEVICE:
${ASR_DEVICE:-NPU}` in `docker-compose.yml`, `.env`'s `ASR_DEVICE=GPU` —
instead of editing the mounted `configs/audio-analyzer/config.yaml` (which
pins `NPU` with its own benchmark rationale). With `queue-service` and
`rtsp-streamer` stopped (freeing CPU) and ASR moved to GPU, two live
conversations measured **4.63s** and **4.999s** voice-to-voice, a marked
improvement over the ~5.7-8.2s CPU-contended baseline.

### 4. Detailed breakdown of a 4.63s turn — root cause of the ~1.5s silence wait
Reconstructed from raw logs (`docker logs | grep <session_id>`) for the turn
"I would like to order. Order one classic chicken burger.":

- **~1.49s silence wait** (`silence_timeout_seconds`) — the single largest
  component of the total.
- The reported `asr=1134ms(4 chunks)` figure is misleading in isolation — it
  includes a first chunk (~334ms) that ran **concurrently while the customer
  was still speaking** (not on the critical path). Only the trailing chunks
  after speech stopped are real added latency.
- **Root cause of the silence wait not shortening**: the codebase already has
  a "completeness shortcut" (`DEFAULT_ENDPOINT_COMPLETE_ENABLED` +
  `DEFAULT_ENDPOINT_SHORT_SECONDS=1.0s`) that should commit at 1.0s instead
  of the full 1.5s when the transcript-so-far already reads as a finished
  sentence, gated on the adaptive-pause flush having landed
  (`_adaptive_flushed and flush_queue.unfinished_tasks == 0`). It never
  fired for this turn. Ground-truth log timeline showed why: the customer
  spoke continuously right up to the last word, so the phrase "Order one
  classic chicken burger." only got flushed for ASR by the **fixed
  chunk-size cap** (`chunk_seconds`, 1.5s) — not the adaptive-pause
  mechanism meant to catch this — because both thresholds were crossed on
  the same frame and the chunk-size-cap check ran first in the loop,
  winning the race via `continue` before the adaptive-pause check ever got
  evaluated. The cap flush landed at essentially the exact same wall-clock
  instant the silence timer also expired (42.888 vs 42.893), so the ASR
  response for the actual content (43.174) arrived *after* the endpoint had
  already committed on the full 1.5s fallback — the completeness shortcut
  had no transcribed content to judge in time.

**Fix implemented**: reordered the two flush checks in
`_process_frame_stream` (`kiosk_core/audio_session.py`) so the
**adaptive-pause flush is checked before the timed chunk-size cap**. No
threshold values changed — purely a priority fix so a genuine
end-of-speech pause always wins the race against the arbitrary fixed-duration
cap, instead of occasionally losing to it. This lets ASR start processing
the customer's final words as soon as a real pause begins, giving the
1.0s completeness shortcut a real chance to fire.

**Verified via file replay** (`tests/rec1_16k.wav`, same "order one classic
chicken..." phrasing): the shortcut fired for the first time observed in
this investigation —
`[ENDPOINT] early commit at 1.40s (transcript reads complete; saved 0.10s)`
— versus no early commit at all (full 1.5s paid) before the fix. Total
voice-to-voice for that replay: **3880ms**. Savings per turn will vary
(0.1-0.5s) depending on how quickly the adaptive-pause flush's ASR
round-trip lands relative to the 1.0s/1.5s checkpoints, but the shortcut is
now reachable where it previously never fired.

**Not done**: no threshold values (`DEFAULT_SILENCE_TIMEOUT_SECONDS`,
`DEFAULT_ENDPOINT_SHORT_SECONDS`) were changed — this fix is pure ordering,
zero added cutoff risk. Tightening the fallback timeout itself remains a
separate, higher-risk lever if further reduction is wanted.

**Other bugs found and fixed this segment (infra, not latency-related):**
- `REGISTRY=true`/`false` in `.env` are Makefile-only boolean flags
  (translated to the real `intel` prefix by `make`'s `_DOCKER_REGISTRY`
  logic); raw `docker compose` commands bypass this translation and use the
  literal value as the image prefix. `REGISTRY=true` was breaking any direct
  `docker compose` compose call (`pull access denied for true/...`). Fixed
  `.env` to `REGISTRY=false` (also the semantically correct "build locally"
  setting) and confirmed via `make show-config`.
- `Makefile`'s `build` target unconditionally pulled `text-to-speech` from
  the registry regardless of `REGISTRY`, silently overwriting the
  custom-built Kokoro-TTS-enabled image (built from a sibling
  `../../edge-ai-libraries/microservices/text-to-speech` checkout) with a
  generic one that only supports `runtime: openvino|pytorch` — breaking
  startup since `configs/text-to-speech/config.yaml` pins `runtime: kokoro`.
  Fixed: `text-to-speech` now always builds locally, unconditionally, in the
  `build` target.

**Current stack state after this segment:** opener disabled, ASR on GPU,
queue-service/rtsp-streamer stopped (per active testing), kiosk-core
rebuilt with the flush-ordering fix and accurate voice-to-voice logging,
text-to-speech rebuilt locally with Kokoro support, `REGISTRY=false` in
`.env`.

---

## Round 4: tool-description scrubbing + first-phrase TTS split

Baseline for this round: **3,952 ms** mean voice-to-voice (`rec1_16k.wav`,
warm runs, opener disabled).

### Three governing constants (re-measured, and one correction)

An earlier round recorded "prefill ≈ 10 ms" and concluded that every
prompt-size optimisation was a dead end. That measurement was taken with a
*tiny* prompt and does not describe the agent. Replaying the agent's real
request against OVMS (real `_AGENT_INSTRUCTION` + real MCP tool schemas,
6,352 + 9,882 chars ≈ 16 KB) gives:

| Quantity | Value |
|---|---|
| Prefill (warm, prefix-cached) | **~480 ms**, and *flat* |
| Marginal decode, agent context | **~30 tok/s** (~33 ms/token) |
| Kokoro TTS | **248 ms fixed + 13.9 ms/char** |

The correction matters, but the *conclusion* survives in a stronger form:
prefill is ~480 ms and **does not move with context size**. Measured across
16,234 → 5,572 chars of context, TTFT stayed within 445–493 ms. So trimming
the system prompt or dropping tools cannot buy prefill time. What context
size *does* change is how many tokens the model emits.

### The actual bottleneck: tokens the model never needed to emit

`ttft_ms` in `[PIPELINE]` is a misnomer — it is the whole agent call, because
LiteLLM buffers a function-call until the arguments are complete. So the
~1,517 ms agent call was ~480 ms prefill + ~1,030 ms decoding a tool call.

`user_id`, `dietary` and `dry_run` are already injected by `_mcp_fn` and
already stripped from the model-visible JSON schema. But the model kept
emitting `"user_id": "kiosk-user"` anyway — because each tool's
**description still documented the parameter** in its `Args:` block, and the
model follows the prose over the schema. Measured directly against OVMS:

| Tool declaration | Emitted tokens | Call latency |
|---|---:|---:|
| Full schemas (nothing hidden) | 36 | 1,671 ms |
| Params hidden from schema only (what we shipped before) | 27 | 1,438 ms |
| **Params also scrubbed from the description** | **17** | **1,079 ms** |
| Descriptions blanket-truncated to 200 chars | 18 | 1,054 ms |
| Only 8 "core" tools advertised | 10 | 783 ms |

The last two rows were **rejected**. Blanket truncation and tool removal are
faster but destroy correctness: with a reduced tool set the model emitted
`place_order{"items": {"classic chicken": "1"}}` — an object instead of the
required list, a display name instead of a `product_id`, and a string
quantity. Truncation also silently deletes the behavioural guidance the
ordering guards depend on (e.g. `get_current_order`'s "never call
`get_order` with a guessed id" paragraph).

**Shipped:** `_scrub_injected_args()` + `_injected_param_names()` in
`plugins/kiosk/ordering_agent.py`, applied to `_mcp_fn.__doc__`. It removes
*only* the `Args:` entries for parameters the runtime injects, leaving every
other line intact. `_injected_param_names()` is shared with the schema-
stripping loop so the schema and the description can never disagree.

Note `plugins/kiosk/ordering_agent.py` is the live file (bind-mounted via
`./plugins:/app/rag-service/plugins:ro`, loaded as `plugins.kiosk.
ordering_agent`). `rag-service/agentic/ordering_agent.py` is an unreferenced
legacy copy — editing it has no effect.

### First-phrase TTS split

Because Kokoro costs 248 ms + 13.9 ms/char, the *first* segment's length is
almost the whole of time-to-first-audio. The opening sentence "I've added
Classic Chicken Burger to your order." (47 chars) cost ~900 ms before any
sound reached the customer.

`BaseAudioSession._split_first_phrase()` splits **only the first segment** of
a turn at a 5-word cap (`KIOSK_CORE_TTS_FIRST_PHRASE_MAX_WORDS`), producing
"I've added Classic Chicken Burger" + "to your order." — a break after the
object noun phrase, not inside it. Later segments are deliberately left
alone: they are already overlapped with playback, so splitting them would
only add 248 ms of fixed cost each plus extra prosody seams. A split is
skipped unless at least 3 words remain
(`KIOSK_CORE_TTS_FIRST_PHRASE_MIN_TAIL_WORDS`), so a 6-word sentence is not
chopped into a 5-word head and a 1-word orphan.

### Results

| Stage | V2V (rec1, warm mean) | Agent call |
|---|---:|---:|
| Round 3 baseline | 3,952 ms | ~1,517 ms |
| + first-phrase split | 3,838 ms | ~1,517 ms |
| + description scrubbing | **3,479 ms** | **~1,190 ms** |

**−473 ms (−12%)**, with the agent call down 327 ms — within noise of the
359 ms predicted by the isolated OVMS measurement. Cart correctness verified
on every run (`classic_chicken`, ₹169); `rec2_16k.wav` regression unchanged
(its unresolved `chicken`/`french fry` references are pre-existing).

### Measured dead ends this round

- **Freeing the iGPU** (ASR `GPU→NPU`, reranker `GPU→CPU`): decode rate
  35.0 vs 34.6 tok/s, V2V 3,947 vs 3,952 ms. Decode is memory-bandwidth
  bound, not contended. Device moves kept (neutral, and ASR-on-NPU matches
  the documented preference) but they buy nothing.
- **Shortening the system instruction** (6,352 → 2,500 chars): made things
  *worse* — emitted tokens rose from 10 to 17, because the guidance that
  keeps the tool call well-formed was part of what was cut.

### Trap worth knowing

OVMS defaults Qwen3 to **thinking mode**. A request without
`chat_template_kwargs={"enable_thinking": false}` burns 115+
`reasoning_content` tokens and never produces content within a 120-token
budget (~3.4 s wasted). `adk_runtime.py` and `ovms_llm_service.py` both set
it correctly today, but any new OVMS caller that forgets will silently
regress by seconds.

---

## Round 5: TTS lead-clause + cart-read templating

Two changes, both aimed at the two costs that actually dominate a turn: the
duration of the *first* TTS segment, and the *second* LLM call.

### Measured first: is the WAV file the problem?

A proposal was raised to stream TTS bytes continuously instead of writing a
WAV file per segment. Measured component-by-component before changing
anything (`kiosk-core`, warm, "I've added Classic Chicken Burger"):

| Component | Time |
|---|---:|
| TTS HTTP call (Kokoro synthesis + full WAV transfer) | 490–613 ms |
| **WAV file write** | **0.1 ms** |
| `_trim_tts_segment` | 1.0 ms |
| `_apply_tts_gain` | 0.6 ms |

**Rejected.** The file write is 0.1 ms — essentially 100% of first-audio
latency is Kokoro synthesis inside the HTTP call. Byte streaming would also
not help *within* a segment: Kokoro's `create_stream()` emits a single chunk
(independently measured by the reference lab, which disabled streaming for
this reason), so there are no partial frames to forward. The only real levers
on first audio are **synthesise less text first** (P1 below) or **synthesise
earlier** (speculative pre-synth, still open).

### P1 — lead-clause in `item_added` (−233 ms)

Kokoro costs ~208 ms fixed + ~9 ms/char, and playback cannot start until
segment 1 exists. Segments split on sentence punctuation, so prefixing a
short sentence makes segment 1 ~5 chars instead of 33–48:

```yaml
item_added: "Done. I've added {items} to your order. Your total is now {currency}{total}."
```

Everything after "Done." synthesises in parallel while it is already playing,
so the extra segment is free on the critical path. It is truthful by
construction — the template is only reachable after the order tool returned
success.

**rec1 V2V: 3,479 → 3,246 ms** (3 warm runs: 3212 / 3360 / 3166). Cart
correct (₹169, 1× BURGER-NV-001).

### P2 — templates for `get_current_order` / `get_popular_products`

"What's in my cart" previously had no template, so it paid a full second LLM
call (~1,200 ms) to restate data already present in the tool result.

* `get_popular_products` returns the same shape as `list_products`, so it
  reuses `speak_catalogue` verbatim — no new templater.
* `get_current_order` gets `speak_current_order`, plus `cart_summary` /
  `cart_empty` in `reply_templates.yaml`.

The old `_CATALOGUE_TOOLS` constant conflated two separate concerns, so it
was split: `_READ_TOOLS` (browse-intent gate) and `_LIST_RESULT_TOOLS`
(list-vs-object decoder choice). `get_current_order` needs the first but not
the second.

**Live cart-read turn: 777 ms total, `llm_calls: 1`, `templated: true`**
(previously 2 calls).

#### Bug caught by the unit test — do not regress this

A transport failure (`unwrap` returns `{"error": ...}`) initially fell
through the falsy-`items` check and was spoken as **"Your cart is empty"** —
a false claim about a cart that merely failed to read, silently erasing the
customer's real items. `speak_current_order` now returns `None` on an error
payload and defers to the model. Any future edit to the empty-cart branch
must keep the error check *above* it.

Similarly, `get_current_order` returning JSON `null` is a **meaningful
answer** (no open order), not a decode failure, so `speak()` special-cases it
rather than falling through to the LLM.

### Diagnosis: the "478 ms warm prefill floor" is not prefill

This number was the largest unexplained block for several rounds. It is now
fully accounted for, and the explanation invalidates two standing hypotheses.

**Raw OVMS TTFT vs prompt size** (direct, bypassing the agent, `stream=true`):

| approx prompt tokens | unique (cache miss) | repeated (cache hit) |
|---:|---:|---:|
| 5 | 119 ms | 108 ms |
| 520 | 123 ms | 117 ms |
| 2,080 | 144 ms | 128 ms |
| 4,160 | 146 ms | 139 ms |

Prefill is **~115 ms fixed + ~26 ms per 4,000 tokens**. Prompt size is
irrelevant, and prefix caching is worth only ~7 ms. No prompt-shrinking or
context-trimming work can pay off — this is now measured twice, two ways.

**Where the 350 ms actually goes.** Same request, only `tools` differs:

| variant | ctx | first content | total |
|---|---:|---:|---:|
| `tools` param | 16,915 | 456 ms | 1,571 ms |
| schemas inlined as text, no `tools` | 17,828 | 124 ms | 849 ms |
| no tools at all | 6,831 | 106 ms | 260 ms |

And it does **not** scale with tool count — 1 tool costs the same as 12:

| tools | first content |
|---:|---:|
| 12 | 462 ms |
| 7 | 490 ms |
| 3 | 468 ms |
| 1 | 452 ms |
| 0 | 106 ms |

It is not guided generation (`--enable_tool_guided_generation` is already
`false`). Timing the raw SSE stream shows what it really is:

| variant | first SSE byte | first content | gap |
|---|---:|---:|---:|
| with tools | 66 ms | 462 ms | **396 ms** |
| without tools | 42 ms | 102 ms | 60 ms |

The stream opens at 66 ms — the model is already decoding. The `hermes3`
tool parser simply cannot emit a structured delta until it has buffered the
`<tool_call>{"name": "place_order", "arguments":` preamble, ~12 tokens at
~33 ms/token. **The floor is generation, not prefill.**

**Consequence — the LLM call is 100% decode-bound.** For the live scrubbed
request: 106 ms prefill + (12 preamble + 20 argument) tokens x 33 ms
= ~1,150 ms, matching the measured 1,154 ms exactly. This explains, after the
fact, why tool pruning bought only ~58 ms (it changes no emitted tokens) and
why description scrubbing bought 359 ms (it removed 19 emitted tokens).
**Emitted token count is the only LLM lever.**

### Rejected: P3, compact tool-call wire format

Renaming `place_order`->`add` and `items`/`product_id`/`quantity`->`i`/`p`/`q`:

| variant | emitted | total | output |
|---|---:|---:|---|
| current | 20 tok | 1,154 ms | `place_order{"items":[{"product_id":"classic_chicken_burger","quantity":1}]}` |
| compact | 19 tok | 1,073 ms | `add{"i":[{"product_id":"Classic Chicken Burger","quantity":"1"}]}` |

**Rejected.** 81 ms, in exchange for the model emitting the *display name*
instead of the `product_id` and a *string* quantity — the same degradation
seen when the tool set was cut. Short keys also gave the model less signal,
and it ignored `p`/`q` anyway. Consistent with the shortened-instruction
result in Round 3: removing text the model leans on costs correctness.

### Implication for the 1-second target

Any turn that makes a tool call pays roughly:

    106 ms prefill + ~32 tokens x 33 ms  ~=  1,150 ms

That is an **architectural floor of ~1.05 s for the LLM call alone**, on this
model and hardware, before ASR or TTS. The <1 s voice-to-voice target is
therefore **not reachable by tuning the LLM path**. Remaining options, in
order of leverage:

1. **Do not call the LLM at all for high-frequency intents.** A deterministic
   intent fast path ("one <product> please", "what's my total") would cut the
   full ~1,150 ms, exactly as `reply_templates` removed the *second* call.
2. **Cut ASR**, now the largest single component (~1,466 ms on rec1, much of
   it the endpointing silence wait).
3. Speculative TTS pre-synthesis to overlap TTS with generation (~250 ms).

Decode speed itself (~30 tok/s for 4B-int4 on this iGPU) is memory-bandwidth
bound; a draft-model speculative-decoding attempt was already tried and
rejected (see docker-compose.yml comments).

---

## Round 6: directive mode — breaking the tool-calling floor

Round 5 established that a tool-calling turn cannot start speaking before
~1,150 ms, because (a) passing `tools` costs a flat ~350 ms and (b) a tool
call produces no words at all until its arguments are complete. Both costs
are structural, so the only way past them is to stop using tool calls for
ordering turns.

This is the approach used by the reference prototype in
`kiosk-voice-lab-main` (`pipeline/cart.py`), and it is why that prototype
reaches ~700 ms. The model emits a compact directive inline, immediately
followed by the prose it should speak, in ONE tool-free generation:

    <act>add|Classic Chicken Burger|1</act>Got it. One chicken burger.

Verified on our own OVMS/Qwen3-4B-int4 before any code was written:

| | first content | 3 prose words (TTS can start) | tokens | total |
|---|---:|---:|---:|---:|
| ours, MCP tool call | 453 ms | never — no speech until 1,154 ms+ | 36 | 1,574 ms |
| directive | **137 ms** | **551 ms** | 20 | **670 ms** |

### What was built

`plugins/kiosk/directive_mode.py`, behind `AGENT_DIRECTIVE_MODE` (default
**false**). It reuses the existing MCP tools as the execution layer — only
the *wire format the model speaks in* changed, so `OrderingService`, the MCP
server, and the ordering guards are all untouched.

Three properties make the weaker text contract acceptable:

1. **The model never says a price.** It is instructed to speak only a
   price-free confirmation; the total and upsell are appended afterwards by
   `_facts_tail()` straight from the tool payload. A hallucinated total is
   therefore not possible — the model never sees one to repeat.
2. **It always falls back.** A missing, malformed, or partially-parseable
   directive, an unresolvable item, a `needs_choice` result, or any exception
   returns `None`, and the authoritative tool-calling path runs instead.
   `parse_directive` rejects the *whole* directive if any clause is bad,
   rather than applying half of "remove the burger; add the wrap".
3. **It is scoped.** Only turns with explicit mutation intent are attempted
   (reusing `reply_templates.is_browse_intent`), since a fallback costs a
   wasted generation and questions have nothing to gain.

The tool call is dispatched the instant `</act>` is seen, so it overlaps the
generation of the prose that follows it. Prose sentences are released to
`on_safe_sentence` as they complete, feeding the *already existing*
`/chat/stream` -> `_get_reply_streaming` -> TTS path.

The model is also told to open with a two-word sentence ("Got it."), which
carries the Round 5 lead-clause trick into this path: the first TTS segment
is ~7 characters (~270 ms) instead of ~34 (~510 ms).

### Results

| stage | rec1 V2V | agent call |
|---|---:|---:|
| session start | 3,952 ms | ~1,517 ms |
| + first-phrase split | 3,838 ms | ~1,517 ms |
| + description scrubbing | 3,479 ms | ~1,190 ms |
| + lead clause + cart templates | 3,140 ms | ~1,190 ms |
| **+ directive mode** | **2,606 ms** | **~746 ms** |

rec2 unchanged at 5,432/5,551 ms — it falls back exactly as designed. Cart
correct throughout. Question, catalogue, and cart-read turns all still route
through their existing paths.

### Read the benchmark carefully

rec1's V2V contains a fixed **~1,300 ms endpointing silence wait**
(`endpoint committed after 1.30s trailing silence`) which no LLM or TTS work
can touch. A live UI turn where the customer stops the recording skips it:
the same turn measured **1,517 ms live** with a 1,182 ms agent call, so at
746 ms it lands near **~1.05 s**. That, not 2,606 ms, is the number to
compare against the 1 s target.

**ASR is now the largest remaining component** (~1,420 ms of rec1), and the
silence wait is the largest single item inside it.

### Pre-existing bug found (not introduced here)

"that's all i'm done" has no mutation-intent match, so it takes the ADK path,
where `confirm_active_order` was *not* templated and the model narrated its
own total: **"Total amount is $1670"** — wrong currency symbol and a
model-spoken price. Two independent defects on a path this round did not
touch. Worth fixing: add the confirm/done phrasing to the mutation-intent
regex, and find why `speak_confirm` did not fire.

---

## Round 7 — Transcript correctness + voice-to-voice observability

Two user-reported symptoms on one live turn (session `87b82f40`, "I would like to
order one classic chicken burger"), plus the UI work they motivated.

### Symptom 1 — truncated transcript shown in the UI

The UI displayed `I would like to order one classic`; the words "chicken burger"
were missing, even though the agent still ordered the right product.

Log evidence:

```
[CHUNK] ... flushing 1.50s of audio, is_final=False, diarization=False
[CHUNK] ... response: 0 segment(s), flat_text='I would like to order one classic'
[CHUNK] ... appending to transcript: 'I would like to order one classic'
[CHUNK] ... flushing 1.07s of audio, is_final=True, diarization=True
[CHUNK] ... response: 2 segment(s), flat_text='I would like to order one classic\nchicken burger.'
[SPEAKER-LOCK] ... primary speaker locked -> SPEAKER_00 (is_primary flag)
[SPEAKER-FILTER] ... is_primary path: kept=1 | final_text='I would like to order one classic'
```

Diarization split a single continuous utterance across two speakers and the
analyzer flagged only the first as `is_primary`. `_filter_target_speaker`
correctly honoured that verdict and dropped the tail.

**Not fixed here — deliberately.** The `is_primary` flag is the anti-bystander
control (`DEFAULT_SPEAKER_STRICT_DROP`); relaxing it so trailing segments are
kept would weaken the guarantee that a bystander cannot inject items into a
customer's order. This needs a product decision, not a silent code change.
Candidate fix if we do take it: treat *all* segments in the chunk that first
establishes the speaker lock as primary, since per-chunk diarization labels are
arbitrary and there is no prior evidence to contradict them.

### Symptom 2 — duplicated transcript

The same turn ended with `transcript=I would like to order one classic I would
like to order one classic`.

Root cause: the analyzer runs in cumulative (`append_to_session`) mode and there
are **two independent dedup cursors that cannot see each other** —

| Path | Cursor | Set when |
|---|---|---|
| flat text (no segments) | `_last_cumulative_flat_text` (string prefix) | diarization off |
| segments | `_last_analyzer_segment_end` (timestamp) | diarization on |

Chunk 1 committed the text via the *flat* path. Chunk 2 came back diarized, and
its first segment *straddled* the 1.50 s cursor, so `end > cursor` judged it
"fresh" and re-appended words already committed.

**Fixed** with a path-agnostic backstop at the append site,
`BaseAudioSession._strip_duplicate_prefix()`: strip a leading run of words the
transcript already ends with. Requires >= `KIOSK_CORE_DUPLICATE_PREFIX_MIN_WORDS`
(default 3) matching words, so natural repetition ("yes yes", "two two please")
is never swallowed. 8/8 unit cases pass, including the exact live failure.

### Symptom 3 — `voice_to_voice_ms` was always `null` on the UI path

`/api/v1/pipeline/latest` returned `voice_to_voice_ms: null` for the very turn
whose log said `voice_to_voice_ms=996`. The log figure came from a *fallback*
anchor (`_t_turn_start`), not from `_t_last_word`.

`_t_last_word` was only ever stamped on the **silence-timeout endpoint** path.
A browser turn normally ends because the customer released the mic
(`POST /audio/end` -> `signal_end()`), which never reaches that path — so V2V was
`None` for essentially every turn driven from the UI, which is exactly where the
metric matters. **Fixed** by stamping `_t_last_word` in
`BrowserStreamSession.signal_end()` (guarded, so the endpoint path still wins).

### Feature — per-request voice-to-voice in the UI

`AI Inference Pipeline` (`kiosk-ui/src/components/Dashboard/PipelineFlow.tsx`)
previously showed only TTFA and E2E chips for the latest turn. Added:

* a **V2V chip** in the header, with the informative figure in its tooltip;
* a **"Voice-to-Voice per Turn"** table — one row per request/response
  (V2V, V2V-informative, endpoint wait, E2E), newest first.

Backend needed no new endpoint: `/api/v1/pipeline/recent` already exposed the
20-deep `TurnTrace` ring buffer. Wired through `constants.ts` ->
`ttsApi.fetchKpis()` -> `KpiBundle.pipelineRecent`, and `PipelineWall` in
`types.ts` was extended with the three wall fields that already existed on the
Python dataclass but were never typed on the client.

**Why V2V is not derivable from E2E:** `turn_total_ms` starts at the endpoint
*decision*, so it excludes the trailing-silence wait the customer sits through.
V2V starts at the customer's last word. They answer different questions and both
are shown.

### Verification

`rec1_16k.wav` x3 (warm), plus a new browser-stream test exercising the real UI
path (`start-stream` -> chunked `/audio` -> `/audio/end`):

| Conversation | V2V | V2V info | Endpoint wait | E2E |
|---|---:|---:|---:|---:|
| bench-v2vfix-1 | 2394.7 ms | 2753.7 ms | 1400 ms | 8355.2 ms |
| bench-v2vfix-2 | 2322.1 ms | 2692.8 ms | 1300 ms | 7633.3 ms |
| bench-v2vfix-3 | 2500.3 ms | 2826.5 ms | 1300 ms | 7812.9 ms |
| streamtest-1 (browser path) | 2494.9 ms | 2777.2 ms | 1400 ms | 7921.2 ms |

Transcript correct on every run (`I would like to Order one classic chicken.
Good.`), no false stripping, cart correct (1x Classic Chicken Burger, Rs.169).

**Best live UI turn observed this session: `voice_to_voice_ms=996` — under the
1 s target.** The benchmark fixture reads higher because it carries ~1.3-1.4 s of
recorded trailing silence that a click-to-stop UI turn never pays.

### Still open

* **ASR endpointing** is now the largest remaining component; the trailing
  silence wait (1.3-1.4 s) dominates the benchmark figure.
* **Diarization tail-drop** (Symptom 1) — needs a product decision.
* **Pre-existing confirm bug** — "that's all i'm done" takes the ADK path and
  speaks *"Total amount is $1670"* (wrong currency, model-spoken total).

---

## Round 8 — The history feedback loop (turn 2 was 44% slower than turn 1)

User report: in one conversation, turn 1 took ~1.5 s and turn 2 took ~2.2 s.

Measured from the trace ring buffer:

| Turn | Utterance | V2V | ASR | LLM | TTS segs |
|---|---|---:|---:|---:|---:|
| 1 | "…one class. classic chicken burger" | **1534 ms** | 1244 ms | 851 ms | 4 |
| 2 | "Can you add classic french fries or also." | **2210 ms** | 1549 ms | **1473 ms** | 6 |

The LLM accounts for **+622 ms of the +675 ms delta**. At the measured
33 ms/token that is ~19 extra emitted tokens — and the logs show exactly where
they went:

```
turn 1: gen=851ms  sentences_streamed=2  reply='Got it. One Classic Chicken Burger. Your total is now Rs.169. ...'
turn 2: gen=1473ms sentences_streamed=4  reply='Got it. One Classic French Fries (Regular). Your total is now
                                                Rs.258. Would you like anything else? Your total is now Rs.258'
```

### Root cause — the composed reply is fed back as if the model wrote it

A directive-mode reply is `model prose + _facts_tail(payload)`. The model is
instructed never to speak money; the totals and the upsell question are composed
from the tool result. But the client displays the **whole** string and then
sends it back as a single assistant history turn.

So on turn 2 the model's own in-context example shows "itself" saying
*"Your total is now Rs.169. Would you also like Classic French Fries (Regular)
(Rs.89)?"* — and it faithfully imitates the pattern, emitting a total and a
closing question. `_facts_tail()` then appends the authoritative version on top,
producing the audible duplication the user heard.

**This compounds.** Every turn adds another example of the tail, so generation
gets longer and more duplicated the deeper into the conversation you go. It is
a self-reinforcing loop, not a one-off.

### Fix (two layers, both in `plugins/kiosk/directive_mode.py`)

1. **`strip_facts_tail()`** — prior assistant turns are cut at the first
   `Your total is…` / `Would you also like…` / `Would you like anything else…`
   before being put in the prompt. The model's in-context examples now show only
   short, price-free prose, which is the behaviour we actually want.
2. **Streaming guard** — if the model starts a facts sentence anyway,
   `prose_closed` is set and all further prose is dropped instead of being sent
   to TTS. This is what stops the duplication being *spoken*, since sentences
   are streamed as they arrive and cannot be unspoken at composition time.
   Required a separate `consumed` counter, because `spoken` is no longer a
   1:1 index into the parsed sentence list.

Layer 1 removes the cause; layer 2 makes the failure mode unreachable.

### Verification

Replaying the exact reported conversation:

| Turn | Before | After |
|---|---:|---:|
| 1 | 851 ms | 772 ms |
| 2 | **1473 ms** | **818 ms** |
| 3 | — | 1013 ms |

`sentences_streamed=2` on every turn (was climbing to 4), and the duplicated
"Your total is now Rs.258" is gone. 12/12 unit cases pass on the strip regex and
the sentence guard, including both live strings and all four YAML templates
(`item_added`, `item_removed`, `cart_summary`, `order_confirmed`).

Regression gate — no change to single-turn benchmarks (as expected, since the
loop needs >= 2 turns to appear):

| Fixture | Before | After |
|---|---|---|
| rec1 x4 | v2v 2322-2500 ms | v2v 2303-2527 ms, llm 746-829 ms, cart correct (Rs.169) |
| rec2 x2 | v2v 5472/5504 ms | v2v 4838/4855 ms |

### Noted, not fixed

* "Add a coke please." resolved to **Cold Coffee** — menu matching picks a poor
  nearest item instead of declining.
* "Make it two burgers." took 4039 ms and failed with *"we don't have Burger-1
  on the menu"* — the model invents a product id for quantity-change phrasing
  against an ambiguous referent, then falls back.

---

## Round 9 — Directive markup leaked to the customer

User report: turn 2 spoke and displayed raw wire format.

```
Got it. One Classic French Fries. <act>add|Classic Chicken Burger|1</act>Got it.
One Classic Chicken Burger. Your total is now Rs.258. …
```

`[DIRECTIVE] turn complete | gen=1606ms sentences_streamed=4`

### Root cause

`run_turn` consumes only the **first** `<act>…</act>`; everything after it is
treated as speech (`buf.split("</act>", 1)[1]`). That assumption holds right up
until the model emits a *second* directive — then the markup is prose by
definition, and goes straight to TTS and the screen.

The cart stayed correct (Rs.258 = 169 + 89, only the first directive executed),
so this was purely a presentation and latency failure — but a customer-visible
one, and the wasted second half of the generation is what pushed the turn to
1606 ms.

Round 8 removed the *price* tail from history, but nothing had ever taught the
model to stop after one action, so it kept starting a fresh turn on its own.

### Fix — three layers in `plugins/kiosk/directive_mode.py`

1. **`scrub_directives()`** — strips complete stray blocks, unclosed
   `<act>` runs, and partial tags still mid-stream (`<`, `<act`, `<act>`).
   Applied per sentence before TTS, to the trailing remainder, and to the final
   composed reply. Markup now has three independent chances to be caught.
2. **Prose truncation** — prose is cut at the first stray `<act>`, so the
   suffix is never even considered for speech.
3. **`_StopGeneration`** — a second directive means the rest of the generation
   is unusable, so the delta callback raises and `stream_completion` closes the
   HTTP stream instead of reading it to completion. This turns the guardrail
   into a latency *win* rather than a cost.

### Verification

Replaying the reported conversation:

| Turn | Before | After |
|---|---:|---:|
| 1 | — | 1239 ms |
| 2 | **1606 ms, markup leaked** | **821 ms, clean** |

The live model did not re-emit the second directive on the replay, so the
guardrail was also verified deterministically by replaying the exact recorded
bad generation through `run_turn` 7 characters at a time (so the second `<act>`
straddles delta boundaries), with `stream_completion` and `_execute` stubbed:

```
streamed to TTS : ['Got it.', 'One Classic French Fries.']
final reply     : Got it. One Classic French Fries. Your total is now Rs.258. Would you also like …
deltas consumed : 13/26 (stream aborted early)
RESULT: PASS
```

Half the generation was skipped, and no markup reached TTS or the reply.
9/9 unit cases pass on `scrub_directives` (including unclosed and partial tags).

Regression gate — rec1 x4: v2v 2410-2516 ms, llm 761-834 ms,
`sentences_streamed=2`, cart correct (Rs.169). No change.

### Still open (unchanged from Round 8)

* "Add a coke please." -> **Cold Coffee**; "And a chocolate shake." ->
  **Chocolate Brownie**. Item matching picks a poor nearest neighbour instead of
  declining. This is now the most user-visible correctness issue.
* Utterances directive mode declines (multi-item, quantity changes) fall back to
  the ADK tool path, which is ~5 s and lets the model narrate its own totals
  ("The total is Rs.347") — bypassing the price-safety guarantee.

---

## Round 10 — UI mic button deadlock ("can't send a prompt")

Reported symptom: from the dashboard UI the mic button "is not stopping, not
taking prompt". Not a latency issue — a client-side state-machine bug.

### Diagnosis

The decisive evidence was an *absence* in the logs. After the page load there
was **no `POST /api/v1/sessions/start-stream` and no `GET
/pcm-capture-processor.js`** — only metrics polling. The browser therefore
never got past the first two statements of `useVoiceSession.start()`.

The nginx referer showed `http://localhost:7860/` (an SSH port-forward), which
**is** a secure context, so the "requires HTTPS" branch was ruled out.

Root cause: `getUserMedia` can stay **pending forever** when a Chrome
permission prompt is left unanswered. In that window:

| state | value |
|---|---|
| `phase` | `'listening'` (set before the `try`) |
| `recordingRef.current` | `false` (only set *after* the await) |
| `conversationModeRef.current` | `true` |

Every recovery path then failed:

* `endConversation()` calls `stop()` only `if (recordingRef.current)` → skipped.
* Its `else if (phase === 'idle')` was also false → **did nothing at all**.
* So `phase` stayed `'listening'`, and every later tap hit `start()`'s own
  `if (recordingRef.current || phase !== 'idle') return;` guard.

The button latched on "Recording… (tap to stop)" and no tap could ever recover
it — exactly the reported symptom, and it explains the total absence of
network activity.

### Fixes

1. **Bounded mic acquisition** — `withTimeout(getUserMedia(...), 15 s)` plus
   `describeMicError()` to turn `NotAllowedError` / `NotFoundError` /
   `NotReadableError` / `OverconstrainedError` into actionable text.
2. **`forceIdle()`** — an unconditional hard reset usable from any phase,
   exported as `reset`. Bumps a `startGenRef` generation counter.
3. **Cancellable `start()`** — captures the generation and re-checks it after
   every await, releasing the `MediaStream` if superseded so a slow
   `getUserMedia` cannot resurrect a torn-down session or leak a live mic.
4. **`endConversation()`** — the `else if (phase === 'idle')` became a plain
   `else { forceIdle(); }`, so there is no longer any phase it cannot recover.
5. **Conversation-mode rollback** — a failed `start()` now clears conversation
   mode, so the button no longer renders a stop square while nothing records.
6. **`deviceId: { exact }` → `{ ideal }`** — a stale device id now degrades to
   the default mic instead of a hard `OverconstrainedError`.
7. **Visible errors** — `App.tsx` renders a red alert with a "Reset
   microphone" action. Previously `useVoiceSession`'s `error` was never
   destructured, so every mic failure was invisible on the dashboard.

### Verification

Rebuilt `kiosk-ui` (the multi-stage build runs `tsc`, so it doubles as the
typecheck — there is no `node` on the host). Replayed the real browser-stream
path through the UI's own nginx proxy with `/tmp/ui_stream_test.sh`:

```
status    : completed
transcript: I would like to order one classic chicken burger.
response  : Got it. One Classic Chicken Burger. Your total is now Rs.169. ...
tts segs  : 4
[VOICE2VOICE] ... event=last_word_spoken (browser signalled end of capture)
[VOICE2VOICE] ... voice_to_voice_ms=1712
```

Transcript is complete (no truncation, no duplication) and `voice_to_voice_ms`
is populated on the browser path, re-confirming the Round 7 `signal_end()` fix.

Note `silence_timeout_seconds` is capped at 10 s by the request model; the UI's
`singleChunkSilenceSeconds` is exactly 10.0.
