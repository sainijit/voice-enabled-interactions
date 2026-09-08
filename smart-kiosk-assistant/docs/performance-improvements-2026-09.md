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

## Remaining opportunity (unchanged, still the big one)

Speculative drafting during speech. The prototype's 97% draft-hit rate is
what buys its median; nothing in this round touches *when* work happens. The
staged path is: a throwaway preview-ASR tick (strictly isolated from the
committed transcript, so the diarization and Whisper-hallucination fixes are
not reintroduced) → speculative agent drafts on changed previews with
newest-wins preemption, read-only turns first so no tool ever executes
speculatively → first-phrase TTS pre-synthesis from the draft prefix, reusing
the existing opener cache.
