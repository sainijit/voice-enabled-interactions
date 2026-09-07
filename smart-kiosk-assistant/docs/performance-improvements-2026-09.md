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
   an int4 model (vs. int8), no conversation history resent for
   retail-mode Q&A (`context_turns=0`), and simple regex-parsed `<act>`
   directives instead of a full JSON-schema tool-calling protocol.
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
