import os


DEFAULT_ANALYZER_URL = os.getenv(
    "KIOSK_CORE_ANALYZER_URL",
    "http://127.0.0.1:8010/v1/audio/transcriptions",
)
DEFAULT_RAG_URL = os.getenv(
    "KIOSK_CORE_RAG_URL",
    "http://127.0.0.1:8020/api/v1/query",
)
DEFAULT_TTS_URL = os.getenv(
    "KIOSK_CORE_TTS_URL",
    "http://127.0.0.1:8011/v1/audio/speech",
)
DEFAULT_TTS_MODEL = os.getenv("KIOSK_CORE_TTS_MODEL", "qwen-tts")
DEFAULT_TTS_VOICE = os.getenv("KIOSK_CORE_TTS_VOICE")
DEFAULT_TTS_LANGUAGE = os.getenv("KIOSK_CORE_TTS_LANGUAGE", "English")
# ASR language hint sent with every transcription request. Left unset, Whisper
# auto-detects per 5-second chunk and regularly mis-fires on short kiosk
# utterances — measured over 25 real session chunks it produced Spanish output
# ("¿No puedo abrir el restaurante...") and one runaway repetition loop.
# Forcing "en" cut word error rate from 11.0% to 7.0% on CPU, GPU and NPU
# alike. Set to an empty string to restore auto-detection.
DEFAULT_ASR_LANGUAGE = os.getenv("KIOSK_CORE_ASR_LANGUAGE", "en") or None
# Whisper prompt injected with every transcription request. Primes the model
# with restaurant/menu vocabulary so it prefers domain-specific spellings over
# phonetically similar common words (e.g. "Classic Chicken Burger" over
# "classic senses price", "Do you serve burgers" over "Are you sir burgers?").
# The prompt is NOT instruction text — Whisper treats it as prior transcript
# context, so it should read like natural speech, not a list.
#
# Built from the live product catalogue rather than hardcoded: a hand-written
# list silently drifts from the menu. It previously omitted every
# Indian-origin item, which are exactly the names an English-forced Whisper
# mishears — "Aloo Tikki Burger" was transcribed as "and 2,000" and the item
# never reached the agent.
#
# Whisper's prompt window is 224 tokens; the catalogue is well inside that.
# Falls back to a static list when the YAML is unreadable (e.g. CI) so ASR is
# never broken by a missing seed file.
_ASR_PROMPT_FALLBACK = (
    "QuickBite Express restaurant. Classic Chicken Burger, Classic French Fries,"
    " Margherita Pizza, Mango Lassi, Cold Coffee, Fresh Lime Soda, Pepsi, 7UP."
    " Peri Peri Fries, Chocolate Lava Cake."
    " Menu, order, remove, add, confirm, cancel."
)


def _build_asr_prompt() -> str:
    """Compose the Whisper priming prompt from the product catalogue.

    Returns:
        Natural-language prompt naming every menu item, or a static fallback
        when the catalogue cannot be read.
    """
    try:
        import yaml  # local import: keeps config importable without PyYAML

        path = os.getenv(
            "KIOSK_CORE_PRODUCTS_YAML", "./configs/ordering/products.yaml"
        )
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw = data.get("products") if isinstance(data, dict) else data
        names: list[str] = []
        for product in raw or []:
            name = str(product.get("name", "")).strip()
            if not name:
                continue
            # Drop size/qty parentheticals ("(330 ml)", "(Regular)") — they are
            # packaging metadata, not words the customer says.
            name = name.split("(")[0].strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return _ASR_PROMPT_FALLBACK
        return (
            "QuickBite Express restaurant. "
            + ", ".join(names)
            + ". Menu, order, remove, add, confirm, cancel."
        )
    except Exception:
        return _ASR_PROMPT_FALLBACK


DEFAULT_ASR_PROMPT = os.getenv("KIOSK_CORE_ASR_PROMPT") or _build_asr_prompt()
DEFAULT_TTS_INSTRUCTIONS = os.getenv("KIOSK_CORE_TTS_INSTRUCTIONS")
DEFAULT_SAMPLE_RATE = int(os.getenv("KIOSK_CORE_SAMPLE_RATE", "16000"))

# Audio capture source override.
# When HOST_MIC=true the backend captures audio directly from the host machine's
# microphone; otherwise the browser captures audio and streams it to the backend.
# This lets the same build work both locally (host mic) and against a
# remote/headless kiosk-core (browser mic) without auto-detection surprises.
HOST_MIC = os.getenv("HOST_MIC", "false").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# TTS segment silence trimming
# ---------------------------------------------------------------------------
# The response is split at clause boundaries (",", ".", "!") so synthesis can
# start streaming after the first fragment — measured time-to-first-audio is
# 387 ms this way versus 1769 ms when splitting on sentences only, so the split
# is worth keeping. The cost is that every fragment arrives with its own
# leading/trailing silence baked in by the TTS model: across 10 real production
# segments that was 1.76 s of dead air (8% of the reply), heard as a long pause
# at every comma.
#
# Trimming each segment back to a deliberate pad turns that accidental pause
# into a controlled one. The pad is intentionally non-zero — clauses should
# breathe — but short enough to sound continuous.
DEFAULT_TTS_TRIM_ENABLED = os.getenv("KIOSK_CORE_TTS_TRIM_ENABLED", "true").lower() not in ("false", "0", "no")

# When list_products is called with no category, return a per-category summary
# instead of every product. The catalogue is 26 items: reciting it costs ~19 s
# of LLM generation and ~40 s of speech, and no kiosk customer listens to a
# 26-item list. Prompt rules alone did not stop the model reciting it.
# Set false to restore the full listing.
DEFAULT_LIST_PRODUCTS_SUMMARY = os.getenv(
    "KIOSK_CORE_LIST_PRODUCTS_SUMMARY", "true"
).lower() not in ("false", "0", "no")
# Silence kept before speech starts, in every segment.
DEFAULT_TTS_LEAD_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_LEAD_PAD_MS", "20"))
# Trailing silence for a fragment ending mid-sentence (",", ":", ";").
DEFAULT_TTS_CLAUSE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_CLAUSE_PAD_MS", "60"))
# Trailing silence for a fragment ending a sentence (".", "!", "?").
DEFAULT_TTS_SENTENCE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_SENTENCE_PAD_MS", "150"))
# Amplitude below this fraction of the segment peak counts as silence.
DEFAULT_TTS_SILENCE_FLOOR = float(os.getenv("KIOSK_CORE_TTS_SILENCE_FLOOR", "0.02"))
# Linear fade-in/fade-out applied to the very edges of every trimmed segment,
# in milliseconds. The trim cut lands on whatever raw sample index the pad
# window computes to — not a zero-crossing — so the waveform can (and does,
# depending on the phoneme at that exact point) have a non-zero value right
# at the edge. The UI schedules segments back-to-back on one Web Audio
# timeline with no gap and no crossfade (see useAudioQueue.ts), so any such
# jump between one segment's last sample and the next segment's first sample
# is heard as an audible click — intermittently, only on the sentences whose
# cut points happen to land off a zero-crossing. This ramps every segment's
# edges to true zero unconditionally, which fully removes that click
# regardless of content. Kept well inside the lead/clause/sentence pads above
# so it never touches actual speech.
DEFAULT_TTS_FADE_MS = float(os.getenv("KIOSK_CORE_TTS_FADE_MS", "5"))

# TTS output loudness. SpeechT5's vocoder outputs a quiet, roughly constant
# level regardless of which speaker embedding is selected — swapping voices
# (e.g. Ryan -> Kabir) changes timbre, not level, so a soft-sounding kiosk is
# a gain problem, not a voice problem. There is no gain/volume control in the
# text-to-speech service itself, so kiosk-core normalizes + boosts every
# synthesized segment in place before playback.
#
# Two knobs stack: first each segment's peak is normalized up (or down) to
# TARGET_PEAK, then an extra flat boost of GAIN_DB is applied on top. Total
# applied gain is hard-clamped at GAIN_MAX_DB so that a near-silent/failed
# synthesis (e.g. a clipped word) can't be amplified into harsh noise.
#
# TARGET_PEAK is intentionally kept a little below full scale (not 1.0):
# observed live, pushing peaks to 100% FS left zero headroom, so any
# downstream OS/mixer volume above unity (venue speaker volume nudged up on
# demo day, PipeWire sink gain >100%, etc.) clipped and was heard as
# crackling/distortion. 0.9 keeps ~1 dB of headroom against exactly that.
DEFAULT_TTS_GAIN_ENABLED = os.getenv("KIOSK_CORE_TTS_GAIN_ENABLED", "true").lower() not in ("false", "0", "no")
# Fraction of full-scale (int16 max) each segment's peak is normalized to.
DEFAULT_TTS_TARGET_PEAK = float(os.getenv("KIOSK_CORE_TTS_TARGET_PEAK", "0.9"))
# Extra flat boost in dB applied on top of peak normalization. Raise this
# first if the kiosk still sounds quiet over venue noise after normalization.
DEFAULT_TTS_GAIN_DB = float(os.getenv("KIOSK_CORE_TTS_GAIN_DB", "4.0"))
# Safety ceiling on total applied gain (normalization + boost combined).
DEFAULT_TTS_GAIN_MAX_DB = float(os.getenv("KIOSK_CORE_TTS_GAIN_MAX_DB", "15.0"))

# Metrics collector – base URL of the standalone metrics-collector container.
# Within Docker the service is reachable as http://metrics-collector:9000.
METRICS_COLLECTOR_URL = os.getenv(
    "KIOSK_CORE_METRICS_URL",
    "http://metrics-collector:9000",
)
# Hard cap on how much audio accumulates before a chunk is force-flushed.
#
# Raised 2.5 → 6.0s. At 2.5s this cap fired mid-utterance on almost every turn,
# because a typical kiosk request ("Can you suggest something to drink") is
# 2.5-4s of *continuous* speech with no pause long enough to trigger the
# adaptive flush. Two distinct defects followed from that:
#
#   1. ASR quality — the cut landed mid-word, so Whisper saw fragments.
#      Observed verbatim in production logs: "something" was split into
#      "Can you suggest some" + "thing to drink." This is the same failure the
#      adaptive_flush_pause 0.30 → 0.70 change fixed, arriving via a different
#      path.
#   2. Speaker identity — the severed tail is a 0.5-1.5s fragment. Speaker
#      embeddings computed over such a short span are unreliable, so the
#      analyzer clustered the tail as a *different* speaker and the
#      diarization filter discarded the customer's own words.
#
# 6.0s lets a normal single utterance complete inside one chunk. The adaptive
# pause flush (0.70s) still cuts at natural boundaries, so latency for ordinary
# speech is unchanged — this cap now only bites on genuinely continuous speech,
# where a longer chunk also yields better ASR context and a more reliable
# speaker embedding.
DEFAULT_CHUNK_SECONDS = float(os.getenv("KIOSK_CORE_CHUNK_SECONDS", "6.0"))
# Trailing silence that ends a turn.
#
# INVARIANT: must be strictly greater than DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS.
# If it is not, the endpoint fires before the adaptive flush can ever run (its
# guard requires silence_run < silence_timeout) and the pre-warm optimisation
# below is silently dead. The kiosk-ui previously sent 0.65s while the flush
# pause was 0.70s, which is exactly what happened.
#
# Note the browser UI overrides this per session via silence_timeout_seconds
# (kiosk-ui/src/constants.ts), so that constant governs the kiosk; this default
# applies to microphone and file sessions.
#
# 1.5s: at 0.65s a customer pausing to read the menu was cut off mid-sentence,
# so Whisper only ever received 1.8-2.5s of truncated audio and had to guess
# the item name. A genuine end-of-turn pause is ~1.0-1.5s, so this tolerates
# hesitation while keeping the reply prompt.
DEFAULT_SILENCE_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_SILENCE_TIMEOUT_SECONDS", "1.5"))
# Adaptive mid-utterance flush: when silence reaches this threshold but hasn't
# yet hit silence_timeout_seconds, flush the accumulated chunk to the background
# ASR worker so processing starts immediately. The tail chunk at true endpoint
# will then be short (only the frames since the last adaptive flush), cutting
# critical-path ASR from up to chunk_seconds down to ~0.3-0.5s of audio.
# Raised from 0.30 → 0.70s: at 0.30s, natural in-phrase pauses (e.g. between
# "chicken" and "burger") triggered an adaptive flush mid-word, clearing
# chunk_frames so the next word had no sentence context — Whisper then
# hallucinated ("chip" for "chicken") or misread the isolated tail ("Kin Burger"
# for "burger").  0.70s is still well below a genuine inter-utterance pause
# (~1.0-1.5s) but avoids splitting mid-sentence breathing pauses.
# Kept at 0.70 (not lowered) when the endpoint moved to 1.5s: 0.70 is the value
# proven to avoid the mid-word splits above, and it only became effective again
# because the endpoint is now longer than it. See the INVARIANT note above.
DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS = float(os.getenv("KIOSK_CORE_ADAPTIVE_FLUSH_PAUSE_SECONDS", "0.70"))
DEFAULT_MAX_SESSION_SECONDS = float(os.getenv("KIOSK_CORE_MAX_SESSION_SECONDS", "20.0"))
DEFAULT_SILENCE_THRESHOLD = int(os.getenv("KIOSK_CORE_SILENCE_THRESHOLD", "900"))

# ── Adaptive VAD (noise-floor calibration) ────────────────────────────────────
# DEFAULT_SILENCE_THRESHOLD is an absolute int16 RMS value, which is only ever
# correct for the microphone and room it was measured on. Measured on the demo
# unit (PCM2902 USB codec, quiet room, nobody speaking) the *silence* floor was
# RMS ~1076 — i.e. above the 900 gate, so every frame classified as speech,
# silence_run_seconds never accumulated, and neither the silence endpoint nor
# the adaptive flush could ever fire. A louder venue makes that worse.
#
# Rather than hand-tuning the constant per venue (impossible when the venue
# cannot be tested beforehand), the session measures the actual noise floor at
# runtime and places the speech gate a fixed margin above it.
#
# FAIL-OPEN BY CONSTRUCTION: the derived gate is clamped to
# [THRESHOLD_MIN, THRESHOLD_MAX]. If a venue is so loud that floor*margin
# exceeds THRESHOLD_MAX, the gate saturates at THRESHOLD_MAX and behaviour
# degrades to "treat everything as speech" — exactly today's behaviour, never
# worse. A too-HIGH gate is the dangerous direction (speech is never detected
# and nothing is transcribed at all), which the ceiling exists to prevent.
ADAPTIVE_VAD_ENABLED = os.getenv("KIOSK_CORE_ADAPTIVE_VAD_ENABLED", "true").lower() not in ("false", "0", "no")
# Audio observed before the gate is derived. Frames in this window are held in
# the preroll buffer (which is sized to cover it), so nothing is lost.
DEFAULT_VAD_CALIBRATION_SECONDS = float(os.getenv("KIOSK_CORE_VAD_CALIBRATION_SECONDS", "0.5"))
# Low percentile of frame RMS used as the floor estimate. Deliberately low so
# that a customer who starts talking immediately (making some calibration frames
# loud) still yields a floor drawn from the quiet frames between words.
DEFAULT_VAD_FLOOR_PERCENTILE = float(os.getenv("KIOSK_CORE_VAD_FLOOR_PERCENTILE", "20"))
# How far above the measured floor the speech gate sits. 6 dB ~= 2x the floor.
# Measured against real speech mixed over the real PCM2902 noise floor: at 9 dB
# the gate needed ~15 dB SNR before it reliably saw speech, which a kiosk mic at
# arm's length will not deliver. 6 dB detects normal speech from ~9 dB SNR while
# still sitting clear of the floor.
DEFAULT_VAD_MARGIN_DB = float(os.getenv("KIOSK_CORE_VAD_MARGIN_DB", "6.0"))
DEFAULT_VAD_THRESHOLD_MIN = int(os.getenv("KIOSK_CORE_VAD_THRESHOLD_MIN", "300"))
DEFAULT_VAD_THRESHOLD_MAX = int(os.getenv("KIOSK_CORE_VAD_THRESHOLD_MAX", "4000"))
# Once calibrated, the floor keeps tracking on non-speech frames only (standard
# VAD practice — never adapt the noise estimate while speech is present, or the
# gate climbs during a long utterance and cuts the customer off).
#
# Adaptation is ASYMMETRIC. Quiet frames *within* an utterance (gaps between
# words) fall below the gate and are therefore seen as "non-speech"; with a
# symmetric EMA they dragged the floor upward mid-sentence — measured drift was
# 1098 -> 1642, which pushed the gate up and swallowed the rest of the
# utterance. Downward moves (room genuinely got quieter) are safe and track
# quickly; upward moves are deliberately ~10x slower.
DEFAULT_VAD_FLOOR_ADAPT_DOWN = float(os.getenv("KIOSK_CORE_VAD_FLOOR_ADAPT_DOWN", "0.05"))
DEFAULT_VAD_FLOOR_ADAPT_UP = float(os.getenv("KIOSK_CORE_VAD_FLOOR_ADAPT_UP", "0.005"))
DEFAULT_BLOCK_DURATION_SECONDS = float(os.getenv("KIOSK_CORE_BLOCK_DURATION_SECONDS", "0.1"))
DEFAULT_PREROLL_SECONDS = float(os.getenv("KIOSK_CORE_PREROLL_SECONDS", "0.3"))
DEFAULT_HTTP_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_HTTP_TIMEOUT_SECONDS", "300.0"))

# Wake-word activation (openwakeword)
WAKEWORD_ENABLED = os.getenv("KIOSK_CORE_WAKEWORD_ENABLED", "false").lower() not in ("false", "0", "no")
DEFAULT_WAKEWORD_MODEL = os.getenv("KIOSK_CORE_WAKEWORD_MODEL", "hey jarvis")
DEFAULT_WAKEWORD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_THRESHOLD", "0.5"))
DEFAULT_WAKEWORD_VAD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_VAD_THRESHOLD", "0.4"))
DEFAULT_WAKEWORD_PATIENCE_FRAMES = int(os.getenv("KIOSK_CORE_WAKEWORD_PATIENCE_FRAMES", "2"))
DEFAULT_WAKEWORD_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_WAKEWORD_TIMEOUT_SECONDS", "0"))
DEFAULT_WAKEWORD_INFERENCE_FRAMEWORK = os.getenv("KIOSK_CORE_WAKEWORD_INFERENCE_FRAMEWORK", "onnx")

# Speaker diarization — master switch and semantic fallback sensitivity.
# Set KIOSK_CORE_DIARIZATION_ENABLED=false to revert to flat-text behavior
# (no speaker filtering; all segments forwarded as-is).
DEFAULT_DIARIZATION_ENABLED = os.getenv("KIOSK_CORE_DIARIZATION_ENABLED", "true").lower() not in ("false", "0", "no")
# Minimum domain-keyword overlap ratio to accept a fallback segment when the
# primary customer is silent for an entire chunk.
DEFAULT_SEMANTIC_FALLBACK_THRESHOLD = float(os.getenv("KIOSK_CORE_SEMANTIC_FALLBACK_THRESHOLD", "0.10"))
# When the audio-analyzer holds an enrolled reference voice it tags every
# segment with is_primary. If it marks them all non-primary that is a positive
# rejection and the chunk is dropped, so a bystander cannot inject orders.
# Set KIOSK_CORE_SPEAKER_STRICT_DROP=false to fall back to the first-speaker /
# semantic heuristics instead — an escape hatch for when voice enrollment is
# mistuned and starts rejecting the real customer.
DEFAULT_SPEAKER_STRICT_DROP = os.getenv("KIOSK_CORE_SPEAKER_STRICT_DROP", "true").lower() not in ("false", "0", "no")

# Spoken replies used when a turn produces no usable transcript. The two cases
# are NOT interchangeable and must never share a message:
#   * NO_SPEECH   — the microphone captured nothing (true silence). Prompting
#                   the customer to order is the right response.
#   * UNRECOGNIZED— speech WAS captured but every segment was rejected by the
#                   speaker filter (analyzer marked it non-primary, or it came
#                   from a bystander). Replying with the generic greeting here
#                   is actively misleading: the customer spoke, was ignored,
#                   and is given no hint that they need to retry.
DEFAULT_NO_SPEECH_PROMPT = os.getenv(
    "KIOSK_CORE_NO_SPEECH_PROMPT",
    "How can I help you?",
)
DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT = os.getenv(
    "KIOSK_CORE_UNRECOGNIZED_SPEAKER_PROMPT",
    "Sorry, I couldn't clearly recognise your voice. Could you please repeat that?",
)

# A single rejected turn is unreliable evidence of a real bystander — it is
# just as often a Whisper hallucination or TTS echo bleeding into the mic from
# the kiosk's own previous reply (see the rationale where this constant is
# consumed, in BaseAudioSession._finalize_run). Speaking
# DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT after every single rejection reintroduces
# that false-positive noise. Requiring this many CONSECUTIVE rejected turns in
# the SAME conversation (tracked by agent_session_id, reset the moment a turn
# produces a real transcript) before speaking distinguishes a persistent
# bystander/misconfigured enrollment from a one-off echo, while still telling
# a genuinely ignored customer something after a couple of silently dropped
# turns rather than leaving them with no feedback at all.
DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD = int(
    os.getenv("KIOSK_CORE_CONSECUTIVE_REJECTION_THRESHOLD", "2")
)

# ── Ordering & Agent feature ─────────────────────────────────────────────────
# Set KIOSK_CORE_ORDERING_ENABLED=false to disable the ordering/agent feature
# and keep the legacy RAG-only Q&A flow.
ORDERING_ENABLED = os.getenv("KIOSK_CORE_ORDERING_ENABLED", "true").lower() not in ("false", "0", "no")

# Single shared kiosk identity used for ordering when no per-user login is
# wired into the request (this kiosk currently serves one customer at a time).
DEFAULT_ORDERING_USER_ID = os.getenv("KIOSK_CORE_DEFAULT_USER_ID", "kiosk-user")

# RAG-service agent chat endpoint (for ordering turns).
DEFAULT_AGENT_URL = os.getenv(
    "KIOSK_CORE_AGENT_URL",
    "http://127.0.0.1:8020/api/v1/agent/chat",
)

# Consume the agent's streaming endpoint so complete sentences reach TTS as
# they are generated instead of after the whole turn.
#
# Measured: the first sentence of a reply exists at ~700 ms while the full
# reply takes 1.3-4.0 s, so time-to-first-audio drops from ~5.0 s to ~2.8 s.
#
# OFF by default, and it must stay paired with AGENT_STREAM_SENTENCES on
# rag-service: the agent only releases sentences that provably cannot be
# rewritten by a later guard, and this client still re-validates the
# authoritative reply before speaking any remainder. Set to false to fall back
# to the buffered endpoint without a rebuild.
AGENT_STREAM_ENABLED = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_ENABLED", "false"
).lower() in ("true", "1", "yes")

# Derived from DEFAULT_AGENT_URL so both point at the same service.
DEFAULT_AGENT_STREAM_URL = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_URL",
    DEFAULT_AGENT_URL.rstrip("/") + "/stream",
)

# SQLite database file path (ordering domain).
KIOSK_DB_PATH = os.getenv("KIOSK_CORE_DB_PATH", "./kiosk.db")

# YAML seed files for product catalogue and upsell rules.
PRODUCTS_YAML_PATH = os.getenv(
    "KIOSK_CORE_PRODUCTS_YAML",
    "./configs/ordering/products.yaml",
)
UPSELL_RULES_YAML_PATH = os.getenv(
    "KIOSK_CORE_UPSELL_RULES_YAML",
    "./configs/ordering/upsell_rules.yaml",
)
# Maximum number of upsell suggestions attached to a place_order/update_order
# result.  Every extra suggestion the agent has to speak costs ~8 output tokens
# (~350 ms of LLM decode on Panther Lake iGPU), so this directly trades upsell
# breadth against spoken-reply latency.  1 keeps the turn inside the 3-4 s SLA.
UPSELL_MAX_SUGGESTIONS = int(os.getenv("KIOSK_CORE_UPSELL_MAX_SUGGESTIONS", "1"))

# ── Identity / biometric authentication feature ──────────────────────────────
# Master switch for the multimodal (face + voice) identity subsystem.  When
# false, kiosk-core does not mount the identity router, does not construct the
# IdentityClient, and the standalone identity-service container is never called.
# Set KIOSK_CORE_IDENTITY_ENABLED=true to turn the feature on (the
# identity-service container must also be started, e.g. via the `identity`
# compose profile).
IDENTITY_ENABLED = os.getenv("KIOSK_CORE_IDENTITY_ENABLED", "false").lower() not in ("false", "0", "no")

# Base URL of the standalone identity-service.  Within Docker the service is
# reachable as http://identity-service:8013.
IDENTITY_SERVICE_URL = os.getenv(
    "KIOSK_CORE_IDENTITY_URL",
    "http://127.0.0.1:8013",
)

# ---------------------------------------------------------------------------
# Queue-service integration (dynamic peak-hour menu)
# ---------------------------------------------------------------------------
# When enabled, the queue-service exposes a queue count that kiosk-core can
# query (future server-side menu filtering).  The UI also polls this directly
# via /queue-svc/api/v1/queue/count proxied through nginx.
QUEUE_SERVICE_ENABLED = os.getenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "true").lower() not in ("false", "0", "no")

QUEUE_SERVICE_URL = os.getenv(
    "KIOSK_CORE_QUEUE_SERVICE_URL",
    "http://127.0.0.1:8090",
)

# ---------------------------------------------------------------------------
# Conversation recording (offline analysis)
# ---------------------------------------------------------------------------
# Single master switch: when false (default), kiosk_core.conversation_recorder
# does no file I/O at all -- every call is a no-op. When true, every completed
# voice turn (user transcript + assistant reply) is appended as one JSON line
# to <CONVERSATION_LOG_DIR>/<conversation_id>.jsonl, so each full multi-turn
# conversation lives in its own file for later analysis.
CONVERSATION_LOGGING_ENABLED = os.getenv(
    "KIOSK_CORE_CONVERSATION_LOGGING_ENABLED", "false"
).lower() not in ("false", "0", "no")

# Directory conversation transcripts are written to. Relative paths resolve
# against the kiosk-core project root (same convention as KIOSK_DB_PATH).
CONVERSATION_LOG_DIR = os.getenv(
    "KIOSK_CORE_CONVERSATION_LOG_DIR",
    "./conversations",
)
