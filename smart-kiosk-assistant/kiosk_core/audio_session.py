import logging
import math
import hashlib
import shutil
import tempfile
import threading
import time
import wave
import io
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
import re
from typing import Callable
from uuid import uuid4

import numpy as np
import sounddevice as sd

from kiosk_core import config, conversation_recorder
from kiosk_core.agent_client import AgentClient
from kiosk_core.analyzer_client import AnalyzerClient
from kiosk_core.models import FileSessionStartRequest, SessionStartRequest
from kiosk_core.pipeline_latency import (
    AgentSpan, AsrSpan, LlmSpan, PipelineLatencyStore, RetrievalSpan,
    TtsSpan, TurnTrace, WallTimes, pipeline_store,
)
from kiosk_core.rag_client import RagClient
from kiosk_core.tts_client import TtsClient


logger = logging.getLogger(__name__)

# ── Pre-synthesized opener cache ──────────────────────────────────────────
# Rendered at most once per (text, model, voice, language) for the whole
# process and reused by every session, so the opener never costs TTS time on
# the hot path. Guarded by a lock because concurrent sessions can race on the
# first turn after startup.
_opener_lock = threading.Lock()
_opener_cache: dict[tuple[str, str, str | None, str | None], Path | None] = {}


def _render_opener(
    tts_client: TtsClient,
    text: str,
    model: str,
    voice: str | None,
    language: str | None,
    instructions: str | None,
) -> Path | None:
    """Return a cached WAV of ``text``, synthesising it on first use.

    Args:
        tts_client:   Client used for the one-off synthesis.
        text:         Opener phrase. Must be non-committal — see
                      ``config.DEFAULT_OPENER_TEXT``.
        model:        TTS model name.
        voice:        TTS voice, or None for the service default.
        language:     TTS language, or None for the service default.
        instructions: Optional TTS style instructions.

    Returns:
        Path to the rendered WAV, or None when synthesis failed. A failure is
        cached as None so a broken TTS service cannot make every turn pay a
        failed round-trip.
    """
    key = (text, model, voice, language)
    with _opener_lock:
        if key in _opener_cache:
            return _opener_cache[key]

        cache_dir = Path(config.DEFAULT_OPENER_CACHE_DIR)
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parent.parent / cache_dir
        # hashlib, not hash(): str hashing is salted per process (PYTHONHASHSEED),
        # so hash() would pick a new filename on every restart and re-synthesise
        # the opener instead of reusing the cached WAV on disk.
        digest = hashlib.sha256("\x00".join(str(k) for k in key).encode()).hexdigest()[:16]
        path = cache_dir / f"opener_{digest}.wav"

        if path.exists() and path.stat().st_size > 0:
            _opener_cache[key] = path
            return path

        try:
            t0 = time.monotonic()
            tts_client.synthesize_to_file(
                text=text,
                output_path=str(path),
                model=model,
                voice=voice,
                language=language,
                instructions=instructions,
            )
            logger.info(
                "[OPENER] Rendered %r -> %s in %.0f ms",
                text, path.name, (time.monotonic() - t0) * 1000,
            )
            _opener_cache[key] = path
        except Exception:
            logger.exception("[OPENER] Synthesis failed; opener disabled for this process")
            _opener_cache[key] = None
        return _opener_cache[key]


_SENTENCE_PATTERN = re.compile(r"^(.+?[.!?,:;](?:[\"')\]]+)?)(?:\s+|$)", re.DOTALL)
# Whisper hallucination tokens to strip from transcripts
_WHISPER_JUNK = re.compile(
    r"\[(?:BLANK_AUDIO|Music|Noise|Applause|Laughter|Silence|Background Music|noise|music)\]",
    re.IGNORECASE,
)

# ASR homophone normalization: "cart" (the shopping cart) is routinely
# mis-transcribed as "card" — same vowel sound, no acoustic distinction for
# Whisper. Observed live: "remove item in my card" reached the agent
# verbatim; "card" is a legitimately in-domain word (payment method), so no
# existing guard caught it, and the LLM improvised a "contact customer
# support" refusal instead of removing the item — a pure ASR-homophone
# hallucination, not a code bug reachable by any menu/order guard. This is
# corrected at the same layer as _WHISPER_JUNK: a deterministic transcript
# normalization before the text ever reaches the agent.
#
# Scoped narrowly to avoid corrupting genuine payment-card mentions:
#   - Only fires when "card" appears near an order-cart verb/phrase
#     (remove/delete/clear/empty/what's in/add ... to/in my card).
#   - Never fires if the utterance also contains a payment-context word
#     (pay, payment, credit, debit, swipe, tap, cash, upi) anywhere, since a
#     genuine "pay by card" / "swipe my card" must not be rewritten.
#   - Never fires for "gift card" / "loyalty card" / "membership card",
#     which are real nouns distinct from "cart".
_PAYMENT_CONTEXT_RE = re.compile(
    r"\b(?:pay|paying|payment|credit|debit|swipe|tap|paypal|upi|cash)\b",
    re.IGNORECASE,
)
_CARD_CART_HOMOPHONE_RE = re.compile(
    r"\b(?:remove|removing|delete|deleting|take out|taking out|clear|clearing|"
    r"empty|emptying|what'?s|whats|check|show|view)\b(?:\s+\S+){0,6}?\s+"
    r"(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b"
    r"|\badd(?:ing)?\b(?:\s+\S+){0,8}?\s+to\s+(?:my\s+)?"
    r"(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b"
    r"|\bin\s+my\s+(?<!gift\s)(?<!loyalty\s)(?<!membership\s)card\b",
    re.IGNORECASE,
)


def _normalize_card_cart_homophone(text: str) -> str:
    """Rewrite an order-context "card" mis-transcription to "cart".

    Args:
        text: Raw (already Whisper-junk-stripped) transcript text.

    Returns:
        ``text`` unchanged unless an order-cart phrase containing "card"
        is found and no payment-context word is present anywhere in the
        utterance, in which case the matched "card" occurrence(s) are
        rewritten to "cart".
    """
    if not text or _PAYMENT_CONTEXT_RE.search(text):
        return text

    def _swap(match: re.Match) -> str:
        return re.sub(r"\bcard\b", "cart", match.group(0), flags=re.IGNORECASE)

    return _CARD_CART_HOMOPHONE_RE.sub(_swap, text)


# Whisper emits a short stock phrase when handed near-silence. The PyTorch
# "openai" provider suppresses these via no_speech_prob/avg_logprob, but the
# OpenVINO GenAI provider exposes no confidence signal at all (its `scores`
# field is a degenerate 1.0), so a silent chunk transcribes as "you" or
# "Thank you." and would otherwise start a spurious agent turn. Matching the
# whole utterance keeps genuine speech containing these words intact.
_WHISPER_FILLER = re.compile(
    r"^\W*(?:you|thank you|thanks(?: for watching)?|bye|okay|ok|uh|um|"
    r"thank you\.? bye|please subscribe)\W*$",
    re.IGNORECASE,
)

_DEDUP_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _collapse_repeated_phrases(text: str) -> str:
    """Fold consecutive repeated word runs (window 1-8) to a single occurrence.

    Applied to a fixpoint so triples collapse too. Safety net for transcripts
    assembled from multiple ASR chunks where the same utterance can appear
    twice ("How are you today? How are you today?").
    """
    if len(text.split()) < 2:
        return text

    def _one_pass(words: list[str]) -> list[str]:
        result: list[str] = []
        i = 0
        while i < len(words):
            matched = False
            max_window = min(8, (len(words) - i) // 2)
            for w in range(max_window, 0, -1):
                if words[i:i + w] == words[i + w:i + 2 * w]:
                    result.extend(words[i:i + w])
                    i += 2 * w
                    matched = True
                    break
            if not matched:
                result.append(words[i])
                i += 1
        return result

    current = text.split()
    for _ in range(8):
        collapsed = _one_pass(current)
        if collapsed == current:
            break
        current = collapsed
    return " ".join(current)


# ── Adaptive endpoint: does the transcript sound finished? ──────────────────
# A word that ends a transcript but cannot end a sentence. If the customer
# stopped on one of these they are mid-thought, so the turn must keep the full
# silence_timeout_seconds instead of committing early.
_INCOMPLETE_TAIL_WORDS = frozenset(
    # hesitation sounds
    "um uh umm uhh er ah hmm erm".split()
    # contractions that open a clause: "I'd like..." cut after "I'd"
    + "i've i'd i'll i'm we've we'd we'll you've you'd you'll they've they'd "
      "they'll wanna gonna gimme lemme".split()
    # conjunctions, articles and determiners
    + "and or but so like the a an to of my your our their some any those "
      "these that this".split()
    # auxiliaries and pronouns
    + "do does did is are was were can could would will have has had get got "
      "i we you they".split()
    # prepositions that must take an object: "with...", "a burger and fries for..."
    + "for with in on at by from about without".split()
)

_ENDPOINT_WORD_RE = re.compile(r"[^a-z' ]")


def _looks_complete(text: str, min_words: int) -> bool:
    """True when a transcript reads like a finished utterance.

    Used only to SHORTEN the end-of-turn wait, never to extend it, so every
    uncertain case must return False. An empty or not-yet-arrived transcript
    therefore fails closed and the caller keeps the full silence timeout.

    Args:
        text: The transcript assembled so far this turn.
        min_words: Fewest words that may be judged complete.

    Returns:
        Whether the turn may be ended early.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    # A trailing comma is an explicit continuation, whatever the words are.
    if raw.endswith(","):
        return False
    words = _ENDPOINT_WORD_RE.sub("", raw.lower()).split()
    if len(words) < min_words:
        return False
    # Whisper punctuates fragments with "?" and ".", so punctuation cannot
    # rescue a dangling word — judge on the final word itself.
    return words[-1] not in _INCOMPLETE_TAIL_WORDS


def _assemble_transcript(parts: list[str]) -> str:
    """Join transcript chunk parts, dropping consecutive duplicates.

    A chunk's text that repeats the previous kept part (case/punctuation-
    insensitive) is skipped before joining, then any residual repeated phrase
    inside the joined string is collapsed. This is the single assembly point
    for the user-visible transcript, so it guards against duplicate text no
    matter which upstream ASR path produced it.
    """
    deduped: list[str] = []
    prev_norm: str | None = None
    for part in parts:
        if not part:
            continue
        norm = " ".join(_DEDUP_PUNCT_RE.sub(" ", part.lower()).split())
        if norm and norm == prev_norm:
            continue
        deduped.append(part)
        prev_norm = norm
    return _collapse_repeated_phrases(" ".join(deduped).strip())


# Domain vocabulary for the semantic fallback in _filter_target_speaker.
# When the primary customer is silent for an entire chunk this set is used
# to decide whether a background speaker said something kiosk-relevant enough
# to warrant re-assigning the primary (e.g. a new customer stepped up).
_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "order", "orders", "ordering", "menu", "item", "items",
    "burger", "pizza", "sandwich", "wrap", "salad", "combo",
    "fries", "drink", "water", "coffee", "tea", "juice", "soda",
    "price", "cost", "how much", "pay", "payment", "card", "cash",
    "checkout", "bill", "receipt", "change",
    "ticket", "seat", "flight", "hotel", "book", "booking", "reserve",
    "help", "assist", "please", "want", "need", "like", "get",
})

# ── Consecutive speaker-rejection tracking (cross-turn, per conversation) ──
# ``_rejected_speech_chunks`` on a session instance only counts chunks
# rejected WITHIN that one turn. A BaseAudioSession is created fresh per
# voice turn, so distinguishing "this is the first rejected turn" from "this
# conversation has been rejected several turns running" needs state that
# outlives a single instance, keyed by the persistent ``agent_session_id``
# (see its docstring in __init__) shared across every turn of one
# conversation. See config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD for the
# rationale on why a streak, not a single rejection, gates the retry prompt.
_consecutive_rejections_lock = threading.Lock()
_consecutive_rejections: dict[str, int] = {}
# Simple bound so a long-running kiosk process doesn't accumulate one entry
# per conversation forever. A kiosk lane never has anywhere near this many
# conversations in flight at once, so clearing on overflow only ever discards
# stale entries from finished conversations.
_MAX_TRACKED_CONVERSATIONS = 500


def _note_conversation_rejection(agent_session_id: str) -> int:
    """Record a rejected turn for ``agent_session_id`` and return the streak.

    Args:
        agent_session_id: The persistent conversation identifier shared
            across every voice turn of one customer's session.

    Returns:
        The number of consecutive rejected turns recorded so far for this
        conversation, including this one.
    """
    with _consecutive_rejections_lock:
        if (
            len(_consecutive_rejections) > _MAX_TRACKED_CONVERSATIONS
            and agent_session_id not in _consecutive_rejections
        ):
            _consecutive_rejections.clear()
        count = _consecutive_rejections.get(agent_session_id, 0) + 1
        _consecutive_rejections[agent_session_id] = count
        return count


def _reset_conversation_rejections(agent_session_id: str) -> None:
    """Clear the rejection streak for ``agent_session_id``.

    Called whenever a turn produces a real, accepted transcript — the
    customer was successfully heard, so any earlier rejection streak no
    longer says anything about whether they are being ignored now.
    """
    with _consecutive_rejections_lock:
        _consecutive_rejections.pop(agent_session_id, None)


def reset_all_rejection_tracking() -> None:
    """Drop all tracked rejection streaks.

    Test-only entry point — prevents state from one test leaking into the
    next when several tests reuse the same default conversation id.
    """
    with _consecutive_rejections_lock:
        _consecutive_rejections.clear()


class BaseAudioSession:
    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        self.session_id = str(uuid4())
        # Persistent agent session ID — reused across all voice turns in the same
        # conversation so the ADK agent retains order state between mic presses.
        # Falls back to the audio session UUID if no conversation_id was supplied.
        self.agent_session_id: str = request.conversation_id or self.session_id
        self.request = request
        self.on_complete = on_complete
        self.client = AnalyzerClient(request.analyzer_url)
        self.rag_client = RagClient(request.rag_url)
        self.tts_client = TtsClient(request.tts_url)
        # Agent client is used when the ordering feature is enabled.
        # All turns go through the agent — it decides Q&A vs ordering.
        if config.ORDERING_ENABLED:
            agent_url = getattr(request, "agent_url", None) or config.DEFAULT_AGENT_URL
            self.agent_client: AgentClient | None = AgentClient(agent_url)
            logger.info("[SESSION] Agent routing enabled → %s", agent_url)
        else:
            self.agent_client = None
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.status = "created"
        self.end_reason: str | None = None
        self.error: str | None = None
        self.transcript_parts: list[str] = []
        self.response_parts: list[str] = []
        self.tts_audio_segments: list[dict[str, object]] = []
        self.tts_errors: list[str] = []
        self.stop_requested_at: datetime | None = None

        # ── Primary-speaker lock-on ────────────────────────────────────────────
        # The audio-analyzer sets is_primary when it has an enrolled reference
        # voice for the conversation (see speaker_scope_id). kiosk-core honours
        # that flag when present and otherwise falls back to locking onto the
        # first speaker label seen in the session and treating all subsequent
        # segments from that label as the customer (primary).
        # Segments from any different label are unconditionally dropped — the
        # semantic fallback is only used before the primary is established.
        self._primary_speaker_id: str | None = None
        # Number of chunks in this turn that carried real transcribed speech
        # which the speaker filter then rejected. Distinguishes "nobody said
        # anything" from "somebody spoke and we discarded all of it", so
        # _finalize_run can ask the customer to repeat instead of replying with
        # a generic greeting that hides the rejection.
        self._rejected_speech_chunks: int = 0
        # Analyzer's own session_id — passed on every chunk so per-session
        # state (e.g. pyannote enrolled speaker embedding) persists across
        # the many chunked HTTP requests made in this kiosk session.
        # Initialised from our own session_id; the analyzer echoes/normalises
        # it via the X-Session-ID response header, which we then reuse.
        self._analyzer_session_id: str = self.session_id
        # Highest segment end-time already consumed from the analyzer. The
        # analyzer runs in append_to_session mode (session_id is reused) and
        # returns the cumulative segment list on every chunk with offset-
        # adjusted timestamps; without this cursor kiosk-core would re-append
        # every prior primary-speaker segment on each subsequent chunk,
        # duplicating utterances in transcript_parts.
        self._last_analyzer_segment_end: float = 0.0
        # ───────────────────────────────────────────────────────────────────────

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._audio_queue: Queue[np.ndarray] = Queue()
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        # ── ASR chunk-flush worker ──────────────────────────────────────────
        # Chunk transcription is a blocking HTTP round-trip to audio-analyzer
        # (whisper-small on CPU: ~1.5-4.7s depending on chunk duration). It
        # used to run inline in the frame-reading loop (_process_frame_stream),
        # which meant a mid-utterance chunk flush blocked silence-timeout
        # detection and frame ingestion for its entire duration — that
        # latency then serialised on top of the final tail-chunk flush,
        # roughly doubling the delay the customer felt after they stopped
        # talking (observed live: ~5-6s total instead of ~1-1.5s for
        # utterances long enough to trigger a mid-stream flush).
        #
        # A single dedicated worker thread now owns every _flush_chunk() call
        # instead. The main frame-reading loop just enqueues each chunk's
        # frames and keeps consuming new audio / evaluating silence in real
        # time; the worker drains the queue FIFO (one HTTP call at a time,
        # matching audio-analyzer's own one-request-at-a-time model cache),
        # which preserves both transcript_parts ordering and the analyzer's
        # cumulative-segment cursor (_last_analyzer_segment_end) exactly as
        # before. Only the *final* chunk's flush latency remains in the
        # customer-perceived critical path — _finalize_run explicitly waits
        # for the queue to drain (see the join() in _process_frame_stream)
        # before reading the completed transcript.
        self._flush_queue: Queue[list[np.ndarray] | None] = Queue()
        self._flush_thread = threading.Thread(
            target=self._flush_worker, name=f"asr-flush-{self.session_id}", daemon=True,
        )
        self._speech_started = False
        self._captured_samples = 0
        self._source_kind = "audio"
        # True once any speech frame has landed in the *current* chunk_frames
        # buffer since it was last cleared by a flush. Reset alongside
        # chunk_frames on every flush (timed cap, adaptive pause, or final).
        # Used only when KIOSK_CORE_SKIP_EMPTY_FINAL_FLUSH_ENABLED is set —
        # see config.py for the full rationale.
        self._chunk_has_speech = False
        # Count of turns where the final tail-chunk flush was skipped because
        # it held no unflushed speech, purely for observability/metrics.
        self._final_flush_skipped = False

        # ── Adaptive VAD state ─────────────────────────────────────────────
        # `request.silence_threshold` is the seed/fallback gate. When adaptive
        # VAD is enabled the effective gate (`_vad_threshold`) is re-derived
        # from the measured noise floor after the calibration window; until
        # then the seed value is used, so behaviour is unchanged if
        # calibration never completes (e.g. a very short recording).
        self._vad_threshold: float = float(self.request.silence_threshold)
        self._noise_floor: float | None = None
        self._vad_calibrating: bool = config.ADAPTIVE_VAD_ENABLED
        self._vad_calibration_rms: list[float] = []
        # ───────────────────────────────────────────────────────────────────

        # ── Pipeline timing (monotonic clock) ──────────────────────────────────
        # All _t_* fields are set during _finalize_run / _stream_rag_response.
        # Using time.monotonic() for accurate durations; datetime only for display.
        self._t_capture_start: float | None = None  # first speech frame detected
        self._asr_ms_total: float = 0.0             # summed transcribe_file time
        self._asr_chunks: int = 0                   # number of transcribe calls
        self._t_turn_start: float | None = None     # start of _finalize_run
        self._t_agent_start: float | None = None    # just before agent HTTP call
        self._t_agent_end: float | None = None      # agent reply received
        self._t_first_tts: float | None = None      # first TTS sentence queued
        self._t_first_audio: float | None = None    # first audio available (opener or first sentence)
        # Trailing silence the endpoint waited through before committing the
        # turn. Needed to report voice-to-voice latency, because every other
        # timestamp in the trace starts after this wait has already elapsed.
        self._endpoint_wait_seconds: float | None = None
        # First segment carrying the answer, stamped when its WAV is on disk.
        # _t_first_tts marks when a sentence was queued for synthesis, which is
        # ~one TTS call earlier and would understate voice-to-voice latency.
        self._t_first_answer_audio: float | None = None
        self._t_last_tts: float | None = None       # last TTS segment written (in worker thread)
        self._t_turn_end: float | None = None       # after worker.join()
        self._tts_segment_count: int = 0
        # ───────────────────────────────────────────────────────────────────────

        self._frame_samples = max(1, int(self.request.sample_rate * config.DEFAULT_BLOCK_DURATION_SECONDS))
        self._frame_duration_seconds = self._frame_samples / self.request.sample_rate
        self._vad_calibration_frames = max(
            1, int(config.DEFAULT_VAD_CALIBRATION_SECONDS / self._frame_duration_seconds)
        )
        # Preroll must outlast the calibration window: calibration frames are
        # classified as non-speech (they are what defines "non-speech"), so they
        # land in the preroll deque. If the deque were shorter than the window a
        # customer who starts talking immediately would lose their opening word.
        preroll_seconds = config.DEFAULT_PREROLL_SECONDS
        if config.ADAPTIVE_VAD_ENABLED:
            preroll_seconds = max(preroll_seconds, config.DEFAULT_VAD_CALIBRATION_SECONDS + 0.2)
        preroll_frames = max(1, int(preroll_seconds / self._frame_duration_seconds))
        self._preroll_frames: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._session_output_dir = Path(__file__).resolve().parent.parent / "generated_audio" / self.session_id

        # Silero VAD (optional, feature-flagged — see config.KIOSK_CORE_SILERO_VAD_ENABLED).
        # Only constructed when enabled: it loads an onnxruntime session, which
        # is unnecessary overhead for the default RMS-VAD path.
        self._silero_vad = None
        if config.KIOSK_CORE_SILERO_VAD_ENABLED:
            try:
                from kiosk_core.silero_vad import SileroVAD

                self._silero_vad = SileroVAD(
                    config.DEFAULT_SILERO_VAD_MODEL_PATH,
                    sample_rate=self.request.sample_rate,
                    intra_op_threads=config.DEFAULT_SILERO_VAD_INTRA_OP_THREADS,
                )
            except Exception:
                # Fail open: fall back to the RMS VAD rather than breaking the
                # session if the model file/onnxruntime isn't available.
                logger.exception(
                    "session=%s | Silero VAD enabled but failed to initialize; falling back to RMS VAD",
                    self.session_id,
                )
                self._silero_vad = None

    def start(self) -> None:
        with self._lock:
            if self.status != "created":
                raise ValueError("Session already started")
            self.status = "running"
            self.started_at = datetime.now(UTC)
        self._flush_thread.start()
        self._thread.start()

    def stop(self, reason: str = "stopped_by_api") -> None:
        with self._lock:
            if self.status not in {"running", "stopping"}:
                raise ValueError(f"Session is not running: {self.status}")
            self.status = "stopping"
            self.end_reason = reason
            self.stop_requested_at = datetime.now(UTC)
        self._stop_event.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            transcript = _assemble_transcript(self.transcript_parts)
            response_text = "".join(self.response_parts).strip()
            return {
                "session_id": self.session_id,
                "source_kind": self._source_kind,
                "status": self.status,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "stop_requested_at": self.stop_requested_at.isoformat() if self.stop_requested_at else None,
                "end_reason": self.end_reason,
                "error": self.error,
                "speech_started": self._speech_started,
                "noise_floor_rms": round(self._noise_floor, 1) if self._noise_floor is not None else None,
                "vad_threshold": round(self._vad_threshold, 1),
                "vad_calibrated": self._noise_floor is not None,
                "captured_audio_seconds": round(self._captured_samples / self.request.sample_rate, 3),
                "transcript": transcript,
                "partial_transcript": transcript,
                "transcript_parts": list(self.transcript_parts),
                "response": response_text,
                "response_parts": list(self.response_parts),
                "tts_audio_segments": [dict(segment) for segment in self.tts_audio_segments],
                "tts_errors": list(self.tts_errors),
            }

    def _run(self) -> None:
        raise NotImplementedError

    def _process_frame_stream(self, frame_iterator) -> tuple[str, str | None]:
        chunk_frames: list[np.ndarray] = []
        silence_run_seconds = 0.0
        final_status = "completed"
        end_reason = self.end_reason or "completed"

        # Adaptive flush threshold: flush accumulated speech to the background
        # ASR worker as soon as a natural pause appears, so the tail chunk at
        # true endpoint (silence_timeout_seconds) is as short as possible.
        # Only fires when the chunk has genuine speech content (avoids flushing
        # near-empty buffers) and when adaptive_flush_pause_seconds > 0.
        adaptive_pause = self.request.adaptive_flush_pause_seconds
        _adaptive_flushed = False  # guard: only one adaptive flush per silence run

        try:
            for frame in frame_iterator:
                if self._stop_event.is_set():
                    break

                rms = self._rms(frame)
                is_speech = rms >= self._vad_threshold
                # Refine the gate from the measured floor, then re-classify:
                # during the calibration window the seed threshold is still in
                # force, so the first frames after calibration must be judged
                # by the newly derived gate rather than the stale seed.
                # Still computed/updated even when Silero VAD is active: it's
                # cheap and keeps the RMS gate ready as an instant fallback.
                self._update_vad_threshold(rms, is_speech)
                if self._vad_calibrating:
                    # Still measuring the room — hold the frame in preroll
                    # rather than committing to a speech/silence decision.
                    is_speech = False
                else:
                    is_speech = rms >= self._vad_threshold

                if self._silero_vad is not None:
                    # Model-based speech probability replaces the RMS decision
                    # while feature-flagged on. int16-scale frame -> normalized
                    # float32 [-1, 1], as the Silero graph expects.
                    normalized = frame.astype(np.float32) / 32768.0
                    speech_prob = self._silero_vad.prob(normalized)
                    is_speech = speech_prob >= config.DEFAULT_SILERO_VAD_THRESHOLD

                if not self._speech_started:
                    if is_speech:
                        self._speech_started = True
                        if self._t_capture_start is None:
                            self._t_capture_start = time.monotonic()
                        while self._preroll_frames:
                            buffered = self._preroll_frames.popleft()
                            chunk_frames.append(buffered)
                            self._captured_samples += len(buffered)
                        chunk_frames.append(frame)
                        self._captured_samples += len(frame)
                    else:
                        self._preroll_frames.append(frame)
                    continue

                chunk_frames.append(frame)
                self._captured_samples += len(frame)

                if is_speech:
                    silence_run_seconds = 0.0
                    _adaptive_flushed = False  # new speech: allow adaptive flush again
                    self._chunk_has_speech = True
                else:
                    silence_run_seconds += self._frame_duration_seconds

                # ── Timed chunk flush (max chunk size cap) ──────────────────
                if self._chunk_duration_seconds(chunk_frames) >= self.request.chunk_seconds:
                    # Enqueue for the background flush worker instead of
                    # transcribing inline — see the _flush_queue docstring in
                    # __init__ for why this must not block this loop. Not the
                    # final chunk — speech may still be ongoing.
                    self._flush_queue.put((chunk_frames, False))
                    chunk_frames = []
                    silence_run_seconds = 0.0
                    _adaptive_flushed = False
                    self._chunk_has_speech = False
                    continue

                # ── Adaptive pause flush ────────────────────────────────────
                # When the speaker pauses for adaptive_flush_pause_seconds
                # (default 300ms) — but hasn't reached the endpoint yet —
                # flush the current speech to the background worker now so ASR
                # starts immediately. The tail chunk at endpoint will then
                # contain only silence frames (effectively empty), keeping
                # critical-path ASR cost near-zero.
                # Only fire once per silence run; reset when speech resumes.
                # Minimum 0.5s chunk: Whisper has a fixed per-call overhead
                # (~1.2s on CPU, ~150ms on GPU) that dominates sub-0.5s inputs
                # — sending near-empty frames wastes more time than it saves.
                if (
                    adaptive_pause > 0
                    and not _adaptive_flushed
                    and silence_run_seconds >= adaptive_pause
                    and silence_run_seconds < self.request.silence_timeout_seconds
                    and chunk_frames
                    and self._chunk_duration_seconds(chunk_frames) >= 0.5
                ):
                    logger.debug(
                        "[CHUNK] session=%s | adaptive flush at %.2fs pause (%.2fs of audio)",
                        self.session_id,
                        silence_run_seconds,
                        self._chunk_duration_seconds(chunk_frames),
                    )
                    # Not the final chunk — see DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
                    # in config.py for why this one is flagged non-final.
                    self._flush_queue.put((chunk_frames, False))
                    chunk_frames = []
                    _adaptive_flushed = True
                    self._chunk_has_speech = False
                    # Do NOT reset silence_run_seconds — we're still in silence,
                    # the endpoint counter keeps running toward silence_timeout_seconds.

                # ── Endpoint (trailing silence) ─────────────────────────────
                # Two waits, not one. A transcript that reads as a finished
                # sentence commits at endpoint_short_seconds; anything that
                # looks mid-thought keeps the full silence_timeout_seconds.
                # _looks_complete fails closed, so if the adaptive flush's ASR
                # has not landed yet this is exactly the old fixed behaviour.
                if (
                    config.DEFAULT_ENDPOINT_COMPLETE_ENABLED
                    and silence_run_seconds >= config.DEFAULT_ENDPOINT_SHORT_SECONDS
                    and silence_run_seconds < self.request.silence_timeout_seconds
                    # The tail has been flushed, so no un-transcribed speech is
                    # held back. chunk_frames is NOT a usable test here: every
                    # frame is appended to it, silence included, so it refills
                    # the moment the adaptive flush clears it.
                    and _adaptive_flushed
                    # ...and every flushed chunk has been transcribed. Without
                    # this, an in-flight tail could leave a truncated transcript
                    # that still reads as a finished sentence ("Order one
                    # classic"), and the turn would commit on the wrong words.
                    and self._flush_queue.unfinished_tasks == 0
                    and _looks_complete(
                        _assemble_transcript(self.transcript_parts),
                        config.DEFAULT_ENDPOINT_MIN_WORDS,
                    )
                ):
                    logger.info(
                        "[ENDPOINT] session=%s | early commit at %.2fs "
                        "(transcript reads complete; saved %.2fs)",
                        self.session_id,
                        silence_run_seconds,
                        self.request.silence_timeout_seconds - silence_run_seconds,
                    )
                    self._endpoint_wait_seconds = silence_run_seconds
                    end_reason = "silence_timeout"
                    break

                if silence_run_seconds >= self.request.silence_timeout_seconds:
                    self._endpoint_wait_seconds = silence_run_seconds
                    end_reason = "silence_timeout"
                    break

                if (self._captured_samples / self.request.sample_rate) >= self.request.max_session_seconds:
                    end_reason = "max_duration_reached"
                    break

        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Audio session %s failed", self.session_id)

        # The final chunk is enqueued the same way as every mid-stream chunk —
        # the worker's own exception handling (see _flush_worker) already
        # treats any single chunk's ASR failure as non-fatal, so there is
        # nothing left for this call site to catch. This IS the final chunk —
        # always keep full diarization on it regardless of
        # DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED (see config.py).
        #
        # See config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED: when set, and
        # nothing in chunk_frames is unflushed speech (the adaptive-pause
        # flush already sent every real word — chunk_frames is trailing
        # silence only), skip this call entirely rather than pay Whisper's
        # fixed per-call round trip to transcribe silence into "".
        if (
            chunk_frames
            and self._speech_started
            and (config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED is False or self._chunk_has_speech)
        ):
            self._flush_queue.put((chunk_frames, True))
        elif chunk_frames and self._speech_started:
            self._final_flush_skipped = True
            logger.info(
                "[CHUNK] session=%s | skipped empty final tail-chunk flush "
                "(%.2fs of silence, no unflushed speech since last flush)",
                self.session_id,
                self._chunk_duration_seconds(chunk_frames),
            )

        # Signal the worker to stop after draining everything queued so far,
        # then block until it has actually finished — _finalize_run (called
        # right after this returns) needs the complete transcript, and the
        # worker is what now owns every transcript_parts append. This is the
        # only wait left in the critical path: just the last chunk's ASR
        # latency, no longer serialised behind an earlier chunk's.
        self._flush_queue.put(None)
        self._flush_queue.join()

        return final_status, end_reason

    def _flush_worker(self) -> None:
        """Background worker that transcribes queued chunks one at a time.

        Runs on its own thread so the frame-reading loop in
        ``_process_frame_stream`` is never blocked waiting on an
        audio-analyzer round-trip. A single worker draining a FIFO queue
        guarantees chunks are still flushed in capture order, which
        ``_flush_chunk`` depends on for both ``transcript_parts`` ordering and
        the analyzer's cumulative-segment cursor (``_last_analyzer_segment_end``).

        A ``None`` item is the stop sentinel (queued once, after the last real
        chunk, by ``_process_frame_stream``). Any exception from an individual
        chunk's ``_flush_chunk`` call is caught and logged here rather than
        propagated, so one bad chunk can never lose the rest of an otherwise
        good utterance — the same reasoning the previous synchronous code
        already applied to the final chunk only; this now applies uniformly
        to every chunk.
        """
        while True:
            item = self._flush_queue.get()
            try:
                if item is None:
                    break
                frames, is_final = item
                try:
                    self._flush_chunk(frames, is_final)
                except Exception:
                    logger.exception(
                        "Audio session %s: chunk flush failed (non-fatal)", self.session_id,
                    )
            finally:
                self._flush_queue.task_done()

    def _finalize_run(self, final_status: str, end_reason: str) -> None:
        # Attempt RAG whenever there is a transcript, even if the session
        # ended with an error mid-stream (e.g. a transient ASR failure on one
        # chunk).  Only skip entirely when NO audio was captured at all.
        self._t_turn_start = time.monotonic()
        transcript = _assemble_transcript(self.transcript_parts)
        if transcript:
            # A real, accepted transcript ends any rejection streak for this
            # conversation — the customer was just heard, so a stale streak
            # from earlier turns must not trigger the retry prompt later.
            _reset_conversation_rejections(self.agent_session_id)
            try:
                self._stream_rag_response(transcript)
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                logger.exception("RAG query failed for session %s", self.session_id)
        elif final_status == "completed":
            # The transcript is empty — the kiosk usually has nothing
            # meaningful to say. Log why the turn produced no output, then
            # decide whether this specific case still warrants speaking.
            #
            # Rationale for staying silent by default:
            #  - "stopped_by_api" / "no_speech_detected": user explicitly stopped
            #    without speaking — speaking any prompt looks like the button
            #    had no effect.
            #  - Rejected-speech (speaker filter / diarization): often triggered
            #    by Whisper hallucinations ("you", "thank you", etc.) on
            #    background noise or TTS echo from the previous turn. These are
            #    not real utterances, so "I couldn't recognise your voice" is
            #    a false alarm on the FIRST occurrence.
            #  - "silence_timeout": VAD ended the turn with no speech — the
            #    customer is either not there or not ready; prompting can feel
            #    intrusive and the UI already shows "🎧 Listening…" or re-arms
            #    automatically in conversation mode.
            #
            # Exception: rejected speech is escalated once it becomes a
            # STREAK (config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
            # consecutive rejected turns in the same conversation). Staying
            # silent forever there regressed to the exact problem the retry
            # prompt originally existed for — a genuinely ignored customer
            # (real bystander, or a mistuned enrollment rejecting them) gets
            # zero feedback and the kiosk looks unresponsive. An explicit stop
            # never escalates: speaking here would look like the stop button
            # didn't work.
            if self._rejected_speech_chunks and end_reason != "stopped_by_api":
                streak = _note_conversation_rejection(self.agent_session_id)
                threshold = config.DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD
                if streak >= threshold:
                    logger.info(
                        "Session %s: %d chunk(s) rejected AND %d consecutive "
                        "rejected turn(s) for conversation %s (threshold=%d) "
                        "— asking the customer to repeat",
                        self.session_id, self._rejected_speech_chunks, streak,
                        self.agent_session_id, threshold,
                    )
                    self._synthesize_response(config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT)
                    _reset_conversation_rejections(self.agent_session_id)
                else:
                    logger.info(
                        "Session %s: %d chunk(s) of speech were rejected by the speaker "
                        "filter (likely hallucination or echo) — staying silent "
                        "(streak=%d/%d)",
                        self.session_id, self._rejected_speech_chunks, streak, threshold,
                    )
            else:
                logger.info(
                    "Session %s: empty transcript (end_reason=%s) — staying silent",
                    self.session_id, end_reason,
                )

        with self._lock:
            if final_status == "completed" and self.end_reason == "stopped_by_api":
                end_reason = "stopped_by_api"
            self.status = final_status
            self.completed_at = datetime.now(UTC)
            self.end_reason = end_reason

        # Record the completed turn for offline analysis. Controlled entirely
        # by config.CONVERSATION_LOGGING_ENABLED -- a no-op (no file I/O) when
        # the flag is off, and never raises when it's on (see
        # conversation_recorder.record_turn).
        conversation_recorder.record_turn(
            conversation_id=self.agent_session_id,
            turn_id=self.session_id,
            user_text=transcript,
            assistant_text="".join(str(part) for part in self.response_parts),
            end_reason=end_reason,
        )

        logger.info(
            "Session %s ended with reason=%s transcript=%s",
            self.session_id,
            self.end_reason,
            " ".join(self.transcript_parts).strip(),
        )
        if self.on_complete is not None:
            self.on_complete(self.session_id)

    def _synthesize_response(self, text: str) -> None:
        """Speak a fixed response directly via TTS, without calling RAG."""
        with self._lock:
            self.response_parts.append(text)
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        worker = threading.Thread(target=self._tts_worker, args=(sentence_queue,), daemon=True)
        worker.start()
        sentence_queue.put((1, text))
        sentence_queue.put((None, None))
        worker.join()

    def _emit_opener(self) -> None:
        """Play a cached, non-committal opener while the agent turn runs.

        Ordering turns emit only a tool call — the spoken reply is templated
        after the tool returns — so there is no model text to stream during the
        ~2 s the LLM spends generating tool-call JSON. This publishes a
        pre-rendered segment at index 0 (model sentences start at 1) so the
        customer hears something immediately.

        The segment is a plain file copy, so it costs no TTS round-trip. Any
        failure is swallowed: the opener is a latency optimisation and must
        never be able to break a turn.
        """
        if not config.DEFAULT_OPENER_ENABLED:
            return
        text = (config.DEFAULT_OPENER_TEXT or "").strip()
        if not text:
            return

        source = _render_opener(
            self.tts_client,
            text,
            self.request.tts_model,
            self.request.tts_voice,
            self.request.tts_language,
            self.request.tts_instructions,
        )
        if source is None:
            return

        try:
            destination = self._session_output_dir / "response_000.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            with self._lock:
                if self._t_first_audio is None:
                    self._t_first_audio = time.monotonic()
                self._t_last_tts = time.monotonic()
                self.tts_audio_segments.append(
                    {
                        "index": 0,
                        "text": text,
                        "audio_file": str(destination),
                    }
                )
            logger.info("[OPENER] session=%s emitted %r", self.session_id, text)
        except Exception:
            logger.exception("[OPENER] Failed to publish opener for session %s", self.session_id)

    def _stream_rag_response(self, transcript: str) -> None:
        pending_text = ""
        # Emitted before the agent call so the customer hears it while the
        # model is still generating the turn's tool call.
        self._emit_opener()
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        worker = threading.Thread(target=self._tts_worker, args=(sentence_queue,), daemon=True)
        worker.start()

        history = list(getattr(self.request, "history", []) or [])

        # Route through the ordering agent when enabled; fall back to direct RAG.
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
            token_source = self.agent_client.get_reply(
                transcription=transcript,
                session_id=self.agent_session_id,  # persistent across voice turns
                user_id=getattr(self.request, "user_id", None) or config.DEFAULT_ORDERING_USER_ID,
                history=history,
            )
            label = "Agent"
        else:
            token_source = self.rag_client.stream_answer(transcript, history=history)
            label = "RAG"

        print(f"\n{label} response for session {self.session_id}:\n", end="", flush=True)
        sentence_index = 0
        _first_token_seen = False
        _tool_calls: list[str] = []
        _llm_ms: float | None = None
        _llm_ttft_ms: float | None = None
        _llm_calls: int = 0
        _retrieval_ms: float | None = None
        # t_agent_start set here — generator body (HTTP call) runs on first iteration
        if self.agent_client is not None:
            self._t_agent_start = time.monotonic()
        try:
            for token in token_source:
                # Handle metadata sentinel from AgentClient BEFORE appending to response_parts
                # to avoid dict items in response_parts (which causes TypeError in snapshot())
                if isinstance(token, dict) and "_tool_calls" in token:
                    _tool_calls = token["_tool_calls"]
                    _llm_ms = token.get("_llm_ms")
                    _llm_ttft_ms = token.get("_llm_ttft_ms")
                    _llm_calls = token.get("_llm_calls", 0)
                    _retrieval_ms = token.get("_retrieval_ms")
                    continue

                with self._lock:
                    self.response_parts.append(token)
                print(token, end="", flush=True)

                if not _first_token_seen:
                    _first_token_seen = True
                    self._t_agent_end = time.monotonic()

                pending_text += token
                complete_sentences, pending_text = self._drain_complete_sentences(pending_text)
                for sentence in complete_sentences:
                    sentence_index += 1
                    if sentence_index == 1:
                        self._t_first_tts = time.monotonic()
                        if self._t_first_audio is None:
                            self._t_first_audio = self._t_first_tts
                    sentence_queue.put((sentence_index, sentence))

            trailing_text = pending_text.strip()
            if trailing_text:
                sentence_index += 1
                if sentence_index == 1:
                    self._t_first_tts = time.monotonic()
                    if self._t_first_audio is None:
                        self._t_first_audio = self._t_first_tts
                sentence_queue.put((sentence_index, trailing_text))

            # If agent_end wasn't set (empty reply), set it now
            if self._t_agent_start is not None and self._t_agent_end is None:
                self._t_agent_end = time.monotonic()

        finally:
            sentence_queue.put((None, None))
            worker.join()
            self._t_turn_end = time.monotonic()
            self._tts_segment_count = sentence_index
            print(flush=True)

        # ── Record pipeline turn trace ──────────────────────────────────────
        self._record_turn_trace(
            _tool_calls, _llm_ms, _llm_calls, _retrieval_ms, _llm_ttft_ms
        )

    def _record_turn_trace(
        self,
        tool_calls: list[str],
        llm_ms: float | None = None,
        llm_calls: int = 0,
        retrieval_ms: float | None = None,
        llm_ttft_ms: float | None = None,
    ) -> None:
        """Build and persist a TurnTrace for the completed voice turn."""
        t0 = self._t_turn_start
        t_agent_s = self._t_agent_start
        t_agent_e = self._t_agent_end
        t_first = self._t_first_tts
        t_last = self._t_last_tts
        t_end = self._t_turn_end

        def _ms(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return round((b - a) * 1000, 1)

        # ASR = summed analyzer round-trips measured in _flush_chunk. It cannot
        # be derived from turn_start, because every chunk is already transcribed
        # by the time _finalize_run stamps t0.
        asr_ms = round(self._asr_ms_total, 1) if self._asr_chunks else None
        # Agent TTFT = from agent_start to when first token (reply) arrived
        ttft_ms = _ms(t_agent_s, t_agent_e)
        # Agent total = from agent_start to last TTS segment done (whole orchestration)
        agent_total_ms = _ms(t_agent_s, t_end)
        # TTS = from first sentence queued to last segment written
        tts_ms = _ms(t_first, t_last)
        # Time to first audio = from end of speech (t0, stamped by
        # _finalize_run) to the moment the first segment became available.
        #
        # Measured from t0 rather than from agent_start because the opener is
        # published *before* the agent call starts, which would otherwise make
        # this negative. t0 is also the more honest boundary: it is what the
        # customer actually waits through after they stop talking.
        ttfa_ms = _ms(t0, self._t_first_audio or t_first)
        # Voice to voice: the customer's last word to the first sound. t0 is
        # stamped at the endpoint decision, so the silence the customer sat
        # through has to be added back on. This is the clock external
        # voice-kiosk numbers use; ttfa_ms alone understates the wait.
        endpoint_wait_ms = (
            round(self._endpoint_wait_seconds * 1000, 1)
            if self._endpoint_wait_seconds is not None
            else None
        )
        v2v_ms = (
            round(endpoint_wait_ms + ttfa_ms, 1)
            if endpoint_wait_ms is not None and ttfa_ms is not None
            else None
        )
        # Same clock, but to the first sound that actually answers. The opener
        # is deliberately excluded here: it breaks the silence but tells the
        # customer nothing, so counting it as "the reply" would flatter the
        # number. Uses the on-disk stamp, not the queue stamp.
        informative_ms = _ms(t0, self._t_first_answer_audio)
        v2v_informative_ms = (
            round(endpoint_wait_ms + informative_ms, 1)
            if endpoint_wait_ms is not None and informative_ms is not None
            else None
        )
        # Wall E2E: genuinely end-to-end — from the first speech frame captured
        # (so audio capture and ASR are included) to the last TTS segment
        # written. Falls back to turn_start when no speech was ever detected.
        t_e2e_start = self._t_capture_start or t0
        wall_total_ms = _ms(t_e2e_start, t_end)

        retrieval_invoked = any(
            "retrieval" in tc.lower() or "knowledge" in tc.lower() or "lookup" in tc.lower()
            for tc in tool_calls
        )

        trace = TurnTrace(
            turn_id=self.session_id,
            conversation_id=self.agent_session_id,
            started_at=self.started_at.isoformat() if self.started_at else datetime.now(UTC).isoformat(),
            ended_at=datetime.now(UTC).isoformat(),
            wall=WallTimes(
                turn_total_ms=wall_total_ms,
                time_to_first_audio_ms=ttfa_ms,
                endpoint_wait_ms=endpoint_wait_ms,
                voice_to_voice_ms=v2v_ms,
                voice_to_voice_informative_ms=v2v_informative_ms,
            ),
            asr=AsrSpan(ms=asr_ms, chunks=self._asr_chunks, final_flush_skipped=self._final_flush_skipped),
            agent=AgentSpan(
                ttft_ms=ttft_ms,
                total_ms=agent_total_ms,
                retrieval=RetrievalSpan(
                    invoked=retrieval_invoked,
                    ms=retrieval_ms,
                ),
                llm=LlmSpan(
                    ms=llm_ms,
                    ttft_ms=llm_ttft_ms,
                    calls=llm_calls,
                    device="GPU",
                ),
            ),
            tts=TtsSpan(
                ms=tts_ms,
                segments=self._tts_segment_count,
                overlapped_with_agent=True,
            ),
        )
        pipeline_store.record(trace)
        logger.info(
            "[PIPELINE] turn=%s wall=%.0fms asr=%.0fms(%d) ttft=%.0fms tts=%.0fms "
            "llm=%.0fms(%d) retrieval=%s/%.0fms tools=%s",
            self.session_id,
            wall_total_ms or 0,
            asr_ms or 0,
            self._asr_chunks,
            ttft_ms or 0,
            tts_ms or 0,
            llm_ms or 0,
            llm_calls,
            retrieval_invoked,
            retrieval_ms or 0,
            tool_calls,
        )

    @staticmethod
    def _drain_complete_sentences(buffer: str) -> tuple[list[str], str]:
        sentences: list[str] = []
        remaining = buffer
        while True:
            match = _SENTENCE_PATTERN.match(remaining.lstrip())
            if match is None:
                break
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining.lstrip()[match.end() :]
        return sentences, remaining

    def _tts_worker(self, sentence_queue: Queue[tuple[int | None, str | None]]) -> None:
        while True:
            sentence_index, sentence = sentence_queue.get()
            if sentence_index is None or sentence is None:
                return

            output_path = self._session_output_dir / f"response_{sentence_index:03d}.wav"
            try:
                self.tts_client.synthesize_to_file(
                    text=sentence,
                    output_path=str(output_path),
                    model=self.request.tts_model,
                    voice=self.request.tts_voice,
                    language=self.request.tts_language,
                    instructions=self.request.tts_instructions,
                )
                if config.DEFAULT_TTS_TRIM_ENABLED:
                    self._trim_tts_segment(output_path, sentence)
                if config.DEFAULT_TTS_GAIN_ENABLED:
                    self._apply_tts_gain(output_path)
                with self._lock:
                    self._t_last_tts = time.monotonic()
                    # First segment that carries the ANSWER (the opener is
                    # index 0 and is written by _emit_opener, never here).
                    # Stamped on write, not on queue: queuing a sentence does
                    # not make a sound, and the synthesis in between is a real
                    # part of what the customer waits through.
                    if self._t_first_answer_audio is None:
                        self._t_first_answer_audio = self._t_last_tts
                    self.tts_audio_segments.append(
                        {
                            "index": sentence_index,
                            "text": sentence,
                            "audio_file": str(output_path),
                        }
                    )
            except Exception as exc:
                logger.exception("TTS synthesis failed for session %s sentence %s", self.session_id, sentence_index)
                with self._lock:
                    self.tts_errors.append(f"sentence {sentence_index}: {exc}")

    def _trim_tts_segment(self, path: Path, sentence: str) -> None:
        """Trim baked-in silence from a synthesised segment to a fixed pad.

        Segments are synthesised per clause so playback can start early, which
        leaves each one padded with its own lead-in and lead-out silence. Played
        back to back those pads stack into an audible stall at every comma. This
        rewrites the file in place keeping a short, deliberate pad instead —
        longer at a sentence end than mid-sentence, so prosody still breathes.

        Failures are swallowed: a segment that cannot be parsed is simply left
        as synthesised, since degraded pacing is preferable to a lost reply.

        Args:
            path: WAV file to rewrite in place.
            sentence: Text the segment was synthesised from; its final
                punctuation selects the trailing pad.
        """
        try:
            with wave.open(str(path), "rb") as wav_in:
                n_channels = wav_in.getnchannels()
                sample_width = wav_in.getsampwidth()
                frame_rate = wav_in.getframerate()
                frames = wav_in.readframes(wav_in.getnframes())

            # Only 16-bit mono is handled; anything else is left untouched
            # rather than risking a corrupted rewrite.
            if sample_width != 2 or n_channels != 1 or not frames:
                return

            samples = np.frombuffer(frames, dtype=np.int16)
            if samples.size == 0:
                return

            envelope = np.abs(samples.astype(np.int32))
            peak = int(envelope.max())
            if peak <= 0:
                return

            floor = max(peak * config.DEFAULT_TTS_SILENCE_FLOOR, 1.0)
            voiced = np.flatnonzero(envelope > floor)
            if voiced.size == 0:
                return

            trailing_ms = (
                config.DEFAULT_TTS_SENTENCE_PAD_MS
                if sentence.rstrip()[-1:] in ".!?"
                else config.DEFAULT_TTS_CLAUSE_PAD_MS
            )
            lead_pad = int(frame_rate * config.DEFAULT_TTS_LEAD_PAD_MS / 1000.0)
            trail_pad = int(frame_rate * trailing_ms / 1000.0)

            start = max(0, int(voiced[0]) - lead_pad)
            end = min(samples.size, int(voiced[-1]) + 1 + trail_pad)
            if end - start >= samples.size:
                return

            segment = samples[start:end].astype(np.float64)
            # Ramp both edges to true zero so back-to-back playback of
            # separately-synthesised segments never has a sample-level jump
            # at the seam (see DEFAULT_TTS_FADE_MS docstring in config.py).
            fade_samples = min(
                int(frame_rate * config.DEFAULT_TTS_FADE_MS / 1000.0),
                segment.size // 4,
            )
            if fade_samples > 1:
                ramp = np.linspace(0.0, 1.0, fade_samples)
                segment[:fade_samples] *= ramp
                segment[-fade_samples:] *= ramp[::-1]
            segment = np.clip(segment, -32768, 32767).astype(np.int16)

            with wave.open(str(path), "wb") as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(frame_rate)
                wav_out.writeframes(segment.tobytes())

            logger.debug(
                "[TTS] session=%s trimmed %s: %.2fs -> %.2fs",
                self.session_id, path.name,
                samples.size / frame_rate, (end - start) / frame_rate,
            )
        except Exception:
            logger.warning(
                "[TTS] session=%s could not trim %s; using untrimmed audio",
                self.session_id, path.name, exc_info=True,
            )

    def _apply_tts_gain(self, path: Path) -> None:
        """Boost a synthesized segment's loudness for kiosk speakers.

        SpeechT5's vocoder outputs a quiet, roughly constant level regardless
        of which speaker embedding is used, so a soft-sounding kiosk is a
        level problem, not a voice-choice problem, and there is no gain
        control in the text-to-speech service itself. This peak-normalizes
        the segment to ``DEFAULT_TTS_TARGET_PEAK`` of full scale, then applies
        an extra flat boost (``DEFAULT_TTS_GAIN_DB``), with the combined gain
        hard-clamped at ``DEFAULT_TTS_GAIN_MAX_DB`` so a near-silent or failed
        synthesis can't be amplified into distortion/noise.

        Failures are swallowed: a segment that cannot be parsed is left as
        synthesised, since a quiet reply is preferable to a corrupted one.

        Args:
            path: WAV file to rewrite in place.
        """
        try:
            with wave.open(str(path), "rb") as wav_in:
                n_channels = wav_in.getnchannels()
                sample_width = wav_in.getsampwidth()
                frame_rate = wav_in.getframerate()
                frames = wav_in.readframes(wav_in.getnframes())

            # Only 16-bit mono is handled; anything else is left untouched
            # rather than risking a corrupted rewrite.
            if sample_width != 2 or n_channels != 1 or not frames:
                return

            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            if samples.size == 0:
                return

            peak = float(np.abs(samples).max())
            if peak <= 0:
                return

            int16_max = 32767.0
            normalize_gain = (int16_max * config.DEFAULT_TTS_TARGET_PEAK) / peak
            extra_gain = 10.0 ** (config.DEFAULT_TTS_GAIN_DB / 20.0)
            max_gain = 10.0 ** (config.DEFAULT_TTS_GAIN_MAX_DB / 20.0)
            # `normalize_gain * extra_gain` alone can push the segment's peak
            # past full scale (extra_gain is a flat boost stacked *on top of*
            # normalization, not a replacement for it), which used to get
            # silently hard-clipped by np.clip below. That is real clipping
            # distortion, not a client-side volume issue: observed live on a
            # real synthesized sentence, 14 separate clipped runs (up to 6
            # consecutive samples each) — audible as a subtle crackle/buzz on
            # every loud syllable. `no_clip_gain` caps the total at the exact
            # gain that brings the peak to (not past) full scale, so this
            # segment is never actually clipped, only ever soft-limited.
            no_clip_gain = int16_max / peak
            total_gain = min(normalize_gain * extra_gain, max_gain, no_clip_gain)

            if abs(total_gain - 1.0) < 1e-3:
                return  # already at target level; skip a no-op rewrite

            boosted = np.clip(samples * total_gain, -int16_max, int16_max).astype(np.int16)

            with wave.open(str(path), "wb") as wav_out:
                wav_out.setnchannels(n_channels)
                wav_out.setsampwidth(sample_width)
                wav_out.setframerate(frame_rate)
                wav_out.writeframes(boosted.tobytes())

            logger.debug(
                "[TTS] session=%s gained %s: peak %d -> target %.0f%% FS (total gain %.1f dB)",
                self.session_id, path.name, int(peak),
                config.DEFAULT_TTS_TARGET_PEAK * 100,
                20.0 * math.log10(total_gain),
            )
        except Exception:
            logger.warning(
                "[TTS] session=%s could not apply gain to %s; using unmodified audio",
                self.session_id, path.name, exc_info=True,
            )

    def _on_audio(self, indata, frames, time, status) -> None:
        del frames, time
        if status:
            logger.warning("Audio callback status for %s: %s", self.session_id, status)
        self._audio_queue.put(indata[:, 0].copy())

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        samples = frame.astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples)))

    def _update_vad_threshold(self, rms: float, is_speech: bool) -> None:
        """Derive the speech gate from the measured background noise level.

        During the calibration window every frame is treated as background and
        collected; once enough frames exist the floor is taken as a low
        percentile of them (robust to a customer who starts talking straight
        away, since the quiet gaps between words still dominate the low
        percentile). Afterwards the floor keeps tracking, but only on frames
        classified as non-speech — adapting during speech would make the gate
        climb mid-utterance and cut the customer off.

        The resulting gate is always clamped to
        ``[DEFAULT_VAD_THRESHOLD_MIN, DEFAULT_VAD_THRESHOLD_MAX]`` so a freak
        measurement can never push it high enough to suppress all speech.

        Args:
            rms: Energy of the current frame.
            is_speech: Current classification of this frame, used to freeze
                floor adaptation while the customer is talking.
        """
        if not config.ADAPTIVE_VAD_ENABLED:
            return

        if self._vad_calibrating:
            self._vad_calibration_rms.append(rms)
            if len(self._vad_calibration_rms) < self._vad_calibration_frames:
                return
            floor = float(
                np.percentile(self._vad_calibration_rms, config.DEFAULT_VAD_FLOOR_PERCENTILE)
            )
            self._vad_calibrating = False
            self._vad_calibration_rms = []
        elif not is_speech:
            current = self._noise_floor if self._noise_floor is not None else rms
            # Asymmetric: track a quietening room quickly, but resist being
            # dragged upward by the quiet gaps inside an utterance.
            alpha = (
                config.DEFAULT_VAD_FLOOR_ADAPT_DOWN
                if rms < current
                else config.DEFAULT_VAD_FLOOR_ADAPT_UP
            )
            floor = (1.0 - alpha) * current + alpha * rms
        else:
            return

        gate = floor * (10.0 ** (config.DEFAULT_VAD_MARGIN_DB / 20.0))
        clamped = min(
            max(gate, float(config.DEFAULT_VAD_THRESHOLD_MIN)),
            float(config.DEFAULT_VAD_THRESHOLD_MAX),
        )

        first_time = self._noise_floor is None
        self._noise_floor = floor
        self._vad_threshold = clamped

        if first_time:
            # Logged at INFO deliberately: this single line is the on-site
            # diagnostic for whether the venue's noise floor is workable.
            logger.info(
                "[VAD] session=%s | noise floor RMS=%.0f (%.1f dBFS) | gate=%.0f%s | seed was %d",
                self.session_id,
                floor,
                20.0 * np.log10(max(floor, 1.0) / 32767.0),
                clamped,
                " (CLAMPED — venue louder than gate ceiling, "
                "falling back to permissive detection)"
                if clamped < gate
                else "",
                self.request.silence_threshold,
            )

    def _chunk_duration_seconds(self, frames: list[np.ndarray]) -> float:
        total_samples = sum(len(frame) for frame in frames)
        return total_samples / self.request.sample_rate

    def _flush_chunk(self, frames: list[np.ndarray], is_final: bool = True) -> None:
        audio = np.concatenate(frames, axis=0)
        temp_path = self._write_temp_wav(audio)
        try:
            duration = len(audio) / self.request.sample_rate
            # Full diarization always runs on the final tail chunk. On
            # intermediate chunks (max-chunk-size-cap flush, adaptive-pause
            # pre-warm flush) it is gated by DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
            # — see config.py for the latency/accuracy tradeoff this encodes.
            diarization_requested = config.DEFAULT_DIARIZATION_ENABLED and (
                is_final or config.DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
            )
            logger.info(
                "[CHUNK] session=%s | flushing %.2fs of audio, is_final=%s, diarization=%s",
                self.session_id, duration, is_final, diarization_requested,
            )
            _t_asr = time.monotonic()
            payload = self.client.transcribe_file(
                temp_path,
                language=self.request.language,
                temperature=self.request.temperature,
                diarization=diarization_requested,
                session_id=self._analyzer_session_id,
                speaker_scope_id=self.agent_session_id,
                # Bias Whisper towards the real menu vocabulary. Without it the
                # decoder spells product names phonetically ("aloo tiki",
                # "Kin Burger"), which the catalogue's fuzzy resolver then
                # fails to match, so the item silently never reaches the cart.
                prompt=config.DEFAULT_ASR_PROMPT,
            )
            # Accumulate genuine ASR time. Chunks are transcribed as they are
            # flushed during capture, long before _finalize_run runs, so this
            # is the only place the cost can be observed.
            self._asr_ms_total += (time.monotonic() - _t_asr) * 1000
            self._asr_chunks += 1
            # Latch the analyzer's assigned session id from the first
            # response so subsequent chunks reuse the same server-side
            # session (and its enrolled primary voice embedding).
            assigned = (
                payload.get("_analyzer_session_id")
                if isinstance(payload, dict)
                else None
            )
            if assigned and assigned != self._analyzer_session_id:
                logger.info(
                    "[CHUNK] session=%s | analyzer session pinned to %s",
                    self.session_id, assigned,
                )
                self._analyzer_session_id = assigned
            segments: list[dict] = payload.get("segments", []) if isinstance(payload, dict) else []
            raw_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else str(payload).strip()

            logger.info(
                "[CHUNK] session=%s | audio-analyzer response: %d segment(s), flat_text=%r",
                self.session_id, len(segments), raw_text[:120],
            )

            # ── Dedupe against analyzer's cumulative session state ──────────
            # The analyzer reuses our session_id in append_to_session mode and
            # returns EVERY segment ever produced for the session (with
            # timestamps offset by the accumulated duration). Keep only the
            # segments that start after the last end-time we've already seen;
            # otherwise every prior utterance gets re-appended to the
            # transcript on each new chunk.
            if segments:
                fresh_segments = [
                    s for s in segments
                    if float(s.get("end", 0.0)) > self._last_analyzer_segment_end + 1e-3
                ]
                if len(fresh_segments) != len(segments):
                    logger.info(
                        "[CHUNK] session=%s | deduped %d cumulative segment(s) → %d fresh (cursor=%.2fs)",
                        self.session_id,
                        len(segments) - len(fresh_segments),
                        len(fresh_segments),
                        self._last_analyzer_segment_end,
                    )
                segments = fresh_segments
                if segments:
                    self._last_analyzer_segment_end = max(
                        self._last_analyzer_segment_end,
                        max(float(s.get("end", 0.0)) for s in segments),
                    )
                    # Rebuild raw_text from fresh segments so the flat-text
                    # fallback (used when diarization is off or returns no
                    # segments) is also free of the cumulative duplicates.
                    raw_text = " ".join(
                        s.get("text", "").strip() for s in segments if s.get("text", "").strip()
                    ).strip()
                else:
                    raw_text = ""

            if segments and diarization_requested:
                text = self._filter_target_speaker(segments)
            else:
                if diarization_requested and not segments:
                    logger.info(
                        "[CHUNK] session=%s | diarization enabled but no segments returned — using flat text",
                        self.session_id,
                    )
                text = raw_text

            if text:
                # Strip Whisper hallucination tokens (e.g. [BLANK_AUDIO], [Music])
                text = _WHISPER_JUNK.sub("", text).strip()
            if text:
                # Correct the "cart" -> "card" ASR homophone before the agent
                # ever sees the transcript (see _normalize_card_cart_homophone).
                normalized = _normalize_card_cart_homophone(text)
                if normalized != text:
                    logger.info(
                        "[CHUNK] session=%s | normalized card->cart homophone: %r -> %r",
                        self.session_id, text[:120], normalized[:120],
                    )
                    text = normalized
            if text and _WHISPER_FILLER.fullmatch(text):
                logger.info(
                    "[CHUNK] session=%s | dropping filler-only transcription: %r",
                    self.session_id, text[:60],
                )
                text = ""
            if text:
                logger.info(
                    "[CHUNK] session=%s | appending to transcript: %r",
                    self.session_id, text[:120],
                )
                with self._lock:
                    self.transcript_parts.append(text)
            else:
                logger.info(
                    "[CHUNK] session=%s | chunk produced no usable text (filtered or empty)",
                    self.session_id,
                )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _note_rejected_speech(self, segments: list[dict], reason: str) -> None:
        """Record that transcribed speech was discarded by the speaker filter.

        Only counts chunks that actually carried words. A chunk of pure
        silence yields empty segment text and must stay classified as "no
        speech", otherwise every silent moment would trigger the "please
        repeat" prompt.

        Args:
            segments: The diarized segments that were rejected.
            reason: Short tag naming which filter rule rejected them.
        """
        spoken = " ".join(s.get("text", "") for s in segments).strip()
        if not spoken:
            return
        # Mirrors the defensive getattr used for _primary_speaker_id: the
        # filter is exercised directly by unit tests against bare instances.
        self._rejected_speech_chunks = getattr(self, "_rejected_speech_chunks", 0) + 1
        logger.info(
            "[SPEAKER-FILTER] session=%s | rejected speech recorded (%s) "
            "| rejected_chunks=%d | text=%r",
            self.session_id, reason, self._rejected_speech_chunks, spoken[:120],
        )

    def _filter_target_speaker(self, segments: list[dict]) -> str:
        """Filter diarized segments to keep only the primary customer's speech.

        Primary speaker is determined in order of precedence:
        1. Analyzer-provided ``is_primary`` flag on the segment — honoured
           directly when present, so the analyzer's own enrollment logic wins.
           Its verdict is authoritative in *both* directions: if it marks every
           segment non-primary, the chunk is dropped outright.
        2. First-speaker lock-on — the first speaker label seen in this session
           is treated as the customer.  Every subsequent segment from that label
           is kept; any other label is unconditionally dropped.

        Semantic fallback is used only before the primary speaker is
        established (i.e. on the very first chunk that has no clear speaker
        label), so a legitimate opening utterance is never lost.
        """
        if not segments:
            return ""

        # Defensive read — guard against subclasses or test stubs that may not
        # have called BaseAudioSession.__init__.
        primary_speaker_id: str | None = getattr(self, "_primary_speaker_id", None)

        # ── Honor analyzer-provided is_primary when present ──────────────────
        # The flag only appears once the analyzer holds an enrolled reference
        # voice for this conversation (scoped by speaker_scope_id), so when it
        # is present its verdict is authoritative and must be honoured in BOTH
        # directions. In particular "no segment is primary" means the analyzer
        # has positively matched this speech against the enrolled customer and
        # rejected it — the chunk must be dropped, never fall through to the
        # first-speaker rule below (which would lock on to the interloper,
        # because _primary_speaker_id resets on every new audio session).
        if any("is_primary" in s for s in segments):
            primary_segments = [s for s in segments if s.get("is_primary")]
            if not primary_segments:
                if not config.DEFAULT_SPEAKER_STRICT_DROP:
                    logger.warning(
                        "[SPEAKER-FILTER] session=%s | analyzer rejected all %d segment(s) but "
                        "strict drop is DISABLED — falling back to heuristics",
                        self.session_id, len(segments),
                    )
                else:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | analyzer rejected all %d segment(s) as "
                        "non-primary — chunk DROPPED | text=%r",
                        self.session_id, len(segments),
                        " ".join(s.get("text", "") for s in segments).strip()[:120],
                    )
                    self._note_rejected_speech(segments, "analyzer_non_primary")
                    return ""
            else:
                # Also update / initialise the lock-on label from the first
                # primary segment so label-based filtering stays in sync.
                label = primary_segments[0].get("speaker", "")
                if label and primary_speaker_id is None:
                    self._primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (is_primary flag)",
                        self.session_id, label,
                    )
                final_text = " ".join(seg.get("text", "") for seg in primary_segments).strip()
                logger.info(
                    "[SPEAKER-FILTER] session=%s | is_primary path: kept=%d | final_text=%r",
                    self.session_id, len(primary_segments), final_text[:120],
                )
                return final_text

        # ── Lock on to the first speaker seen in this session ────────────────
        if primary_speaker_id is None:
            for seg in segments:
                label = seg.get("speaker", "")
                if label:
                    self._primary_speaker_id = label
                    primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (first speaker rule)",
                        self.session_id, primary_speaker_id,
                    )
                    break

        # ── Classify segments as primary / non-primary ───────────────────────
        if primary_speaker_id:
            kept_segments = [s for s in segments if s.get("speaker") == primary_speaker_id]
            discarded_segments = [s for s in segments if s.get("speaker") != primary_speaker_id]
        else:
            # No speaker label at all — treat everything as potentially primary
            kept_segments = []
            discarded_segments = list(segments)

        logger.info(
            "[SPEAKER-FILTER] session=%s | processing %d segment(s): %d primary, %d non-primary",
            self.session_id, len(segments), len(kept_segments), len(discarded_segments),
        )
        for i, segment in enumerate(segments):
            is_primary = primary_speaker_id is not None and segment.get("speaker") == primary_speaker_id
            logger.info(
                "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s is_primary=%s → %s | text=%r",
                self.session_id, i, segment.get("speaker", "UNKNOWN"), is_primary,
                "KEEP" if is_primary else "DISCARD", segment.get("text", "")[:80],
            )

        # ── Semantic fallback — only before primary is established ───────────
        if not kept_segments and discarded_segments:
            if primary_speaker_id:
                # Primary is known — non-primary utterances are noise, drop them.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | non-primary speech only, primary=%s — chunk DROPPED (no fallback)",
                    self.session_id, primary_speaker_id,
                )
            else:
                # Primary not yet established — run semantic fallback so the
                # very first domain utterance (before a clean speaker label
                # arrives) is not silently discarded.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | primary not yet established — running semantic fallback on %d segment(s)",
                    self.session_id, len(discarded_segments),
                )
                best_score = 0.0
                best_segment: dict | None = None
                for segment in discarded_segments:
                    words = segment.get("text", "").lower().split()
                    if not words:
                        continue
                    overlap = sum(1 for w in words if w in _DOMAIN_KEYWORDS)
                    score = overlap / max(len(words), 1)
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | fallback score speaker=%s score=%.2f | text=%r",
                        self.session_id, segment.get("speaker", "UNKNOWN"), score, segment.get("text", "")[:80],
                    )
                    if score > best_score:
                        best_score = score
                        best_segment = segment

                if best_segment is not None and best_score >= config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback ACCEPTED speaker=%s score=%.2f | text=%r",
                        self.session_id, best_segment.get("speaker", "UNKNOWN"), best_score, best_segment.get("text", "")[:80],
                    )
                    kept_segments = [best_segment]
                else:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback found no domain match (best_score=%.2f, threshold=%.2f) → chunk DROPPED",
                        self.session_id, best_score, config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD,
                    )

        final_text = " ".join(seg.get("text", "") for seg in kept_segments).strip()
        if not final_text:
            # Covers both remaining rejection routes: a known primary was
            # silent while somebody else spoke, and the semantic fallback
            # finding no domain match before a primary was established.
            self._note_rejected_speech(discarded_segments, "no_primary_segment_kept")
        logger.info(
            "[SPEAKER-FILTER] session=%s | RESULT: kept=%d dropped=%d | final_text=%r",
            self.session_id,
            len(kept_segments),
            len(discarded_segments) - (1 if kept_segments and discarded_segments else 0),
            final_text[:120],
        )
        return final_text

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(prefix=f"{self.session_id}-", suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.request.sample_rate)
            wav_file.writeframes(audio.astype(np.int16).tobytes())

        return temp_path


class BrowserStreamSession(BaseAudioSession):
    """Session that receives audio chunks pushed from the browser via HTTP.

    Call push_audio(wav_bytes) from the HTTP handler each time a chunk arrives.
    The session applies the same RMS silence detection and chunk-flushing logic
    as MicrophoneSession.  It ends automatically when:
      - silence_timeout_seconds of silence follows detected speech, OR
      - max_session_seconds of captured audio have been processed, OR
      - stop() is called explicitly (e.g. user clicks stop-recording in browser).
    """

    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"browser-session-{self.session_id}", daemon=True)
        self._source_kind = "browser"
        # Sentinel: None means end-of-stream
        self._push_queue: Queue[np.ndarray | None] = Queue()

    def push_audio(self, wav_bytes: bytes) -> None:
        """Called from the HTTP handler for each incoming audio chunk."""
        # Browser chunks arrive as WAV containers. Decode them first so we only
        # enqueue actual PCM frames (not RIFF/WAV headers), which keeps RMS/VAD
        # and ASR input stable across chunk boundaries.
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                raw = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")

            audio = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)[:, 0]

            if sample_rate != self.request.sample_rate:
                logger.warning(
                    "[CHUNK] session=%s | WAV sample rate %s differs from session sample_rate %s",
                    self.session_id,
                    sample_rate,
                    self.request.sample_rate,
                )
        except (wave.Error, EOFError) as exc:
            logger.debug(
                "[CHUNK] session=%s | WAV decode failed; falling back to raw PCM (%s)",
                self.session_id,
                exc,
            )
            # Backward-compat fallback for legacy clients that might post raw
            # PCM bytes directly instead of a WAV container.
            audio = np.frombuffer(wav_bytes, dtype=np.int16)

        # Split into frame-sized pieces so _process_frame_stream sees uniform frames
        for start in range(0, len(audio), self._frame_samples):
            frame = audio[start : start + self._frame_samples]
            if len(frame) > 0:
                self._push_queue.put(frame.copy())

    def signal_end(self) -> None:
        """Signal that the browser has stopped recording (enqueue sentinel)."""
        self._push_queue.put(None)

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_push_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Browser stream session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)

    def _iter_push_frames(self):
        while not self._stop_event.is_set():
            try:
                frame = self._push_queue.get(timeout=0.25)
            except Empty:
                continue
            if frame is None:
                # End-of-stream sentinel from signal_end()
                break
            yield frame


class MicrophoneSession(BaseAudioSession):
    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        self._source_kind = "microphone"

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            with sd.InputStream(
                samplerate=self.request.sample_rate,
                blocksize=self._frame_samples,
                channels=1,
                dtype="int16",
                device=self.request.device,
                callback=self._on_audio,
            ):
                def iter_frames():
                    while not self._stop_event.is_set():
                        try:
                            yield self._audio_queue.get(timeout=0.25)
                        except Empty:
                            continue

                final_status, end_reason = self._process_frame_stream(iter_frames())
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Microphone session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)


class TextQuerySession(BaseAudioSession):
    """A typed-question turn. Skips audio capture and ASR entirely, feeding the
    text straight through the RAG/agent pipeline. Per the UI's text-input
    design the answer is returned as text only (no speech is synthesized)."""

    def __init__(
        self,
        request: SessionStartRequest,
        query_text: str,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._query_text = query_text
        self._thread = threading.Thread(target=self._run, name=f"text-session-{self.session_id}", daemon=True)
        self._source_kind = "text"

    def start(self) -> None:
        # No audio capture ??? the ASR flush worker is never needed; only the
        # main thread (which drives RAG) runs.
        with self._lock:
            if self.status != "created":
                raise ValueError("Session already started")
            self.status = "running"
            self.started_at = datetime.now(UTC)
        self._thread.start()

    def _run(self) -> None:
        final_status = "completed"
        end_reason = "text_query"
        try:
            with self._lock:
                self._speech_started = True
                self.transcript_parts.append(self._query_text)
        except Exception as exc:  # noqa: BLE001
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Text session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)

    def _tts_worker(self, sentence_queue: Queue[tuple[int | None, str | None]]) -> None:
        # Text queries are answered in text only. Drain the sentence queue
        # without synthesizing so _stream_rag_response still accumulates the
        # response text and completes normally.
        while True:
            sentence_index, sentence = sentence_queue.get()
            if sentence_index is None or sentence is None:
                return

    def _emit_opener(self) -> None:
        # Text queries produce no audio at all, so there is no silent gap to
        # fill and an opener segment would be a spurious audio artefact.
        return


class FileAudioSession(BaseAudioSession):
    def __init__(
        self,
        request: FileSessionStartRequest,
        audio_file_path: str,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self.request = request
        self.audio_file_path = audio_file_path
        self._thread = threading.Thread(target=self._run, name=f"file-session-{self.session_id}", daemon=True)
        self._source_kind = "file"

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_file_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("File session %s failed", self.session_id)
        finally:
            Path(self.audio_file_path).unlink(missing_ok=True)
            self._finalize_run(final_status, end_reason)

    def _iter_file_frames(self):
        with wave.open(self.audio_file_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()

            if sample_width != 2:
                raise ValueError("Only 16-bit PCM WAV files are supported for file-based testing")
            if sample_rate != self.request.sample_rate:
                raise ValueError(
                    f"Uploaded WAV sample rate {sample_rate} Hz does not match the "
                    f"session sample_rate parameter ({self.request.sample_rate} Hz). "
                    f"Either convert the file to {self.request.sample_rate} Hz 16-bit mono PCM "
                    f"(e.g. ffmpeg -i input.wav -ar {self.request.sample_rate} -ac 1 -sample_fmt s16 output.wav) "
                    f"or pass sample_rate={sample_rate} when starting the session."
                )

            while not self._stop_event.is_set():
                raw = wav_file.readframes(self._frame_samples)
                if not raw:
                    break

                frame = np.frombuffer(raw, dtype=np.int16)
                if channels > 1:
                    frame = frame.reshape(-1, channels)[:, 0]

                if len(frame) == 0:
                    continue

                yield frame.copy()

                if self.request.realtime_factor > 0:
                    time.sleep((len(frame) / self.request.sample_rate) / self.request.realtime_factor)
