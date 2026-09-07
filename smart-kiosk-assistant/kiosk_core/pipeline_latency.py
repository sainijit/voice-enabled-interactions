"""
Turn-level AI pipeline latency tracker.

Captures one TurnTrace per completed voice turn and holds the last N in a
thread-safe ring buffer.  Exposed via GET /api/v1/pipeline/latest and
GET /api/v1/pipeline/recent on kiosk-core.

Design principles
─────────────────
* Wall-clock E2E is MEASURED (ended_at − started_at), never summed from stages,
  so the TTS-overlap with LLM generation is handled correctly.
* Spans are NESTED: retrieval and llm live under agent (matching runtime reality).
* An ``invoked`` flag on retrieval prevents stale latency leaking across turns.
* All durations use monotonic clock (time.monotonic); ISO timestamps use datetime.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RetrievalSpan:
    invoked: bool = False
    ms: float | None = None


@dataclass
class LlmSpan:
    ms: float | None = None        # total cumulative LLM time this turn
    ttft_ms: float | None = None   # cumulative prefill; ms - ttft_ms = decode
    calls: int = 0
    device: str = "GPU"


@dataclass
class AgentSpan:
    ttft_ms: float | None = None   # time-to-first-token (perceived latency)
    total_ms: float | None = None  # full agent round-trip
    retrieval: RetrievalSpan = field(default_factory=RetrievalSpan)
    llm: LlmSpan = field(default_factory=LlmSpan)


@dataclass
class AsrSpan:
    ms: float | None = None
    device: str = "CPU"
    chunks: int = 0                # number of transcribe calls summed into ms
    final_flush_skipped: bool = False  # see config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED


@dataclass
class TtsSpan:
    ms: float | None = None
    device: str = "CPU"
    segments: int = 0
    overlapped_with_agent: bool = True   # always true — TTS runs concurrently


@dataclass
class WallTimes:
    turn_total_ms: float | None = None
    time_to_first_audio_ms: float | None = None
    # How long the endpoint waited in trailing silence before committing the
    # turn (1.5s fixed, or KIOSK_CORE_ENDPOINT_SHORT_SECONDS when the
    # transcript already read as a finished sentence). The customer sits
    # through this, so any voice-to-voice figure has to include it.
    endpoint_wait_ms: float | None = None
    # Customer's last word -> first sound out of the speaker. This is the
    # "voice to voice" clock, the one a customer actually feels, and the only
    # one comparable to external voice-kiosk figures. The other timings in this
    # trace start at the endpoint decision instead, which excludes the wait.
    voice_to_voice_ms: float | None = None
    # Customer's last word -> first sound that carries the ANSWER. The opener
    # ("One moment.") is real audio and legitimately stops the silence, but it
    # is not informative, so it is reported separately rather than allowed to
    # flatter voice_to_voice_ms.
    voice_to_voice_informative_ms: float | None = None


@dataclass
class TurnTrace:
    turn_id: str
    conversation_id: str
    started_at: str       # ISO8601 UTC
    ended_at: str | None
    wall: WallTimes = field(default_factory=WallTimes)
    asr: AsrSpan = field(default_factory=AsrSpan)
    agent: AgentSpan = field(default_factory=AgentSpan)
    tts: TtsSpan = field(default_factory=TtsSpan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineLatencyStore:
    """Thread-safe ring buffer of the last ``maxlen`` TurnTrace records."""

    def __init__(self, maxlen: int = 20) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[TurnTrace] = deque(maxlen=maxlen)

    def record(self, trace: TurnTrace) -> None:
        with self._lock:
            self._buffer.append(trace)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].to_dict()

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
        return [t.to_dict() for t in items[-n:]]


# Module-level singleton — imported by audio_session and main.py
pipeline_store = PipelineLatencyStore()
