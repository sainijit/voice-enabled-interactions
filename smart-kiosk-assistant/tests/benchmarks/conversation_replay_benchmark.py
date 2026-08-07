#!/usr/bin/env python3
"""Replay a recorded kiosk conversation and profile per-service latency.

Why this exists
---------------
``agent_latency_benchmark.py`` measures a *single* utterance repeated N times.
That is the right tool for A/B-ing one optimisation, but it cannot answer
"where does the time go across a real customer conversation?" — and it hides
the effects that only appear with a growing session:

  * the agent prompt grows with conversation history, so LLM prefill grows
  * cart state accumulates, so tool results get larger
  * the reply gets longer, so TTS cost rises

This script replays every user utterance from a ``conversations/*.jsonl``
transcript, in order, through the *full* voice pipeline
(TTS-synthesised audio -> kiosk-core -> ASR -> agent -> TTS), keeping one
``conversation_id`` for the whole run so the agent accumulates real state.

Per-stage numbers come from kiosk-core's own trace endpoint
(``/api/v1/pipeline/latest``), not from wall-clock guesses, so the attribution
is the service's own instrumentation.

Usage::

    python tests/benchmarks/conversation_replay_benchmark.py \
        --conversation conversations/04b03082-....jsonl --label baseline

Stdlib only, consistent with the sibling benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_latency_benchmark import (  # noqa: E402
    CONTAINER_ANALYZER_URL,
    CONTAINER_RAG_URL,
    CONTAINER_TTS_URL,
    CORE_BASE_URL,
    collect_stack_config,
    fetch_pipeline_trace,
    http_post_multipart,
    poll_session,
    synthesize_prompt_wav,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TurnMeasurement:
    """One replayed turn, with kiosk-core's own stage attribution."""

    index: int
    prompt: str
    expected_reply: str = ""
    transcript: str = ""
    reply: str = ""

    # Harness cost — synthesising the *user's* voice. NOT kiosk latency;
    # tracked only so it can be excluded from every aggregate.
    harness_tts_ms: float | None = None

    # kiosk-core trace
    wall_total_ms: float | None = None
    time_to_first_audio_ms: float | None = None
    asr_ms: float | None = None
    asr_chunks: int | None = None
    agent_ttft_ms: float | None = None
    agent_total_ms: float | None = None
    llm_ms: float | None = None
    llm_calls: int | None = None
    llm_device: str | None = None
    retrieval_invoked: bool | None = None
    retrieval_ms: float | None = None
    tts_ms: float | None = None
    tts_segments: int | None = None
    tts_overlapped: bool | None = None

    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.wall_total_ms is not None


@dataclass
class ReplayReport:
    label: str
    conversation_file: str
    conversation_id: str
    started_at: str
    finished_at: str = ""
    stack: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnMeasurement] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------


def load_prompts(path: Path, limit: int | None) -> list[tuple[str, str]]:
    """Read ``(user, assistant)`` pairs from a conversation JSONL file."""
    pairs: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        user = (rec.get("user") or "").strip()
        if not user:
            continue
        pairs.append((user, (rec.get("assistant") or "").strip()))
    return pairs[:limit] if limit else pairs


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_turn(
    index: int,
    prompt: str,
    expected: str,
    conversation_id: str,
    tts_model: str,
    tts_language: str,
    session_timeout: float,
    max_session_seconds: float,
) -> TurnMeasurement:
    """Synthesise one utterance, push it through kiosk-core, read the trace."""
    m = TurnMeasurement(index=index, prompt=prompt, expected_reply=expected)

    try:
        wav, synth_ms = synthesize_prompt_wav(prompt, tts_model, tts_language)
        m.harness_tts_ms = round(synth_ms, 1)
    except Exception as exc:  # noqa: BLE001
        m.error = f"harness tts failed: {exc}"
        return m

    fields = {
        "sample_rate": "16000",
        "chunk_seconds": "4.0",
        "silence_timeout_seconds": "1.5",
        "max_session_seconds": str(max_session_seconds),
        "silence_threshold": "900",
        "temperature": "0.0",
        # Replay the WAV as fast as kiosk-core allows: we are measuring
        # processing latency, not the customer's speaking rate.
        "realtime_factor": "100.0",
        "analyzer_url": CONTAINER_ANALYZER_URL,
        "rag_url": CONTAINER_RAG_URL,
        "tts_url": CONTAINER_TTS_URL,
        "tts_model": tts_model,
        "tts_language": tts_language,
        # The whole point: one conversation across all turns.
        "conversation_id": conversation_id,
    }

    try:
        started = http_post_multipart(
            f"{CORE_BASE_URL}/api/v1/sessions/start-file",
            fields=fields,
            file_field="file",
            filename=f"turn{index}.wav",
            file_bytes=wav,
            timeout=300.0,
        )
        snapshot = poll_session(str(started.get("session_id")), timeout=session_timeout)
        m.transcript = snapshot.get("transcript", "") or ""
        m.reply = snapshot.get("response", "") or ""
        if snapshot.get("error"):
            m.error = str(snapshot["error"])
    except Exception as exc:  # noqa: BLE001
        m.error = str(exc)
        return m

    trace = fetch_pipeline_trace() or {}
    wall = trace.get("wall", {}) or {}
    asr = trace.get("asr", {}) or {}
    agent = trace.get("agent", {}) or {}
    llm = agent.get("llm", {}) or {}
    retrieval = agent.get("retrieval", {}) or {}
    tts = trace.get("tts", {}) or {}

    m.wall_total_ms = wall.get("turn_total_ms")
    m.time_to_first_audio_ms = wall.get("time_to_first_audio_ms")
    m.asr_ms = asr.get("ms")
    m.asr_chunks = asr.get("chunks")
    m.agent_ttft_ms = agent.get("ttft_ms")
    m.agent_total_ms = agent.get("total_ms")
    m.llm_ms = llm.get("ms")
    m.llm_calls = llm.get("calls")
    m.llm_device = llm.get("device")
    m.retrieval_invoked = retrieval.get("invoked")
    m.retrieval_ms = retrieval.get("ms")
    m.tts_ms = tts.get("ms")
    m.tts_segments = tts.get("segments")
    m.tts_overlapped = tts.get("overlapped_with_agent")
    return m


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stats(values: list[float]) -> dict[str, float] | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "p90": round(vals_sorted[max(0, int(len(vals_sorted) * 0.9) - 1)], 1),
    }


def build_summary(turns: list[TurnMeasurement]) -> dict[str, Any]:
    ok = [t for t in turns if t.ok]
    fields_of_interest = {
        "wall_total_ms": [t.wall_total_ms for t in ok],
        "time_to_first_audio_ms": [t.time_to_first_audio_ms for t in ok],
        "asr_ms": [t.asr_ms for t in ok],
        "agent_ttft_ms": [t.agent_ttft_ms for t in ok],
        "agent_total_ms": [t.agent_total_ms for t in ok],
        "llm_ms": [t.llm_ms for t in ok],
        "tts_ms": [t.tts_ms for t in ok],
        "harness_tts_ms": [t.harness_tts_ms for t in ok],
    }
    stages = {k: _stats([v for v in vs if v is not None]) for k, vs in fields_of_interest.items()}

    # Share of the turn each stage accounts for. ASR and agent are sequential;
    # TTS overlaps the agent (streamed sentence-by-sentence), so these
    # deliberately do not sum to 100% and are reported against the wall clock.
    wall_mean = (stages.get("wall_total_ms") or {}).get("mean") or 0.0
    share = {}
    if wall_mean:
        for key in ("asr_ms", "agent_total_ms", "llm_ms", "tts_ms"):
            st = stages.get(key)
            if st:
                share[key] = round(st["mean"] / wall_mean * 100, 1)

    llm_calls = [t.llm_calls for t in ok if t.llm_calls]
    return {
        "turns_total": len(turns),
        "turns_ok": len(ok),
        "turns_failed": len(turns) - len(ok),
        "stages_ms": stages,
        "share_of_wall_pct": share,
        "llm_calls_per_turn_mean": round(statistics.fmean(llm_calls), 2) if llm_calls else None,
        "retrieval_invoked_turns": sum(1 for t in ok if t.retrieval_invoked),
        "tts_overlapped_turns": sum(1 for t in ok if t.tts_overlapped),
    }


def print_report(report: ReplayReport) -> None:
    s = report.summary
    print("\n" + "=" * 78)
    print(f"CONVERSATION REPLAY — {report.label}")
    print("=" * 78)
    print(f"transcript : {report.conversation_file}")
    print(f"turns      : {s['turns_ok']} ok / {s['turns_total']} total")
    print()

    print(f"{'#':>2}  {'wall':>8} {'TTFA':>8} {'ASR':>7} {'agentTot':>9} "
          f"{'LLM':>7} {'TTS':>7}  utterance")
    print("-" * 78)
    for t in report.turns:
        if not t.ok:
            print(f"{t.index:>2}  {'FAILED':>8}  {t.error}")
            continue

        def f(v: float | None) -> str:
            return f"{v:,.0f}" if isinstance(v, (int, float)) else "-"

        print(
            f"{t.index:>2}  {f(t.wall_total_ms):>8} {f(t.time_to_first_audio_ms):>8} "
            f"{f(t.asr_ms):>7} {f(t.agent_total_ms):>9} {f(t.llm_ms):>7} "
            f"{f(t.tts_ms):>7}  {t.prompt[:28]}"
        )

    print("\n" + "-" * 78)
    print("STAGE STATISTICS (ms)")
    print("-" * 78)
    print(f"{'stage':<26}{'mean':>9}{'median':>9}{'p90':>9}{'min':>9}{'max':>9}")
    for key, st in (s.get("stages_ms") or {}).items():
        if not st:
            continue
        print(f"{key:<26}{st['mean']:>9,.0f}{st['median']:>9,.0f}"
              f"{st['p90']:>9,.0f}{st['min']:>9,.0f}{st['max']:>9,.0f}")

    print("\nSHARE OF WALL CLOCK (stages overlap; will not sum to 100%)")
    for key, pct in (s.get("share_of_wall_pct") or {}).items():
        bar = "█" * int(pct / 2)
        print(f"  {key:<20}{pct:>6.1f}%  {bar}")

    print(f"\nLLM calls/turn (mean) : {s.get('llm_calls_per_turn_mean')}")
    print(f"Retrieval invoked     : {s.get('retrieval_invoked_turns')} turns")
    print(f"TTS overlapped agent  : {s.get('tts_overlapped_turns')} turns")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conversation", type=Path, required=True)
    p.add_argument("--label", default="replay")
    p.add_argument("--limit", type=int, default=None, help="Replay only the first N turns")
    p.add_argument("--tts-model", default="qwen-tts")
    p.add_argument("--tts-language", default="English")
    p.add_argument("--session-timeout", type=float, default=180.0)
    p.add_argument("--max-session-seconds", type=float, default=25.0)
    p.add_argument("--settle-seconds", type=float, default=1.0,
                   help="Pause between turns, mimicking a customer thinking")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    prompts = load_prompts(args.conversation, args.limit)
    if not prompts:
        print(f"No user utterances found in {args.conversation}", file=sys.stderr)
        return 2

    conversation_id = f"replay-{uuid.uuid4().hex[:8]}"
    report = ReplayReport(
        label=args.label,
        conversation_file=str(args.conversation),
        conversation_id=conversation_id,
        started_at=datetime.now(UTC).isoformat(),
        stack=collect_stack_config(),
    )

    print(f"Replaying {len(prompts)} turns from {args.conversation.name}")
    print(f"conversation_id={conversation_id}\n")

    for i, (prompt, expected) in enumerate(prompts, start=1):
        print(f"[{i}/{len(prompts)}] {prompt[:70]!r}")
        m = replay_turn(
            index=i,
            prompt=prompt,
            expected=expected,
            conversation_id=conversation_id,
            tts_model=args.tts_model,
            tts_language=args.tts_language,
            session_timeout=args.session_timeout,
            max_session_seconds=args.max_session_seconds,
        )
        report.turns.append(m)
        if m.ok:
            print(f"        wall={m.wall_total_ms:,.0f}ms  asr={m.asr_ms}  "
                  f"agent={m.agent_total_ms}  tts={m.tts_ms}")
            print(f"        heard: {m.transcript[:70]!r}")
        else:
            print(f"        FAILED: {m.error}")
        time.sleep(args.settle_seconds)

    report.finished_at = datetime.now(UTC).isoformat()
    report.summary = build_summary(report.turns)
    print_report(report)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"replay-{args.label}.json"
    out.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
