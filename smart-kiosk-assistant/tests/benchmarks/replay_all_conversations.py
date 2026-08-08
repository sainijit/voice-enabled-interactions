#!/usr/bin/env python3
"""Replay every recorded conversation through the live agent (text-only, fast).

Unlike ``conversation_replay_benchmark.py`` (which drives the *full* voice
pipeline — TTS synthesis, ASR, kiosk-core session polling — to profile
latency), this script hits ``rag-service``'s ``/api/v1/agent/chat`` directly
with each transcript's recorded user utterances, in order, on one fresh
session/user per conversation file. It is a correctness sweep, not a latency
one: it flags turns that error out, come back empty, or trip one of the
truthfulness guards (menu_guard / removal_guard / order_claim_guard), since a
guard firing means the model attempted an unsupported claim even though it
was caught.

Usage::

    python tests/benchmarks/replay_all_conversations.py
    python tests/benchmarks/replay_all_conversations.py --pattern 'conversations/2*.jsonl'
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_latency_benchmark import http_post_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERSATIONS_DIR = REPO_ROOT / "conversations"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
AGENT_CHAT_URL = "http://localhost:8020/api/v1/agent/chat"

# Guard-replacement stock phrases (see order_claim_guard.py, removal_guard.py,
# menu_guard.py). A reply matching one of these means the model tried to claim
# an order action the tool results didn't support and a guard rewrote it —
# functionally "safe" (the customer wasn't lied to) but worth flagging
# separately from a hard failure, since ideally the model never needs
# correcting in the first place.
_GUARD_CORRECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"wasn't able to add that to your order", re.IGNORECASE),
    re.compile(r"wasn't able to remove that from your order", re.IGNORECASE),
    re.compile(r"couldn't confirm your order just now", re.IGNORECASE),
    re.compile(r"could not complete that just now", re.IGNORECASE),
    re.compile(r"wasn't able to remove that", re.IGNORECASE),
    re.compile(r"you don't have an open order to cancel", re.IGNORECASE),
)

# Hard-failure signatures: the customer-visible "something broke" message, or
# an empty/missing reply.
_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sorry,?\s+i\s+encountered\s+an\s+error", re.IGNORECASE),
    re.compile(r"please\s+try\s+again", re.IGNORECASE),
)


@dataclass
class TurnResult:
    index: int
    user: str
    expected_assistant: str
    reply: str = ""
    tool_calls: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None
    guard_corrected: bool = False
    hard_failed: bool = False


@dataclass
class ConversationResult:
    file: str
    conversation_id: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(not t.error and not t.hard_failed for t in self.turns)

    @property
    def guard_corrections(self) -> int:
        return sum(1 for t in self.turns if t.guard_corrected)


def load_turns(path: Path) -> list[tuple[str, str]]:
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
    return pairs


def replay_conversation(path: Path, settle_seconds: float) -> ConversationResult:
    conversation_id = f"replay-{uuid.uuid4().hex[:8]}"
    # A dedicated user_id per replay keeps carts isolated across conversation
    # files — sharing "anonymous" would let one file's leftover cart bleed
    # into the next file's totals.
    user_id = f"replay-user-{uuid.uuid4().hex[:8]}"
    result = ConversationResult(file=str(path.relative_to(REPO_ROOT)), conversation_id=conversation_id)

    for i, (user, expected) in enumerate(load_turns(path), start=1):
        turn = TurnResult(index=i, user=user, expected_assistant=expected)
        payload = {
            "transcription": user,
            "user_id": user_id,
            "session_id": conversation_id,
            "history": [],
        }
        t0 = time.perf_counter()
        try:
            data = http_post_json(AGENT_CHAT_URL, payload, timeout=120.0)
            turn.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
            turn.reply = data.get("reply", "") or ""
            turn.tool_calls = list(data.get("tool_calls", []) or [])
        except Exception as exc:  # noqa: BLE001
            turn.error = str(exc)
            turn.hard_failed = True
            result.turns.append(turn)
            continue

        if not turn.reply:
            turn.hard_failed = True
            turn.error = "empty reply"
        elif any(p.search(turn.reply) for p in _ERROR_PATTERNS):
            turn.hard_failed = True
            turn.error = "error-signature reply"
        if any(p.search(turn.reply) for p in _GUARD_CORRECTION_PATTERNS):
            turn.guard_corrected = True

        result.turns.append(turn)
        time.sleep(settle_seconds)

    return result


def print_report(results: list[ConversationResult]) -> None:
    print("\n" + "=" * 90)
    print("CONVERSATION REPLAY — CORRECTNESS SWEEP")
    print("=" * 90)
    total_turns = sum(len(r.turns) for r in results)
    total_failed_turns = sum(sum(1 for t in r.turns if t.hard_failed) for r in results)
    total_guard = sum(r.guard_corrections for r in results)
    passed_files = [r for r in results if r.passed]
    failed_files = [r for r in results if not r.passed]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n[{status}] {r.file}  ({len(r.turns)} turns, "
              f"{r.guard_corrections} guard-corrected)")
        for t in r.turns:
            if t.hard_failed:
                print(f"    ✗ turn {t.index}: {t.user[:60]!r}")
                print(f"        error: {t.error}")
                if t.reply:
                    print(f"        reply: {t.reply[:120]!r}")
            elif t.guard_corrected:
                print(f"    ⚠ turn {t.index}: {t.user[:60]!r}")
                print(f"        guard-corrected reply: {t.reply[:120]!r}")

    print("\n" + "-" * 90)
    print(f"Conversations : {len(passed_files)}/{len(results)} passed")
    print(f"Turns         : {total_turns - total_failed_turns}/{total_turns} passed "
          f"({total_guard} guard-corrected)")
    if failed_files:
        print("\nFailed conversations:")
        for r in failed_files:
            print(f"  - {r.file}")
    print("=" * 90)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="*.jsonl", help="Glob pattern within conversations/")
    p.add_argument("--settle-seconds", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    files = sorted(CONVERSATIONS_DIR.glob(args.pattern))
    if not files:
        print(f"No conversation files matched {args.pattern!r} in {CONVERSATIONS_DIR}", file=sys.stderr)
        return 2

    print(f"Replaying {len(files)} conversation file(s) from {CONVERSATIONS_DIR}")
    results: list[ConversationResult] = []
    for i, path in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {path.name} ...")
        results.append(replay_conversation(path, args.settle_seconds))

    print_report(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / "replay-all-conversations.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
