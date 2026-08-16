"""Per-turn LLM timing accumulator.

Google ADK issues more than one LLM round-trip per tool-calling turn (one to
decide the tool call, one to compose the final reply).  The agent's overall
latency therefore mixes LLM time with MCP tool time and ADK overhead, which
made the UI's "LLM" figure meaningless.

This module accumulates genuine LLM time for the *current* turn.  A
``ContextVar`` is used rather than an instance attribute so that concurrent
requests handled by the same singleton agent never contaminate each other's
measurements.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Holds the in-flight turn's LLM timings:
#   ms       -> cumulative FULL round-trip time (prefill + decode)
#   ttft_ms  -> cumulative time-to-first-token (prefill only)
#   calls    -> number of round-trips
#
# Both are tracked because they answer different questions: `ttft_ms` is
# prefill-bound (prompt size, prefix-cache hit rate), while `ms - ttft_ms` is
# decode-bound (reply length, max_tokens). Optimising the wrong one wastes
# effort, so the trace exposes them separately.
_llm_stats: ContextVar[Dict[str, float] | None] = ContextVar("_llm_stats", default=None)

# Holds {"ms": float, "calls": int} of retrieval work done by tools this turn.
_retrieval_stats: ContextVar[Dict[str, float] | None] = ContextVar(
    "_retrieval_stats", default=None
)

# Holds {"ms": float, "calls": int} of MCP tool-call round-trips this turn
# (network + kiosk-core request handling, including its own SQLite time —
# rag-service never touches SQLite directly, so MCP round-trip time is the
# only vantage point this service has on it). Tracked separately from
# ``_llm_stats`` so the two can be told apart when explaining where the ~4s
# total turn latency goes (requirement: LLM vs MCP vs guard vs total).
_mcp_stats: ContextVar[Dict[str, float] | None] = ContextVar("_mcp_stats", default=None)

# Holds {"ms": float, "calls": int} of truthfulness-guard processing this turn
# (menu_guard/removal_guard/confirm_guard record_tool_result + validate_reply,
# plus the ordering_agent.py whole-reply guards). These are pure, in-process
# functions, so this number should stay small (<1ms typically) — tracked
# mainly to rule guard overhead in or out as a latency contributor rather
# than assuming it is negligible.
_guard_stats: ContextVar[Dict[str, float] | None] = ContextVar("_guard_stats", default=None)

# Holds {"ms": float, "calls": int} of deterministic response-template
# rendering this turn (agentic/reply_templates.py). Its value is not the time
# it costs — pure string formatting, microseconds — but the time it SAVES:
# a non-zero ``calls`` here means a whole Qwen inference (~2-3s) was skipped
# for this turn. Read alongside ``llm_calls`` to see the optimisation working.
_template_stats: ContextVar[Dict[str, float] | None] = ContextVar("_template_stats", default=None)

# Stores the spoken text produced by the most recent template render this turn.
# When skip_summarization=True, ADK emits no model text event, so reply_parts
# would be empty. This var lets chat() recover the template text as the reply.
_template_reply: ContextVar[str] = ContextVar("_template_reply", default="")


def reset() -> None:
    """Begin a fresh measurement window for the current turn."""
    _llm_stats.set({"ms": 0.0, "ttft_ms": 0.0, "calls": 0})
    _retrieval_stats.set({"ms": 0.0, "calls": 0})
    _mcp_stats.set({"ms": 0.0, "calls": 0})
    _guard_stats.set({"ms": 0.0, "calls": 0})
    _template_stats.set({"ms": 0.0, "calls": 0})
    _template_reply.set("")


def record(elapsed_ms: float, ttft_ms: float | None = None) -> None:
    """Add one completed LLM round-trip to the current window.

    Args:
        elapsed_ms: Full round-trip duration, i.e. until the stream is
            exhausted. This previously stopped at the first chunk, which
            silently excluded all decode time and under-reported LLM cost by
            roughly 2x on generation-heavy turns.
        ttft_ms: Time to the first streamed chunk, when known.
    """
    stats = _llm_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["ttft_ms"] += ttft_ms if ttft_ms is not None else elapsed_ms
    stats["calls"] += 1


def record_retrieval(elapsed_ms: float) -> None:
    """Add one completed knowledge-base retrieval to the current window."""
    stats = _retrieval_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def record_mcp(elapsed_ms: float) -> None:
    """Add one completed MCP tool round-trip to the current window."""
    stats = _mcp_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def record_guard(elapsed_ms: float) -> None:
    """Add one completed guard-processing span to the current window."""
    stats = _guard_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def retrieval_snapshot() -> Dict[str, Any]:
    """Return accumulated retrieval timings for the current turn.

    Returns:
        dict with ``ms`` (float or None when nothing was retrieved) and
        ``calls`` (int). ``ms`` stays None when retrieval never ran so the UI
        can distinguish "not invoked" from "took 0 ms".
    """
    stats = _retrieval_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}


def mcp_snapshot() -> Dict[str, Any]:
    """Return accumulated MCP tool round-trip timings for the current turn.

    Returns:
        dict with ``ms`` (float or None when no tool was called) and
        ``calls`` (int).
    """
    stats = _mcp_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}


def guard_snapshot() -> Dict[str, Any]:
    """Return accumulated guard-processing timings for the current turn.

    Returns:
        dict with ``ms`` (float or None when no guard ran) and ``calls``
        (int).
    """
    stats = _guard_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}


def record_template(elapsed_ms: float) -> None:
    """Add one deterministic response-template render to the current window."""
    stats = _template_stats.get()
    if stats is None:
        return
    stats["ms"] += elapsed_ms
    stats["calls"] += 1


def template_snapshot() -> Dict[str, Any]:
    """Return deterministic response-template timings for the current turn.

    Returns:
        dict with ``ms`` (float or None when no template was attempted) and
        ``calls`` (int).
    """
    stats = _template_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "calls": 0}
    return {"ms": round(stats["ms"], 1), "calls": int(stats["calls"])}


def snapshot() -> Dict[str, Any]:
    """Return the accumulated timings for the current turn.

    Returns:
        dict with ``ms`` (cumulative full LLM time), ``ttft_ms`` (cumulative
        prefill time) and ``calls``. ``ms``/``ttft_ms`` are None when no LLM
        call was recorded, so callers can distinguish "not measured" from
        "zero".
    """
    stats = _llm_stats.get()
    if stats is None or stats["calls"] == 0:
        return {"ms": None, "ttft_ms": None, "calls": 0}
    return {
        "ms": round(stats["ms"], 1),
        "ttft_ms": round(stats["ttft_ms"], 1),
        "calls": int(stats["calls"]),
    }


def set_template_reply(text: str) -> None:
    """Store the spoken text produced by a template render for this turn.

    Called inside the MCP tool callback when skip_summarization=True is set.
    Because ADK emits no model text event in that case, chat() would otherwise
    assemble an empty reply. chat() reads this back as the authoritative reply.
    """
    _template_reply.set(text)


def get_template_reply() -> str:
    """Return the template-rendered reply stored for this turn, or empty string."""
    return _template_reply.get()
