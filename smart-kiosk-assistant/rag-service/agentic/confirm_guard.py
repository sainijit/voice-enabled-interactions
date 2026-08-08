"""Guard against the agent claiming an order was confirmed when it was not.

Background
----------
``_strip_false_confirmation`` in ``ordering_agent.py`` already catches the
case where the reply claims a confirmation and **no** confirm tool ran at
all. It does not catch a narrower, equally damaging case: the model calls
``confirm_order`` with a **hallucinated** ``order_id`` (observed live: the
model repeatedly guessed the literal placeholder-shaped integer ``12345``
instead of using ``confirm_active_order``, which needs no id and resolves the
customer's real open draft). ``confirm_order(12345)`` returns
``{"error": "Order not found: 12345"}`` — nothing is confirmed — yet the
model's second, narrating call still said "Your order is confirmed... Thank
you!" because *a* confirm tool name appeared in ``tool_calls`` this turn, and
the existing check is invocation-only.

This is the exact "invocation is not success" gap that ``menu_guard`` and
``removal_guard`` already close for additions and removals, just not yet for
confirmation — the single most damaging claim this kiosk can make. This
module closes it the same way: pure functions over recorded tool *results*,
turn-scoped via a ``contextvars.ContextVar`` (the agent is a process-wide
singleton across concurrent sessions), no I/O, unit-testable like its
siblings.
"""

from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass
from typing import Any

from agentic import action_result

logger = logging.getLogger(__name__)

# Only these tools' results legitimise a "your order is confirmed" claim.
# Sourced from action_result.CLAIM_TOOLS — see that module for why this must
# not be a locally redefined set.
_CONFIRM_TOOLS = action_result.CLAIM_TOOLS[action_result.ORDER_CONFIRMED]


@dataclass
class _TurnState:
    """Outcome of the confirm-tool calls observed during one turn.

    Attributes:
        attempted: True once a confirm tool was invoked at all this turn.
        succeeded: True once a confirm tool call actually returned a
            confirmed order (``status == "confirmed"`` and a real
            ``order_id``).
        error_message: The tool's own error string, if the last confirm
            attempt failed — used to compose an honest refusal.
    """

    attempted: bool = False
    succeeded: bool = False
    error_message: str = ""


_turn_state: contextvars.ContextVar[_TurnState] = contextvars.ContextVar(
    "confirm_guard_turn_state", default=_TurnState()
)


def begin_turn() -> _TurnState:
    """Reset per-turn tool-outcome tracking.

    Must be called at the start of every ``OrderingAgent.chat()`` turn, next
    to ``menu_guard.begin_turn()`` / ``removal_guard.begin_turn()``.

    Returns:
        The fresh state object, mainly so tests can inspect it.
    """
    state = _TurnState()
    _turn_state.set(state)
    return state


def current_state() -> _TurnState:
    """Return the confirm-tool-outcome state for the turn running in this context."""
    return _turn_state.get()


def record_tool_result(tool_name: str, raw: Any) -> None:
    """Classify one confirm-tool result as success or failure.

    Args:
        tool_name: Name of the MCP tool that was just invoked.
        raw: The raw value returned by ``mcp_client.call_tool``.
    """
    if tool_name not in _CONFIRM_TOOLS:
        return

    state = _turn_state.get()
    state.attempted = True
    result = action_result.classify(tool_name, raw)
    if result.code == "UNDECODABLE":
        logger.warning(
            "[CONFIRM-GUARD] Could not decode result of %s — treating as unsuccessful",
            tool_name,
        )
        return

    if not result.success:
        state.error_message = result.message
        logger.warning(
            "[CONFIRM-GUARD] %s failed | order_id=%s error=%s",
            tool_name, result.data.get("order_id"), result.message,
        )
        return

    payload = result.data
    if payload.get("status") == "confirmed" and payload.get("order_id") is not None:
        state.succeeded = True
    else:
        logger.warning(
            "[CONFIRM-GUARD] %s returned an unexpected shape — treating as "
            "unsuccessful | payload=%s", tool_name, str(payload)[:200],
        )


_REFUSAL_GENERIC = (
    "Sorry, I couldn't confirm your order just now. Please say \"confirm my order\" "
    "once more, or ask a member of staff."
)


def build_refusal(state: _TurnState | None = None) -> str:
    """Compose the spoken refusal for a confirmation that never happened.

    Args:
        state: Turn state to read. Defaults to the current context's state.

    Returns:
        A short, markup-free sentence — never repeats the tool's raw error
        text, which is worded for the model, not the customer.
    """
    return _REFUSAL_GENERIC
