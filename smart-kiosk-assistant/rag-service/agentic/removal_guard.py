"""Guard against the agent claiming a cart item was removed when it was not.

Background
----------
``kiosk-core``'s ``remove_from_order`` MCP tool is careful about what it
reports: a reference that does not match anything in the cart comes back as
``{"error": ..., "cart_items": [...]}`` and nothing is deleted, and a partial
match reports the untouched items in ``not_in_cart`` alongside whatever really
was removed.

Nothing previously reconciled the spoken reply against that result. Every
pre-existing guard reasons about **confirm** and **add** claims:

* ``order_claim_guard`` / ``chat()``'s ``_ORDER_CLAIM_FALLBACK`` only match
  confirmation phrasing ("your order is confirmed", "order id ...").
* ``_strip_false_confirmation`` only strips confirmation claims.
* ``menu_guard`` only tracks ``place_order`` / ``update_order`` results.

So a turn where the model narrates "I've removed the Pepsi from your order"
while ``remove_from_order`` was never called (or was called and returned
``not_in_cart``/an error) sails through untouched: the customer is told the
item is gone, the cart still has it, and every existing check passes because
none of them look at removal at all. This is the exact "invocation is not
success" gap that ``menu_guard`` closes for additions — this module closes it
for removals.

Design mirrors ``menu_guard.py``: turn state lives in a
:class:`contextvars.ContextVar` (the agent is a process-wide singleton across
concurrent sessions), the guard performs no I/O, and it is pure enough to unit
test as plain functions over text and recorded tool results.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Only this tool's result legitimises a "that's off your order now" claim.
_REMOVAL_TOOLS = frozenset({"remove_from_order"})

# Claims that an item was taken out of the cart. Kept broad, in the same
# spirit as ``menu_guard._ADDED_PATTERNS``: the model has seen a tool result by
# this point and phrases success confidently and variously.
_REMOVED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI(?:'ve| have)\s+removed\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ve| have)\s+taken\b.{0,60}?\boff\b", re.IGNORECASE),
    re.compile(r"\b(?:has|have)\s+been\s+removed\b", re.IGNORECASE),
    re.compile(r"\bremoved\s+(?:the|a|an|\d+)\b.{0,60}?\bfrom\s+your\s+(?:order|cart)\b", re.IGNORECASE),
    re.compile(r"\btaken\s+(?:the|a|an|\d+)\b.{0,60}?\boff\s+your\s+(?:order|cart)\b", re.IGNORECASE),
    re.compile(r"\b(?:that(?:'s| is)|it(?:'s| is))\s+(?:now\s+)?(?:been\s+)?removed\b", re.IGNORECASE),
    re.compile(r"\bno\s+longer\s+in\s+your\s+(?:order|cart)\b", re.IGNORECASE),
)

_REFUSAL_WITH_CART = (
    "Sorry, I couldn't find {item} in your order. "
    "Your order currently has {cart_items}. Which of those would you like removed?"
)

_REFUSAL_GENERIC = (
    "Sorry, I wasn't able to remove that — it doesn't look like it's in your "
    "order. What would you like me to take off?"
)

_MAX_CART_ITEMS = 3


@dataclass
class _TurnState:
    """Outcome of the ``remove_from_order`` calls observed during one turn.

    Attributes:
        succeeded: True once a call actually removed at least one line.
        attempted: True once ``remove_from_order`` was invoked at all, whether
            or not it removed anything. Used to tell "never even tried" apart
            from "tried and failed", which get slightly different wording.
        rejected_refs: References the tool could not match to a cart line.
        cart_items: Names of what is actually left in (or was in) the cart,
            for a grounded "here's what you have" refusal.
    """

    succeeded: bool = False
    attempted: bool = False
    rejected_refs: list[str] = field(default_factory=list)
    cart_items: list[str] = field(default_factory=list)

    @property
    def has_rejection(self) -> bool:
        """True when at least one requested item could not be removed."""
        return bool(self.rejected_refs)


_turn_state: contextvars.ContextVar[_TurnState] = contextvars.ContextVar(
    "removal_guard_turn_state", default=_TurnState()
)


def begin_turn() -> _TurnState:
    """Reset per-turn tool-outcome tracking.

    Must be called at the start of every ``OrderingAgent.chat()`` turn, next
    to ``menu_guard.begin_turn()``.

    Returns:
        The fresh state object, mainly so tests can inspect it.
    """
    state = _TurnState()
    _turn_state.set(state)
    return state


def current_state() -> _TurnState:
    """Return the tool-outcome state for the turn running in this context."""
    return _turn_state.get()


def _unwrap(raw: Any) -> dict[str, Any] | None:
    """Extract the tool's own JSON payload from an MCP response envelope.

    Mirrors ``menu_guard._unwrap`` — see there for the envelope shape.
    """
    if not isinstance(raw, dict):
        return None
    if "error" in raw and "result" not in raw:
        return raw

    result = raw.get("result")
    if isinstance(result, dict):
        return result
    if not isinstance(result, str) or not result:
        return None
    try:
        decoded = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def record_tool_result(tool_name: str, raw: Any) -> None:
    """Classify one ``remove_from_order`` result as success or rejection.

    Args:
        tool_name: Name of the MCP tool that was just invoked.
        raw: The raw value returned by ``mcp_client.call_tool``.
    """
    if tool_name not in _REMOVAL_TOOLS:
        return

    state = _turn_state.get()
    state.attempted = True
    payload = _unwrap(raw)
    if payload is None:
        logger.warning(
            "[REMOVAL-GUARD] Could not decode result of %s — treating as unsuccessful",
            tool_name,
        )
        return

    error = payload.get("error")
    if error:
        # Nothing in the cart matched — kiosk-core lists what is actually
        # there so the refusal can point the customer at real cart lines.
        for entry in payload.get("cart_items") or []:
            if isinstance(entry, dict) and entry.get("name"):
                state.cart_items.append(entry["name"])
        state.rejected_refs.append("")
        logger.warning(
            "[REMOVAL-GUARD] remove_from_order refused — nothing matched the cart "
            "(cart=%s)", state.cart_items,
        )
        return

    removed = payload.get("removed") or []
    not_in_cart = payload.get("not_in_cart") or []
    if removed:
        state.succeeded = True
    state.rejected_refs.extend(str(ref) for ref in not_in_cart if ref)
    for item in payload.get("items") or []:
        if isinstance(item, dict) and item.get("product_name"):
            state.cart_items.append(item["product_name"])

    if not_in_cart:
        logger.warning(
            "[REMOVAL-GUARD] remove_from_order partial miss | removed=%s not_in_cart=%s",
            removed, not_in_cart,
        )


def claims_removal(text: str) -> bool:
    """Return True when ``text`` tells the customer an item left their cart."""
    return bool(text) and any(pattern.search(text) for pattern in _REMOVED_PATTERNS)


def _format_cart(cart_items: list[str]) -> str:
    """Render up to three real cart line names as a spoken list."""
    names = cart_items[:_MAX_CART_ITEMS]
    if not names:
        return "no items"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def build_refusal(state: _TurnState | None = None) -> str:
    """Compose the spoken refusal for a removal that never happened.

    Args:
        state: Turn state to read. Defaults to the current context's state.

    Returns:
        A short, markup-free sentence naming only real cart items.
    """
    state = state if state is not None else _turn_state.get()
    item = next((ref for ref in state.rejected_refs if ref), "")
    if not item or not state.cart_items:
        return _REFUSAL_GENERIC
    return _REFUSAL_WITH_CART.format(item=item, cart_items=_format_cart(state.cart_items))


def validate_reply(reply: str, state: _TurnState | None = None) -> tuple[str, bool]:
    """Reconcile a "removed from your order" claim against real tool outcomes.

    The whole reply is replaced, matching ``menu_guard.validate_reply``: once
    the removal never happened, the item name, new total, and any surrounding
    narration were composed around a false premise and are not salvageable.

    Args:
        reply: The assistant's drafted reply, after the other text guards.
        state: Turn state to read. Defaults to the current context's state.

    Returns:
        ``(reply, corrected)`` — the reply to speak, and whether it was
        rewritten so the caller can log the event.
    """
    if not reply:
        return reply, False

    state = state if state is not None else _turn_state.get()
    if state.succeeded or not claims_removal(reply):
        return reply, False

    refusal = build_refusal(state)
    logger.error(
        "[REMOVAL-GUARD] Reply claimed an item was removed but remove_from_order "
        "never succeeded this turn (attempted=%s rejected=%s). Replacing reply: %r",
        state.attempted, state.rejected_refs, reply[:160],
    )
    return refusal, True
