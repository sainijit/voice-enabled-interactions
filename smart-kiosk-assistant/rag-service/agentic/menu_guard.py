"""Guard against the agent adding items that are not on the menu.

Background
----------
``kiosk-core``'s MCP layer already refuses off-menu items correctly: when
``_resolve_items`` cannot resolve a spoken reference to a catalogue product it
returns ``{"error": ..., "available_products": [...]}`` and writes **nothing**
to the database. The order is untouched.

The model, however, still narrates success. A turn where the customer asks for
a sushi platter produces ``place_order`` → rejection → and a reply of
"I've added the Sushi Platter to your order." The customer walks away believing
they ordered food that the restaurant does not sell.

Every pre-existing guard misses this, because they all reason about which tools
were **invoked**:

* ``chat()``'s ``_ORDER_CLAIM_FALLBACK`` — ``place_order`` *was* invoked.
* ``_strip_false_confirmation`` — the reply claims an addition, not a
  confirmation.
* ``kiosk_core.ordering.order_claim_guard`` — ``place_order`` is in its
  ``_MUTATING_TOOLS`` set, so the claim is considered supported.

Invocation is not success. This module closes that gap by reasoning about tool
**results**: a reply may only claim an addition when a mutating tool actually
succeeded during the turn.

Design
------
The guard is deterministic and model-independent, matching the stance of the
other guards in this service: prompt wording alone was already tried for this
exact failure and ignored, because a language model reproduces the shape of a
successful reply regardless of what happened.

Turn state lives in a :class:`contextvars.ContextVar` rather than on
``OrderingAgent``: the agent is a process-wide singleton serving concurrent
sessions, and per-instance state would let one customer's rejected item rewrite
another customer's reply. A ContextVar is naturally scoped to the asyncio task
running the turn.

The module performs no I/O and makes no LLM round-trip, so it cannot itself
hallucinate and can be unit-tested as pure functions.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Tools whose success is what legitimises a "that's in your order now" claim.
# Kept in sync with ``ordering_agent._MUTATING_TOOLS`` — adding a new
# cart-mutating MCP tool means adding it in both places, or this guard silently
# stops covering it.
_MUTATING_TOOLS = frozenset({"place_order", "update_order"})

# Claims that an item was put into the cart. Deliberately broader than
# ``order_claim_guard._ADDED_PATTERNS``: that guard only has to catch replies
# where *no* tool ran, whereas here the model has seen a tool result and phrases
# the claim more confidently and more variously ("that's been added",
# "I've put a sushi platter in your order").
_ADDED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bI(?:'ve| have)\s+added\b", re.IGNORECASE),
    re.compile(r"\bI(?:'ve| have)\s+put\b.{0,60}?\b(?:in|into|on)\s+your\s+order\b", re.IGNORECASE),
    re.compile(r"\b(?:has|have|is|are)\s+been\s+added\b", re.IGNORECASE),
    re.compile(r"\badded\s+(?:the|a|an|\d+)\b.{0,60}?\bto\s+your\s+(?:order|cart)\b", re.IGNORECASE),
    re.compile(r"\b(?:that(?:'s| is)|it(?:'s| is))\s+(?:now\s+)?(?:been\s+)?added\b", re.IGNORECASE),
    re.compile(r"\badded\s+to\s+your\s+(?:order|cart)\b", re.IGNORECASE),
    re.compile(r"\byour\s+(?:order|cart)\s+now\s+(?:contains|has|includes)\b", re.IGNORECASE),
)

# Spoken at the kiosk, so it must be short and contain no markup. The wording
# states the fact (not on the menu), then hands the customer a concrete next
# step — a bare refusal leaves a voice customer with nowhere to go, since they
# cannot see the menu.
_REFUSAL_WITH_ALTERNATIVES = (
    "Sorry, we don't have {item} on the menu at the moment. "
    "Please choose from our menu — we do have {alternatives}. "
    "Would you like one of those?"
)

_REFUSAL_PLAIN = (
    "Sorry, we don't have {item} on the menu at the moment. "
    "Please choose an item from our menu — what would you like instead?"
)

_REFUSAL_GENERIC = (
    "Sorry, that isn't on our menu at the moment. "
    "Please choose an item from our menu — what would you like instead?"
)

# How many alternatives to speak. Three is the practical ceiling for a voice
# reply; beyond that the customer stops retaining them.
_MAX_ALTERNATIVES = 3


@dataclass
class _TurnState:
    """Outcomes of the mutating tool calls observed during one agent turn.

    Attributes:
        succeeded: True once any mutating tool returned a real order payload,
            which makes an "added" claim truthful.
        rejected_refs: Product references kiosk-core refused because they are
            not in the catalogue, in the order they were rejected.
        alternatives: Grounded ``{name, price}`` rows taken from the tool's
            ``available_products``. Only these may be offered back to the
            customer — anything else would be invented.
    """

    succeeded: bool = False
    rejected_refs: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_rejection(self) -> bool:
        """True when at least one item was refused as off-menu this turn."""
        return bool(self.rejected_refs)


_turn_state: contextvars.ContextVar[_TurnState] = contextvars.ContextVar(
    "menu_guard_turn_state", default=_TurnState()
)

# Pulls the offending reference out of the kiosk-core error sentence, which is
# formatted as: "'sushi platter' is not on the menu. ...". Parsing the sentence
# is acceptable coupling because the same module that formats it is the only
# producer, and the fallback below degrades to a generic refusal rather than
# failing.
_REJECTED_REF_RE = re.compile(r"^'([^']+)'\s+is not on the menu", re.IGNORECASE)


def begin_turn() -> _TurnState:
    """Reset per-turn tool-outcome tracking.

    Must be called at the start of every ``OrderingAgent.chat()`` turn.
    Without a reset, a rejection from an earlier turn would suppress a
    legitimate addition later in the same conversation.

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

    ``mcp_client.call_tool`` returns ``{"status": "success", "result": "<json>"}``
    on success and ``{"error": "..."}`` on a transport failure. The tool's real
    payload — including an off-menu rejection — is the JSON *inside* ``result``.

    Args:
        raw: The value returned by ``call_tool``.

    Returns:
        The decoded tool payload, or None when there is nothing decodable.
        A transport-level error is returned as-is so the caller can treat it
        as "not a success" without misreading it as an off-menu rejection.
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
    """Classify one mutating tool result as success or off-menu rejection.

    Non-mutating tools are ignored: ``list_products`` returning nothing is a
    catalogue question, not a failed addition, and conflating the two would
    make the kiosk refuse items it actually sells.

    Args:
        tool_name: Name of the MCP tool that was just invoked.
        raw: The raw value returned by ``mcp_client.call_tool``.
    """
    if tool_name not in _MUTATING_TOOLS:
        return

    state = _turn_state.get()
    payload = _unwrap(raw)
    if payload is None:
        # Undecodable payload: treat as not-a-success. The turn then falls back
        # to the existing name-based guards, which is the prior behaviour.
        logger.warning(
            "[MENU-GUARD] Could not decode result of %s — treating as unsuccessful",
            tool_name,
        )
        return

    error = payload.get("error")
    if not error:
        state.succeeded = True
        return

    ref_match = _REJECTED_REF_RE.match(str(error))
    if ref_match:
        state.rejected_refs.append(ref_match.group(1))
    else:
        # A non-catalogue failure (e.g. a database error). Record it without a
        # reference so the reply is still corrected, but with generic wording —
        # naming an item we did not parse would be a guess.
        state.rejected_refs.append("")

    for product in payload.get("available_products") or []:
        if not isinstance(product, dict):
            continue
        name = product.get("name")
        if not name or any(a.get("name") == name for a in state.alternatives):
            continue
        state.alternatives.append({"name": name, "price": product.get("price")})

    logger.warning(
        "[MENU-GUARD] Off-menu item refused by %s | ref=%r alternatives=%s",
        tool_name,
        state.rejected_refs[-1],
        [a["name"] for a in state.alternatives],
    )


def claims_addition(text: str) -> bool:
    """Return True when ``text`` tells the customer an item is in their order."""
    return bool(text) and any(pattern.search(text) for pattern in _ADDED_PATTERNS)


def _format_price(price: Any) -> str:
    """Render a price for speech, dropping a meaningless trailing ``.0``."""
    try:
        value = float(price)
    except (TypeError, ValueError):
        return ""
    return str(int(value)) if value.is_integer() else f"{value:.2f}"


def _format_alternatives(alternatives: list[dict[str, Any]]) -> str:
    """Render up to three grounded alternatives as a spoken list."""
    parts: list[str] = []
    for product in alternatives[:_MAX_ALTERNATIVES]:
        price = _format_price(product.get("price"))
        parts.append(f"{product['name']} at {price} rupees" if price else str(product["name"]))
    if len(parts) <= 1:
        return "".join(parts)
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def build_refusal(state: _TurnState | None = None) -> str:
    """Compose the spoken refusal for an off-menu request.

    Args:
        state: Turn state to read. Defaults to the current context's state.

    Returns:
        A short, markup-free sentence naming only real catalogue items.
    """
    state = state if state is not None else _turn_state.get()
    item = next((ref for ref in state.rejected_refs if ref), "")
    if not item:
        return _REFUSAL_GENERIC

    alternatives = _format_alternatives(state.alternatives)
    template = _REFUSAL_WITH_ALTERNATIVES if alternatives else _REFUSAL_PLAIN
    return template.format(item=item, alternatives=alternatives)


def _mentions_alternative(reply: str, state: _TurnState) -> bool:
    """Return True when ``reply`` names at least one grounded alternative.

    Used to catch the case where the model neither claims a false success
    nor invents anything, but simply drops the real ``available_products``
    the tool handed it — e.g. replying "we currently don't have any items in
    your order" to an ambiguous "burger" instead of naming the five real
    burgers on the menu. That reply is not false, but it strands a voice
    customer who cannot see a screen of alternatives, so it must still be
    replaced by the guard.
    """
    lowered = reply.lower()
    return any(
        isinstance(product.get("name"), str) and product["name"].lower() in lowered
        for product in state.alternatives
    )


def validate_reply(reply: str, state: _TurnState | None = None) -> tuple[str, bool]:
    """Reconcile a reply against a real off-menu/ambiguous-item rejection.

    Two failure shapes are corrected, both stemming from the same root cause
    (the model narrating a turn it did not fully understand):

    1. A false "added"/"confirmed" claim after every mutating call was
       refused as off-menu — the surrounding text (item name, price, total,
       upsell) was composed around that false premise and is not salvageable.
    2. A reply that makes no false claim but also drops the real
       ``available_products`` alternatives the tool provided, leaving a
       voice customer with no path forward (see ``_mentions_alternative``).

    The whole reply is replaced rather than edited sentence-by-sentence in
    both cases, for the same reason: the text was built around a premise
    (success, or "nothing to offer") that a grounded refusal fully replaces.

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
    if state.succeeded or not state.has_rejection:
        return reply, False

    added_claim = claims_addition(reply)
    if not added_claim and _mentions_alternative(reply, state):
        return reply, False

    refusal = build_refusal(state)
    if added_claim:
        logger.error(
            "[MENU-GUARD] Reply claimed an item was added but every mutating tool call "
            "was refused as off-menu (refs=%s). Replacing reply: %r",
            state.rejected_refs,
            reply[:160],
        )
    else:
        logger.warning(
            "[MENU-GUARD] Reply omitted the real menu alternatives after an off-menu/"
            "ambiguous rejection (refs=%s). Replacing reply: %r",
            state.rejected_refs,
            reply[:160],
        )
    return refusal, True
