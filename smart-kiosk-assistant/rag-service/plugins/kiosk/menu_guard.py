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
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agentic import action_result
from agentic import domain_config

logger = logging.getLogger(__name__)

# Tools whose success is what legitimises a "that's in your order now" claim.
# Sourced from action_result.CLAIM_TOOLS — the single registry also consumed
# by ordering_agent._SentenceGate, so a new cart-mutating tool only needs to
# be added in one place instead of silently going unguarded here.
_MUTATING_TOOLS = action_result.CLAIM_TOOLS[action_result.ITEM_ADDED]

# Claims that an item was put into the cart. Deliberately broader than
# ``order_claim_guard._ADDED_PATTERNS``: that guard only has to catch replies
# where *no* tool ran, whereas here the model has seen a tool result and phrases
# the claim more confidently and more variously ("that's been added",
# "I've put a sushi platter in your order").
_ADDED_PATTERNS_DEFAULT: tuple[re.Pattern[str], ...] = (
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
_REFUSAL_WITH_ALTERNATIVES_DEFAULT = (
    "Sorry, we don't have {item} on the menu at the moment. "
    "Please choose from our menu — we do have {alternatives}. "
    "Would you like one of those?"
)

_REFUSAL_PLAIN_DEFAULT = (
    "Sorry, we don't have {item} on the menu at the moment. "
    "Please choose an item from our menu — what would you like instead?"
)

_REFUSAL_GENERIC_DEFAULT = (
    "Sorry, that isn't on our menu at the moment. "
    "Please choose an item from our menu — what would you like instead?"
)

# Spoken when kiosk-core refused the quantity rather than the product. The
# item is deliberately not named: the number was misheard, so the reference
# attached to it is not trustworthy enough to repeat back as fact.
_REFUSAL_QUANTITY_DEFAULT = (
    "Sorry, I didn't catch how many you'd like, so I haven't added anything "
    "yet. How many would you like?"
)

# How many alternatives to speak. Three is the practical ceiling for a voice
# reply; beyond that the customer stops retaining them.
_MAX_ALTERNATIVES = 3

# The rejected reference is whatever string the model passed as a tool
# argument — untrusted text, not a fixed vocabulary. It is usually a clean
# dish name ("sushi platter"), but the model can also pass a garbled fragment
# of the whole request (observed live: asking to add "all the burgers to my
# cart" produced the tool argument "ll the burgers to my cart", which then
# spoke back as "we don't have ll the burgers to my cart on the menu" —
# nonsensical and undermines the customer's confidence in every other guard).
# A real dish name is short and never contains a cart/order verb, so anything
# outside that shape is not echoed verbatim; the refusal falls back to the
# generic, item-less wording instead.
_NON_ITEM_MARKER_RE = re.compile(r"\b(?:cart|order|checkout|basket|please)\b", re.IGNORECASE)
_MAX_ITEM_WORDS = 5

_added_patterns: tuple[re.Pattern[str], ...] | None = None


def _get_refusal(key: str, default: str) -> str:
    return domain_config.get_guard_rule("menu_guard", key, default) or default


def _build_added_patterns() -> tuple[re.Pattern[str], ...]:
    patterns = domain_config.get_guard_patterns("menu_guard")
    if patterns:
        return tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    return _ADDED_PATTERNS_DEFAULT


def _get_added_patterns() -> tuple[re.Pattern[str], ...]:
    global _added_patterns
    if _added_patterns is None:
        _added_patterns = _build_added_patterns()
    return _added_patterns


def _looks_like_item_name(ref: str) -> bool:
    """Return True when ``ref`` is safe to speak back to the customer verbatim."""
    if not ref:
        return False
    if _NON_ITEM_MARKER_RE.search(ref):
        return False
    return len(ref.split()) <= _MAX_ITEM_WORDS


@dataclass
class _TurnState:
    """Outcomes of the mutating tool calls observed during one agent turn.

    Attributes:
        succeeded: True once any mutating tool returned a real order payload,
            which makes an "added" claim truthful.
        rejected_refs: Product references from a call that resolved **no**
            items at all (a fully-failed call), in the order they were
            rejected. Deliberately separate from ``partial_refs`` below: a
            turn where an earlier call fails outright and a *later* call
            fully succeeds is treated as resolved by that later success (see
            ``has_rejection`` / the full-refusal path in ``validate_reply``),
            whereas a call that ADDS some items and refuses others in the
            very same result must always be disclosed, regardless of what
            any other call in the turn did.
        alternatives: Grounded ``{name, price}`` rows for ``rejected_refs``,
            taken from the tool's ``available_products``.
        partial_refs: Product references refused by a call that also
            resolved and added at least one other item in the *same* call
            (kiosk-core's ``_resolve_items`` is per-item — see
            ``mcp_server.py`` — so ``place_order``/``update_order`` can
            return both ``just_added`` and ``unavailable_items`` on one
            success). Observed live: a customer asked to order "all pizza
            available", one fabricated id among four was refused, three
            pizzas were genuinely added, and the model's reply implied all
            four were added without ever mentioning the refusal.
        partial_alternatives: Grounded ``{name, price}`` rows for
            ``partial_refs``.
        partial_quantity_refs: Product references whose quantity was refused
            as implausible by a call that also resolved and added at least
            one other item in the *same* call (kiosk-core's
            ``_split_implausible_quantities`` is per-item — see
            ``mcp_server.py`` — so ``place_order``/``update_order`` can
            return both ``just_added`` and ``quantity_rejected_items`` on one
            success). Kept separate from ``partial_refs`` because the two
            need different spoken disclosures: an off-menu refusal offers
            real alternatives, a quantity refusal asks the customer to
            repeat the number instead.
    """

    succeeded: bool = False
    rejected_refs: list[str] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    partial_refs: list[str] = field(default_factory=list)
    partial_alternatives: list[dict[str, Any]] = field(default_factory=list)
    quantity_refused: bool = False
    partial_quantity_refs: list[str] = field(default_factory=list)

    @property
    def has_rejection(self) -> bool:
        """True when a fully-failed call refused an item this turn."""
        return bool(self.rejected_refs)

    @property
    def has_partial_rejection(self) -> bool:
        """True when an otherwise-successful call also refused an item."""
        return bool(self.partial_refs)

    @property
    def has_partial_quantity_rejection(self) -> bool:
        """True when an otherwise-successful call also skipped an implausible quantity."""
        return bool(self.partial_quantity_refs)


_turn_state: contextvars.ContextVar[_TurnState | None] = contextvars.ContextVar(
    "menu_guard_turn_state", default=None
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
    """Return the tool-outcome state for the turn running in this context.

    ``contextvars.ContextVar`` can only be given one shared default object,
    not a per-context factory — using a single ``_TurnState()`` instance as
    that default (the previous implementation) meant every context that never
    called ``begin_turn()`` (e.g. a request whose context didn't propagate, or
    a test that calls ``record_tool_result`` directly) mutated that *same*
    shared object, permanently leaking one context's rejection/success flags
    into every other context that also fell through to the default —
    including, in production, unrelated concurrent requests. A fresh state is
    materialised here instead, so "no ``begin_turn()`` yet" always means an
    empty, unshared state.
    """
    state = _turn_state.get()
    if state is None:
        state = _TurnState()
        _turn_state.set(state)
    return state


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

    state = current_state()
    result = action_result.classify(tool_name, raw)
    if result.code == "UNDECODABLE":
        # Undecodable payload: treat as not-a-success. The turn then falls back
        # to the existing name-based guards, which is the prior behaviour.
        logger.warning(
            "[MENU-GUARD] Could not decode result of %s — treating as unsuccessful",
            tool_name,
        )
        return

    if result.success:
        state.succeeded = True
        # A multi-item place_order/update_order call resolves items
        # independently (see mcp_server._resolve_items): one fabricated
        # reference alongside otherwise-valid ones does NOT fail the whole
        # call — kiosk-core writes the valid items and reports the rest in
        # ``unavailable_items``/``unavailable_message`` on the *same*,
        # overall-successful payload. ``action_result.classify`` only checks
        # for a top-level ``error`` key, so this partial-rejection detail was
        # previously silently dropped: ``state.succeeded`` short-circuited
        # ``validate_reply`` and nothing ever checked whether the model's
        # narration disclosed the dropped item. Observed live: a customer
        # asked to order "all pizza available" (4 items), one fabricated
        # product_id was refused, only 3 were actually added, and the model's
        # reply implied all 4 were added (or simply announced the total and
        # jumped straight to upsell/confirm) without ever mentioning the
        # refusal. Recording the rejection here — without flipping
        # ``succeeded`` back to False, since some items really were added —
        # lets ``validate_reply`` require that partial outcome to be
        # disclosed.
        payload = result.data if isinstance(result.data, dict) else {}
        _record_rejection(state, payload.get("unavailable_items"), payload.get("available_products"))
        _record_quantity_rejection(state, payload.get("quantity_rejected_items"))
        return

    payload = result.data

    # A quantity refusal is a rejection, but not an off-menu one: the product
    # is real and on the menu, only the number was implausible (ASR heard
    # "one and 2,000" for "one or two"). Falling through to the off-menu path
    # would tell the customer their burger "isn't on our menu", which is both
    # false and unactionable — they would keep renaming a product that was
    # never the problem. kiosk-core marks this payload with ``max_quantity``.
    if isinstance(payload, dict) and payload.get("max_quantity"):
        state.quantity_refused = True
        state.rejected_refs.append("")
        logger.warning(
            "[MENU-GUARD] Implausible quantity refused by %s | detail=%r",
            tool_name, result.message[:160],
        )
        return

    ref_match = _REJECTED_REF_RE.match(result.message)
    if ref_match:
        state.rejected_refs.append(ref_match.group(1))
    else:
        # A non-catalogue failure (e.g. a database error). Record it without a
        # reference so the reply is still corrected, but with generic wording —
        # naming an item we did not parse would be a guess.
        state.rejected_refs.append("")

    _record_alternatives(state.alternatives, payload.get("available_products"))

    logger.warning(
        "[MENU-GUARD] Off-menu item refused by %s | ref=%r alternatives=%s",
        tool_name,
        state.rejected_refs[-1],
        [a["name"] for a in state.alternatives],
    )


def _record_alternatives(target: list[dict[str, Any]], available_products: Any) -> None:
    """Merge ``available_products`` rows into ``target`` (deduped by name)."""
    for product in available_products or []:
        if not isinstance(product, dict):
            continue
        name = product.get("name")
        if not name or any(a.get("name") == name for a in target):
            continue
        target.append({"name": name, "price": product.get("price")})


def _record_rejection(
    state: _TurnState, unavailable_items: Any, available_products: Any
) -> None:
    """Record a partial rejection alongside an otherwise-successful mutation.

    Populates ``state.partial_refs``/``partial_alternatives`` — kept separate
    from ``rejected_refs``/``alternatives`` (used for calls that resolved NO
    items) so that a later, cleanly-successful call in the same turn does not
    erase a genuine partial-rejection disclosure requirement, while a
    fully-failed call followed by an unrelated clean success still resolves
    as before (see the ``_TurnState`` docstring).

    Args:
        state: The current turn's tool-outcome state.
        unavailable_items: ``payload["unavailable_items"]`` — the raw
            references kiosk-core could not resolve, if any.
        available_products: ``payload["available_products"]`` — grounded
            substitutes for the unresolved references, if any.
    """
    if not unavailable_items:
        return
    for ref in unavailable_items:
        state.partial_refs.append(str(ref) if ref else "")
    _record_alternatives(state.partial_alternatives, available_products)
    logger.warning(
        "[MENU-GUARD] Partial success: mutating call also refused %d item(s) as "
        "off-menu | refused=%s alternatives=%s",
        len(unavailable_items),
        unavailable_items,
        [a["name"] for a in state.partial_alternatives],
    )


def _record_quantity_rejection(state: _TurnState, quantity_rejected_items: Any) -> None:
    """Record a partial implausible-quantity skip alongside a successful mutation.

    Populates ``state.partial_quantity_refs`` — the quantity analogue of
    ``_record_rejection`` above. Kept as its own list (not merged into
    ``partial_refs``) because the spoken disclosure differs: an off-menu
    refusal offers real menu alternatives, a quantity refusal has none to
    offer — it just needs the customer to repeat the number.

    Args:
        state: The current turn's tool-outcome state.
        quantity_rejected_items: ``payload["quantity_rejected_items"]`` — the
            references whose quantity kiosk-core judged implausible, if any.
    """
    if not quantity_rejected_items:
        return
    for ref in quantity_rejected_items:
        state.partial_quantity_refs.append(str(ref) if ref else "")
    logger.warning(
        "[MENU-GUARD] Partial success: mutating call also skipped %d item(s) for "
        "an implausible quantity | refs=%s",
        len(quantity_rejected_items),
        quantity_rejected_items,
    )


def claims_addition(text: str) -> bool:
    """Return True when ``text`` tells the customer an item is in their order."""
    return bool(text) and any(pattern.search(text) for pattern in _get_added_patterns())


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
    state = state if state is not None else current_state()
    if state.quantity_refused:
        return _get_refusal("refusal_quantity", _REFUSAL_QUANTITY_DEFAULT)
    item = next((ref for ref in state.rejected_refs if ref), "")
    if not item or not _looks_like_item_name(item):
        return _get_refusal("refusal_generic", _REFUSAL_GENERIC_DEFAULT)

    alternatives = _format_alternatives(state.alternatives)
    template = (
        _get_refusal("refusal_with_alternatives", _REFUSAL_WITH_ALTERNATIVES_DEFAULT)
        if alternatives
        else _get_refusal("refusal_plain", _REFUSAL_PLAIN_DEFAULT)
    )
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


# Phrases that tell the customer *something* was not added — used only to
# decide whether a PARTIAL success (some items added, one refused) already
# disclosed the refusal. Deliberately broader/looser than ``claims_addition``:
# here we want to avoid appending a redundant disclosure, so a false negative
# (missing a real disclosure) is the safe direction, not a false positive.
_DISCLOSURE_HINT_RE = re.compile(
    r"\b(?:unavailable|not available|don'?t have|do\s+not\s+have|"
    r"not\s+on\s+(?:the\s+)?menu|could\s?n'?t\s+(?:add|find)|"
    r"can'?t\s+add|wasn'?t\s+added|weren'?t\s+added|not\s+added)\b",
    re.IGNORECASE,
)


def _partial_success_disclosed(reply: str, state: _TurnState) -> bool:
    """True when ``reply`` already tells the customer about the dropped item."""
    return bool(_DISCLOSURE_HINT_RE.search(reply))


# Same intent as _DISCLOSURE_HINT_RE, plus "how many"/"quantity" phrasing —
# the natural way a reply asks the customer to repeat an implausible number,
# which does not necessarily use "wasn't added" wording.
_QUANTITY_DISCLOSURE_HINT_RE = re.compile(
    r"\b(?:unavailable|not available|don'?t have|do\s+not\s+have|"
    r"not\s+on\s+(?:the\s+)?menu|could\s?n'?t\s+(?:add|find)|"
    r"can'?t\s+add|wasn'?t\s+added|weren'?t\s+added|not\s+added|"
    r"how\s+many|\bquantity\b)\b",
    re.IGNORECASE,
)


def _partial_quantity_disclosed(reply: str) -> bool:
    """True when ``reply`` already tells the customer a quantity was unclear."""
    return bool(_QUANTITY_DISCLOSURE_HINT_RE.search(reply))


def _format_partial_disclosure(state: _TurnState) -> str:
    """Compose a short spoken sentence disclosing a partially-rejected item.

    Only echoes the raw fabricated reference when it reads like a real dish
    name (see ``_looks_like_item_name`` — the same guard ``build_refusal``
    uses), since the reference is untrusted model-authored text and can be a
    garbled fragment rather than a clean item name.
    """
    item = next((ref for ref in state.partial_refs if ref and _looks_like_item_name(ref)), "")
    alternatives = _format_alternatives(state.partial_alternatives)
    if item and alternatives:
        return (
            f"One note: {item} isn't on our menu, so it wasn't added — "
            f"we do have {alternatives} if you'd like that instead."
        )
    if alternatives:
        return (
            f"One note: one of those isn't on our menu, so it wasn't added — "
            f"we do have {alternatives} if you'd like that instead."
        )
    return "One note: one of those isn't on our menu, so it wasn't added."


def _reconcile_partial_success(reply: str, state: _TurnState) -> tuple[str, bool]:
    """Append a disclosure to a reply that hid a partial off-menu rejection.

    Some items in a multi-item ``place_order``/``update_order`` call really
    were added; at least one other was refused as off-menu in that same call
    (``state.has_partial_rejection``). Observed live: the model's reply named
    the successful items (or simply announced a total and moved on to
    upsell/confirmation) without ever mentioning the one that was refused —
    implying more was ordered than actually was. The reply's claims about
    what *did* succeed are left untouched (they are true); only the missing
    disclosure is appended.

    Args:
        reply: The assistant's drafted reply, after the other text guards.
        state: Turn state to read.

    Returns:
        ``(reply, corrected)``.
    """
    if _partial_success_disclosed(reply, state):
        return reply, False
    disclosure = _format_partial_disclosure(state)
    logger.warning(
        "[MENU-GUARD] Reply after a PARTIAL success did not disclose the refused "
        "item(s) (refs=%s) — appending disclosure to: %r",
        state.partial_refs,
        reply[:160],
    )
    return f"{reply.rstrip()} {disclosure}", True


def _format_partial_quantity_disclosure(state: _TurnState) -> str:
    """Compose a short spoken sentence disclosing a skipped implausible quantity.

    No alternatives to offer here (the item is real, only the number was
    unclear) — the customer just needs to be asked to repeat the quantity.
    """
    item = next(
        (ref for ref in state.partial_quantity_refs if ref and _looks_like_item_name(ref)), ""
    )
    if item:
        return f"One note: I didn't catch how many {item} you wanted, so that wasn't added yet."
    return "One note: I didn't catch one of the quantities you wanted, so that wasn't added yet."


def _reconcile_partial_outcomes(reply: str, state: _TurnState) -> tuple[str, bool]:
    """Append disclosures for every partial-rejection shape a turn hit.

    A single ``place_order``/``update_order`` call can carry both an
    off-menu partial rejection and an implausible-quantity partial rejection
    at once (e.g. one fabricated item AND one absurd quantity in the same
    multi-item request) — both must be disclosed, not just whichever the
    guard happens to check first.

    Each disclosure-already-present check is run against the model's
    ORIGINAL reply, not a reply already amended by the other disclosure:
    the off-menu disclosure text itself contains "wasn't added", which also
    matches the quantity hint pattern — checking the amended reply would
    make the quantity check see a false disclosure it never actually made.
    """
    original = reply
    corrected = False
    if state.has_partial_rejection:
        reply, changed = _reconcile_partial_success(reply, state)
        corrected = corrected or changed
    if state.has_partial_quantity_rejection and not _partial_quantity_disclosed(original):
        disclosure = _format_partial_quantity_disclosure(state)
        logger.warning(
            "[MENU-GUARD] Reply after a PARTIAL success did not disclose the skipped "
            "implausible-quantity item(s) (refs=%s) — appending disclosure to: %r",
            state.partial_quantity_refs,
            original[:160],
        )
        reply = f"{reply.rstrip()} {disclosure}"
        corrected = True
    return reply, corrected


def validate_reply(reply: str, state: _TurnState | None = None) -> tuple[str, bool]:
    """Reconcile a reply against a real off-menu/ambiguous-item rejection.

    Four failure shapes are corrected, all stemming from the same root cause
    (the model narrating a turn it did not fully understand):

    1. A false "added"/"confirmed" claim after every mutating call was
       refused as off-menu — the surrounding text (item name, price, total,
       upsell) was composed around that false premise and is not salvageable.
    2. A reply that makes no false claim but also drops the real
       ``available_products`` alternatives the tool provided, leaving a
       voice customer with no path forward (see ``_mentions_alternative``).
    3. A **partial** success: some items in a multi-item request really were
       added, but at least one other was refused as off-menu in the same
       call, and the reply never discloses that — implying the whole request
       succeeded. Unlike (1) and (2), the reply is not wholesale-replaced
       here (its claims about what *did* succeed are true); only the missing
       disclosure is appended (see ``_reconcile_partial_success``).
    4. The quantity analogue of (3): some items were added, but one line's
       quantity was implausible (e.g. ASR mis-hearing) and was skipped rather
       than aborting the whole call — the reply must disclose that too (see
       ``_reconcile_partial_outcomes``).

    Args:
        reply: The assistant's drafted reply, after the other text guards.
        state: Turn state to read. Defaults to the current context's state.

    Returns:
        ``(reply, corrected)`` — the reply to speak, and whether it was
        rewritten so the caller can log the event.
    """
    if not reply:
        return reply, False

    state = state if state is not None else current_state()
    if state.has_partial_rejection or state.has_partial_quantity_rejection:
        return _reconcile_partial_outcomes(reply, state)
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
