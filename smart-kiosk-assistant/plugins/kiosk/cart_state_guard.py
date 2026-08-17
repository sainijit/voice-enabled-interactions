"""Guard against a multi-item order call silently re-adding cart contents
or auto-accepting an unconfirmed upsell suggestion.

Background
----------
Observed live::

    assistant: "I've added Classic Chicken Burger to your order. Your total
                is now 169. Would you also like Classic French Fries
                (Regular) (89)?"
    customer:  "I was wondering if you can add one dosa also"
    tool call: place_order(items=[
                   {"product_id": "classic_chicken_burger", "quantity": 1},
                   {"product_id": "classic_french_fries",   "quantity": 1},
                   {"product_id": "dosa",                   "quantity": 1},
               ])

The customer named exactly one new item ("dosa", correctly refused as
off-menu by kiosk-core's ``_resolve_items``). But the model's tool call also
restated the burger already in the cart — ``place_order``/``update_order``
are documented as *additive* ("safe to call for follow-up items ... added to
it"), so re-sending an existing line **increments its quantity**, silently
billing the customer for a second burger they never asked for — and it
included the fries from the still-open upsell offer, which the customer
never accepted. Both are real, DB-committed writes; ``menu_guard`` (which
reasons about whether a *reply* is supported by what a tool call returned)
has nothing to catch here, because the call itself is exactly what the
customer's words license it to write for "dosa" — the surrounding two items
are the defect, and they must be stopped before the call is made, on the
same principle as ``item_intent_guard``.

Design
------
Deliberately narrow, to avoid the false-positive this module's sibling
guard (``item_intent_guard``) already warns about — a customer who is asked
"would you like another burger?" and simply says "yes" must not have that
burger stripped out just because it wasn't re-named this turn.

* Only **multi-item** calls are touched. A single-item call has nothing to
  drop without also dropping the customer's only request, and
  ``item_intent_guard`` already owns the single-item swap-vs-trust decision.
* Filtering only activates when :func:`item_intent_guard.extract_named_item`
  finds a concrete, non-anaphoric item the customer explicitly named this
  turn. A bare confirmation ("yes", "sure") finds nothing, and this guard
  then does nothing at all — the model's full multi-item resolution is
  trusted exactly as it is today. This is what protects the legitimate
  "yes, add another burger" case: no named item this turn means no filtering.
* Once a named item is found, any *other* item in the call is dropped only
  when it token-overlaps either:
    (a) a product already present in the session's known cart (a stale
        restatement of something already written), or
    (b) the single most recent upsell suggestion offered to this session
        that the customer has not since named or explicitly accepted (a
        silent auto-accept).
  Anything that matches neither is left alone — including genuinely new
  items in a real compound request ("a burger and a dosa"), since both
  survive the named-item token check in that case.

Cart/upsell state is tracked by ``OrderingAgent`` per ``session_id`` and
must be updated by the caller after every successful place_order/
update_order/remove_from_order result — see ``OrderingAgent._cart_states``.
This module itself performs no I/O and holds no state; it is a pure
function over the state handed to it, so it stays unit-testable like every
other guard in this package.
"""

from __future__ import annotations

import logging
import re

from plugins.kiosk import item_intent_guard

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "one", "some", "please", "order", "add", "get",
        "me", "to", "want", "like", "would", "will", "now", "instead",
        "well", "as", "too", "for", "have", "take", "can", "i", "also",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens with stopwords removed, for overlap comparison."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _item_reference(item: object) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("product_id") or item.get("name") or item.get("product") or "")


# Minimum token-overlap ratio (overlap / smaller set size) to treat two
# references as "the same product" — the same 0.6 cutoff
# ``OrderingService.resolve_product`` uses for its own fuzzy name matching.
# A plain non-empty intersection is too loose here: many catalogue names
# share one generic modifier ("Classic Chicken Burger" vs "Classic French
# Fries" both contain "classic"), which would otherwise misclassify an
# unrelated product as a stale cart duplicate. Requiring most of the
# *shorter* reference's tokens to overlap keeps a single shared adjective
# from causing a false match while still catching the exact-name-repeated
# case this guard exists for.
_OVERLAP_CUTOFF = 0.6


def _overlaps(tokens_a: set[str], tokens_b: set[str]) -> bool:
    if not tokens_a or not tokens_b:
        return False
    smaller = min(len(tokens_a), len(tokens_b))
    ratio = len(tokens_a & tokens_b) / smaller
    return ratio >= _OVERLAP_CUTOFF


def filter_stale_and_unconfirmed_items(
    items: list,
    utterance: str,
    known_cart_items: list[dict],
    pending_upsell: dict | None,
) -> list:
    """Drop cart-duplicate/unconfirmed-upsell items from a multi-item call.

    Args:
        items: The tool call's ``items`` argument as supplied by the model.
        utterance: The raw customer message for the current turn.
        known_cart_items: The session's most recently known cart contents,
            each ``{"product_id"/"name"/"product_name": str, "quantity": int}``.
        pending_upsell: The single most recent upsell suggestion offered to
            this session, ``{"product_id": str, "name": str}``, or ``None``.

    Returns:
        The (possibly shortened) items list. Unchanged unless a concrete
        named item was found this turn and at least one other item in the
        call matches an existing cart line or the pending upsell.
    """
    if not isinstance(items, list) or len(items) < 2:
        # Single-item calls are item_intent_guard's territory (reference
        # *swap*, not filtering); nothing to safely drop without discarding
        # the customer's only request.
        return items

    named_item = item_intent_guard.extract_named_item(utterance)
    if not named_item:
        # Bare confirmation or vague/anaphoric reference: trust the model's
        # own multi-item resolution, exactly as item_intent_guard does for
        # the single-item case.
        return items
    named_tokens = _tokens(named_item)
    if not named_tokens:
        return items

    known_cart_token_sets = [
        _tokens(c.get("name") or c.get("product_name") or "")
        for c in (known_cart_items or [])
    ]
    known_cart_token_sets = [t for t in known_cart_token_sets if t]

    upsell_tokens = (
        _tokens(pending_upsell.get("name") or "") if pending_upsell else set()
    )

    kept: list = []
    for item in items:
        reference = _item_reference(item)
        ref_tokens = _tokens(reference)

        if not ref_tokens or _overlaps(ref_tokens, named_tokens):
            # Either unresolvable (leave alone, not this guard's job) or
            # this is the item the customer just named — always kept.
            kept.append(item)
            continue

        if any(_overlaps(ref_tokens, cart_tokens) for cart_tokens in known_cart_token_sets):
            logger.warning(
                "[CART-STATE-GUARD] dropping stale re-add of %r — already in "
                "cart and not named this turn (utterance=%r)",
                reference, utterance,
            )
            continue

        if upsell_tokens and _overlaps(ref_tokens, upsell_tokens):
            logger.warning(
                "[CART-STATE-GUARD] dropping unconfirmed upsell auto-accept "
                "of %r — offered but never accepted this turn (utterance=%r)",
                reference, utterance,
            )
            continue

        kept.append(item)

    return kept
