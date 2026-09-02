"""Guard against adding a stale/pending item instead of the one just named.

Background
----------
Observed live: the customer had an item pending confirmation ("Would you like
to proceed with ordering French fries?"), then said "Go ahead and add a
pizza." The model's ``update_order``/``place_order`` tool call still carried
``"french fries"`` as the item reference — it resolved successfully (French
fries genuinely are on the menu), so ``menu_guard`` sees a clean success and
has nothing to correct. The customer is told a pizza was added while French
fries land in the cart, and the wrong item is written to the database.

This is not an off-menu problem (``menu_guard``'s territory) or a false
confirm-without-a-tool-call problem (``order_claim_guard``/
``_ORDER_CLAIM_FALLBACK``). It is the tool call itself carrying a stale
argument the model failed to update against what the customer just said.

Design
------
Rather than detect the bad write after it lands and speak a correction (which
still leaves a wrong row in the cart until something rolls it back), this
guard corrects the **tool-call argument before the call is made** — the same
philosophy as the dietary-preference injection in ``ordering_agent.py``:
deterministic ground truth from the current turn's raw utterance overrides
whatever the model decided to pass.

Scope is deliberately narrow to avoid false corrections:

* Only single-item ``place_order``/``update_order`` calls are considered.
  A multi-item call ("fries and a pizza") is left untouched — safely
  disambiguating which of several references is stale is out of scope, and
  leaving the call alone preserves today's (correct, most of the time)
  behaviour rather than risking making it worse.
* The utterance must contain an explicit, extractable "add/order X" phrase.
  A bare confirmation ("yes", "go ahead", "sounds good") has no such phrase,
  so the guard does nothing and the pending item is correctly used as-is.
* The correction only fires when the named phrase shares **no** meaningful
  token overlap with the tool call's reference — a paraphrase of the same
  item ("the classic burger" vs. "classic chicken burger") still overlaps and
  is left alone; only a clearly *different* item swaps the reference.

The module performs no I/O and makes no LLM round-trip, so it can be unit
tested as pure functions over text.
* The named phrase must not be *anaphoric*. This is the constraint whose
  absence made the first version of this guard actively harmful. "Yes, one of
  those" and "I'd like to try all of them" both match the add/order pattern,
  but they point at products established earlier in the conversation, which
  the model had already resolved correctly. Substituting the literal words
  produced replies like "Sorry, we don't have of those on the menu" and threw
  away a valid four-item order. When the customer is vague, the model's
  context-resolved reference wins.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from agentic import action_result

logger = logging.getLogger(__name__)

# Verbs that introduce a fresh item reference. Deliberately anchored to the
# start of an imperative/desire clause — "go ahead and add a pizza",
# "I'd like a pizza", "order me a pizza", "get me a pizza" — rather than any
# mention of the verb anywhere in the sentence, to avoid capturing unrelated
# clauses.
_NAMED_ITEM_RE = re.compile(
    r"""
    \b(?:
        add(?:ing)?|order|get\s+me|bring\s+me|
        i(?:'d|'ll|\s+would|\s+will)?\s+(?:like|want|take|have)|
        i\s+want|can\s+i\s+(?:get|have|order)
    )\b
    \s+(?:to\s+)?(?:order\s+)?
    (?:me\s+)?
    (?:a|an|one|\d+|the|some)?\s*
    (?P<item>[a-z][a-z\s'-]{2,40}?)
    (?:\s*(?:please|now|instead|as\s+well|too|for\s+me))?\s*[.!?]*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare agreement/confirmation phrases carry no new item and must never be
# treated as naming one, even though some contain a word like "order".
_CONFIRMATION_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"yes|yeah|yep|sure|ok(?:ay)?|go\s+ahead|sounds?\s+good|"
    r"please\s+(?:proceed|confirm|do)|proceed|confirm(?:\s+it)?|"
    r"that('?s| is)\s+(?:right|correct|it)|correct"
    r")\s*[.!?]*\s*$",
    re.IGNORECASE,
)

# Tentative/exploratory phrases — the customer is browsing, not placing an
# order. Matched against the whole utterance; when detected, extract_named_item
# returns None so the guard never treats the embedded item as a direct order
# reference. Examples:
#   "I was thinking of ordering a burger"
#   "Maybe a pizza"
#   "I might want fries"
#   "I'm considering a coffee"
_TENTATIVE_RE = re.compile(
    r"""^\s*(?:
        i\s+(?:was|am|'m)\s+(?:thinking|considering|looking|going)\s+(?:of|about|at|to)
        |i\s+might(?:\s+want|\s+like|\s+have|\s+get)?
        |maybe\s+a?(?:\s+i\s+(?:could|can|would))?
        |possibly\s+a?
        |what\s+about\s+a?
        |how\s+about\s+a?
        |i\s+(?:could|can|would)\s+(?:try|have|get)\s+a?
        |(?:just\s+)?(?:browsing|looking)
        |i\s+was\s+(?:going\s+to|planning\s+to)\s+(?:get|order|have)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# Common stopwords stripped before token-overlap comparison so they never
# count as "overlap" between two different items.
_STOPWORDS = frozenset(
    {
        "a", "an", "the", "one", "some", "please", "order", "add", "get",
        "me", "to", "want", "like", "would", "will", "now", "instead",
        "well", "as", "too", "for", "have", "take", "can", "i",
    }
)


def _tokens(text: str) -> set[str]:
    """Lowercase word tokens with stopwords removed, for overlap comparison."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS}


# Anaphora and collective quantifiers. These point at something established
# earlier in the conversation rather than naming a product, so the model's own
# reference — which was resolved from that context — must win. Matched against
# the whole extracted phrase, so "one of those" is rejected while a real
# product containing a listed word (e.g. "Cold Coffee") is unaffected.
_ANAPHORIC_RE = re.compile(
    r"""^\s*(?:
        (?:try\s+|have\s+|take\s+)?
        (?:all|both|each|any|either|one|some|two|three|\d+)?\s*
        (?:of\s+)?
        (?:them|those|these|it|that|this|the\s+(?:same|other|first|second|last|one))
        (?:\s+one)?
        |everything|anything|something|the\s+usual|same\s+again|same|usual
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)

def _is_concrete_item(phrase: str) -> bool:
    """Return True when ``phrase`` could plausibly identify a real product.

    Anaphora ("one of those", "all of them") points at something established
    earlier in the conversation, which the model already resolved into a
    concrete product. Substituting the literal words destroys that resolution
    and guarantees a spurious off-menu refusal, so those are rejected.

    A bare *category* ("pizza") is deliberately **not** rejected: it is exactly
    the original motivating bug ("go ahead and add a pizza" while fries are
    pending), where letting the stale reference through would silently add the
    wrong item. The category is allowed to flow into the tool call, where
    ``mcp_server._resolve_items`` turns it into a "which pizza?" disambiguation
    rather than an off-menu refusal.

    Args:
        phrase: The item phrase extracted from the customer's utterance.

    Returns:
        True when the phrase names something specific enough to act on,
        False for anaphora and collective quantifiers.
    """
    if _ANAPHORIC_RE.match(phrase):
        return False
    return bool(_tokens(phrase))


def extract_named_item(utterance: str) -> str | None:
    """Return the item the customer explicitly named this turn, if any.

    Args:
        utterance: The raw customer message for the current turn (not the
            ``[dietary=...]``/``[customer_name=...]`` tag-prefixed version).

    Returns:
        The captured item phrase (untrimmed of internal words), or ``None``
        when the utterance is a bare confirmation, does not match an explicit
        "add/order X" pattern, or names something too vague to identify a
        product (anaphora or a bare category).
    """
    if not utterance or _CONFIRMATION_ONLY_RE.match(utterance):
        return None
    # Exploratory/tentative utterances ("I was thinking of ordering a burger",
    # "maybe a pizza") are NOT order commands — the model should browse and
    # clarify. Returning None here prevents the stale-reference guard from
    # treating the embedded item as a confirmed item reference, which would
    # make the guard appear to confirm a tentative statement as an order intent.
    if _TENTATIVE_RE.match(utterance):
        return None
    match = _NAMED_ITEM_RE.search(utterance)
    if not match:
        return None
    item = match.group("item").strip()
    if not item or not _tokens(item):
        return None
    if not _is_concrete_item(item):
        # Vague reference: the model's context-resolved product wins.
        return None
    return item


def mismatches(reference: str, named_item: str) -> bool:
    """Return True when ``named_item`` shares no meaningful token with ``reference``.

    Args:
        reference: The product reference currently in the tool-call args
            (what the model is about to send to kiosk-core).
        named_item: The item phrase just extracted from the customer's
            current utterance via :func:`extract_named_item`.

    Returns:
        True when the two token sets are disjoint, meaning the tool call is
        very likely still carrying a stale/pending reference rather than the
        item the customer just named.
    """
    ref_tokens = _tokens(reference)
    named_tokens = _tokens(named_item)
    if not ref_tokens or not named_tokens:
        return False
    return ref_tokens.isdisjoint(named_tokens)


def corrected_reference(
    tool_name: str, items: object, utterance: str
) -> str | None:
    """Compute a corrected single-item reference, if a correction is warranted.

    Args:
        tool_name: The MCP tool about to be called (only ``place_order`` and
            ``update_order`` carry an ``items`` argument worth checking).
        items: The tool call's ``items`` argument as supplied by the model.
        utterance: The raw customer message for the current turn.

    Returns:
        The customer's just-named item phrase when it should replace the
        existing single-item reference, or ``None`` when nothing should
        change (multi-item calls, bare confirmations, or no mismatch).
    """
    if tool_name not in action_result.CLAIM_TOOLS[action_result.ITEM_ADDED]:
        return None
    if not isinstance(items, list) or len(items) != 1:
        # Multi-item calls are left untouched — see module docstring.
        return None
    item = items[0]
    if not isinstance(item, dict):
        return None
    reference = item.get("product_id") or item.get("name") or item.get("product") or ""
    if not reference:
        return None

    named_item = extract_named_item(utterance)
    if not named_item:
        return None

    if is_anaphoric_product_ref(named_item):
        # "Yes, I would like to order it." names no product — it points back at
        # one. Treating the pronoun as the customer's just-named item would
        # overwrite a perfectly good reference (observed live: the model sent
        # "Cafe Latte", this guard replaced it with "order it", and the
        # customer was told a just-quoted item was off-menu).
        return None

    if not mismatches(reference, named_item):
        return None

    logger.warning(
        "[ITEM-INTENT-GUARD] tool=%s stale reference=%r did not match customer's "
        "just-named item=%r — correcting to the named item",
        tool_name, reference, named_item,
    )
    return named_item


# ---------------------------------------------------------------------------
# Anaphoric tool arguments ("order it")
#
# The guard above corrects a *stale but real* product reference. This section
# handles the opposite failure, observed live: the assistant answered a price
# question ("The Cafe Latte (250 ml) is available for ₹109. Would you like to
# order it?"), the customer said "Yes, I would like to order it.", and the
# model passed ``product_id="order it"`` — echoing the customer's words instead
# of resolving the pronoun against its own previous sentence.
#
# Nothing downstream can recover from that: kiosk-core looks up "order it",
# finds no such product, and menu_guard tells the customer the item is not on
# the menu — while the item is on the menu and was quoted seconds earlier.
#
# The fix is to remember which catalogue product the customer named in an
# earlier turn and substitute it when the model sends a pronoun. Same
# philosophy as the dietary/user_id injection: deterministic context beats
# whatever the model chose to pass.
# ---------------------------------------------------------------------------

# Leading action verbs stripped before an argument is tested for anaphora, so
# "order it" / "add that one" reduce to the bare pronoun the pattern expects.
_ACTION_PREFIX_RE = re.compile(
    r"""^\s*(?:
        please|just|kindly|
        i|we|
        would|will|'d|'ll|
        like|want|wish|
        to|
        order|add|get|have|take|buy|purchase|place|make|do
    )\b\s*""",
    re.IGNORECASE | re.VERBOSE,
)


def is_anaphoric_product_ref(reference: str) -> bool:
    """Return True when a tool's product argument is a pronoun, not a product.

    Args:
        reference: The ``product_id``/``name`` the model passed to the tool.

    Returns:
        True when the reference points at something established earlier in the
        conversation ("it", "order it", "that one") or carries no meaningful
        token at all, rather than identifying a product. A real product id or
        name — including one that merely *contains* a listed word, such as
        "Cold Coffee" — returns False.
    """
    if not reference or not isinstance(reference, str):
        return False
    cleaned = reference.strip()
    # Strip stacked lead-ins ("I would like to order it") one at a time.
    previous = None
    while previous != cleaned and cleaned:
        previous = cleaned
        cleaned = _ACTION_PREFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return True
    if _CONFIRMATION_ONLY_RE.match(cleaned):
        return True
    return bool(_ANAPHORIC_RE.match(cleaned)) or not _tokens(cleaned)


def _fold(text: str) -> str:
    """Lowercase and strip accents so "café latte" matches "Cafe Latte"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _spoken_name(product_name: str) -> str:
    """Drop the trailing size/volume qualifier customers rarely say aloud."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", product_name).strip()


def product_named_in(utterance: str, products: list) -> dict | None:
    """Find the one catalogue product the customer named in this utterance.

    Used to remember a referent across turns, so a later pronoun ("order it")
    can be resolved back to a real product.

    Args:
        utterance: The customer's raw message for the turn.
        products: Product dicts as returned by a catalogue tool; each is
            expected to carry ``product_id`` and ``name``.

    Returns:
        The single matching product dict, or ``None`` when the utterance named
        no product or named more than one (ambiguous — nothing is remembered
        rather than remembering the wrong item).
    """
    if not utterance or not isinstance(products, list):
        return None
    folded = _fold(utterance)
    spoken = [
        (product, _spoken_name(_fold(product["name"])))
        for product in products
        if isinstance(product, dict) and isinstance(product.get("name"), str)
    ]
    matches = [product for product, name in spoken if name and name in folded]
    if len(matches) != 1:
        return None

    # Guard against a multi-item utterance ("a latte and an espresso") whose
    # full-name match happens to land on only one of them: remembering that one
    # would silently resolve a later "add it" to the wrong product. Any token
    # pointing at a *second* product makes the referent ambiguous, so nothing
    # is remembered.
    said = _tokens(folded)
    hinted = {
        product["product_id"]
        for product, name in spoken
        if said & _tokens(name)
    }
    if len(hinted) > 1:
        return None
    return matches[0]
