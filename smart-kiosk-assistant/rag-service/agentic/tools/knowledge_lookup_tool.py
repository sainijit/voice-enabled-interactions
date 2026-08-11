"""knowledge_lookup_tool — retrieves kiosk knowledge-base context for the agent.

The tool deliberately returns *retrieved context*, not a generated answer.

Earlier this wrapped the full RAG pipeline (``pipeline.stream_answer``), which
ran a complete LLM generation inside the tool. Because Google ADK already makes
two LLM round-trips per tool-calling turn (one to pick the tool, one to compose
the reply), that meant three sequential generations per question — the answer
was written by the RAG pipeline and then rewritten by the agent. Measured on a
"what are your opening hours?" turn, the tool alone cost 8.76 s of an 18.3 s
turn while retrieval itself was only 91 ms.

Returning context lets the agent compose the reply once, from the same
knowledge, inside its existing second call.
"""

from __future__ import annotations

import logging
import os
import re
import time

from agentic import llm_metrics
from utils.latency_store import retrieval_latency

logger = logging.getLogger(__name__)

# Char budget for the context handed back to the agent. The agent's reply call
# must prefill whatever we return, so this trades answer coverage against
# time-to-first-token.
#
# It is also the only reliable brake on reply length. The 4B int4 model
# paraphrases roughly everything it is handed, so context size sets answer size:
# measured on "opening and closing hours", 2001 chars of context produced a
# 277-char reply — 20 seconds of speech on a voice kiosk. Prompt instructions to
# Lower means shorter spoken answers and less prefill; too low starts
# dropping facts the customer actually asked for. Measured on that same
# question: 2000 chars -> 315-char reply / 6835 ms LLM; 900 -> 218 / 5727;
# 600 -> 147 / 4401 but the reply became factually WRONG ("open 8 AM to 12 AM
# on weekdays and weekends", collapsing distinct weekday and weekend hours).
# 900 is the floor that still answered accurately, so it is the default.
_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1800"))

_NO_CONTEXT = "No relevant information was found in the knowledge base."

# ── pinned root-section context ──────────────────────────────────────
#
# Ingestion prefixes every chunk with its heading path, e.g.
#   "[Context: QuickBite Express — Restaurant Knowledge Base > MENU 5: Pizza]"
# The chunk whose path has no ">" is the document's *root section* — the
# overview block that carries the venue-level facts (brand name, hours,
# breakfast hours, kitchen close, delivery, parking, wi-fi).
#
# That single block answers almost every question this tool legitimately
# receives, because catalogue questions are refused above and routed to
# `list_products`. Yet it is the block retrieval is worst at finding, since
# it covers ~14 unrelated topics at once and so embeds as a weak match for
# all of them. Measured rank of the root chunk over the QuickBite KB:
#
#   query                        ANN rank   rerank rank
#   "restaurant name"                   4           29
#   "opening and closing hours"         1            2
#   "breakfast timings"                16           25
#   "do you have parking"               2            1
#   "when does the kitchen close"       5            1
#
# The reranker actively demotes it for the two questions where it is the
# only source of truth. With top_k=3 and a char budget smaller than one
# chunk, the correct block was retrieved for "opening hours" at rank 2 and
# then discarded by truncation — the kiosk answered "all day from 11 AM"
# (a snack-counter line) instead of its real hours.
#
# Pinning it removes retrieval from the path for global facts entirely.
# This is deterministic, unlike tuning a corpus-dependent score threshold.
_PIN_ROOT_SECTION = os.getenv("RAG_PIN_ROOT_SECTION", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
_PIN_MAX_CHARS = int(os.getenv("RAG_PIN_MAX_CHARS", "1300"))
_ROOT_SECTION_RE = re.compile(r"^\[Context:\s*([^\]]*)\]")

# Administrative / regulatory fields that must never reach the model.
# Stripped from the pinned root section at build time so the model cannot
# echo them (which triggers the _ADMIN_LEAK_RE guard and substitutes a
# fallback, causing the "I don't have that detail" regression).
#
# Pattern strips field segments at three granularities:
#  - inline in blockquote: "· Outlet Code: QBE-CHN-001 ·" or "· Outlet Code: QBE-CHN-001"
#  - bullet sub-field: "| **FSSAI License**: 10015033005321"
#  - whole bullet: "- **GST Registration**: 33AAACQ5678G1ZM"
_ADMIN_PIN_FIELD_RE = re.compile(
    r"(?:(?:\s*[|·]\s*)?(?:\*\*)?(?:Outlet Code|FSSAI License|GST Registration|Parent Company)"
    r"(?:\*\*)?\s*:?[^|·\n]*(?:[|·])?)",
    re.IGNORECASE,
)

_pinned_context: str | None = None
_root_facts_cache: dict[str, str] | None = None

# Parses "- **Field Name**: value" bullets out of the pinned root section.
# Several fields may share one markdown bullet, separated by " | ".
_ROOT_FIELD_RE = re.compile(r"\*\*([^*]+)\*\*:\s*([^|\n]+)")


def reset_pinned_context() -> None:
    """Drop the cached root-section pin.

    Must be called whenever the vector store changes. The pin is resolved once
    and cached for the process lifetime, so without this a re-ingested or
    cleared knowledge base keeps serving the previous document's global facts
    (restaurant name, opening hours) as authoritative — the customer is told
    the old restaurant's details with full confidence.
    """
    global _pinned_context, _root_facts_cache
    if _pinned_context:
        logger.info("[TOOL:knowledge_lookup] Knowledge base changed — clearing pinned root section")
    _pinned_context = None
    _root_facts_cache = None


def root_facts(pipeline=None) -> dict[str, str]:
    """Return venue-level facts parsed out of the pinned root section.

    The root section (see ``_root_section``) is markdown bullets of the form
    ``- **Field**: value``. Parsing it into a dict lets simple, single-value
    facts (restaurant name, hours, breakfast hours) be spoken back verbatim
    without an LLM paraphrasing them — see ``reply_templates.speak_root_fact``
    for why that matters: a quantised model measurably merges distinct hour
    ranges and invents details (e.g. a public-holiday closure) that are not in
    the source text.

    Args:
        pipeline: The shared ``RagPipeline``, or ``None`` to fetch it lazily
            (avoids importing ``pipeline`` at module load for tests).

    Returns:
        A dict keyed by lower-cased, stripped field name (e.g. ``"hours"``,
        ``"brand name"``, ``"breakfast hours"``). Empty when no root section
        was found — callers must treat that as "fall back to the LLM path".
    """
    global _root_facts_cache
    if _root_facts_cache is not None:
        return _root_facts_cache

    if pipeline is None:
        from pipeline import get_shared_pipeline  # rag-service module

        pipeline = get_shared_pipeline()

    text = _root_section(pipeline)
    facts: dict[str, str] = {}
    for match in _ROOT_FIELD_RE.finditer(text):
        key = match.group(1).strip().lower()
        value = match.group(2).strip().rstrip(".")
        if key and value:
            facts[key] = value
    _root_facts_cache = facts
    return facts


def _root_section(pipeline) -> str:
    """Return the document's root-section chunk, cached after first lookup.

    Args:
        pipeline: The shared ``RagPipeline``.

    Returns:
        The root-section text, or an empty string when it cannot be
        identified. Never raises — a missing pin degrades to plain
        retrieval rather than failing the customer's question.
    """
    global _pinned_context
    if _pinned_context is not None:
        return _pinned_context

    _pinned_context = ""
    if not _PIN_ROOT_SECTION:
        return _pinned_context

    try:
        roots = []
        for doc in pipeline.iter_documents():
            text = doc.strip()
            match = _ROOT_SECTION_RE.match(text)
            if match and ">" not in match.group(1):
                roots.append(text)
        if roots:
            # Prefer the densest root block when a corpus has several docs.
            raw = max(roots, key=len)[:_PIN_MAX_CHARS]
            # Strip admin/regulatory fields before the model sees the context.
            cleaned = _ADMIN_PIN_FIELD_RE.sub("", raw)
            # Drop lines that became empty after stripping (e.g. a bullet that
            # held only the GST registration).
            lines = [
                ln for ln in cleaned.splitlines()
                if ln.strip() and ln.strip() not in {"-", "·", "|"}
            ]
            _pinned_context = "\n".join(lines)
            logger.info(
                "[TOOL:knowledge_lookup] Pinned root section (%d chars, %d stripped)",
                len(_pinned_context),
                len(raw) - len(_pinned_context),
            )
        else:
            logger.warning(
                "[TOOL:knowledge_lookup] No root section found — global facts "
                "will depend on retrieval ranking",
            )
    except Exception as exc:  # noqa: BLE001 - pin is best-effort
        logger.warning("[TOOL:knowledge_lookup] Root-section pin unavailable: %s", exc)

    return _pinned_context

# Questions about what is sold and what it costs must be answered from the
# product database via ``list_products``, never from knowledge-base prose. The
# knowledge base is marketing/menu narrative: it mentions dish names loosely and
# does not define the catalogue, so composing an answer from it invents products.
# Observed in production — a single "price of chicken burger" turn reported
# seven chicken burgers of which four ("Grilled Chicken Burger", "Chicken BBQ
# Burger", "Chicken Tikka Burger", "Chicken Schnitzel Sandwich") do not exist in
# the catalogue, each with a fabricated price.
#
# Prompt guidance alone did not stop this, so the tool refuses the query itself.
_PRICE_QUERY_RE = re.compile(
    r"\b(?:price|prices|pricing|cost|costs|how much|rate|rates|"
    r"cheapest|expensive|menu items?|what.{0,20}(?:do you (?:have|serve|sell))|"
    r"available|availability)\b",
    re.IGNORECASE,
)
# Terms that indicate a genuine knowledge-base question even when a price-ish
# word appears (e.g. "how much sugar is in the lassi", "is the offer free").
_KB_OVERRIDE_RE = re.compile(
    r"\b(?:hour|hours|open|close|closing|timing|address|location|contact|"
    r"vegan|vegetarian|allerg\w*|gluten|ingredient\w*|calorie\w*|nutrition\w*|"
    r"spicy|policy|refund|delivery|parking|wifi|offer|discount|combo deal)\b",
    re.IGNORECASE,
)

_USE_LIST_PRODUCTS = (
    "This question is about products, prices, or availability. The knowledge "
    "base is not authoritative for the catalogue and using it would invent "
    "items that do not exist. Call `list_products` instead and answer only "
    "from its result. Do not answer this question from the knowledge base or "
    "from memory."
)

# Same failure mode as _PRICE_QUERY_RE, different phrasing: "what's popular /
# your favourite / most ordered" is still a catalogue question — the answer is
# a specific dish name and price, not marketing prose. Observed live: "show me
# restaurant favourite dishes" against this knowledge base returned one real
# item (Chicken Tikka Kathi Roll, ₹169 — correctly [HIT]-tagged in the source)
# plus two fabricated ones ("Butter Chicken Wrap ₹179", "Crispy Chicken
# Burrito ₹179") that do not exist anywhere in the catalogue. The bestseller
# flag is catalogue data (``Product.is_bestseller``, seeded from the same
# products.yaml that prices come from) — ``get_popular_products`` is the
# authoritative source, never free-text retrieval.
_POPULARITY_QUERY_RE = re.compile(
    r"\b(?:popular|favou?rite\w*|bestsell\w*|best.selling|most.order\w*|"
    r"top.selling|recommend\w*|suggest\w*)\b",
    re.IGNORECASE,
)

_USE_GET_POPULAR_PRODUCTS = (
    "This question asks what is popular/most-ordered/recommended. The "
    "knowledge base is not authoritative for this and using it would invent "
    "dishes that do not exist. Call `get_popular_products` instead and answer "
    "only from its result. Do not answer this question from the knowledge "
    "base or from memory."
)

# Questions fully answered by the pinned root section (see _root_section
# above): the same ~14 venue-level facts the root chunk covers in one block.
# Retrieval scores are not a reliable relevance signal at this corpus size —
# measured live, an unrelated "Breakfast Menu > Veg Breakfast" chunk (Poha,
# Upma prices) scored -6.84, just inside the -7.0 rerank threshold, and was
# appended to a "restaurant name and opening hours" question the pinned
# section already answered in full. Supplementary retrieval adds noise, not
# coverage, for these questions, so it is skipped outright rather than tuned
# via a corpus-dependent threshold.
_ROOT_ONLY_QUERY_RE = re.compile(
    r"\b(?:restaurant('?s)? name|name of (?:the|this) (?:restaurant|outlet|place)|"
    r"(?:what|which).{0,20}(?:restaurant|outlet|place).{0,10}(?:called|name)|"
    r"opening hours?|closing hours?|operating hours?|business hours?|"
    r"what time (?:do|does) (?:you|it|the restaurant|the outlet).{0,10}(?:open|close)|"
    r"(?:open|close|working|business) (?:on|hours?)|"
    r"breakfast (?:hours?|timings?)|kitchen clos\w*|last order\w*|"
    r"parking|wi-?fi\b|wifi\b|delivery (?:radius|available|options?|area)|"
    r"take.?away|dine.?in|seating capacity|loyalty program|"
    r"catering|bulk orders?|outlet code|fssai|gst\b|"
    r"(?:restaurant|outlet) address|(?:your|the) address|location\b|"
    r"phone number|contact (?:number|details)|payment methods?)\b",
    re.IGNORECASE,
)


def _format_sources(records, budget: int = _MAX_CONTEXT_CHARS) -> str:
    """Render retrieval records into a compact, numbered context block."""
    blocks: list[str] = []
    used = 0
    for index, record in enumerate(records, start=1):
        content = (getattr(record, "content", "") or "").strip()
        if not content:
            continue
        block = f"[{index}] {content}"
        if used + len(block) > budget:
            remaining = budget - used
            # Keep a partial block only when enough room is left to be useful.
            if remaining > 200:
                blocks.append(block[:remaining].rstrip())
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


# Results that are NOT knowledge-base context: a redirect to the catalogue
# tool, or an honest "nothing found". Callers that want to *pre-ground* a turn
# with retrieved context (see ordering_agent's PREGROUND_KNOWLEDGE) must be
# able to tell these apart from real context, since injecting either into the
# prompt as if it were fact would be actively misleading.
NON_CONTEXT_RESULTS = frozenset({_NO_CONTEXT, _USE_LIST_PRODUCTS, _USE_GET_POPULAR_PRODUCTS})


async def knowledge_lookup(question: str) -> str:
    """Look up facts about hours, ingredients, allergens, policies, or outlet information.

    Use this tool for questions like "What are your opening hours?", "Is the
    Paneer Tikka Burger vegetarian?", or "Do you have parking?".

    Do NOT use this tool for prices, product availability, or menu listings —
    call ``list_products`` for those, it is the only authoritative source for
    product names and prices. Do NOT use it for "what's popular / your
    favourites / most ordered / what do you recommend" — call
    ``get_popular_products`` for those. Do NOT use it for placing, updating, or
    confirming orders.

    Args:
        question: The user's question about menu or outlet information.

    Returns:
        Knowledge-base excerpts relevant to the question. Compose the customer
        reply from these excerpts; do not read the numbering back to the user.
    """
    logger.info("[TOOL:knowledge_lookup] question=%r", question[:120])

    # Refuse catalogue questions outright — see _PRICE_QUERY_RE.
    if _PRICE_QUERY_RE.search(question) and not _KB_OVERRIDE_RE.search(question):
        logger.info(
            "[TOOL:knowledge_lookup] Redirecting product/price question to list_products: %r",
            question[:120],
        )
        return _USE_LIST_PRODUCTS

    # Refuse popularity/recommendation questions — see _POPULARITY_QUERY_RE.
    # Checked before the KB-override carve-out below would otherwise apply,
    # since "popular"/"recommend" never co-occurs with a genuine KB override
    # term in a way that changes which tool is authoritative.
    if _POPULARITY_QUERY_RE.search(question):
        logger.info(
            "[TOOL:knowledge_lookup] Redirecting popularity question to get_popular_products: %r",
            question[:120],
        )
        return _USE_GET_POPULAR_PRODUCTS

    started = time.perf_counter()
    try:
        from pipeline import get_shared_pipeline  # rag-service module

        pipeline = get_shared_pipeline()
        pinned = _root_section(pipeline)

        root_only = bool(pinned) and bool(_ROOT_ONLY_QUERY_RE.search(question))
        if root_only:
            # The pinned block already answers this question in full — skip
            # retrieval entirely rather than risk a marginal-score, off-topic
            # chunk (e.g. a menu item) being appended as if it were relevant.
            records = []
            logger.info(
                "[TOOL:knowledge_lookup] Root-covered question — skipping "
                "supplementary retrieval: %r", question[:120],
            )
        else:
            records = pipeline.retrieve(question)

        # The root section is frequently also returned by retrieval. Paying
        # for it twice would evict the question-specific chunk that retrieval
        # was actually needed for.
        if pinned:
            head = pinned[:80]
            records = [
                record
                for record in records
                if not (getattr(record, "content", "") or "").strip().startswith(head)
            ]

        elapsed_ms = (time.perf_counter() - started) * 1000
        retrieval_latency.record(elapsed_ms)
        llm_metrics.record_retrieval(elapsed_ms)

        remaining = _MAX_CONTEXT_CHARS - (len(pinned) + 2 if pinned else 0)
        retrieved = _format_sources(records, budget=max(remaining, 0))
        blocks = [block for block in (pinned, retrieved) if block]
        context = "\n\n".join(blocks)
        logger.info(
            "[TOOL:knowledge_lookup] retrieved=%d pinned=%d context_chars=%d elapsed_ms=%.0f",
            len(records), len(pinned), len(context), elapsed_ms,
        )
        return context or _NO_CONTEXT
    except Exception as exc:
        logger.error("[TOOL:knowledge_lookup] Retrieval error: %s", exc, exc_info=True)
        return f"I'm unable to look that up right now ({exc})."
