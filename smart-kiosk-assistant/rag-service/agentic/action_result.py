"""Shared foundation for the truthfulness guards: envelope unwrap, tool-name
policy, and a normalized ``ActionResult`` classification.

Background
----------
``menu_guard.py``, ``removal_guard.py``, and ``confirm_guard.py`` each grew an
identical ``_unwrap()`` (MCP envelope decode) and an identical-in-spirit
tool-name set (``_MUTATING_TOOLS`` / ``_REMOVAL_TOOLS`` / ``_CONFIRM_TOOLS``),
and ``ordering_agent.py`` keeps a *fourth* copy of the same tool sets
(``_ORDER_TOOLS``) for ``_SentenceGate``. That duplication already caused one
real bug this session: ``_SentenceGate._is_safe()`` checked confirm-tool
*invocation* instead of *result* because the streaming gate's copy of the
policy was not updated when ``confirm_guard`` learned to distinguish
attempted-and-failed from never-attempted.

This module is the single source for:

* ``unwrap`` — decode the MCP transport envelope into the tool's own JSON
  payload. One copy instead of four.
* ``CLAIM_TOOLS`` — which MCP tools legitimise which claim type. Consumed by
  every guard *and* by ``_SentenceGate``, so a new mutating tool is wired into
  every consumer by adding it in exactly one place.
* ``ActionResult`` — a normalized ``{success, operation, code, data|message}``
  view of a tool outcome, independent of each tool's ad hoc payload shape.
  Guards keep their own richer, already-tested state (alternatives, cart
  items, "no open order") because that detail is genuinely different per
  claim type — but they now classify success/failure through one function
  instead of three near-identical inline checks.

Deliberately NOT changed by this module: the MCP tool return payloads
themselves (``kiosk_core/ordering/mcp_server.py``). Reshaping those is a
larger, cross-service change with its own test surface; this module adapts to
the existing payload shapes instead, per "preserve compatibility with the
existing MCP interface where practical."
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claim-type constants (stable identifiers — never change these).
# ---------------------------------------------------------------------------

ITEM_ADDED = "ITEM_ADDED"
ITEM_REMOVED = "ITEM_REMOVED"
ORDER_CONFIRMED = "ORDER_CONFIRMED"


# ---------------------------------------------------------------------------
# Claim-type → tool-name policy.
#
# CLAIM_TOOLS maps claim type → _MutableToolSet.  The sets start with Safe
# Smart Kiosk defaults (so guards work before MCP bootstrap completes) and
# are updated in-place after bootstrap via update_claim_tools_from_classified().
#
# Guards snapshot the _MutableToolSet OBJECT at module load, not its contents.
# Because _MutableToolSet is a mutable container, in-place updates via
# .update() are immediately visible to every guard that holds a reference —
# this is the same mutable-holder pattern used by _TemplateReplyHolder in
# llm_metrics.py.
# ---------------------------------------------------------------------------

class _MutableToolSet:
    """Set-like container whose contents can be updated after MCP bootstrap.

    Supports ``in``, ``len``, and iteration so it is a drop-in replacement for
    the plain frozenset that guards previously snapshotted.
    """

    def __init__(self, initial: frozenset[str] = frozenset()) -> None:
        self._names: set[str] = set(initial)

    # Membership test used by guards: ``tool_name not in _MUTATING_TOOLS``
    def __contains__(self, item: object) -> bool:
        return item in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"_MutableToolSet({sorted(self._names)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _MutableToolSet):
            return self._names == other._names
        if isinstance(other, (frozenset, set)):
            return self._names == other
        return NotImplemented

    def __hash__(self) -> None:  # type: ignore[override]
        raise TypeError(f"unhashable type: {type(self).__name__!r}")

    def update(self, names) -> None:
        """Replace the current contents with *names*."""
        self._names = set(names)

    def union(self, other) -> frozenset[str]:
        """Return a plain frozenset union (used to build ORDER_TOOLS)."""
        return frozenset(self._names | set(other))

    def __sub__(self, other) -> frozenset[str]:
        return frozenset(self._names - set(other))

    def __rsub__(self, other) -> frozenset[str]:
        return frozenset(set(other) - self._names)

    def __or__(self, other) -> frozenset[str]:
        return frozenset(self._names | set(other))

    def __ror__(self, other) -> frozenset[str]:
        return frozenset(set(other) | self._names)

    def __and__(self, other) -> frozenset[str]:
        return frozenset(self._names & set(other))


# Defaults match the original Smart Kiosk tool names so existing deployments
# that don't mount a domain config see zero behaviour change.
CLAIM_TOOLS: dict[str, _MutableToolSet] = {
    ITEM_ADDED:      _MutableToolSet(frozenset({"place_order", "update_order"})),
    ITEM_REMOVED:    _MutableToolSet(frozenset({"remove_from_order", "cancel_order"})),
    ORDER_CONFIRMED: _MutableToolSet(frozenset({"confirm_order", "confirm_active_order"})),
}

# Read-only order-state tools (not a mutation claim, so not in CLAIM_TOOLS).
ORDER_READ_TOOLS: _MutableToolSet = _MutableToolSet(frozenset({"get_order", "get_current_order"}))

# Union of all CLAIM_TOOLS + ORDER_READ_TOOLS.  Rebuilt after bootstrap.
def _make_mutating() -> _MutableToolSet:
    combined: set[str] = set()
    for ts in CLAIM_TOOLS.values():
        combined.update(ts)
    return _MutableToolSet(frozenset(combined))

MUTATING_TOOLS: _MutableToolSet = _make_mutating()

def _make_order_tools() -> _MutableToolSet:
    return _MutableToolSet(frozenset(MUTATING_TOOLS) | frozenset(ORDER_READ_TOOLS))

ORDER_TOOLS: _MutableToolSet = _make_order_tools()


def update_claim_tools_from_classified(classified: dict[str, frozenset[str]]) -> None:
    """Update CLAIM_TOOLS and derived sets in-place from MCP-discovered tools.

    Called once by ordering_agent.bootstrap() after MCP tool discovery.
    Because the sets are mutable holders (not snapshots), all guards that hold
    a reference to CLAIM_TOOLS[...] / MUTATING_TOOLS / ORDER_TOOLS will see
    the updated contents immediately with no re-import required.

    Args:
        classified: Output of ``domain_config.classify_discovered_tools()``.
    """
    if classified.get("cart_mutate"):
        CLAIM_TOOLS[ITEM_ADDED].update(classified["cart_mutate"])
    if classified.get("cart_remove"):
        CLAIM_TOOLS[ITEM_REMOVED].update(classified["cart_remove"])
    if classified.get("cart_confirm"):
        CLAIM_TOOLS[ORDER_CONFIRMED].update(classified["cart_confirm"])
    if classified.get("cart_read"):
        ORDER_READ_TOOLS.update(classified["cart_read"])

    # Rebuild derived sets from the now-updated sources.
    MUTATING_TOOLS.update(t for ts in CLAIM_TOOLS.values() for t in ts)
    ORDER_TOOLS.update(frozenset(MUTATING_TOOLS) | frozenset(ORDER_READ_TOOLS))

    logger.info(
        "[action_result] CLAIM_TOOLS updated from MCP bootstrap — "
        "ITEM_ADDED=%s ITEM_REMOVED=%s ORDER_CONFIRMED=%s",
        sorted(CLAIM_TOOLS[ITEM_ADDED]),
        sorted(CLAIM_TOOLS[ITEM_REMOVED]),
        sorted(CLAIM_TOOLS[ORDER_CONFIRMED]),
    )


def unwrap(raw: Any) -> dict[str, Any] | None:
    """Extract the tool's own JSON payload from an MCP response envelope.

    ``mcp_client.call_tool`` returns ``{"status": "success", "result": "<json>"}``
    on success and ``{"error": "..."}`` on a transport failure. The tool's
    real payload — including an off-menu rejection or a confirm failure — is
    the JSON *inside* ``result``.

    Args:
        raw: The value returned by ``call_tool``.

    Returns:
        The decoded tool payload, or None when there is nothing decodable.
        A transport-level error is returned as-is so the caller can treat it
        as "not a success" without misreading it as a domain-level rejection.
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


def unwrap_any(raw: Any) -> Any:
    """Like :func:`unwrap`, but preserves a JSON *list* payload.

    ``unwrap`` narrows its return to ``dict | None`` because every guard that
    consumes it immediately calls ``.get()``; handing those a list would turn
    a shape mismatch into an ``AttributeError`` deep inside a truthfulness
    check. Catalogue tools (``list_products``/``list_categories``) legitimately
    return a top-level JSON array, so the response templates need a decoder
    that keeps it.

    Args:
        raw: The value returned by ``call_tool``.

    Returns:
        The decoded payload — ``dict``, ``list``, or ``None`` when there is
        nothing decodable.
    """
    if not isinstance(raw, dict):
        return None
    if "error" in raw and "result" not in raw:
        return raw

    result = raw.get("result")
    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str) or not result:
        return None
    try:
        decoded = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    return decoded if isinstance(decoded, (dict, list)) else None


@dataclass
class ActionResult:
    """Normalized view of one tool outcome, independent of its raw payload shape.

    Attributes:
        success: True when the domain operation actually happened.
        operation: The MCP tool name that produced this result.
        code: A short machine-readable outcome code, e.g. ``"ITEM_ADDED"``,
            ``"PRODUCT_NOT_FOUND"``, ``"ORDER_NOT_FOUND"``, ``"UNDECODABLE"``.
        data: Structured fields useful to a caller on success (e.g.
            ``just_added``, ``total``, ``order_id``) — never free text.
        message: The tool's own (untrusted, model-facing) error string, kept
            only for logging — never spoken to the customer verbatim.
    """

    success: bool
    operation: str
    code: str
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


def classify(tool_name: str, raw: Any) -> ActionResult:
    """Classify a raw MCP tool result into a normalized :class:`ActionResult`.

    This performs only the generic success/failure classification shared by
    every guard (decode the envelope, check for an ``error`` key). Claim-type
    specific detail — rejected item alternatives, cart contents, "no open
    order" — stays in each guard, which already has tested, claim-specific
    logic for it.

    Args:
        tool_name: The MCP tool that was just invoked.
        raw: The raw value returned by ``mcp_client.call_tool``.

    Returns:
        An :class:`ActionResult` describing the outcome.
    """
    payload = unwrap(raw)
    if payload is None:
        return ActionResult(
            success=False, operation=tool_name, code="UNDECODABLE",
            message="Tool result could not be decoded",
        )

    error = payload.get("error")
    if error:
        return ActionResult(
            success=False, operation=tool_name, code="OPERATION_FAILED",
            data=payload, message=str(error),
        )

    return ActionResult(success=True, operation=tool_name, code="OK", data=payload)
