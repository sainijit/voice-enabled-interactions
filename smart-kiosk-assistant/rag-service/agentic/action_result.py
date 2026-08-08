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
# Claim-type -> tool-name policy.
#
# This is the CLAIM_POLICIES registry: the single place that says "these
# tools' results are what legitimise this kind of claim". Every guard and
# _SentenceGate must read tool sets from here, not redefine them.
# ---------------------------------------------------------------------------

ITEM_ADDED = "ITEM_ADDED"
ITEM_REMOVED = "ITEM_REMOVED"
ORDER_CONFIRMED = "ORDER_CONFIRMED"

CLAIM_TOOLS: dict[str, frozenset[str]] = {
    ITEM_ADDED: frozenset({"place_order", "update_order"}),
    ITEM_REMOVED: frozenset({"remove_from_order", "cancel_order"}),
    ORDER_CONFIRMED: frozenset({"confirm_order", "confirm_active_order"}),
}

# Union of every claim-bearing tool. This replaces ordering_agent.py's
# _ORDER_TOOLS (which also includes the read-only get_order — kept as an
# explicit superset below rather than folded into CLAIM_TOOLS, since
# get_order legitimises "here is your order" claims but not a mutation claim).
MUTATING_TOOLS: frozenset[str] = frozenset(
    tool for tools in CLAIM_TOOLS.values() for tool in tools
)

ORDER_TOOLS: frozenset[str] = MUTATING_TOOLS | frozenset({"get_order"})


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
