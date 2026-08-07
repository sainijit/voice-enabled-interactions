"""Customer-facing sentences for deterministic order-mutation tool outcomes.

Background
----------
Every mutating tool call today costs **two** LLM round-trips: one to decide
the tool and its arguments, and a second, purely to narrate a result the
server already knows in full — order_id, item names, total, upsell display
strings. On Panther Lake's single iGPU that second call is ~half the LLM time
in a typical turn (see tests/benchmarks/results/replay-qwen3-4b-postfix.json).

Google ADK supports skipping that second call natively: a tool sets
``tool_context.actions.skip_summarization = True`` and returns a plain string,
and ``Event.is_final_response()`` treats that as the end of the turn — no
second model call happens. See ``agentic/ordering_agent.py::_make_mcp_callable``
for where this module is used.

**Why these templates are written fresh here rather than reusing
``kiosk_core.ordering.mcp_server``'s ``unavailable_message`` /
``choice_message`` strings**: those strings are authored *for the model*, not
the customer. They read like "Do not invent them and do not ask the customer
to try again. Tell them those are unavailable and offer these real
alternatives instead: ..." — instructive prose meant to steer the LLM's next
completion. Speaking that verbatim over TTS would be nonsense. Every sentence
below is built only from *structured* fields (names, prices, totals, ids) that
the server already validated, never from those instruction strings.

Design
------
Every function here returns ``None`` when the tool outcome does not cleanly
match one of the templates below (e.g. a partial success mixed with a
rejection). Returning ``None`` means "fall through to the normal LLM
narration call" — the safe default. A function is never called on a result it
cannot describe with total confidence; when in doubt, the LLM still speaks.

Pure functions, no I/O, no LLM round-trip — unit-testable exactly like the
guards in this package.
"""

from __future__ import annotations

import json
from typing import Any

# How many upsell/alternative suggestions to mention in one spoken sentence.
# Matches the ceiling already used by menu_guard and mcp_server for the same
# reason: a voice customer stops retaining a list beyond this.
_MAX_SPOKEN_ALTERNATIVES = 3


def unwrap(raw: Any) -> dict[str, Any] | None:
    """Extract the tool's own JSON payload from the MCP response envelope.

    Mirrors ``menu_guard._unwrap`` / ``removal_guard._unwrap`` — see there for
    the full rationale. Duplicated rather than imported to keep this module
    self-contained and independently testable, matching the existing pattern
    between those two guards.

    Args:
        raw: The value returned by ``mcp_client.call_tool``.

    Returns:
        The decoded tool payload, or ``None`` when there is nothing decodable
        (a transport-level error is returned as-is).
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


def _money(value: Any) -> str:
    """Render a price/total as ``"79"`` or ``"78.50"`` — never ``"79.00"``.

    TTS pronounces ``79.00`` as "seventy-nine point zero zero"; whole-rupee
    values are the overwhelming majority in this catalogue.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else f"{f:.2f}"


def _join_names(names: list[str]) -> str:
    """Join a list of item names into natural spoken English."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def speak_order_mutation(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean ``place_order``/``update_order`` outcome.

    Args:
        payload: The tool's own JSON result (already unwrapped from the MCP
            transport envelope), as produced by ``mcp_server.place_order`` /
            ``update_order``.

    Returns:
        A spoken sentence for an unambiguous success, or ``None`` when the
        result is a rejection, a category-disambiguation, or a mix of
        success and failure — those are left to the LLM, which can weave a
        compound reply together better than a fixed template can.
    """
    if not isinstance(payload, dict):
        return None
    # Any of these mean the outcome is not a clean, fully-resolved success —
    # fall through and let the model narrate the nuance.
    if "error" in payload or "needs_choice" in payload or "unavailable_message" in payload:
        return None
    just_added = payload.get("just_added")
    total = payload.get("total")
    if not just_added or total is None:
        return None

    names = _join_names([it.get("name", "") for it in just_added])
    if not names:
        return None

    sentence = f"I've added {names} to your order. Your total is now ₹{_money(total)}."

    upsell = payload.get("upsell_suggestions") or []
    if upsell and isinstance(upsell[0], dict) and upsell[0].get("display"):
        sentence += f" Would you also like {upsell[0]['display']}?"
    else:
        sentence += " Would you like anything else?"
    return sentence


def speak_confirm(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean order confirmation.

    Args:
        payload: The tool's own JSON result from ``confirm_order`` /
            ``confirm_active_order``.

    Returns:
        A spoken confirmation sentence, or ``None`` for any error — the
        error paths here (empty cart, order not found, already confirmed)
        are worded for the model, not the customer, so they always fall
        through to normal narration.
    """
    if not isinstance(payload, dict) or "error" in payload:
        return None
    order_id = payload.get("order_id")
    total = payload.get("total")
    if order_id is None or total is None or payload.get("status") != "confirmed":
        return None
    return (
        f"Your order is confirmed! Order number {order_id}, total ₹{_money(total)}. "
        f"Thank you!"
    )


def speak_removal(payload: dict[str, Any]) -> str | None:
    """Template a reply for a clean ``remove_from_order`` outcome.

    Args:
        payload: The tool's own JSON result from ``remove_from_order``.

    Returns:
        A spoken sentence for a fully-resolved removal (every requested item
        was actually in the cart), or ``None`` when nothing was removed or
        the removal was only partial — those read better as an LLM-narrated
        explanation of what was and wasn't found.
    """
    if not isinstance(payload, dict) or "error" in payload:
        return None
    removed = payload.get("removed")
    not_in_cart = payload.get("not_in_cart")
    total = payload.get("total")
    if not removed or not_in_cart or total is None:
        return None

    names = _join_names(removed)
    if payload.get("cart_empty"):
        return f"I've removed {names}. Your cart is now empty."
    return f"I've removed {names}. Your new total is ₹{_money(total)}."


# Dispatch table used by ordering_agent.py — keeps the "which tools are
# speakable" decision in one place, next to the templates themselves.
_TEMPLATES = {
    "place_order": speak_order_mutation,
    "update_order": speak_order_mutation,
    "confirm_order": speak_confirm,
    "confirm_active_order": speak_confirm,
    "remove_from_order": speak_removal,
}

# Public: which tools this module can ever speak for. Checked by
# ordering_agent.py before even attempting ``speak()``, so a tool this module
# doesn't know about is never a candidate for skipping narration.
SPEAKABLE_TOOLS = frozenset(_TEMPLATES)


def speak(tool_name: str, raw_result: Any) -> str | None:
    """Return a spoken reply for ``tool_name``'s result, or None to defer to the LLM.

    Args:
        tool_name: The MCP tool that was just called.
        raw_result: The raw value returned by ``mcp_client.call_tool`` (the
            MCP transport envelope, not yet decoded).

    Returns:
        A ready-to-speak sentence, or ``None`` when this tool/outcome
        combination is not one of the templates above.
    """
    template = _TEMPLATES.get(tool_name)
    if template is None:
        return None
    payload = unwrap(raw_result)
    if payload is None:
        return None
    return template(payload)
