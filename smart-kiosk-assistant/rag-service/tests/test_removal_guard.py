"""Tests for the unbacked cart-removal claim guard (agentic/removal_guard.py).

These are pure functions over reply text and recorded tool outcomes — no LLM,
no network, no database. They must stay that way.
"""

from __future__ import annotations

import json

import pytest

from agentic import removal_guard


def _mcp_envelope(payload: dict) -> dict:
    """Wrap a tool payload the way mcp_client.call_tool returns it."""
    return {"status": "success", "result": json.dumps(payload)}


def _not_in_cart_payload(ref: str = "sushi platter") -> dict:
    """Reproduce kiosk-core's "nothing matched the cart" rejection verbatim."""
    return {
        "error": (
            f"None of those items are in the cart. The cart currently contains: "
            f"Pepsi (330 ml). Tell the customer what is actually in their order and "
            f"ask which of those to remove. Do not claim you removed anything."
        ),
        "cart_items": [
            {"product_id": "DRINK-001", "name": "Pepsi (330 ml)", "quantity": 2},
        ],
    }


def _success_payload() -> dict:
    return {
        "order_id": 87,
        "status": "draft",
        "total": 59.0,
        "items": [
            {"product_id": "DRINK-002", "product_name": "7UP (330 ml)",
             "quantity": 1, "price": 59.0},
        ],
        "removed": ["Pepsi (330 ml)"],
        "not_in_cart": [],
        "cart_empty": False,
    }


def _partial_miss_payload() -> dict:
    """One item removed, one requested reference never matched the cart."""
    return {
        "order_id": 87,
        "status": "draft",
        "total": 59.0,
        "items": [
            {"product_id": "DRINK-002", "product_name": "7UP (330 ml)",
             "quantity": 1, "price": 59.0},
        ],
        "removed": ["Pepsi (330 ml)"],
        "not_in_cart": ["fries"],
        "cart_empty": False,
    }


@pytest.fixture(autouse=True)
def fresh_turn():
    """Every test starts from a clean per-turn state."""
    removal_guard.begin_turn()


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def test_not_in_cart_rejection_is_recorded_with_cart_contents() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))

    state = removal_guard.current_state()
    assert state.succeeded is False
    assert state.attempted is True
    assert state.rejected_refs == [""]
    assert state.cart_items == ["Pepsi (330 ml)"]


def test_successful_removal_is_recorded_as_success() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_success_payload()))

    assert removal_guard.current_state().succeeded is True


def test_partial_miss_is_recorded_as_success_with_a_rejected_ref() -> None:
    """Removing one of two requested items is still a real, partial success."""
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_partial_miss_payload()))

    state = removal_guard.current_state()
    assert state.succeeded is True
    assert state.rejected_refs == ["fries"]


def test_non_removal_tools_are_ignored() -> None:
    removal_guard.record_tool_result("update_order", _mcp_envelope({"error": "boom"}))

    state = removal_guard.current_state()
    assert state.succeeded is False
    assert state.attempted is False


def test_transport_error_is_not_treated_as_success() -> None:
    removal_guard.record_tool_result("remove_from_order", {"error": "Tool remove_from_order timed out"})

    state = removal_guard.current_state()
    assert state.succeeded is False
    assert state.attempted is True


def test_undecodable_result_is_not_a_success() -> None:
    removal_guard.record_tool_result(
        "remove_from_order", {"status": "success", "result": "<not json>"}
    )

    assert removal_guard.current_state().succeeded is False


def test_cancel_order_success_is_recorded_with_items_removed() -> None:
    payload = {"cancelled": True, "order_id": 42, "items_removed": ["Pepsi (330 ml)", "Fries"]}
    removal_guard.record_tool_result("cancel_order", _mcp_envelope(payload))

    state = removal_guard.current_state()
    assert state.succeeded is True
    assert state.attempted is True
    assert state.no_open_order is False
    assert state.cart_items == ["Pepsi (330 ml)", "Fries"]


def test_cancel_order_with_no_open_order_is_recorded_distinctly() -> None:
    payload = {"error": "There is no open order to cancel for this customer."}
    removal_guard.record_tool_result("cancel_order", _mcp_envelope(payload))

    state = removal_guard.current_state()
    assert state.succeeded is False
    assert state.no_open_order is True


def test_cancel_order_claim_with_no_open_order_is_replaced() -> None:
    removal_guard.record_tool_result(
        "cancel_order",
        _mcp_envelope({"error": "There is no open order to cancel."}),
    )
    reply, corrected = removal_guard.validate_reply(
        "I've cancelled your order. Anything else?"
    )
    assert corrected is True
    assert reply == removal_guard._REFUSAL_NO_OPEN_ORDER


def test_cancel_order_claim_backed_by_success_is_left_alone() -> None:
    removal_guard.record_tool_result(
        "cancel_order",
        _mcp_envelope({"cancelled": True, "order_id": 5, "items_removed": ["Pepsi"]}),
    )
    reply, corrected = removal_guard.validate_reply(
        "Your order has been cancelled. Would you like to start a new one?"
    )
    assert corrected is False
    assert "cancelled" in reply


# ---------------------------------------------------------------------------
# Reply validation — the behaviour that was reported broken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I've removed the Sushi Platter from your order. Anything else?",
        "I have removed a sushi platter from your order.",
        "The Sushi Platter has been removed from your order.",
        "Removed the Sushi Platter from your cart.",
        "Taken the Sushi Platter off your order.",
        "That's been removed! Anything else?",
    ],
)
def test_unmatched_removal_claim_is_replaced(reply: str) -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))

    corrected, changed = removal_guard.validate_reply(reply)

    assert changed is True
    assert "removed" not in corrected.lower()


def test_removal_claim_with_no_tool_call_at_all_is_replaced() -> None:
    """The model narrated success without invoking remove_from_order at all."""
    reply = "I've removed the Pepsi from your order."

    corrected, changed = removal_guard.validate_reply(reply)

    assert changed is True
    assert "removed" not in corrected.lower()


def test_successful_removal_claim_is_left_alone() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_success_payload()))
    reply = "I've removed the Pepsi from your order. Your total is now 59 rupees."

    corrected, changed = removal_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_non_removal_reply_on_a_rejected_turn_is_left_alone() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))
    reply = "Sorry, that's not in your order. Your cart has a Pepsi — remove that instead?"

    corrected, changed = removal_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_begin_turn_clears_previous_rejection() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))
    removal_guard.begin_turn()
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_success_payload()))

    reply = "I've removed the Pepsi from your order."
    corrected, changed = removal_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_empty_reply_is_untouched() -> None:
    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))

    assert removal_guard.validate_reply("") == ("", False)


# ---------------------------------------------------------------------------
# Streaming gate integration
# ---------------------------------------------------------------------------


def test_sentence_gate_withholds_unbacked_removal_claim() -> None:
    """Speech cannot be recalled, so the gate must not release the false claim."""
    from agentic.ordering_agent import _SentenceGate

    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_not_in_cart_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("remove the sushi platter", spoken.append)

    gate.feed("I've removed the Sushi Platter from your order. ", ["remove_from_order"])

    assert spoken == []


def test_sentence_gate_releases_backed_removal_claim() -> None:
    from agentic.ordering_agent import _SentenceGate

    removal_guard.record_tool_result("remove_from_order", _mcp_envelope(_success_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("remove the pepsi", spoken.append)

    gate.feed("I've removed the Pepsi from your order. ", ["remove_from_order"])

    assert spoken == ["I've removed the Pepsi from your order."]
