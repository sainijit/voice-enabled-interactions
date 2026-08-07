"""Tests for the off-menu addition guard (agentic/menu_guard.py).

These are pure functions over reply text and recorded tool outcomes — no LLM,
no network, no database. They must stay that way.
"""

from __future__ import annotations

import json

import pytest

from agentic import menu_guard


def _mcp_envelope(payload: dict) -> dict:
    """Wrap a tool payload the way mcp_client.call_tool returns it."""
    return {"status": "success", "result": json.dumps(payload)}


def _off_menu_payload(ref: str = "sushi platter") -> dict:
    """Reproduce kiosk-core's _resolve_items rejection payload verbatim."""
    return {
        "error": (
            f"'{ref}' is not on the menu. Do not invent it and do not ask the "
            f"customer to try again. Tell them it is unavailable and offer these "
            f"real alternatives instead: Veg Crunch Wrap (149)."
        ),
        "available_products": [
            {"product_id": "WRAP-VG-001", "name": "Veg Crunch Wrap", "price": 149.0},
            {"product_id": "BURGER-NV-002", "name": "Spicy Chicken Burger", "price": 169.0},
        ],
    }


def _success_payload() -> dict:
    return {
        "order_id": 7,
        "status": "draft",
        "total": 149.0,
        "items": [
            {"product_id": "WRAP-VG-001", "product_name": "Veg Crunch Wrap",
             "quantity": 1, "price": 149.0},
        ],
    }


@pytest.fixture(autouse=True)
def fresh_turn():
    """Every test starts from a clean per-turn state."""
    menu_guard.begin_turn()


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def test_off_menu_rejection_is_recorded_with_reference_and_alternatives() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))

    state = menu_guard.current_state()
    assert state.succeeded is False
    assert state.rejected_refs == ["sushi platter"]
    assert [a["name"] for a in state.alternatives] == [
        "Veg Crunch Wrap", "Spicy Chicken Burger"
    ]


def test_successful_order_is_recorded_as_success() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_success_payload()))

    assert menu_guard.current_state().succeeded is True


def test_non_mutating_tools_are_ignored() -> None:
    """An empty list_products result is a catalogue answer, not a failed add."""
    menu_guard.record_tool_result("list_products", _mcp_envelope({"error": "boom"}))

    state = menu_guard.current_state()
    assert state.succeeded is False
    assert state.has_rejection is False


def test_transport_error_is_not_treated_as_off_menu() -> None:
    """A timeout must not make the kiosk claim the item is off the menu."""
    menu_guard.record_tool_result("place_order", {"error": "Tool place_order timed out"})

    state = menu_guard.current_state()
    assert state.succeeded is False
    assert state.rejected_refs == [""]
    assert menu_guard.build_refusal(state) is not None


def test_undecodable_result_is_not_a_success() -> None:
    menu_guard.record_tool_result("place_order", {"status": "success", "result": "<not json>"})

    state = menu_guard.current_state()
    assert state.succeeded is False
    assert state.has_rejection is False


# ---------------------------------------------------------------------------
# Reply validation — the behaviour that was reported broken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I've added the Sushi Platter to your order. Anything else?",
        "I have added a sushi platter to your order.",
        "The Sushi Platter has been added to your order.",
        "Added the Sushi Platter to your cart.",
        "That's been added! Would you like a drink?",
        "Your order now contains a Sushi Platter.",
        "I've put a sushi platter in your order.",
    ],
)
def test_off_menu_addition_claim_is_replaced(reply: str) -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    assert "added" not in corrected.lower()
    assert "sushi platter" in corrected.lower()
    assert "don't have" in corrected.lower()
    # Only grounded alternatives may be offered.
    assert "Veg Crunch Wrap at 149 rupees" in corrected
    assert "Spicy Chicken Burger at 169 rupees" in corrected


def test_refusal_is_speakable_plain_text() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))

    corrected, _ = menu_guard.validate_reply("I've added the Sushi Platter to your order.")

    assert "*" not in corrected and "`" not in corrected and "\n" not in corrected
    assert len(corrected) < 220


def test_successful_addition_claim_is_left_alone() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_success_payload()))
    reply = "I've added the Veg Crunch Wrap to your order. Your total is 149 rupees."

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_partial_turn_with_a_later_success_is_left_alone() -> None:
    """A failed attempt followed by a real addition must not be rewritten."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    menu_guard.record_tool_result("update_order", _mcp_envelope(_success_payload()))

    reply = "I've added the Veg Crunch Wrap to your order."
    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_non_addition_reply_on_a_rejected_turn_is_left_alone() -> None:
    """The guard corrects false claims, it does not rewrite honest refusals."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    reply = "Sorry, we don't serve that. Would you like a Veg Crunch Wrap instead?"

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_no_alternatives_falls_back_to_plain_refusal() -> None:
    payload = {
        "error": "'unicorn steak' is not on the menu and there are no similar items to suggest.",
        "available_products": [],
    }
    menu_guard.record_tool_result("place_order", _mcp_envelope(payload))

    corrected, changed = menu_guard.validate_reply("I've added the unicorn steak to your order.")

    assert changed is True
    assert "unicorn steak" in corrected
    assert "choose an item from our menu" in corrected


def test_begin_turn_clears_previous_rejection() -> None:
    """A rejection must not suppress a legitimate addition on a later turn."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    menu_guard.begin_turn()
    menu_guard.record_tool_result("place_order", _mcp_envelope(_success_payload()))

    reply = "I've added the Veg Crunch Wrap to your order."
    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_empty_reply_is_untouched() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))

    assert menu_guard.validate_reply("") == ("", False)


# ---------------------------------------------------------------------------
# Streaming gate integration
# ---------------------------------------------------------------------------


def test_sentence_gate_withholds_unbacked_addition_claim() -> None:
    """Speech cannot be recalled, so the gate must not release the false claim."""
    from agentic.ordering_agent import _SentenceGate

    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("I'd like a sushi platter", spoken.append)

    gate.feed("I've added the Sushi Platter to your order. ", ["place_order"])

    assert spoken == []


def test_sentence_gate_releases_backed_addition_claim() -> None:
    from agentic.ordering_agent import _SentenceGate

    menu_guard.record_tool_result("place_order", _mcp_envelope(_success_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("I'd like a veg crunch wrap", spoken.append)

    gate.feed("I've added the Veg Crunch Wrap to your order. ", ["place_order"])

    assert spoken == ["I've added the Veg Crunch Wrap to your order."]
