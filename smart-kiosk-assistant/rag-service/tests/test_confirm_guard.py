"""Tests for the unbacked order-confirmation claim guard (agentic/confirm_guard.py).

These are pure functions over reply text and recorded tool outcomes — no LLM,
no network, no database. They must stay that way.
"""

from __future__ import annotations

import json

import pytest

from agentic import confirm_guard


def _mcp_envelope(payload: dict) -> dict:
    """Wrap a tool payload the way mcp_client.call_tool returns it."""
    return {"status": "success", "result": json.dumps(payload)}


def _success_payload(order_id: int = 87) -> dict:
    return {
        "order_id": order_id,
        "status": "confirmed",
        "total": 268.0,
        "items": [
            {"product_id": "BURGER-002", "product_name": "Spicy Chicken Crunch Burger",
             "quantity": 1, "price": 179.0},
        ],
    }


def _not_found_payload(order_id: int = 12345) -> dict:
    """Reproduce kiosk-core's "Order not found" error for a hallucinated id."""
    return {"error": f"Order not found: {order_id}"}


@pytest.fixture(autouse=True)
def fresh_turn():
    """Every test starts from a clean per-turn state."""
    confirm_guard.begin_turn()


# ---------------------------------------------------------------------------
# Result classification
# ---------------------------------------------------------------------------


def test_successful_confirm_is_recorded() -> None:
    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_success_payload()))

    state = confirm_guard.current_state()
    assert state.attempted is True
    assert state.succeeded is True


def test_successful_confirm_active_order_is_recorded() -> None:
    confirm_guard.record_tool_result("confirm_active_order", _mcp_envelope(_success_payload()))

    assert confirm_guard.current_state().succeeded is True


def test_hallucinated_order_id_failure_is_recorded_as_unsuccessful() -> None:
    """The exact live bug: confirm_order(12345) — 'Order not found', nothing confirmed."""
    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))

    state = confirm_guard.current_state()
    assert state.attempted is True
    assert state.succeeded is False
    assert "12345" in state.error_message


def test_non_confirm_tools_are_ignored() -> None:
    confirm_guard.record_tool_result("update_order", _mcp_envelope({"error": "boom"}))

    state = confirm_guard.current_state()
    assert state.attempted is False
    assert state.succeeded is False


def test_transport_error_is_not_treated_as_success() -> None:
    confirm_guard.record_tool_result("confirm_order", {"error": "Tool confirm_order timed out"})

    state = confirm_guard.current_state()
    assert state.attempted is True
    assert state.succeeded is False


def test_undecodable_result_is_not_a_success() -> None:
    confirm_guard.record_tool_result(
        "confirm_order", {"status": "success", "result": "<not json>"}
    )

    state = confirm_guard.current_state()
    assert state.attempted is True
    assert state.succeeded is False


def test_unexpected_shape_is_not_a_success() -> None:
    """No error, but also no 'confirmed' status/order_id — treat as unsuccessful."""
    confirm_guard.record_tool_result(
        "confirm_order", _mcp_envelope({"status": "draft", "order_id": 87})
    )

    assert confirm_guard.current_state().succeeded is False


def test_begin_turn_clears_previous_state() -> None:
    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))
    confirm_guard.begin_turn()

    state = confirm_guard.current_state()
    assert state.attempted is False
    assert state.succeeded is False
    assert state.error_message == ""


# ---------------------------------------------------------------------------
# build_refusal
# ---------------------------------------------------------------------------


def test_build_refusal_never_echoes_raw_tool_error() -> None:
    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))

    refusal = confirm_guard.build_refusal(confirm_guard.current_state())

    assert "12345" not in refusal
    assert "Order not found" not in refusal


# ---------------------------------------------------------------------------
# _strip_false_confirmation integration — the behaviour that was reported broken
# ---------------------------------------------------------------------------


def test_false_confirmation_after_hallucinated_id_is_replaced_with_refusal() -> None:
    """The exact live repro: confirm_order(12345) fails, reply still claims success."""
    from agentic.ordering_agent import _strip_false_confirmation

    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))
    reply = "Your order is confirmed!"

    cleaned, changed = _strip_false_confirmation(reply, ["confirm_order"])

    assert changed is True
    assert "confirmed" not in cleaned.lower()
    assert cleaned == confirm_guard.build_refusal()


def test_false_confirmation_with_no_attempt_uses_generic_tail() -> None:
    """No confirm tool ran at all this turn — different wording from a failed attempt."""
    from agentic.ordering_agent import _strip_false_confirmation, _UNCONFIRMED_TAIL

    reply = "Your order is confirmed!"

    cleaned, changed = _strip_false_confirmation(reply, [])

    assert changed is True
    assert cleaned == _UNCONFIRMED_TAIL


def test_genuine_confirmation_is_left_alone() -> None:
    from agentic.ordering_agent import _strip_false_confirmation

    confirm_guard.record_tool_result("confirm_active_order", _mcp_envelope(_success_payload()))
    reply = "Your order is confirmed! Thank you for ordering with us."

    cleaned, changed = _strip_false_confirmation(reply, ["confirm_active_order"])

    assert changed is False
    assert cleaned == reply


def test_confirm_invoked_but_failed_is_not_treated_as_confirm_tool_presence() -> None:
    """Regression guard: mere tool-name presence in tool_calls must not bypass the check."""
    from agentic.ordering_agent import _strip_false_confirmation

    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))
    reply = "All set — your order is confirmed and on its way!"

    # tool_calls still lists confirm_order (it *was* invoked), but the result
    # failed — the old invocation-only check would have let this through.
    cleaned, changed = _strip_false_confirmation(reply, ["confirm_order"])

    assert changed is True
    assert "confirmed" not in cleaned.lower()


# ---------------------------------------------------------------------------
# Streaming gate integration
# ---------------------------------------------------------------------------


def test_sentence_gate_withholds_unbacked_confirmation_claim() -> None:
    """Speech cannot be recalled, so the gate must not release the false claim."""
    from agentic.ordering_agent import _SentenceGate

    confirm_guard.record_tool_result("confirm_order", _mcp_envelope(_not_found_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("confirm my order", spoken.append)

    gate.feed("Your order is confirmed! ", ["confirm_order"])

    assert spoken == []


def test_sentence_gate_releases_backed_confirmation_claim() -> None:
    """Condition (b) unconditionally withholds every sentence on a confirm-intent
    utterance (the whole reply may still be replaced by _force_confirm()), so this
    uses a neutral message to isolate condition (c2) — a genuinely successful
    confirm tool result must not be gated on its own."""
    from agentic.ordering_agent import _SentenceGate

    confirm_guard.record_tool_result("confirm_active_order", _mcp_envelope(_success_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("yes go ahead", spoken.append)

    gate.feed("Your order is confirmed! ", ["confirm_active_order"])

    assert spoken == ["Your order is confirmed!"]
