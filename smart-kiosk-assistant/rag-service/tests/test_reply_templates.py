"""Tests for agentic/reply_templates.py — the 2nd-LLM-call skip templates.

Pure functions over already-decoded tool JSON. No LLM, no network, no
database. They must stay that way.
"""

from __future__ import annotations

import json

from agentic import reply_templates as rt


def _envelope(payload: dict) -> dict:
    """Wrap a payload the way mcp_client.call_tool wraps a real MCP result."""
    return {"status": "success", "result": json.dumps(payload)}


class TestSpeakOrderMutation:
    def test_clean_success_names_items_and_total(self):
        payload = {
            "order_id": 12,
            "total": 268.0,
            "items": [{"product_name": "Margherita Pizza", "quantity": 1}],
            "just_added": [{"name": "Margherita Pizza", "quantity": 1}],
            "upsell_suggestions": [],
        }
        sentence = rt.speak("place_order", _envelope(payload))
        assert sentence is not None
        assert "Margherita Pizza" in sentence
        assert "₹268" in sentence

    def test_whole_rupee_total_has_no_decimal(self):
        payload = {
            "order_id": 1,
            "total": 89.0,
            "just_added": [{"name": "French Fries", "quantity": 1}],
        }
        sentence = rt.speak("update_order", _envelope(payload))
        assert "₹89." in sentence or sentence.endswith("₹89")
        assert "89.00" not in sentence

    def test_mentions_upsell_display_verbatim(self):
        payload = {
            "order_id": 1,
            "total": 89.0,
            "just_added": [{"name": "French Fries", "quantity": 1}],
            "upsell_suggestions": [{"display": "Pepsi (₹59)", "reason": "pairs well"}],
        }
        sentence = rt.speak("place_order", _envelope(payload))
        assert "Pepsi (₹59)" in sentence

    def test_multiple_items_joined_naturally(self):
        payload = {
            "order_id": 1,
            "total": 300.0,
            "just_added": [
                {"name": "Burger", "quantity": 1},
                {"name": "Fries", "quantity": 1},
                {"name": "Coke", "quantity": 1},
            ],
        }
        sentence = rt.speak("place_order", _envelope(payload))
        assert "Burger, Fries, and Coke" in sentence

    def test_rejection_defers_to_llm(self):
        payload = {
            "unavailable_items": ["sushi platter"],
            "unavailable_message": "Do not invent it...",
            "available_products": [{"product_id": "x", "name": "Wrap", "price": 149}],
        }
        assert rt.speak("place_order", _envelope(payload)) is None

    def test_needs_choice_defers_to_llm(self):
        payload = {
            "needs_choice": True,
            "category": "pizza",
            "choice_message": "Ask the customer which one...",
            "available_products": [{"product_id": "x", "name": "Margherita Pizza", "price": 179}],
        }
        assert rt.speak("place_order", _envelope(payload)) is None

    def test_partial_success_defers_to_llm(self):
        """A mix of added items and a rejection is too compound to template."""
        payload = {
            "order_id": 1,
            "total": 89.0,
            "just_added": [{"name": "French Fries", "quantity": 1}],
            "unavailable_message": "Do not invent it...",
            "available_products": [],
        }
        assert rt.speak("place_order", _envelope(payload)) is None

    def test_missing_just_added_defers_to_llm(self):
        payload = {"order_id": 1, "total": 89.0}
        assert rt.speak("place_order", _envelope(payload)) is None


class TestSpeakConfirm:
    def test_clean_confirm(self):
        payload = {"order_id": 42, "total": 178.0, "status": "confirmed"}
        sentence = rt.speak("confirm_active_order", _envelope(payload))
        assert sentence is not None
        assert "42" in sentence
        assert "₹178" in sentence

    def test_error_defers_to_llm(self):
        payload = {"error": "The cart is empty, so there is nothing to confirm."}
        assert rt.speak("confirm_active_order", _envelope(payload)) is None

    def test_non_confirmed_status_defers(self):
        payload = {"order_id": 1, "total": 10.0, "status": "draft"}
        assert rt.speak("confirm_order", _envelope(payload)) is None


class TestSpeakRemoval:
    def test_clean_removal(self):
        payload = {"removed": ["Chicken Burger"], "not_in_cart": [], "total": 89.0, "cart_empty": False}
        sentence = rt.speak("remove_from_order", _envelope(payload))
        assert sentence is not None
        assert "Chicken Burger" in sentence
        assert "₹89" in sentence

    def test_cart_now_empty(self):
        payload = {"removed": ["Fries"], "not_in_cart": [], "total": 0.0, "cart_empty": True}
        sentence = rt.speak("remove_from_order", _envelope(payload))
        assert "empty" in sentence.lower()

    def test_partial_removal_defers_to_llm(self):
        payload = {"removed": ["Fries"], "not_in_cart": ["Pizza"], "total": 59.0}
        assert rt.speak("remove_from_order", _envelope(payload)) is None

    def test_nothing_removed_defers_to_llm(self):
        payload = {
            "error": "None of those items are in the cart.",
            "cart_items": [{"product_id": "x", "name": "Fries", "quantity": 1}],
        }
        assert rt.speak("remove_from_order", _envelope(payload)) is None


class TestSpeakDispatch:
    def test_unknown_tool_defers(self):
        assert rt.speak("list_products", _envelope({"anything": 1})) is None

    def test_transport_error_defers(self):
        assert rt.speak("place_order", {"error": "Tool place_order timed out"}) is None

    def test_undecodable_result_defers(self):
        assert rt.speak("place_order", {"status": "success", "result": "not json"}) is None

    def test_speakable_tools_set(self):
        assert rt.SPEAKABLE_TOOLS == {
            "place_order", "update_order", "confirm_order",
            "confirm_active_order", "remove_from_order",
        }
