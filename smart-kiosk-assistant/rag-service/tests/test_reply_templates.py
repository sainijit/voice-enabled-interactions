"""Tests for agentic/reply_templates.py — the 2nd-LLM-call skip templates.

Pure functions over already-decoded tool JSON. No LLM, no network, no
database. They must stay that way.
"""

from __future__ import annotations

import json

from agentic import reply_templates as rt


def _envelope(payload) -> dict:
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


class TestSpeakCatalogue:
    """Catalogue reads are the largest deterministic win — see speak_catalogue."""

    def test_browsed_category_lists_every_row_with_price(self):
        payload = [
            {"product_id": "B1", "name": "Aloo Tikki Burger", "category": "burgers", "price": 119.0},
            {"product_id": "B2", "name": "Classic Chicken Burger", "category": "burgers", "price": 169.0},
            {"product_id": "B3", "name": "Paneer Tikka Burger", "category": "burgers", "price": 159.0},
        ]
        sentence = rt.speak("list_products", _envelope(payload), "I would like to explore burgers.")
        assert sentence is not None
        # Every row must survive verbatim — omitting one is the failure the
        # system prompt calls out explicitly as WRONG.
        for name, price in (("Aloo Tikki Burger", "119"),
                            ("Classic Chicken Burger", "169"),
                            ("Paneer Tikka Burger", "159")):
            assert name in sentence
            assert f"₹{price}" in sentence
        assert "119.00" not in sentence  # TTS would say "point zero zero"
        assert sentence.endswith("Which one would you like to try?")

    def test_category_summary_is_spoken_with_counts(self):
        payload = [
            {"category": "burgers", "item_count": 7},
            {"category": "pizza", "item_count": 4},
        ]
        sentence = rt.speak("list_categories", _envelope(payload), "what do you serve")
        assert sentence is not None
        assert "burgers (7 items)" in sentence
        assert "pizza (4 items)" in sentence
        assert "explore" in sentence

    def test_unknown_category_is_refused_without_offering_unrelated_items(self):
        payload = {
            "category_not_found": True,
            "requested": "dosa",
            "message": "Do not invent a product or offer unrelated items ...",
            "categories": ["burgers", "pizza", "sides"],
        }
        sentence = rt.speak("list_products", _envelope(payload), "do you have dosa")
        assert sentence is not None
        assert "don't have dosa" in sentence
        assert "burgers" in sentence and "pizza" in sentence
        # The payload's own message is authored for the model, never spoken.
        assert "Do not invent" not in sentence

    def test_mutation_intent_defers_to_the_llm(self):
        """skip_summarization ends the turn — never cut a pending order short."""
        payload = [{"name": "Aloo Tikki Burger", "price": 119.0, "category": "burgers",
                    "product_id": "B1"}]
        for utterance in (
            "I would like to order one double chicken tower",
            "add a coke",
            "remove the fries",
            "please confirm my order",
        ):
            assert rt.speak("list_products", _envelope(payload), utterance) is None, utterance

    def test_browse_phrasings_are_templated(self):
        payload = [{"name": "Aloo Tikki Burger", "price": 119.0, "category": "burgers",
                    "product_id": "B1"}]
        for utterance in (
            "I would like to explore burgers.",
            "show me the burgers",
            "what burgers do you have",
            "which pizzas are available",
        ):
            assert rt.speak("list_products", _envelope(payload), utterance) is not None, utterance

    def test_row_without_a_price_defers(self):
        """A row we cannot describe faithfully must be narrated, not guessed."""
        payload = [{"name": "Mystery Item", "category": "burgers", "product_id": "B9"}]
        assert rt.speak("list_products", _envelope(payload), "show me burgers") is None

    def test_empty_catalogue_defers(self):
        assert rt.speak("list_products", _envelope([]), "show me burgers") is None


class TestSpeakDispatch:
    def test_unrecognised_shape_defers(self):
        assert rt.speak("list_products", _envelope({"anything": 1}), "show me") is None

    def test_unknown_tool_defers(self):
        assert rt.speak("get_upsell_suggestions", _envelope({"anything": 1})) is None

    def test_transport_error_defers(self):
        assert rt.speak("place_order", {"error": "Tool place_order timed out"}) is None

    def test_undecodable_result_defers(self):
        assert rt.speak("place_order", {"status": "success", "result": "not json"}) is None

    def test_speakable_tools_set(self):
        assert rt.SPEAKABLE_TOOLS == {
            "place_order", "update_order", "confirm_order",
            "confirm_active_order", "remove_from_order",
            "list_products", "list_categories",
        }

    def test_mutation_templates_ignore_utterance(self):
        """The browse gate must never suppress a mutation template."""
        payload = {
            "order_id": 12, "total": 219.0,
            "just_added": [{"name": "Double Chicken Tower", "quantity": 1}],
            "upsell_suggestions": [],
        }
        spoken = rt.speak("place_order", _envelope(payload),
                          "I would like to order one double chicken tower")
        assert spoken is not None and "Double Chicken Tower" in spoken
