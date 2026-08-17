"""Tests for the cart-duplicate/unconfirmed-upsell guard
(agentic/cart_state_guard.py).

Pure functions over item lists, cart snapshots, and text — no LLM, no
network, no database. They must stay that way.

The core regression this file guards against, observed live::

    assistant: "I've added Classic Chicken Burger to your order. Your total
                is now 169. Would you also like Classic French Fries
                (Regular) (89)?"
    customer:  "I was wondering if you can add one dosa also"
    tool call: place_order(items=[classic_chicken_burger, classic_french_fries, dosa])

The customer named only "dosa" (correctly refused as off-menu). The burger
was already in the cart and got silently re-added (duplicate quantity, real
money); the fries were only ever a still-open upsell offer, never accepted.
"""

from __future__ import annotations

from plugins.kiosk import cart_state_guard as guard

_BURGER_ITEM = {"product_id": "classic_chicken_burger", "quantity": 1}
_FRIES_ITEM = {"product_id": "classic_french_fries", "quantity": 1}
_DOSA_ITEM = {"product_id": "dosa", "quantity": 1}

_KNOWN_CART = [
    {"product_id": "BURGER-NV-001", "product_name": "Classic Chicken Burger", "quantity": 1}
]
_PENDING_UPSELL = {"product_id": "SIDE-001", "name": "Classic French Fries (Regular)"}


class TestFilterStaleAndUnconfirmedItems:
    def test_live_bug_drops_stale_burger_and_unconfirmed_fries_keeps_dosa(self):
        """Reproduces the exact live conversation that motivated this guard."""
        items = [_BURGER_ITEM, _FRIES_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items,
            "I was wondering if you can add one dosa also",
            _KNOWN_CART,
            _PENDING_UPSELL,
        )
        assert result == [_DOSA_ITEM]

    def test_single_item_call_is_never_touched(self):
        """item_intent_guard owns single-item calls; nothing to safely drop
        here without discarding the customer's only request."""
        items = [_BURGER_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add one dosa also", _KNOWN_CART, _PENDING_UPSELL
        )
        assert result == items

    def test_bare_confirmation_leaves_multi_item_call_untouched(self):
        """A bare 'yes' names nothing this turn, so the model's own
        multi-item resolution must be trusted — this is what protects a
        legitimate 'yes, add another burger' follow-up from being stripped."""
        items = [_BURGER_ITEM, _FRIES_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "yes please", _KNOWN_CART, _PENDING_UPSELL
        )
        assert result == items

    def test_genuine_compound_request_keeps_both_new_items(self):
        """'a burger and a dosa' should not be treated as a stale restatement
        just because 'burger' also happens to be in the cart already — the
        customer explicitly named it this turn."""
        items = [{"product_id": "burger"}, {"product_id": "dosa"}]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add a burger and a dosa", [], None
        )
        assert result == items

    def test_explicit_reorder_of_cart_item_by_name_is_kept(self):
        """Explicitly naming an item already in the cart ('one more classic
        chicken burger') must not be dropped as a stale duplicate."""
        items = [_BURGER_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items,
            "add one more classic chicken burger and a dosa",
            _KNOWN_CART,
            None,
        )
        assert result == items

    def test_shared_generic_word_does_not_cause_a_false_cart_match(self):
        """'Classic' appears in both 'Classic Chicken Burger' (in cart) and
        'Classic French Fries' (not in cart, only offered as upsell) — a
        single shared adjective must not misclassify the fries as a cart
        duplicate instead of an unconfirmed upsell (or keep it wrongly)."""
        items = [_FRIES_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add one dosa also", _KNOWN_CART, _PENDING_UPSELL
        )
        # Fries dropped (unconfirmed upsell), dosa kept (named this turn).
        assert result == [_DOSA_ITEM]

    def test_no_pending_upsell_still_drops_stale_cart_duplicate(self):
        items = [_BURGER_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add one dosa also", _KNOWN_CART, None
        )
        assert result == [_DOSA_ITEM]

    def test_empty_known_cart_and_no_upsell_keeps_everything_named_or_unmatched(self):
        items = [_BURGER_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add one dosa also", [], None
        )
        assert result == items

    def test_non_dict_items_are_left_alone(self):
        items = ["not-a-dict", _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add one dosa also", _KNOWN_CART, _PENDING_UPSELL
        )
        assert result == items

    def test_upsell_explicitly_named_is_accepted_and_kept(self):
        """When the customer's named phrase actually matches the pending
        upsell item, it must be kept even though it wasn't previously in the
        cart — this is a genuine acceptance, not an auto-accept."""
        items = [_FRIES_ITEM]
        # Single-item call: guard doesn't touch it (item_intent_guard's turf).
        result = guard.filter_stale_and_unconfirmed_items(
            items, "yes add the french fries too", _KNOWN_CART, _PENDING_UPSELL
        )
        assert result == items

    def test_multi_item_with_explicit_upsell_acceptance_keeps_both(self):
        items = [_FRIES_ITEM, _DOSA_ITEM]
        result = guard.filter_stale_and_unconfirmed_items(
            items, "add french fries and a dosa", _KNOWN_CART, _PENDING_UPSELL
        )
        assert result == items
