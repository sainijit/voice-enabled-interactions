"""Tests for the stale-item-reference guard (agentic/item_intent_guard.py).

Pure functions over text and tool-call args — no LLM, no network, no
database. They must stay that way.
"""

from __future__ import annotations

import pytest

from agentic import item_intent_guard as guard


class TestExtractNamedItem:
    def test_extracts_item_after_add(self):
        assert guard.extract_named_item("Go ahead and add a pizza.") == "pizza"

    def test_extracts_item_after_i_would_like(self):
        assert guard.extract_named_item("I'd like a burger") == "burger"

    def test_extracts_multi_word_item(self):
        assert (
            guard.extract_named_item("add the classic chicken burger")
            == "classic chicken burger"
        )

    def test_bare_confirmation_returns_none(self):
        for utt in ("yes", "go ahead", "sounds good", "please confirm", "correct"):
            assert guard.extract_named_item(utt) is None, utt

    def test_empty_utterance_returns_none(self):
        assert guard.extract_named_item("") is None

    def test_no_add_phrase_returns_none(self):
        assert guard.extract_named_item("What are your opening hours?") is None


class TestMismatches:
    def test_disjoint_tokens_mismatch(self):
        assert guard.mismatches("french fries", "pizza") is True

    def test_overlapping_tokens_not_mismatch(self):
        assert guard.mismatches("classic chicken burger", "chicken burger") is False

    def test_empty_reference_not_mismatch(self):
        assert guard.mismatches("", "pizza") is False

    def test_empty_named_item_not_mismatch(self):
        assert guard.mismatches("french fries", "") is False


class TestCorrectedReference:
    def test_corrects_stale_single_item(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("update_order", items, "Go ahead and add a pizza.")
        assert result == "pizza"

    def test_leaves_matching_reference_untouched(self):
        items = [{"product_id": "classic chicken burger", "quantity": 1}]
        result = guard.corrected_reference(
            "place_order", items, "add the classic chicken burger"
        )
        assert result is None

    def test_leaves_bare_confirmation_untouched(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("update_order", items, "yes go ahead")
        assert result is None

    def test_skips_multi_item_calls(self):
        items = [
            {"product_id": "french fries", "quantity": 1},
            {"product_id": "pepsi", "quantity": 1},
        ]
        result = guard.corrected_reference("update_order", items, "add a pizza too")
        assert result is None

    def test_skips_non_order_tools(self):
        items = [{"product_id": "french fries", "quantity": 1}]
        result = guard.corrected_reference("list_products", items, "add a pizza")
        assert result is None

    def test_skips_when_items_missing(self):
        assert guard.corrected_reference("place_order", None, "add a pizza") is None

    def test_skips_when_reference_empty(self):
        items = [{"product_id": "", "quantity": 1}]
        result = guard.corrected_reference("place_order", items, "add a pizza")
        assert result is None
