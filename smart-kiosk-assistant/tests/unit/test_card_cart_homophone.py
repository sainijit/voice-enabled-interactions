"""Unit tests for ``_normalize_card_cart_homophone``.

Regression coverage for a real bug: "remove item in my card" reached the
agent verbatim (ASR mis-heard "cart" as "card"), and because "card" is a
legitimately in-domain word (payment method), no existing guard caught it —
the LLM fabricated a "contact customer support" refusal instead of removing
the item. This normalization corrects the homophone before the agent ever
sees the transcript, mirroring the ``_WHISPER_JUNK`` layer.
"""
import pytest

from kiosk_core.audio_session import _normalize_card_cart_homophone


@pytest.mark.tier1
class TestCardCartHomophoneNormalization:
    def test_remove_in_my_card_is_rewritten_to_cart(self):
        assert (
            _normalize_card_cart_homophone("remove item in my card")
            == "remove item in my cart"
        )

    def test_remove_from_my_card_is_rewritten_to_cart(self):
        assert (
            _normalize_card_cart_homophone("please remove the burger from my card")
            == "please remove the burger from my cart"
        )

    def test_whats_in_my_card_is_rewritten_to_cart(self):
        assert (
            _normalize_card_cart_homophone("what's in my card")
            == "what's in my cart"
        )

    def test_add_item_to_my_card_is_rewritten_to_cart(self):
        assert (
            _normalize_card_cart_homophone("add pepsi and burger to my card")
            == "add pepsi and burger to my cart"
        )

    def test_clear_and_empty_card_phrasing_is_rewritten_to_cart(self):
        assert _normalize_card_cart_homophone("clear my card please") == "clear my cart please"
        assert _normalize_card_cart_homophone("empty my card") == "empty my cart"

    def test_genuine_payment_by_card_is_left_alone(self):
        """The single most important guard: a real payment mention must
        never be corrupted into nonsense ("pay by cart")."""
        assert (
            _normalize_card_cart_homophone("i want to pay by card")
            == "i want to pay by card"
        )

    def test_swipe_my_card_is_left_alone(self):
        assert _normalize_card_cart_homophone("can i swipe my card") == "can i swipe my card"

    def test_gift_card_is_left_alone(self):
        assert (
            _normalize_card_cart_homophone("do you accept gift card")
            == "do you accept gift card"
        )

    def test_loyalty_card_is_left_alone(self):
        assert (
            _normalize_card_cart_homophone("do you have a loyalty card")
            == "do you have a loyalty card"
        )

    def test_utterance_with_no_cart_context_is_left_alone(self):
        assert (
            _normalize_card_cart_homophone("is my card empty")
            == "is my card empty"
        )

    def test_empty_string_is_returned_unchanged(self):
        assert _normalize_card_cart_homophone("") == ""

    def test_utterance_without_the_word_card_is_unaffected(self):
        assert (
            _normalize_card_cart_homophone("remove the burger from my order")
            == "remove the burger from my order"
        )
