"""Tests for the off-menu addition guard (agentic/menu_guard.py).

These are pure functions over reply text and recorded tool outcomes — no LLM,
no network, no database. They must stay that way.
"""

from __future__ import annotations

import json

import pytest

from plugins.kiosk import menu_guard


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


def test_garbled_reference_falls_back_to_generic_refusal() -> None:
    """Regression for a live bug: the model passed the mangled tool argument
    "ll the burgers to my cart" (from "add all the burgers to my cart"), and
    the old refusal echoed it verbatim: "we don't have ll the burgers to my
    cart on the menu at the moment" — nonsensical to a voice customer. A
    reference containing a cart/order verb, or too many words to plausibly be
    a dish name, must not be spoken back; use the generic refusal instead.
    """
    menu_guard.record_tool_result(
        "place_order", _mcp_envelope(_off_menu_payload(ref="ll the burgers to my cart"))
    )

    corrected, changed = menu_guard.validate_reply(
        "I've added ll the burgers to my cart to your order."
    )

    assert changed is True
    assert "ll the burgers to my cart" not in corrected
    assert corrected == menu_guard._REFUSAL_GENERIC_DEFAULT


def test_short_clean_reference_is_still_named_in_the_refusal() -> None:
    """A genuine dish name must still be named — the fix only guards against
    garbled/oversized references, not legitimate off-menu items."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload(ref="sushi platter")))

    corrected, changed = menu_guard.validate_reply("I've added the sushi platter to your order.")

    assert changed is True
    assert "sushi platter" in corrected


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
    from plugins.kiosk.ordering_agent import _SentenceGate

    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("I'd like a sushi platter", spoken.append)

    gate.feed("I've added the Sushi Platter to your order. ", ["place_order"])

    assert spoken == []


def test_sentence_gate_releases_backed_addition_claim() -> None:
    from plugins.kiosk.ordering_agent import _SentenceGate

    menu_guard.record_tool_result("place_order", _mcp_envelope(_success_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("I'd like a veg crunch wrap", spoken.append)

    gate.feed("I've added the Veg Crunch Wrap to your order. ", ["place_order"])

    assert spoken == ["I've added the Veg Crunch Wrap to your order."]


# ---------------------------------------------------------------------------
# Partial success: some items added, one refused, in the SAME tool call
# ---------------------------------------------------------------------------
#
# Reproduces a live defect: a customer asked to order "all pizza available"
# (4 items). kiosk-core's per-item resolution (mcp_server._resolve_items)
# added 3 real pizzas and refused one fabricated product_id in the same
# place_order call, returning both "just_added" and "unavailable_items" on
# one successful payload. action_result.classify only checks for a top-level
# "error" key, so this used to short-circuit straight past every guard: the
# model's reply implied all 4 pizzas were added (or simply announced the
# total and moved to upsell/confirm) without ever mentioning the refusal.


def _partial_success_payload() -> dict:
    """A place_order result where 3 of 4 requested items were actually added."""
    return {
        "order_id": 381,
        "status": "draft",
        "total": 667.0,
        "items": [
            {"product_id": "PIZZA-VEG-001", "product_name": "Margherita Pizza (Regular)",
             "quantity": 1, "price": 179.0},
            {"product_id": "PIZZA-VEG-002", "product_name": "Paneer Makhani Pizza (Regular)",
             "quantity": 1, "price": 219.0},
            {"product_id": "PIZZA-NV-002", "product_name": "Pepperoni-Style Chicken Pizza (Regular)",
             "quantity": 1, "price": 269.0},
        ],
        "just_added": [
            {"name": "Margherita Pizza (Regular)", "quantity": 1},
            {"name": "Paneer Makhani Pizza (Regular)", "quantity": 1},
            {"name": "Pepperoni-Style Chicken Pizza (Regular)", "quantity": 1},
        ],
        "unavailable_items": ["CHICKEN_BEEF_PIZZA"],
        "unavailable_message": (
            "'CHICKEN_BEEF_PIZZA' is not on the menu. Do not invent it and do not "
            "ask the customer to try again. Tell them it is unavailable and offer "
            "these real alternatives instead: Chicken BBQ Pizza (Regular) (249)."
        ),
        "available_products": [
            {"product_id": "PIZZA-NV-001", "name": "Chicken BBQ Pizza (Regular)", "price": 249.0},
        ],
    }


def test_partial_success_is_recorded_without_flipping_succeeded_false() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_success_payload()))

    state = menu_guard.current_state()
    assert state.succeeded is True
    assert state.has_partial_rejection is True
    assert state.partial_refs == ["CHICKEN_BEEF_PIZZA"]
    assert state.partial_alternatives == [{"name": "Chicken BBQ Pizza (Regular)", "price": 249.0}]
    # The full-failure fields are untouched by a partial success.
    assert state.rejected_refs == []
    assert state.has_rejection is False


def test_overclaiming_all_items_added_gets_disclosure_appended() -> None:
    """The reported live bug: reply implies all 4 pizzas were added."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_success_payload()))
    reply = (
        "I've added Margherita, Paneer Makhani, Pepperoni-Style Chicken, and "
        "Chicken BBQ Pizza to your order. Your total is now ₹667. "
        "Would you like anything else?"
    )

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    # The true parts of the reply are preserved, not wholesale-replaced.
    assert corrected.startswith(reply)
    assert "Chicken BBQ Pizza" in corrected  # named as the real alternative
    assert "isn't on our menu" in corrected or "not on our menu" in corrected


def test_terse_reply_that_omits_the_refusal_gets_disclosure_appended() -> None:
    """Reproduces the exact live reply: total + upsell, no mention of the miss."""
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_success_payload()))
    reply = "Total: ₹667. Would you like to add a Pepsi (330 ml) (₹59)?"

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    assert corrected.startswith(reply)
    assert "Chicken BBQ Pizza" in corrected


def test_reply_that_already_discloses_the_refusal_is_left_alone() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_success_payload()))
    reply = (
        "I've added Margherita, Paneer Makhani, and Pepperoni-Style Chicken pizzas. "
        "Chicken BBQ Pizza wasn't added — it's not on the menu. Total is ₹667."
    )

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_garbled_reference_is_not_echoed_in_the_disclosure() -> None:
    """Same safety net build_refusal already has for non-item-shaped refs."""
    payload = _partial_success_payload()
    payload["unavailable_items"] = ["ll the burgers to my cart"]
    menu_guard.record_tool_result("place_order", _mcp_envelope(payload))
    reply = "Total: ₹667."

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    assert "ll the burgers to my cart" not in corrected
    assert "one of those isn't on our menu" in corrected


def test_sequential_full_failure_then_clean_success_is_still_left_alone() -> None:
    """Pre-existing behaviour: an EARLIER call's total failure, followed by a
    LATER call's clean (non-partial) success, must not trigger a disclosure —
    only a single call that itself mixes success and rejection should.
    """
    menu_guard.record_tool_result("place_order", _mcp_envelope(_off_menu_payload()))
    menu_guard.record_tool_result("update_order", _mcp_envelope(_success_payload()))

    reply = "I've added the Veg Crunch Wrap to your order."
    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_sentence_gate_withholds_sentences_on_partial_success_turn() -> None:
    """Speech cannot be recalled — no sentence may release before the
    disclosure requirement is known to be satisfied or not.
    """
    from plugins.kiosk.ordering_agent import _SentenceGate

    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_success_payload()))
    spoken: list[str] = []
    gate = _SentenceGate("order all pizza available", spoken.append)

    gate.feed("I've added all the pizzas to your order. ", ["place_order"])

    assert spoken == []


# ---------------------------------------------------------------------------
# Implausible quantity (kiosk-core _implausible_quantity_payload)
# ---------------------------------------------------------------------------


def _quantity_payload(ref: str = "Classic Chicken Burger", qty: int = 2001) -> dict:
    """Reproduce kiosk-core's quantity rejection payload verbatim."""
    return {
        "error": (
            f"A quantity of {qty} for '{ref}' exceeds the limit of 20 per item "
            f"and was almost certainly misheard. Nothing was added to the order. "
            f"Ask the customer how many '{ref}' they would like — do not guess a "
            f"number, and do not say anything was added."
        ),
        "max_quantity": 20,
    }


def test_quantity_rejection_is_recorded_as_a_rejection() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_quantity_payload()))
    state = menu_guard.current_state()
    assert state.quantity_refused is True
    assert state.has_rejection is True
    assert state.succeeded is False


def test_quantity_refusal_does_not_claim_the_item_is_off_menu() -> None:
    # Live regression: ASR heard "one and 2,000" for "one or two", the model
    # ordered 2001 burgers and the kiosk announced a total of 338169. Capping
    # the quantity fixed the write, but the refusal then reached the customer
    # as "that isn't on our menu" — false, and unactionable, since renaming a
    # real product can never resolve a quantity problem.
    menu_guard.record_tool_result("place_order", _mcp_envelope(_quantity_payload()))
    refusal = menu_guard.build_refusal()
    assert "menu" not in refusal.lower()
    assert "how many" in refusal.lower()


def test_quantity_refusal_replaces_a_false_addition_claim() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_quantity_payload()))
    reply, corrected = menu_guard.validate_reply(
        "I've added Classic Chicken Burger to your order. Your total is now 338169."
    )
    assert corrected is True
    assert "338169" not in reply
    assert "how many" in reply.lower()


# ---------------------------------------------------------------------------
# Partial implausible-quantity rejection (kiosk-core
# _split_implausible_quantities / _quantity_rejection_payload)
# ---------------------------------------------------------------------------
#
# Live regression: "one mango lassi and 100 pepsi" — a two-item place_order
# call — refused the ENTIRE call the moment it saw the implausible 100, so
# even the perfectly clear "one mango lassi" was silently dropped. Once
# kiosk-core resolves quantity implausibility per item (like it already does
# for off-menu references), a successful call can carry BOTH ``just_added``
# and ``quantity_rejected_items`` — the same "partial success" shape as the
# off-menu case above, and it needs the analogous disclosure guarantee.


def _partial_quantity_payload() -> dict:
    """A place_order result where 1 of 2 requested items was actually added."""
    return {
        "order_id": 512,
        "status": "draft",
        "total": 89.0,
        "items": [
            {"product_id": "BEV-004", "product_name": "Mango Lassi (300 ml)",
             "quantity": 1, "price": 89.0},
        ],
        "just_added": [
            {"name": "Mango Lassi (300 ml)", "quantity": 1},
        ],
        "quantity_rejected_items": ["Pepsi (330 ml)"],
        "quantity_rejected_message": (
            "The quantity given for 'Pepsi (330 ml)' was implausibly large (over "
            "20 per item) and almost certainly misheard, so was not added. Do not "
            "guess a number for 'Pepsi (330 ml)' and do not say it was added — ask "
            "the customer how many they would like."
        ),
        "max_quantity": 20,
    }


def test_partial_quantity_rejection_is_recorded_without_flipping_succeeded_false() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_quantity_payload()))

    state = menu_guard.current_state()
    assert state.succeeded is True
    assert state.has_partial_quantity_rejection is True
    assert state.partial_quantity_refs == ["Pepsi (330 ml)"]
    # The full-failure/off-menu fields are untouched by a partial quantity skip.
    assert state.rejected_refs == []
    assert state.partial_refs == []
    assert state.quantity_refused is False


def test_reply_ignoring_the_skipped_quantity_item_gets_disclosure_appended() -> None:
    """The reported live bug: "1 mango lassi + 100 pepsi" — reply must not
    imply the pepsi was added too, nor silently omit that it wasn't.
    """
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_quantity_payload()))
    reply = "I've added Mango Lassi to your order. Your total is now ₹89."

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    # The true claim (lassi added) is preserved, not wholesale-replaced.
    assert corrected.startswith(reply)
    assert "how many" in corrected.lower() or "quantity" in corrected.lower()


def test_reply_that_already_asks_about_the_quantity_is_left_alone() -> None:
    menu_guard.record_tool_result("place_order", _mcp_envelope(_partial_quantity_payload()))
    reply = (
        "I've added your Mango Lassi. How many Pepsi did you want? "
        "I didn't quite catch the number."
    )

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is False
    assert corrected == reply


def test_garbled_quantity_reference_is_not_echoed_in_the_disclosure() -> None:
    payload = _partial_quantity_payload()
    payload["quantity_rejected_items"] = ["ll the burgers and 100 more"]
    menu_guard.record_tool_result("place_order", _mcp_envelope(payload))
    reply = "I've added Mango Lassi to your order."

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    assert "ll the burgers and 100 more" not in corrected
    assert "one of the quantities" in corrected


def test_both_off_menu_and_quantity_partial_rejections_are_both_disclosed() -> None:
    """A single call can carry both shapes at once — both must be spoken."""
    payload = _partial_success_payload()
    payload["quantity_rejected_items"] = ["Pepsi (330 ml)"]
    menu_guard.record_tool_result("place_order", _mcp_envelope(payload))
    reply = "I've added your pizzas to your order. Your total is now ₹667."

    corrected, changed = menu_guard.validate_reply(reply)

    assert changed is True
    assert "Chicken BBQ Pizza" in corrected  # off-menu disclosure
    assert "how many" in corrected.lower() or "quantity" in corrected.lower()  # quantity disclosure


def test_pronoun_refusal_does_not_claim_the_item_is_off_menu() -> None:
    # Live regression (conversation 246bfdf3): the assistant quoted "The Cafe
    # Latte (250 ml) is available for ₹109. Would you like to order it?", the
    # customer said "Yes, I would like to order it", and the model passed
    # product_id="order it". Failing to resolve a pronoun is not evidence that
    # anything is off-menu, so the refusal must not say a real, just-quoted
    # item is unavailable — it must ask which item they meant.
    menu_guard.record_tool_result(
        "place_order", _mcp_envelope(_off_menu_payload("order it"))
    )
    refusal = menu_guard.build_refusal()
    assert "menu" not in refusal.lower()
    assert "which item" in refusal.lower()


def test_pronoun_refusal_replaces_a_false_addition_claim() -> None:
    menu_guard.record_tool_result(
        "place_order", _mcp_envelope(_off_menu_payload("order it"))
    )
    cleaned, replaced = menu_guard.validate_reply(
        "I've added the Cafe Latte to your order."
    )
    assert replaced is True
    assert "menu" not in cleaned.lower()


def test_real_off_menu_item_still_says_off_menu() -> None:
    """The pronoun branch must not weaken a genuine off-menu refusal."""
    menu_guard.record_tool_result(
        "place_order", _mcp_envelope(_off_menu_payload("sushi platter"))
    )
    refusal = menu_guard.build_refusal()
    assert "menu" in refusal.lower()
