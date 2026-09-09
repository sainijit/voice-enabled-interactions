"""Directive-mode ordering turn — one LLM call that yields action *and* speech.

Why this exists
---------------
The tool-calling path has a hard latency floor that no amount of tuning
removes. Measured against this exact OVMS/Qwen3-4B-int4 deployment:

* Passing ``tools`` costs a flat ~350 ms regardless of tool count (1 tool and
  12 tools both cost ~460 ms to first content; 0 tools costs ~106 ms). It is
  not guided generation — that is already disabled. Timing the raw SSE stream
  shows the model starts decoding at ~66 ms, and the ``hermes3`` parser simply
  cannot emit a structured delta until it has buffered the
  ``<tool_call>{"name": ..., "arguments":`` preamble.
* Worse, a tool call produces **no speech at all**. The arguments must arrive
  complete before a reply template can turn them into words, so TTS cannot
  overlap generation. First audio therefore cannot start before ~1,150 ms.

This module removes both costs by having the model emit a compact directive
inline, immediately followed by the prose it should say, in a single
generation with no ``tools`` parameter:

    <act>add|Classic Chicken Burger|1</act>One classic chicken burger, got it.

Measured on the same deployment: first content at 137 ms, three speakable
prose words by ~551 ms, whole generation 670 ms — against 1,154-1,574 ms
before *any* speech existed. The approach is taken from the reference
prototype in ``kiosk-voice-lab-main`` (``pipeline/cart.py``).

The safety trade
----------------
Free-form text is weaker than a JSON schema, so this module deliberately
narrows what the model is trusted with: **it never speaks a price or a
total.** The model produces only a short, price-free confirmation. Every
number the customer hears is appended afterwards from the tool's own result
by :mod:`reply_templates`. A directive that cannot be parsed, or that names an
item the catalogue cannot resolve, returns ``None`` so the caller falls back
to the authoritative tool-calling path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Callable

import httpx

from agentic import config as agent_cfg
from agentic.action_result import unwrap
from agentic.mcp_client import call_tool

logger = logging.getLogger(__name__)

# Express cart mutations as an inline <act> directive instead of a structured
# tool call. Lives here rather than in agentic/config.py because this is a
# kiosk-plugin feature, and because plugins/ is bind-mounted — toggling it
# needs only a restart, not an image rebuild.
#
# OFF by default. Free-form text is a weaker contract than a JSON schema, so
# this path never lets the model speak a price, and hands the turn back to the
# tool-calling path whenever the directive is missing, unparseable, or names
# an item the catalogue cannot resolve.
DIRECTIVE_MODE: bool = os.getenv(
    "AGENT_DIRECTIVE_MODE", "false"
).lower() in ("true", "1", "yes")

# Read timeout for the tool-free directive completion.
LLM_TIMEOUT: float = float(os.getenv("AGENT_LLM_TIMEOUT", "120"))

# A directive is only honoured at the very start of the reply, which is where
# the prompt tells the model to put it. Anything later is prose that happens
# to contain the tag and must not silently mutate the customer's cart.
_ACT_RE = re.compile(r"^\s*<act>(.*?)</act>", re.DOTALL)

# Sentence boundary for releasing speech to TTS early. Deliberately requires
# whitespace-or-end after the terminator so "₹169." or "No. 2" do not split.
_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")

# Any further directive block after the first one, plus a trailing partial tag
# still being streamed ("<", "<act", "<act>add|Fries"). Directive markup is an
# internal wire format — it must never reach TTS or the screen.
_STRAY_ACT_RE = re.compile(
    r"<act>.*?</act>"      # a complete stray directive block
    r"|<act>.*$"           # an unclosed directive running to the end
    r"|<a?c?t?/?>?$"       # a partial opening tag still streaming
    r"|<(?:a(?:c(?:t)?)?)?$",
    re.DOTALL,
)


class _StopGeneration(Exception):
    """Raised from the delta callback to abort a stream that has gone wrong.

    Once the model starts a second directive there is nothing useful left in the
    turn — every further token is wasted latency — so the HTTP stream is closed
    rather than read to completion.
    """


def scrub_directives(text: str) -> str:
    """Strip directive markup from text destined for the customer.

    ``run_turn`` only consumes the *first* ``<act>…</act>``; everything after it
    is treated as speech. A model that emits a second directive would therefore
    have raw markup spoken aloud and printed on screen. Observed live:

        'Got it. One Classic French Fries. <act>add|Classic Chicken Burger|1</act>Got it. …'

    Args:
        text: Candidate prose.

    Returns:
        ``text`` with directive blocks and partial tags removed.
    """
    return _STRAY_ACT_RE.sub("", text).strip()


# Verbs the parser accepts. Anything else is treated as unparseable and falls
# back, rather than being guessed at.
_ADD, _REMOVE, _CONFIRM = "add", "remove", "confirm"


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Split ``text`` into complete sentences plus an unterminated remainder.

    Args:
        text: Prose accumulated so far this turn.

    Returns:
        ``(complete, remainder)`` where ``complete`` holds sentences that are
        safe to synthesise now and ``remainder`` is the partial tail still
        being generated.
    """
    parts = _SENTENCE_RE.split(text)
    if not parts:
        return [], text
    remainder = parts[-1]
    complete = [p.strip() for p in parts[:-1] if p.strip()]
    return complete, remainder


def parse_directive(raw: str) -> list[dict[str, Any]] | None:
    """Parse the ``<act>…</act>`` block into structured actions.

    Args:
        raw: The directive body, e.g. ``add|Classic Chicken Burger|1;confirm``.

    Returns:
        A list of action dicts, or ``None`` when any clause is malformed —
        a partially-understood directive is never applied, because dropping
        half of "remove the burger; add the wrap" would corrupt the order.
    """
    actions: list[dict[str, Any]] = []
    for clause in (raw or "").split(";"):
        clause = clause.strip()
        if not clause:
            continue
        fields = [f.strip() for f in clause.split("|")]
        verb = fields[0].lower()
        if verb == _CONFIRM:
            actions.append({"verb": _CONFIRM})
            continue
        if verb not in (_ADD, _REMOVE) or len(fields) < 2 or not fields[1]:
            return None
        quantity = 1
        if len(fields) >= 3 and fields[2]:
            if not fields[2].isdigit():
                return None
            quantity = int(fields[2])
            if quantity < 1:
                return None
        actions.append({"verb": verb, "name": fields[1], "quantity": quantity})
    return actions or None


_MENU_CACHE: str | None = None


async def get_menu_block() -> str:
    """Return the catalogue rendered for the prompt, fetched once per process.

    The catalogue is static configuration seeded at startup, so re-fetching it
    every turn would add an MCP round-trip (~70 ms) to the critical path for
    data that never changes.

    Returns:
        One ``Name (currency price)`` entry per line, or ``""`` when the
        catalogue could not be read (the caller then falls back).
    """
    global _MENU_CACHE
    if _MENU_CACHE is not None:
        return _MENU_CACHE
    try:
        from agentic.action_result import unwrap_any
        from agentic import domain_config

        # list_products with no category returns a category *summary*
        # ([{category, item_count}, ...]), not products — so the catalogue has
        # to be walked one category at a time. Done once per process.
        summary = unwrap_any(await call_tool("list_products", {}))
        if not isinstance(summary, list) or not summary:
            logger.warning("[DIRECTIVE] catalogue summary empty — directive mode inactive")
            return ""
        categories = [
            c["category"] for c in summary
            if isinstance(c, dict) and c.get("category")
        ]
        currency = domain_config.get_currency_symbol()
        lines: list[str] = []
        for category in categories:
            products = unwrap_any(await call_tool("list_products", {"category": category}))
            if not isinstance(products, list):
                continue
            lines.extend(
                f"{p['name']} ({currency}{p['price']:g})"
                for p in products
                if isinstance(p, dict) and p.get("name") and p.get("price") is not None
            )
        if not lines:
            logger.warning("[DIRECTIVE] no products resolved — directive mode inactive")
            return ""
        _MENU_CACHE = "\n".join(lines)
        logger.info(
            "[DIRECTIVE] menu cached (%d items across %d categories)",
            len(lines), len(categories),
        )
        return _MENU_CACHE
    except Exception:
        logger.warning("[DIRECTIVE] menu fetch failed", exc_info=True)
        return ""


def build_directive_spec(menu_block: str) -> str:
    """Build the directive instructions appended to the system prompt.

    Args:
        menu_block: A one-line-per-item catalogue rendering, used so the model
            emits names the ordering tools can actually resolve.

    Returns:
        The prompt fragment describing the directive contract.
    """
    return f"""
ORDERING FORMAT — follow this exactly.
When the customer orders, changes, or cancels something, begin your reply with ONE directive
describing ONLY what this sentence changes, then speak a SHORT confirmation of that change:
<act>add|EXACT ITEM NAME|QUANTITY</act>
<act>remove|EXACT ITEM NAME|QUANTITY</act>
<act>confirm</act>
Several changes in one sentence go in one directive separated by semicolons.
Use item names EXACTLY as they appear in the menu below.
Never speak a directive out loud. Never reply with a directive alone. Questions get no directive.

CRITICAL: never say any price, total, or amount of money. Do not offer or suggest extra items.
Those are added for you automatically.
Begin the spoken part with a TWO-WORD acknowledgement sentence ending in a full stop
("Got it." / "Sure thing." / "All set."), then ONE short confirmation sentence naming the item.

Example:
Customer: two classic chicken burgers
You: <act>add|Classic Chicken Burger|2</act>Got it. Two chicken burgers.

MENU:
{menu_block}
"""


async def _execute(
    actions: list[dict[str, Any]], user_id: str
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Apply parsed actions through the existing MCP ordering tools.

    Args:
        actions: Output of :func:`parse_directive`.
        user_id: Customer identity injected into every ordering call.

    Returns:
        ``(payload, tools_called, committed)``. ``payload`` is the final tool
        result, or ``None`` when a call failed or was rejected. ``committed``
        lists only the tools that actually changed cart state.

        The distinction matters: a ``None`` payload makes the caller fall back
        to the authoritative tool-calling path, which re-interprets the SAME
        utterance and would re-apply anything already written. "add a burger,
        remove the fries" where the add succeeds and the remove fails would
        otherwise add a second burger. When ``committed`` is non-empty the
        caller must not fall back.
    """
    payload: dict[str, Any] | None = None
    called: list[str] = []
    committed: list[str] = []
    adds = [a for a in actions if a["verb"] == _ADD]
    if adds:
        payload = unwrap(await call_tool("place_order", {
            "user_id": user_id,
            "items": [{"product_id": a["name"], "quantity": a["quantity"]} for a in adds],
        }))
        called.append("place_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        # An add the catalogue could not resolve, or one that needs the
        # customer to choose between variants, is exactly the nuance the
        # template cannot narrate. Hand the turn back.
        if payload.get("needs_choice") or payload.get("unavailable_message"):
            return None, called, committed
        if not payload.get("just_added"):
            return None, called, committed
        committed.append("place_order")

    for action in (a for a in actions if a["verb"] == _REMOVE):
        payload = unwrap(await call_tool("remove_from_order", {
            "user_id": user_id,
            "items": [{"product_id": action["name"], "quantity": action["quantity"]}],
        }))
        called.append("remove_from_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        committed.append("remove_from_order")

    if any(a["verb"] == _CONFIRM for a in actions):
        payload = unwrap(await call_tool("confirm_active_order", {"user_id": user_id}))
        called.append("confirm_active_order")
        if not isinstance(payload, dict) or "error" in payload:
            return None, called, committed
        committed.append("confirm_active_order")

    if payload is None:
        return None, called, committed
    return payload, called, committed


_FACTS_TAIL_RE = re.compile(
    r"\s*(?:Your (?:new )?total is(?: now)?\b|Would you also like\b|Would you like anything else\b).*",
    re.IGNORECASE | re.DOTALL,
)

# Spoken when some cart writes committed but a later one failed. The turn
# cannot fall back (that would re-apply the committed writes) and cannot claim
# success either, so it asks the customer to re-state rather than guessing.
_PARTIAL_FAILURE_REPLY = (
    "Sorry, I only managed part of that. Could you tell me again what you'd "
    "like, and I'll check your order?"
)


def strip_facts_tail(content: str) -> str:
    """Remove the composed, price-bearing tail from a prior assistant reply.

    A directive-mode reply is ``model prose + _facts_tail(payload)``. The client
    displays that whole string and then feeds it back as one assistant history
    turn, so from the model's point of view it looks like *it* produced the
    prices and the upsell question. On the next turn it dutifully imitates the
    pattern — emitting a total and a closing question itself — which then gets
    the real tail appended on top.

    Observed live: turn 1 generated 2 sentences in 851 ms; turn 2 generated 4 in
    1473 ms and spoke "Your total is now Rs.258" twice. The loop compounds, so
    every turn is slower and more duplicated than the last.

    Stripping the tail out of history keeps the model's in-context examples
    matching the behaviour we actually want: short prose, no money.

    Args:
        content: A prior assistant turn as displayed to the customer.

    Returns:
        Just the prose portion, or the original string when no tail is found.
    """
    stripped = _FACTS_TAIL_RE.sub("", content).strip()
    return stripped or content


async def stream_completion(
    system_prompt: str,
    message: str,
    history: list[dict[str, str]] | None,
    on_delta: Callable[[str], None],
) -> str:
    """Stream one tool-free completion from the LLM, invoking ``on_delta``.

    Args:
        system_prompt: Full system instruction including the directive spec.
        message: The customer's utterance for this turn.
        history: Prior turns as ``{"role", "content"}`` dicts.
        on_delta: Called with each raw text delta as it arrives.

    Returns:
        The complete generated text.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in history or []:
        role, content = turn.get("role"), turn.get("content")
        if role in ("user", "assistant") and content:
            if role == "assistant":
                content = strip_facts_tail(content)
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    body = {
        "model": agent_cfg.LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": agent_cfg.MAX_TOKENS,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": agent_cfg.ENABLE_THINKING},
    }
    url = agent_cfg.LLM_URL.rstrip("/") + "/chat/completions"

    out: list[str] = []
    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            try:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    text = delta.get("content") or ""
                    if text:
                        out.append(text)
                        on_delta(text)
            except _StopGeneration:
                # The callback decided the rest of the generation is unusable.
                # Returning here closes the stream and skips the wasted tokens.
                pass
    return "".join(out)


def _facts_tail(payload: dict[str, Any]) -> str:
    """Build the price-bearing part of the reply from the tool result.

    The model is forbidden from speaking money, so the total and the upsell
    offer are composed here, straight from the payload. This is why directive
    mode cannot misquote a price: the model never sees one to repeat.

    Args:
        payload: The ordering tool's result.

    Returns:
        A sentence or two of factual tail, or ``""`` when the payload carries
        no total (e.g. a removal that emptied the cart).
    """
    from agentic import domain_config
    from plugins.kiosk.reply_templates import _money

    total = payload.get("total")
    if total is None:
        return ""
    currency = domain_config.get_currency_symbol()
    out = f"Your total is now {currency}{_money(total)}."
    upsell = payload.get("upsell_suggestions") or []
    if upsell and isinstance(upsell[0], dict) and upsell[0].get("display"):
        out += f" Would you also like {upsell[0]['display']}?"
    else:
        out += " Would you like anything else?"
    return out


async def run_turn(
    message: str,
    user_id: str,
    history: list[dict[str, str]] | None,
    menu_block: str,
    base_instruction: str,
    on_safe_sentence: Callable[[str], None] | None,
) -> dict[str, Any] | None:
    """Run one ordering turn in directive mode.

    The tool call is dispatched the moment ``</act>`` is seen, so it overlaps
    the generation of the prose that follows it. Prose sentences are released
    to TTS as they complete, which is the entire point of this path: audio
    starts while the model is still generating.

    Args:
        message: The customer's utterance.
        user_id: Customer identity for the ordering tools.
        history: Prior conversation turns.
        menu_block: Catalogue rendering for the prompt.
        base_instruction: The agent's existing system instruction.
        on_safe_sentence: Callback invoked with each sentence cleared for TTS.

    Returns:
        A turn result dict compatible with the tool-calling path, or ``None``
        to signal the caller must fall back to that path.
    """


    t0 = time.monotonic()
    system_prompt = base_instruction + build_directive_spec(menu_block)

    state: dict[str, Any] = {
        "buf": "", "task": None, "spoken": [], "rest": "", "seen_act": False,
        # Sentences pulled off the stream so far (>= len(spoken), because
        # facts-bearing sentences are consumed but never spoken).
        "consumed": 0,
        # Set once the model starts narrating prices; all further prose is
        # dropped in favour of the payload-derived tail.
        "prose_closed": False,
    }

    def on_delta(_text: str) -> None:
        state["buf"] += _text
        buf: str = state["buf"]

        # Dispatch the cart mutation as soon as the directive is closed,
        # rather than waiting for the rest of the sentence to generate.
        if not state["seen_act"] and "</act>" in buf:
            state["seen_act"] = True
            match = _ACT_RE.match(buf)
            if match:
                actions = parse_directive(match.group(1))
                if actions:
                    state["task"] = asyncio.create_task(_execute(actions, user_id))
                else:
                    state["task"] = "unparseable"

        if not state["seen_act"]:
            return

        # Everything after the directive is speech.
        prose = buf.split("</act>", 1)[1] if "</act>" in buf else ""

        # A second directive means the model has started a fresh turn on its
        # own. Nothing after this point is usable, so stop reading the stream
        # instead of paying for tokens that will only be thrown away.
        if "<act>" in prose:
            if not state["prose_closed"]:
                logger.info(
                    "[DIRECTIVE] second directive emitted — closing prose and aborting stream"
                )
            state["prose_closed"] = True
            prose = prose.split("<act>", 1)[0]

        complete, remainder = _split_sentences(prose)
        state["rest"] = remainder
        new = complete[state["consumed"]:]
        state["consumed"] = len(complete)
        for sentence in new:
            # The model is told never to speak money, but in-context drift can
            # make it start doing so anyway. Anything from the first such
            # sentence onward duplicates the authoritative tail composed from
            # the tool payload, so refuse to speak it: stop accepting prose and
            # let _facts_tail() supply the numbers.
            if _FACTS_TAIL_RE.match(sentence):
                if not state["prose_closed"]:
                    logger.info(
                        "[DIRECTIVE] model began speaking facts — suppressing prose from %r",
                        sentence[:60],
                    )
                state["prose_closed"] = True
            if state["prose_closed"]:
                continue
            sentence = scrub_directives(sentence)
            if not sentence:
                continue
            state["spoken"].append(sentence)
            if on_safe_sentence is not None:
                on_safe_sentence(sentence)

        if state["prose_closed"]:
            raise _StopGeneration

    try:
        raw = await stream_completion(system_prompt, message, history, on_delta)
    except Exception:
        # The dispatch task is fire-and-forget, so an exception here would
        # otherwise leave it running (and possibly mutating the cart) while
        # chat() swallows the error and falls back — the same double-apply
        # hazard as the committed-then-failed path below. Settle it first.
        task = state.get("task")
        if isinstance(task, asyncio.Task):
            try:
                _, _, committed = await task
                if committed:
                    logger.error(
                        "[DIRECTIVE] generation failed after committing %s — "
                        "cart already changed; fallback will double-apply",
                        committed,
                    )
            except Exception:
                logger.exception("[DIRECTIVE] dispatch task failed during generation error")
        raise
    gen_ms = (time.monotonic() - t0) * 1000.0

    if not state["seen_act"] or state["task"] == "unparseable" or state["task"] is None:
        logger.info(
            "[DIRECTIVE] no usable directive (seen_act=%s) — falling back | gen=%.0fms raw=%r",
            state["seen_act"], gen_ms, raw[:120],
        )
        return None

    executed, tools_called, committed = await state["task"]
    if executed is None:
        # Falling back re-runs the SAME utterance through the tool-calling
        # path. That is only safe while nothing has been written yet — once a
        # cart mutation is committed, a fallback would apply it a second time
        # (the "add succeeded, remove failed" case). Speak an honest partial
        # failure instead and end the turn here.
        if committed:
            logger.error(
                "[DIRECTIVE] tools committed %s then a later call failed — "
                "cannot fall back without double-applying | gen=%.0fms",
                committed, gen_ms,
            )
            return {
                "reply": _PARTIAL_FAILURE_REPLY,
                "streamed": " ".join(state["spoken"]).strip(),
                "tool_calls": tools_called,
                "llm_calls": 1,
                "llm_ms": gen_ms,
                "templated": True,
                "directive": True,
            }
        logger.info("[DIRECTIVE] tool execution rejected the action — falling back")
        return None
    payload = executed

    # The model spoke a price-free confirmation; every number the customer
    # hears is appended here, verbatim from the tool result.
    spoken = " ".join(state["spoken"]).strip()
    tail = (state["rest"] or "").strip()
    if tail and not state["prose_closed"] and not _FACTS_TAIL_RE.match(tail):
        spoken = f"{spoken} {scrub_directives(tail)}".strip()
    # Last line of defence: whatever happens above, directive markup never
    # reaches TTS or the screen.
    reply = scrub_directives(f"{spoken} {_facts_tail(payload)}".strip())

    logger.info(
        "[DIRECTIVE] turn complete | gen=%.0fms sentences_streamed=%d reply=%r",
        gen_ms, len(state["spoken"]), reply[:120],
    )
    return {
        "reply": reply,
        "streamed": " ".join(state["spoken"]).strip(),
        "tool_calls": tools_called,
        "llm_calls": 1,
        "llm_ms": gen_ms,
        "templated": True,
        "directive": True,
    }
