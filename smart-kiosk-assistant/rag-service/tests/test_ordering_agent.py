"""Tests for OrderingAgent orchestration with mocked ADK/LLM events."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic import ordering_agent
from agentic.mcp_client import MCPTool
from agentic.ordering_agent import OrderingAgent


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_make_mcp_callable_invokes_mcp_tool(monkeypatch) -> None:
    """The generated ADK callable delegates to mcp_client.call_tool."""
    fake_call_tool = AsyncMock(return_value={"status": "success", "result": "updated"})
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)
    mcp_tool = MCPTool(name="update_order", server="core", description="Add items")

    fn = OrderingAgent._make_mcp_callable("update_order", mcp_tool)
    result = _run(fn(order_id="ORD-1", items=[{"product_id": "coke", "quantity": 1}]))

    assert result == {"status": "success", "result": "updated"}
    fake_call_tool.assert_awaited_once_with(
        "update_order",
        {"order_id": "ORD-1", "items": [{"product_id": "coke", "quantity": 1}]},
    )
    assert fn.__name__ == "update_order"
    assert fn.__schema__["name"] == "update_order"


def test_make_mcp_callable_infers_list_type_for_optional_anyof_items(monkeypatch) -> None:
    """Regression test for a real bug: an optional ``items`` parameter (as on
    ``remove_from_order``) is emitted by FastMCP as ``anyOf: [array, null]``
    with no top-level ``"type"``. The naive lookup fell through to ``str``,
    ADK told the model ``items`` was a string, and the model then sent a
    JSON-encoded or comma-joined string that kiosk-core's Pydantic validation
    rejected outright — silently breaking every multi-item removal.
    """
    fake_call_tool = AsyncMock(return_value={"status": "success", "result": "removed"})
    monkeypatch.setattr(ordering_agent, "call_tool", fake_call_tool)
    mcp_tool = MCPTool(
        name="remove_from_order",
        server="core",
        description="Remove items",
        input_schema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "default": "anonymous"},
                "items": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "object"}},
                        {"type": "null"},
                    ],
                    "default": None,
                },
            },
            "required": [],
        },
    )

    fn = OrderingAgent._make_mcp_callable("remove_from_order", mcp_tool)

    assert fn.__annotations__["items"] is list
    assert fn.__annotations__["user_id"] is str

    _run(fn(user_id="anonymous", items=[{"product_id": "coke", "quantity": 1}]))
    fake_call_tool.assert_awaited_once_with(
        "remove_from_order",
        {"user_id": "anonymous", "items": [{"product_id": "coke", "quantity": 1}]},
    )


@pytest.mark.parametrize(
    ("message", "expected_tools"),
    [
        ("Do you have vegan burgers?", ["knowledge_lookup"]),
        ("Place an order for a paneer burger", ["list_products", "place_order", "get_upsell_suggestions"]),
        ("Add a coke to my order", ["update_order", "get_upsell_suggestions"]),
        ("Confirm my order", ["confirm_order"]),
    ],
)
def test_chat_records_scripted_adk_tool_calls(message: str, expected_tools: list[str]) -> None:
    """Scripted ADK runner events model OVMS tool-call choices without OVMS."""
    pytest.importorskip("google.adk")

    agent = OrderingAgent()
    agent._bootstrapped = True
    agent._agent = SimpleNamespace(name="kiosk_ordering_agent")
    agent._session_service = _FakeSessionService()
    agent._runner = _FakeRunner()

    result = _run(agent.chat(message=message, session_id="session-1", user_id="user-1"))

    assert result["tool_calls"] == expected_tools
    assert "reply" in result


class _FakeSessionService:
    """Session service that makes _ensure_session a no-op."""

    async def get_session(self, app_name: str, user_id: str, session_id: str) -> object:
        return object()


class _FakeRunner:
    """ADK Runner replacement that yields scripted tool-call events."""

    async def run_async(
        self,
        user_id: str,
        session_id: str,
        new_message: object,
    ) -> Any:
        text = new_message.parts[0].text.lower()
        if "vegan" in text:
            tools = ["knowledge_lookup"]
            reply = "Yes, we have vegan options."
        elif "place" in text:
            tools = ["list_products", "place_order", "get_upsell_suggestions"]
            reply = "I created a draft order."
        elif "add a coke" in text:
            tools = ["update_order", "get_upsell_suggestions"]
            reply = "I added a coke."
        elif "confirm" in text:
            tools = ["confirm_order"]
            reply = "Your order is confirmed!"
        else:
            tools = []
            reply = "How can I help?"

        for tool_name in tools:
            yield SimpleNamespace(tool_call=SimpleNamespace(name=tool_name), content=None)
        yield SimpleNamespace(tool_call=None, content=SimpleNamespace(parts=[SimpleNamespace(text=reply)]))


class TestStripKnowledgeMarkers:
    """Pre-grounding delimiters must never reach TTS."""

    def test_parroted_block_is_unwrapped_keeping_content(self):
        reply = ("[knowledge] The restaurant's name is QuickBite Express. "
                 "The operating hours are Mon-Thu 8 AM-11 PM. [/knowledge]")
        out = ordering_agent._strip_knowledge_markers(reply)
        assert "[knowledge]" not in out
        assert "[/knowledge]" not in out
        assert "QuickBite Express" in out

    def test_case_insensitive(self):
        assert "[" not in ordering_agent._strip_knowledge_markers("[KNOWLEDGE] Hi [/Knowledge]")

    def test_clean_reply_untouched(self):
        reply = "We open at 8 AM."
        assert ordering_agent._strip_knowledge_markers(reply) == reply

    def test_marker_only_reply_falls_back_to_original(self):
        # Nothing speakable would remain; never return empty.
        assert ordering_agent._strip_knowledge_markers("[knowledge][/knowledge]").strip() != ""

    def test_gate_withholds_sentence_containing_marker(self):
        gate = ordering_agent._SentenceGate(message="hi", emit=None)
        assert gate._is_safe("[knowledge] We open at 8 AM.", ["list_products"]) is False
