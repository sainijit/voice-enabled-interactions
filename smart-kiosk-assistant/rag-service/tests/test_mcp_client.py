"""Tests for the agentic MCP client without a live MCP server."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from types import SimpleNamespace

from agentic import mcp_client


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def _install_fake_mcp(monkeypatch, state: dict[str, Any]) -> None:
    """Install fake mcp modules used by mcp_client imports."""

    class _FakeStreamableHTTP:
        def __init__(self, url: str) -> None:
            self._url = url

        async def __aenter__(self):
            state["stream_urls"].append(self._url)
            return ("read", "write", None)

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

    class _FakeSession:
        def __init__(self, read: str, write: str) -> None:
            self.read = read
            self.write = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def initialize(self) -> None:
            state["initialized"] += 1

        async def list_tools(self):
            return SimpleNamespace(tools=state["tools"])

        async def call_tool(self, tool_name: str, arguments: dict[str, Any]):
            state["tool_calls"].append((tool_name, arguments))
            if state.get("call_tool_error") is not None:
                raise state["call_tool_error"]
            delay = state.get("call_tool_delay", 0.0)
            if delay:
                await asyncio.sleep(delay)
            return SimpleNamespace(content=state["result_content"])

    mcp_mod = types.ModuleType("mcp")
    mcp_mod.ClientSession = _FakeSession
    client_mod = types.ModuleType("mcp.client")
    stream_mod = types.ModuleType("mcp.client.streamable_http")
    stream_mod.streamablehttp_client = lambda url: _FakeStreamableHTTP(url)

    monkeypatch.setitem(sys.modules, "mcp", mcp_mod)
    monkeypatch.setitem(sys.modules, "mcp.client", client_mod)
    monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", stream_mod)


def test_discover_tools_lists_mcp_tools(monkeypatch, reset_mcp_state) -> None:
    """discover_tools converts MCP list_tools result into MCPTool objects."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [
            SimpleNamespace(
                name="place_order",
                description="Create a draft order",
                inputSchema={"type": "object", "properties": {"items": {"type": "array"}}},
            )
        ],
        "result_content": [],
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    tools = _run(mcp_client.discover_tools(server))

    assert state["stream_urls"] == ["http://mcp.local"]
    assert state["initialized"] == 1
    assert [tool.name for tool in tools] == ["place_order"]
    assert tools[0].server == "core"
    assert tools[0].to_function_schema()["parameters"]["properties"]["items"]["type"] == "array"


def test_call_tool_invokes_registered_tool_with_arguments(monkeypatch, reset_mcp_state) -> None:
    """call_tool uses the registered server and forwards args unchanged."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [SimpleNamespace(text='{"order_id":"ORD-1"}')],
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["place_order"] = mcp_client.MCPTool(name="place_order", server="core")

    result = _run(mcp_client.call_tool("place_order", {"user_id": "u1", "items": []}))

    assert result == {"status": "success", "result": '{"order_id":"ORD-1"}'}
    assert state["stream_urls"] == ["http://mcp.local"]
    assert state["initialized"] == 1
    assert state["tool_calls"] == [("place_order", {"user_id": "u1", "items": []})]


def test_call_tool_times_out(monkeypatch, reset_mcp_state) -> None:
    """call_tool timeout returns an error payload instead of raising."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [SimpleNamespace(text="late")],
        "call_tool_delay": 0.05,
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local", timeout=0.01)
    mcp_client._servers["core"] = server
    mcp_client._tools["confirm_order"] = mcp_client.MCPTool(name="confirm_order", server="core")

    assert _run(mcp_client.call_tool("confirm_order", {"order_id": "ORD-1"})) == {
        "error": "Tool confirm_order timed out"
    }


def test_call_tool_returns_error_payload_on_transport_failure(monkeypatch, reset_mcp_state) -> None:
    """MCP transport errors are returned to the agent, not raised."""
    state = {
        "stream_urls": [],
        "initialized": 0,
        "tool_calls": [],
        "tools": [],
        "result_content": [],
        "call_tool_error": RuntimeError("MCP unavailable"),
    }
    _install_fake_mcp(monkeypatch, state)

    server = mcp_client.MCPServerConfig(name="core", url="http://mcp.local")
    mcp_client._servers["core"] = server
    mcp_client._tools["confirm_order"] = mcp_client.MCPTool(name="confirm_order", server="core")

    assert _run(mcp_client.call_tool("confirm_order", {"order_id": "ORD-1"})) == {
        "error": "MCP unavailable"
    }
