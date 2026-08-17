"""Shared pytest setup for rag-service tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


RAG_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(RAG_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICE_ROOT))

# When plugins/ is mounted at rag-service/plugins/ (docker-compose / docker run
# with a volume), sys.path above already covers it.
# When running tests directly from the smart-kiosk-assistant tree (e.g. a
# developer running pytest from outside the container), plugins/ lives one
# directory above rag-service/ — add that parent so `from plugins.kiosk import`
# resolves correctly without needing a volume mount.
_PARENT = RAG_SERVICE_ROOT.parent
_PLUGINS_VIA_PARENT = _PARENT / "plugins"
if _PLUGINS_VIA_PARENT.is_dir() and str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))


try:
    import aiohttp  # noqa: F401
except ImportError:
    aiohttp_stub = types.ModuleType("aiohttp")

    class ClientError(Exception):
        """Minimal aiohttp.ClientError replacement for tests."""

    class ClientTimeout:
        """Minimal aiohttp.ClientTimeout replacement for tests."""

        def __init__(self, total: float | None = None) -> None:
            self.total = total

    class ClientSession:
        """Placeholder that tests monkeypatch before use."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("aiohttp is not installed; tests must monkeypatch ClientSession")

    aiohttp_stub.ClientError = ClientError
    aiohttp_stub.ClientTimeout = ClientTimeout
    aiohttp_stub.ClientSession = ClientSession
    sys.modules["aiohttp"] = aiohttp_stub


@pytest.fixture
def reset_mcp_state() -> None:
    """Reset module-level MCP registries around each MCP test."""
    from agentic import mcp_client

    mcp_client._servers.clear()
    mcp_client._tools.clear()
    yield
    mcp_client._servers.clear()
    mcp_client._tools.clear()


@pytest.fixture(autouse=True)
def reset_guard_turn_state() -> None:
    """Reset menu/removal/confirm-guard turn state before every test.

    Each guard tracks its per-turn tool outcomes in a ``contextvars.ContextVar``
    that ``OrderingAgent.chat()`` resets via ``begin_turn()`` at the start of
    every real conversation turn. Pytest does not give each test its own
    ``contextvars`` context, so a value set with ``.set()`` (e.g. any test that
    calls ``begin_turn()`` or ``record_tool_result()`` directly) otherwise
    persists into every test that runs afterwards in the same process —
    observed live: a test in ``test_menu_guard.py`` that recorded an off-menu
    rejection left ``menu_guard.current_state().has_rejection`` true for the
    rest of the suite, which made an unrelated, later test in
    ``test_removal_guard.py`` fail because ``_SentenceGate._is_safe()`` checks
    ``menu_guard.current_state()`` too. Calling ``begin_turn()`` here — the
    same reset every real turn gets — guarantees every test starts from a
    clean, unshared state regardless of run order.
    """
    from plugins.kiosk import confirm_guard, menu_guard, removal_guard

    menu_guard.begin_turn()
    removal_guard.begin_turn()
    confirm_guard.begin_turn()
