"""Tests for the local knowledge lookup ADK tool."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from types import SimpleNamespace

from agentic.tools import knowledge_lookup_tool
from agentic.tools.knowledge_lookup_tool import knowledge_lookup


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_knowledge_lookup_returns_streamed_rag_context(monkeypatch) -> None:
    """knowledge_lookup formats retrieved context from the shared RAG pipeline."""

    monkeypatch.setattr(knowledge_lookup_tool, "_PIN_ROOT_SECTION", False)
    knowledge_lookup_tool.reset_pinned_context()

    class FakePipeline:
        """Fake RAG pipeline with deterministic retrieval output."""

        def iter_documents(self) -> list[str]:
            return []

        def retrieve(self, question: str):
            assert question == "What are your opening hours?"
            return [SimpleNamespace(content="We are open from 9am to 9pm.")]

    fake_pipeline = FakePipeline()
    pipeline_module = types.ModuleType("pipeline")
    pipeline_module.get_shared_pipeline = lambda: fake_pipeline
    monkeypatch.setitem(sys.modules, "pipeline", pipeline_module)

    answer = _run(knowledge_lookup("What are your opening hours?"))

    assert "We are open from 9am to 9pm." in answer
