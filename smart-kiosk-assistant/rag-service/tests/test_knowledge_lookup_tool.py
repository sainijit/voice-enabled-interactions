"""Tests for the local knowledge lookup ADK tool."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

from agentic.tools.knowledge_lookup_tool import knowledge_lookup


def _run(coro: Any) -> Any:
    """Run an async test helper without depending on pytest-asyncio."""
    return asyncio.run(coro)


def test_knowledge_lookup_returns_streamed_rag_context(monkeypatch) -> None:
    """knowledge_lookup formats retrieved records from the shared RAG pipeline."""
    from agentic.tools import knowledge_lookup_tool as k
    k.reset_pinned_context()

    record = _Record("We are open from 9am to 9pm.", score=1.0)
    fake_pipeline = _FakeRetrievalPipeline(root_text="", extra_records=[record])
    _install_fake_pipeline(monkeypatch, fake_pipeline)

    answer = _run(knowledge_lookup("What are your opening hours?"))

    assert fake_pipeline.retrieve_calls == ["What are your opening hours?"]
    assert "We are open from 9am to 9pm." in answer


class _Record:
    """Minimal stand-in for a retrieval record."""

    def __init__(self, content: str, score: float = -1.0) -> None:
        self.content = content
        self.score = score


class _FakeRetrievalPipeline:
    """Fake pipeline exposing the current retrieve()/iter_documents() API."""

    def __init__(self, root_text: str, extra_records: list) -> None:
        self._root_text = root_text
        self._extra_records = extra_records
        self.retrieve_calls: list[str] = []

    def iter_documents(self):
        return [self._root_text]

    def retrieve(self, question: str):
        self.retrieve_calls.append(question)
        return list(self._extra_records)


def _install_fake_pipeline(monkeypatch, pipeline) -> None:
    module = types.ModuleType("pipeline")
    module.get_shared_pipeline = lambda: pipeline
    monkeypatch.setitem(sys.modules, "pipeline", module)


class TestRootOnlyQuerySkipsRetrieval:
    """Root-covered questions must not pull in unrelated retrieved chunks."""

    ROOT_TEXT = (
        "[Context: QuickBite Express — Restaurant Knowledge Base] "
        "Brand Name: QuickBite Express. Hours: Mon-Thu 8 AM-11 PM, "
        "Fri-Sat 8 AM-12 AM, Sun 9 AM-11 PM."
    )
    UNRELATED = _Record(
        "[Context: QuickBite Express — Restaurant Knowledge Base > MENU 9: "
        "Breakfast Menu > Veg Breakfast] Poha - Rs89, Upma - Rs89",
        score=-6.8,
    )

    def setup_method(self):
        from agentic.tools import knowledge_lookup_tool as k
        k.reset_pinned_context()

    def test_hours_question_skips_supplementary_retrieval(self, monkeypatch):
        from agentic.tools.knowledge_lookup_tool import knowledge_lookup

        pipeline = _FakeRetrievalPipeline(self.ROOT_TEXT, [self.UNRELATED])
        _install_fake_pipeline(monkeypatch, pipeline)

        result = _run(knowledge_lookup(
            "Can you tell me the restaurant name and what are the timings?"
        ))

        assert "Poha" not in result
        assert "QuickBite Express" in result
        assert pipeline.retrieve_calls == []

    def test_parking_question_skips_supplementary_retrieval(self, monkeypatch):
        from agentic.tools.knowledge_lookup_tool import knowledge_lookup

        pipeline = _FakeRetrievalPipeline(self.ROOT_TEXT, [self.UNRELATED])
        _install_fake_pipeline(monkeypatch, pipeline)

        result = _run(knowledge_lookup("Do you have parking?"))

        assert "Poha" not in result
        assert pipeline.retrieve_calls == []

    def test_non_root_question_still_retrieves(self, monkeypatch):
        """A genuine menu-item question must still use full retrieval."""
        from agentic.tools.knowledge_lookup_tool import knowledge_lookup

        allergen_record = _Record(
            "[Context: ... > Dietary Tag System] Paneer Tikka Burger is "
            "vegetarian and contains dairy.",
            score=1.2,
        )
        pipeline = _FakeRetrievalPipeline(self.ROOT_TEXT, [allergen_record])
        _install_fake_pipeline(monkeypatch, pipeline)

        result = _run(knowledge_lookup("Is the Paneer Tikka Burger vegetarian?"))

        assert pipeline.retrieve_calls == ["Is the Paneer Tikka Burger vegetarian?"]
        assert "vegetarian" in result.lower()
