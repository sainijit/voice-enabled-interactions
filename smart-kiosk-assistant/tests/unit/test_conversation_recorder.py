"""Unit tests for kiosk_core.conversation_recorder.

tier1 (no Docker/ML required) — pure file I/O against a temp directory,
gated entirely by config.CONVERSATION_LOGGING_ENABLED.
"""
import json

import pytest

from kiosk_core import config, conversation_recorder


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch, tmp_path):
    """Ensure every test starts from a clean, isolated log directory."""
    monkeypatch.setattr(config, "CONVERSATION_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(config, "CONVERSATION_LOGGING_ENABLED", True)
    yield tmp_path


@pytest.mark.tier1
class TestRecordTurn:
    def test_disabled_flag_writes_nothing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "CONVERSATION_LOGGING_ENABLED", False)
        conversation_recorder.record_turn(
            conversation_id="conv-1",
            turn_id="turn-1",
            user_text="hello",
            assistant_text="hi there",
        )
        assert list(tmp_path.iterdir()) == []

    def test_enabled_creates_one_file_per_conversation(self, tmp_path):
        conversation_recorder.record_turn(
            conversation_id="conv-1",
            turn_id="turn-1",
            user_text="I'd like a burger",
            assistant_text="One burger, coming right up.",
            end_reason="completed",
        )
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "conv-1.jsonl"

    def test_multiple_turns_same_conversation_append_to_one_file(self, tmp_path):
        conversation_recorder.record_turn(
            conversation_id="conv-1", turn_id="turn-1",
            user_text="hi", assistant_text="hello",
        )
        conversation_recorder.record_turn(
            conversation_id="conv-1", turn_id="turn-2",
            user_text="a coffee please", assistant_text="one coffee added",
        )
        lines = (tmp_path / "conv-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        second = json.loads(lines[1])
        assert first["turn_id"] == "turn-1"
        assert second["turn_id"] == "turn-2"
        assert first["conversation_id"] == "conv-1"

    def test_different_conversations_get_separate_files(self, tmp_path):
        conversation_recorder.record_turn(
            conversation_id="conv-A", turn_id="t1", user_text="hi", assistant_text="hey"
        )
        conversation_recorder.record_turn(
            conversation_id="conv-B", turn_id="t2", user_text="yo", assistant_text="hiya"
        )
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["conv-A.jsonl", "conv-B.jsonl"]

    def test_record_contains_expected_fields(self, tmp_path):
        conversation_recorder.record_turn(
            conversation_id="conv-1",
            turn_id="turn-1",
            user_text="what's on the menu",
            assistant_text="We have burgers, fries and drinks.",
            end_reason="completed",
        )
        record = json.loads((tmp_path / "conv-1.jsonl").read_text().strip())
        assert record["user"] == "what's on the menu"
        assert record["assistant"] == "We have burgers, fries and drinks."
        assert record["end_reason"] == "completed"
        assert record["turn_id"] == "turn-1"
        assert "timestamp" in record

    def test_conversation_id_is_sanitized_against_path_traversal(self, tmp_path):
        conversation_recorder.record_turn(
            conversation_id="../../etc/passwd",
            turn_id="t1",
            user_text="x",
            assistant_text="y",
        )
        # Must stay confined to tmp_path -- no file created outside it, and
        # the dangerous characters are replaced rather than used verbatim.
        files = list(tmp_path.rglob("*.jsonl"))
        assert len(files) == 1
        assert files[0].parent == tmp_path
        assert ".." not in files[0].name

    def test_missing_transcript_is_still_recorded(self, tmp_path):
        """Covers the greeting-only turn (no user speech at all)."""
        conversation_recorder.record_turn(
            conversation_id="conv-1",
            turn_id="turn-1",
            user_text="",
            assistant_text="How can I help you?",
            end_reason="completed",
        )
        record = json.loads((tmp_path / "conv-1.jsonl").read_text().strip())
        assert record["user"] == ""
        assert record["assistant"] == "How can I help you?"

    def test_recording_failure_is_swallowed_not_raised(self, tmp_path, monkeypatch):
        def _boom():
            raise OSError("disk full")

        monkeypatch.setattr(conversation_recorder, "_log_dir", _boom)
        # Must not raise -- a broken recorder can never break a live voice turn.
        conversation_recorder.record_turn(
            conversation_id="conv-1", turn_id="t1", user_text="x", assistant_text="y"
        )
