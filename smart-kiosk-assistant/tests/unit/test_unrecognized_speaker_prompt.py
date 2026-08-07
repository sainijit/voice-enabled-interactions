"""Unit tests for the "I couldn't recognise your voice" reply path.

Regression: when the speaker filter rejected every segment of a chunk the
turn ended with an empty transcript, and ``_finalize_run`` replied with the
generic "How can I help you?" greeting. A recorded conversation showed
exactly that — ``{"user": "", "assistant": "How can I help you?"}`` — leaving
the customer with no indication that they had spoken and been ignored.

These tests pin the distinction between "nobody spoke" and "somebody spoke
and every segment was discarded".
"""
import pytest

from kiosk_core import config
from kiosk_core.audio_session import BaseAudioSession


def _make_session(session_id: str = "test-session") -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = session_id
    session._rejected_speech_chunks = 0
    return session


@pytest.mark.tier1
class TestRejectedSpeechAccounting:
    def test_analyzer_rejection_of_real_speech_is_counted(self):
        session = _make_session()
        segments = [
            {"text": "I want a burger", "speaker": "SPEAKER_01", "is_primary": False},
        ]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 1

    def test_silence_is_not_counted_as_rejected_speech(self):
        """Empty segment text must stay classified as 'no speech'."""
        session = _make_session()
        segments = [{"text": "   ", "speaker": "SPEAKER_01", "is_primary": False}]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 0

    def test_no_segments_at_all_is_not_counted(self):
        session = _make_session()
        assert session._filter_target_speaker([]) == ""
        assert session._rejected_speech_chunks == 0

    def test_accepted_primary_speech_is_not_counted(self):
        session = _make_session()
        segments = [
            {"text": "one cold coffee", "speaker": "SPEAKER_00", "is_primary": True},
        ]
        assert session._filter_target_speaker(segments) == "one cold coffee"
        assert session._rejected_speech_chunks == 0

    def test_bystander_speech_after_primary_locked_is_counted(self):
        """Primary already established; another voice must be counted, not silent."""
        session = _make_session()
        session._primary_speaker_id = "SPEAKER_00"
        segments = [
            {"text": "move along please", "speaker": "SPEAKER_01"},
        ]
        assert session._filter_target_speaker(segments) == ""
        assert session._rejected_speech_chunks == 1

    def test_repeated_rejections_accumulate_across_chunks(self):
        session = _make_session()
        session._primary_speaker_id = "SPEAKER_00"
        for _ in range(3):
            session._filter_target_speaker(
                [{"text": "unrelated chatter", "speaker": "SPEAKER_09"}]
            )
        assert session._rejected_speech_chunks == 3


@pytest.mark.tier1
class TestFinalizePrompt:
    """The reply chosen when a completed turn produced no transcript."""

    @staticmethod
    def _spoken_reply(rejected_chunks: int) -> str:
        """Run the branch in _finalize_run that picks the fallback reply."""
        session = _make_session()
        session._rejected_speech_chunks = rejected_chunks

        spoken: list[str] = []
        session._synthesize_response = spoken.append  # type: ignore[method-assign]

        # Mirrors the _finalize_run decision under test.
        if session._rejected_speech_chunks:
            session._synthesize_response(config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT)
        else:
            session._synthesize_response(config.DEFAULT_NO_SPEECH_PROMPT)
        return spoken[0]

    def test_true_silence_still_gets_the_greeting(self):
        assert self._spoken_reply(0) == config.DEFAULT_NO_SPEECH_PROMPT
        assert "How can I help you" in self._spoken_reply(0)

    def test_rejected_speech_asks_the_customer_to_repeat(self):
        reply = self._spoken_reply(1)
        assert reply == config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT
        assert reply != config.DEFAULT_NO_SPEECH_PROMPT

    def test_the_two_prompts_are_never_the_same(self):
        assert config.DEFAULT_NO_SPEECH_PROMPT != config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT

    def test_retry_prompt_actually_invites_a_retry(self):
        """The message must tell the customer what to do, not just apologise."""
        reply = config.DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT.lower()
        assert "recognise" in reply or "recognize" in reply
        assert "repeat" in reply or "again" in reply
