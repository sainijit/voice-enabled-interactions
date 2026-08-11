"""Shared fixtures for unit tests.

``kiosk_core.audio_session`` imports ``sounddevice``, which requires the
PortAudio shared library at import time. PortAudio is only installed inside
the kiosk-core container, so without this stub every unit test module that
touches an audio session fails at *collection* time on a bare host.

Mirrors the mocking already done in ``tests/functional/conftest.py``. None of
these tests exercise a real capture device.
"""
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("sounddevice", MagicMock())


@pytest.fixture(autouse=True)
def _reset_speaker_rejection_tracking():
    """Clear the module-level consecutive-rejection tracker before each test.

    ``kiosk_core.audio_session`` tracks rejected-speech streaks in a
    module-level dict keyed by ``agent_session_id`` so it survives across the
    many short-lived ``BaseAudioSession`` instances of one conversation (see
    ``_note_conversation_rejection``). Several tests reuse the same default
    conversation id ("test-session"), so without this reset a streak left
    over from one test would silently change another test's outcome.
    """
    from kiosk_core.audio_session import reset_all_rejection_tracking

    reset_all_rejection_tracking()
    yield
    reset_all_rejection_tracking()
