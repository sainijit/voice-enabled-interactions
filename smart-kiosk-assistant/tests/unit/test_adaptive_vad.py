"""Unit tests for BaseAudioSession._update_vad_threshold().

These are tier1 (no Docker/ML/audio hardware required) — the method is pure
energy arithmetic over frame RMS values.

Background: ``KIOSK_CORE_SILENCE_THRESHOLD`` is an absolute int16 RMS value.
Measured on the demo unit (PCM2902 USB codec) the *silent room* floor was
RMS ~1076, i.e. above the 900 gate, so every frame classified as speech and
the silence endpoint could never fire. These tests pin the adaptive
replacement, and in particular its fail-open behaviour: the gate must never
rise high enough to suppress all speech.
"""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

# `kiosk_core.audio_session` imports sounddevice, which needs PortAudio at
# import time — unavailable outside the container. Mirrors the mocking done in
# tests/functional/conftest.py. Must run before the kiosk_core import below.
sys.modules.setdefault("sounddevice", MagicMock())

from kiosk_core import config  # noqa: E402
from kiosk_core.audio_session import BaseAudioSession  # noqa: E402

# Representative of the measured PCM2902 noise floor in a quiet room.
QUIET_ROOM_FLOOR = 1076.0


def _make_session(seed_threshold: int = 900) -> BaseAudioSession:
    session = BaseAudioSession.__new__(BaseAudioSession)
    session.session_id = "test-session"
    session.request = types.SimpleNamespace(silence_threshold=seed_threshold)
    session._vad_threshold = float(seed_threshold)
    session._noise_floor = None
    session._vad_calibrating = config.ADAPTIVE_VAD_ENABLED
    session._vad_calibration_rms = []
    session._vad_calibration_frames = max(
        1, int(config.DEFAULT_VAD_CALIBRATION_SECONDS / config.DEFAULT_BLOCK_DURATION_SECONDS)
    )
    return session


def _feed(session: BaseAudioSession, rms_values: list[float]) -> list[bool]:
    """Drive the session the same way _process_frame_stream does."""
    decisions = []
    for rms in rms_values:
        is_speech = rms >= session._vad_threshold
        session._update_vad_threshold(rms, is_speech)
        decisions.append(False if session._vad_calibrating else rms >= session._vad_threshold)
    return decisions


@pytest.mark.tier1
class TestAdaptiveVadCalibration:
    def test_calibrates_gate_above_measured_noise_floor(self):
        session = _make_session()
        _feed(session, [QUIET_ROOM_FLOOR] * 20)

        assert session._noise_floor == pytest.approx(QUIET_ROOM_FLOOR, rel=0.05)
        assert session._vad_threshold > QUIET_ROOM_FLOOR

    def test_silent_room_above_seed_threshold_is_not_classified_as_speech(self):
        """The exact bug this replaces: floor 1076 > seed gate 900."""
        session = _make_session(seed_threshold=900)
        rng = np.random.default_rng(0)
        floor_frames = [QUIET_ROOM_FLOOR + rng.normal(0, 40) for _ in range(60)]

        decisions = _feed(session, floor_frames)

        # Every one of these frames would have been "speech" under the fixed gate.
        assert all(rms >= 900 for rms in floor_frames)
        assert not any(decisions)

    def test_speech_well_above_floor_is_detected(self):
        session = _make_session()
        _feed(session, [QUIET_ROOM_FLOOR] * 10)
        assert _feed(session, [QUIET_ROOM_FLOOR * 4])[0] is True

    def test_gate_is_never_below_configured_minimum(self):
        session = _make_session()
        _feed(session, [1.0] * 20)  # near-digital-silence input
        assert session._vad_threshold >= config.DEFAULT_VAD_THRESHOLD_MIN


@pytest.mark.tier1
class TestAdaptiveVadFailOpen:
    def test_very_loud_venue_clamps_gate_and_stays_permissive(self):
        """A venue louder than the ceiling must degrade to permissive detection.

        Fail-open is the required direction: a gate that is too HIGH means no
        speech is ever detected and nothing is transcribed at all.
        """
        session = _make_session()
        _feed(session, [QUIET_ROOM_FLOOR * 20] * 20)

        assert session._vad_threshold == float(config.DEFAULT_VAD_THRESHOLD_MAX)
        # Speech at the (very high) ambient level still registers.
        assert _feed(session, [QUIET_ROOM_FLOOR * 20])[0] is True

    def test_floor_is_not_dragged_up_by_gaps_within_an_utterance(self):
        """Quiet inter-word frames must not inflate the noise estimate.

        With a symmetric EMA the floor climbed 1098 -> 1642 during a single
        utterance, pushing the gate up and swallowing the rest of the sentence.
        """
        session = _make_session()
        _feed(session, [QUIET_ROOM_FLOOR] * 10)
        floor_after_calibration = session._noise_floor

        # Alternate loud speech and quiet-but-not-silent inter-word gaps.
        for _ in range(40):
            _feed(session, [QUIET_ROOM_FLOOR * 5, QUIET_ROOM_FLOOR * 1.4])

        assert session._noise_floor < floor_after_calibration * 1.25

    def test_floor_tracks_downward_when_room_quietens(self):
        session = _make_session()
        _feed(session, [QUIET_ROOM_FLOOR] * 10)
        _feed(session, [QUIET_ROOM_FLOOR / 4] * 200)

        assert session._noise_floor < QUIET_ROOM_FLOOR / 2


@pytest.mark.tier1
class TestAdaptiveVadDisabled:
    def test_disabled_leaves_seed_threshold_untouched(self, monkeypatch):
        monkeypatch.setattr(config, "ADAPTIVE_VAD_ENABLED", False)
        session = _make_session(seed_threshold=900)
        session._vad_calibrating = False

        _feed(session, [QUIET_ROOM_FLOOR] * 30)

        assert session._vad_threshold == 900.0
        assert session._noise_floor is None
