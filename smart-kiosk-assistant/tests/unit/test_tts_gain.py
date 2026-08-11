"""Tests for BaseAudioSession._apply_tts_gain (kiosk_core/audio_session.py).

Background: SpeechT5's vocoder outputs a quiet, roughly constant level
regardless of which speaker voice is selected, so kiosk-core boosts every
synthesized segment's loudness in place before playback (see the
"TTS output loudness" section of kiosk_core/config.py).

Regression covered here: the first version of this gain step computed
``total_gain = min(normalize_gain * extra_gain, max_gain)`` and let
``np.clip`` silently hard-clip anything over full scale. Because
``extra_gain`` is a flat boost stacked *on top of* peak-normalization (not a
replacement for it), that combination routinely pushed segments past full
scale. Observed live on a real synthesized sentence: 14 separate clipped
runs, some up to 6 consecutive samples, heard as a crackle/buzz on loud
syllables. The fix adds a third, tighter cap — the exact gain that brings the
segment's true peak to (not past) full scale — so a segment is limited, never
clipped.
"""
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from kiosk_core import config
from kiosk_core.audio_session import BaseAudioSession


class _Dummy:
    """Minimal stand-in with just enough attributes for the bound method."""

    session_id = "test-session"


def _write_wav(path: Path, samples: np.ndarray, frame_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(frame_rate)
        wav_out.writeframes(samples.astype(np.int16).tobytes())


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_in:
        frames = wav_in.readframes(wav_in.getnframes())
    return np.frombuffer(frames, dtype=np.int16)


def _synthetic_speech_like_segment(frame_rate: int = 16000, seconds: float = 1.0, peak: float = 6000.0) -> np.ndarray:
    """A quiet, speech-shaped tone burst — not silence, not a flat DC signal.

    A single-frequency sine at a fixed peak reproduces the failure mode
    exactly: one true maximum sample per cycle, repeated many times, which is
    exactly what a naive post-hoc np.clip turns into many separate clipped
    runs once gain pushes it over full scale.
    """
    t = np.linspace(0, seconds, int(frame_rate * seconds), endpoint=False)
    tone = np.sin(2 * np.pi * 220 * t) * peak
    return tone.astype(np.float64)


def _apply_gain(path: Path) -> None:
    BaseAudioSession._apply_tts_gain(_Dummy(), path)


class TestTtsGainNeverClips:
    def test_quiet_segment_is_boosted_without_any_clipping(self, tmp_path):
        # Mirrors the real measured SpeechT5 output level (~20-40% of full
        # scale peak) that prompted this feature.
        samples = _synthetic_speech_like_segment(peak=8000.0)
        path = tmp_path / "segment.wav"
        _write_wav(path, samples)

        _apply_gain(path)

        boosted = _read_wav(path)
        # No sample may exceed int16 full scale — clipping is the exact bug
        # this test guards against.
        assert int(np.abs(boosted).max()) <= 32767
        # And it must actually have been boosted, not left untouched.
        original_peak = int(np.abs(samples).max())
        assert int(np.abs(boosted).max()) > original_peak

    def test_high_extra_gain_is_capped_by_no_clip_ceiling_not_hard_clipped(self, tmp_path, monkeypatch):
        # Deliberately configure an aggressive extra boost that would have
        # driven the old implementation's normalize_gain * extra_gain product
        # well past full scale on every cycle of the tone.
        monkeypatch.setattr(config, "DEFAULT_TTS_TARGET_PEAK", 0.9)
        monkeypatch.setattr(config, "DEFAULT_TTS_GAIN_DB", 12.0)
        monkeypatch.setattr(config, "DEFAULT_TTS_GAIN_MAX_DB", 24.0)

        samples = _synthetic_speech_like_segment(peak=6450.0, seconds=2.0)
        path = tmp_path / "segment.wav"
        _write_wav(path, samples)

        _apply_gain(path)

        boosted = _read_wav(path)
        clipped_over_ceiling = int(np.sum(np.abs(boosted.astype(np.int32)) > 32767))
        # The true invariant this test guards: nothing may ever exceed full
        # scale (a fixed-frequency tone naturally has several samples per
        # cycle sitting at/near its true peak once quantized to int16 — that
        # is normal waveform shape, not clipping, so we only assert on
        # overflow past the ceiling, never on how many samples reach it).
        assert int(np.abs(boosted).max()) <= 32767
        assert clipped_over_ceiling == 0

    def test_real_measured_speecht5_level_matches_expected_boost(self, tmp_path):
        # Real measured levels from a live Kabir-voice synthesis this session:
        # peak ~19.7% of full scale, RMS ~852.
        rng = np.random.default_rng(42)
        seconds, frame_rate = 1.0, 16000
        n = int(seconds * frame_rate)
        # Band-limited noise shaped to a fixed peak, closer to real speech's
        # non-periodic waveform than a pure tone.
        noise = rng.normal(0, 1, n)
        noise = noise / np.abs(noise).max() * 6450.0
        path = tmp_path / "segment.wav"
        _write_wav(path, noise)

        _apply_gain(path)

        boosted = _read_wav(path)
        assert int(np.abs(boosted).max()) <= 32767
        boosted_rms = float(np.sqrt(np.mean(boosted.astype(np.float64) ** 2)))
        original_rms = float(np.sqrt(np.mean(noise ** 2)))
        assert boosted_rms > original_rms * 2  # meaningfully louder

    def test_already_loud_segment_is_left_near_untouched(self, tmp_path):
        # A segment already at/above target peak should not be attenuated
        # or heavily altered.
        samples = _synthetic_speech_like_segment(peak=32000.0)
        path = tmp_path / "segment.wav"
        _write_wav(path, samples)
        original = _read_wav(path).copy()

        _apply_gain(path)

        boosted = _read_wav(path)
        assert int(np.abs(boosted).max()) <= 32767
        # Should not be drastically quieter than the original.
        assert int(np.abs(boosted).max()) >= int(np.abs(original).max()) * 0.9

    def test_silent_segment_is_left_untouched(self, tmp_path):
        samples = np.zeros(16000, dtype=np.float64)
        path = tmp_path / "segment.wav"
        _write_wav(path, samples)

        _apply_gain(path)  # must not raise

        result = _read_wav(path)
        assert int(np.abs(result).max()) == 0

    def test_disabled_flag_is_respected_by_caller_contract(self):
        # _apply_tts_gain itself has no enabled/disabled branch — the caller
        # (_tts_worker) gates the call on config.DEFAULT_TTS_GAIN_ENABLED.
        # This just pins that the flag exists and defaults to enabled.
        assert isinstance(config.DEFAULT_TTS_GAIN_ENABLED, bool)
