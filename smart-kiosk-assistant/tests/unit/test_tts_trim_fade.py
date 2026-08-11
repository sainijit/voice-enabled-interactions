"""Tests for BaseAudioSession._trim_tts_segment's edge-fade behaviour.

Background: each reply is synthesized as one clause/sentence per audio file
so playback can start early (see the docstring in _trim_tts_segment). The UI
(useAudioQueue.ts) schedules these files back-to-back on a single Web Audio
timeline with no gap and no crossfade, for a gapless-sounding reply.

Regression covered here: the original trim implementation sliced the
waveform at whatever raw sample index the lead/clause/sentence pad window
computed to — not a zero-crossing, with no fade. Whether that landed on a
near-zero sample or a mid-waveform peak depended entirely on the specific
phoneme at that exact cut point, which varies by sentence content. Observed
live: some replies played cleanly, others had an audible click/pop at a
clause boundary — "sometimes", exactly matching a content-dependent seam
discontinuity rather than a constant bug. The fix ramps every segment's
first/last few milliseconds to exactly zero, unconditionally, so the click
cannot occur regardless of what the model synthesized at the cut point.
"""
import wave
from pathlib import Path

import numpy as np

from kiosk_core import config
from kiosk_core.audio_session import BaseAudioSession


class _Dummy:
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


def _speech_like_segment(frame_rate: int = 16000, seconds: float = 0.6, peak: float = 12000.0, phase: float = 0.0) -> np.ndarray:
    """A tone burst with silence padding on both sides, like a real segment.

    ``phase`` shifts where the tone starts relative to the pad windows, which
    changes exactly where the trim cut lands on the waveform — this is what
    makes the original bug content-dependent ("sometimes").
    """
    pad = int(frame_rate * 0.05)  # 50ms silence pad each side, pre-trim
    n = int(frame_rate * seconds)
    t = np.linspace(0, seconds, n, endpoint=False)
    tone = np.sin(2 * np.pi * 180 * t + phase) * peak
    return np.concatenate([np.zeros(pad), tone, np.zeros(pad)])


def _trim(path: Path, sentence: str = "Hello there.") -> None:
    BaseAudioSession._trim_tts_segment(_Dummy(), path, sentence)


class TestTtsTrimEdgesAlwaysFadeToZero:
    def test_segment_starts_and_ends_at_exact_zero_regardless_of_phase(self, tmp_path):
        # Sweep several phase offsets so the cut point lands on a variety of
        # non-zero-crossing waveform positions, reproducing the original bug.
        for i, phase in enumerate([0.0, 0.7, 1.5, 2.3, 3.9, 5.1]):
            samples = _speech_like_segment(phase=phase)
            path = tmp_path / f"segment_{i}.wav"
            _write_wav(path, samples)

            _trim(path)

            trimmed = _read_wav(path)
            assert trimmed.size > 0
            assert int(trimmed[0]) == 0, f"phase={phase}: first sample not zero"
            assert int(trimmed[-1]) == 0, f"phase={phase}: last sample not zero"

    def test_fade_does_not_remove_actual_speech_content(self, tmp_path):
        samples = _speech_like_segment(peak=12000.0, seconds=1.0)
        path = tmp_path / "segment.wav"
        _write_wav(path, samples)

        _trim(path)

        trimmed = _read_wav(path)
        # The loud tone in the middle must still be there — the fade only
        # touches the first/last few ms, well inside the silence pad.
        assert int(np.abs(trimmed).max()) > 10000

    def test_no_crash_on_a_segment_shorter_than_the_fade_window(self, tmp_path):
        # A pathologically short segment (shorter than 4x the fade window)
        # must not raise or corrupt the file — the fade is skipped instead.
        tiny = np.array([0, 100, 200, 100, 0], dtype=np.float64)
        path = tmp_path / "tiny.wav"
        _write_wav(path, tiny)

        _trim(path)  # must not raise

    def test_seamless_concatenation_has_no_sample_level_jump(self, tmp_path):
        # Simulates what the UI actually does: two independently-trimmed
        # segments played back to back with zero gap. The seam sample (last
        # of segment A, first of segment B) must not jump.
        seg_a = _speech_like_segment(phase=0.3)
        seg_b = _speech_like_segment(phase=2.8)
        path_a = tmp_path / "a.wav"
        path_b = tmp_path / "b.wav"
        _write_wav(path_a, seg_a)
        _write_wav(path_b, seg_b)

        _trim(path_a)
        _trim(path_b)

        a = _read_wav(path_a)
        b = _read_wav(path_b)
        seam_jump = abs(int(a[-1]) - int(b[0]))
        assert seam_jump == 0
