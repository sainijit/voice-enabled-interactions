"""Unit tests for kiosk_core.silero_vad.SileroVAD.

Requires onnxruntime and the bundled kiosk_core/models/silero_vad.onnx file.
Skipped automatically (via pytest.importorskip) in environments without
onnxruntime installed (e.g. bare host dev shells) — it is a real dependency
of the kiosk-core container (see requirements.txt) and these tests run there
or in any environment where it's installed.
"""
import wave
from pathlib import Path

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from kiosk_core.silero_vad import SileroVAD  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "kiosk_core" / "models" / "silero_vad.onnx"
SAMPLE_SPEECH_WAV = Path(__file__).resolve().parent.parent / "fixtures" / "sample_speech_16k.wav"

# Frame size matching this project's KIOSK_CORE_BLOCK_DURATION_SECONDS default
# (0.1s @ 16kHz = 1600 samples) — the wrapper must handle chunks that aren't a
# multiple of its internal 512-sample FRAME size.
FRAME_SAMPLES = 1600


def _load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        raw = wav_file.readframes(wav_file.getnframes())
    int16 = np.frombuffer(raw, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


@pytest.fixture(scope="module")
def real_speech() -> np.ndarray:
    return _load_wav_float32(SAMPLE_SPEECH_WAV)


class TestSileroVADModelLoads:
    def test_model_file_exists(self) -> None:
        assert MODEL_PATH.is_file(), f"bundled model missing at {MODEL_PATH}"

    def test_constructs_without_error(self) -> None:
        vad = SileroVAD(MODEL_PATH)
        assert vad.state.shape == (2, 1, 128)
        assert vad.context.shape == (64,)  # 16kHz -> 64-sample causal context


class TestSileroVADSpeechDetection:
    def test_silence_yields_low_probability(self) -> None:
        vad = SileroVAD(MODEL_PATH)
        silence = np.zeros(FRAME_SAMPLES * 20, dtype=np.float32)
        prob = 0.0
        # Feed in project-realistic 100ms hops rather than one giant chunk.
        for start in range(0, len(silence), FRAME_SAMPLES):
            prob = vad.prob(silence[start : start + FRAME_SAMPLES])
        assert prob < 0.3, f"expected low speech probability on silence, got {prob}"

    def test_real_speech_yields_high_probability_somewhere(self, real_speech: np.ndarray) -> None:
        vad = SileroVAD(MODEL_PATH)
        probs = []
        for start in range(0, len(real_speech), FRAME_SAMPLES):
            probs.append(vad.prob(real_speech[start : start + FRAME_SAMPLES]))
        # At least one frame during the utterance should clear the standard
        # 0.5 speech gate -- pins the model + causal-context wiring is correct
        # end-to-end (a broken context prepend collapses ALL probabilities
        # near zero, per the BINDING NOTE in silero_vad.py).
        assert max(probs) >= 0.5, f"expected some frame to read as speech, max prob was {max(probs)}"

    def test_handles_chunk_sizes_not_a_multiple_of_frame(self) -> None:
        # 1600 % 512 != 0 -- exercises the internal buffering/remainder path.
        vad = SileroVAD(MODEL_PATH)
        chunk = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        # Must not raise, and buffer should hold the 1600 % 512 == 64 sample
        # remainder between calls.
        vad.prob(chunk)
        assert len(vad.buf) == FRAME_SAMPLES % SileroVAD.FRAME


class TestSileroVADStateHandling:
    def test_context_and_state_are_threaded_across_frames(self, real_speech: np.ndarray) -> None:
        """Sanity check that the wrapper actually carries the causal context
        and recurrent state forward between calls (rather than e.g.
        accidentally re-zeroing them each ``prob()`` call), per the BINDING
        NOTE in silero_vad.py. Both must change from their initial zeroed
        values once real audio has been processed.
        """
        vad = SileroVAD(MODEL_PATH)
        initial_state = vad.state.copy()
        context_was_ever_nonzero = False

        for start in range(0, len(real_speech), FRAME_SAMPLES):
            vad.prob(real_speech[start : start + FRAME_SAMPLES])
            if np.any(vad.context != 0):
                context_was_ever_nonzero = True

        assert not np.array_equal(vad.state, initial_state)
        # Trailing frames of the fixture may be silent (context legitimately
        # zero at the very end), so check it went non-zero at some point
        # rather than asserting on the final value.
        assert context_was_ever_nonzero
        assert vad.context.shape == (64,)

    def test_reset_clears_state(self, real_speech: np.ndarray) -> None:
        vad = SileroVAD(MODEL_PATH)
        for start in range(0, len(real_speech), FRAME_SAMPLES):
            vad.prob(real_speech[start : start + FRAME_SAMPLES])
        assert not np.array_equal(vad.state, np.zeros((2, 1, 128), dtype=np.float32))

        vad.reset()
        assert np.array_equal(vad.state, np.zeros((2, 1, 128), dtype=np.float32))
        assert np.array_equal(vad.context, np.zeros(64, dtype=np.float32))
        assert len(vad.buf) == 0
        assert vad.last_prob == 0.0


class TestSampleRateValidation:
    """Silero v5 only accepts 8kHz/16kHz.

    An unsupported rate loads fine but fails at inference time inside the
    decoder LSTM, which previously surfaced as a mid-session crash rather
    than a clean fallback. The rate is therefore rejected in ``__init__``.
    """

    @pytest.mark.parametrize("rate", [8000, 16000])
    def test_supported_rates_construct(self, rate: int) -> None:
        vad = SileroVAD(MODEL_PATH, sample_rate=rate)
        assert int(vad.sr) == rate

    @pytest.mark.parametrize("rate", [24000, 22050, 44100, 48000, 0, -1])
    def test_unsupported_rates_raise_value_error(self, rate: int) -> None:
        with pytest.raises(ValueError, match="supports"):
            SileroVAD(MODEL_PATH, sample_rate=rate)

    def test_error_message_names_the_offending_rate(self) -> None:
        with pytest.raises(ValueError, match="24000"):
            SileroVAD(MODEL_PATH, sample_rate=24000)

    def test_default_rate_is_supported(self) -> None:
        assert SileroVAD.SUPPORTED_SAMPLE_RATES == (8000, 16000)
        assert int(SileroVAD(MODEL_PATH).sr) == 16000
