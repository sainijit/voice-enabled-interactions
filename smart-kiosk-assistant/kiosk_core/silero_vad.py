"""Silero VAD (v5) wrapper: per-frame speech probability via onnxruntime.

Mirrors the reference implementation in kiosk-voice-lab-main's
``pipeline/vad.py::SileroVAD`` class. This is an optional, feature-flagged
alternative to the existing adaptive RMS VAD in :mod:`kiosk_core.audio_session`
(see ``KIOSK_CORE_SILERO_VAD_ENABLED`` in :mod:`kiosk_core.config`). The RMS
VAD remains the default; this module is only instantiated/used when the flag
is enabled.

BINDING NOTE (critical, easy to get wrong silently): the exported Silero v5
ONNX graph is causal and expects each "input" to be the 512-sample hop PLUS
the trailing 64 samples of the previous hop as left context (576 samples
total at 16 kHz; 32-sample context / 256-sample hop at 8 kHz). Feeding bare
512-sample frames without this context still runs without error -- the
graph's input dim is dynamic -- but produces near-zero probability on every
frame, speech included, since the first causal conv has no valid left
context to read. This was confirmed both in the reference lab code and
independently on this box: probability stayed <0.06 throughout a full
utterance without the context prepend and jumped to ~0.99+ on speech frames
with it. Do not "simplify" this away.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SileroVAD:
    """Per-frame speech probability using Silero VAD v5 (onnxruntime).

    Frames are accumulated internally; callers may feed chunks of any size
    (e.g. this project's 100ms/1600-sample blocks) and the wrapper slices
    them into the model's required 512-sample (16kHz) hops, buffering any
    remainder for the next call. State (the model's internal recurrent
    state) and the causal left-context are carried between calls.
    """

    FRAME = 512

    def __init__(
        self,
        model_path: str | Path,
        sample_rate: int = 16000,
        intra_op_threads: int = 1,
    ) -> None:
        """Load the Silero VAD ONNX model and initialize per-stream state.

        Args:
            model_path: Path to the bundled ``silero_vad.onnx`` file.
            sample_rate: Audio sample rate in Hz. Only 16000 and 8000 are
                supported by the upstream model.
            intra_op_threads: onnxruntime intra-op thread cap. Defaults to 1
                to avoid contending with the ASR/LLM/TTS pipelines, since
                this model is tiny and does not benefit from parallelism.
        """
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_threads
        self.sess = ort.InferenceSession(str(model_path), so)
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.sr = np.array(sample_rate, dtype=np.int64)
        self.buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
        self.context_size = 64 if sample_rate == 16000 else 32
        self.context = np.zeros(self.context_size, dtype=np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        """Feed a chunk of audio; returns the latest per-frame speech probability.

        Args:
            chunk: 1D array of audio samples (any dtype convertible to
                float32; expected range matches the rest of this project's
                pipeline, i.e. normalized float32 in [-1, 1]).

        Returns:
            The most recently computed speech probability in [0, 1]. If the
            accumulated buffer hasn't reached a full 512-sample frame yet,
            returns the previous call's probability unchanged.
        """
        self.buf = np.concatenate([self.buf, np.asarray(chunk, dtype=np.float32)])
        while len(self.buf) >= self.FRAME:
            frame = self.buf[: self.FRAME]
            self.buf = self.buf[self.FRAME :]
            x = np.concatenate([self.context, frame])[None, :].astype(np.float32)
            out = self.sess.run(
                None, {"input": x, "state": self.state, "sr": self.sr}
            )
            self.last_prob = float(np.asarray(out[0]).ravel()[0])
            self.state = out[1]
            self.context = frame[-self.context_size :]
        return self.last_prob

    def reset(self) -> None:
        """Reset internal recurrent state, context, and buffer.

        Must be called between independent audio streams/sessions to avoid
        leaking state across unrelated utterances.
        """
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.buf = np.zeros(0, dtype=np.float32)
        self.last_prob = 0.0
        self.context = np.zeros(self.context_size, dtype=np.float32)
