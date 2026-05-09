from __future__ import annotations

import io
import os
import time
import wave
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import httpx
import numpy as np

from kiosk_core import config as kiosk_config


KIOSK_CORE_URL = os.getenv("KIOSK_CORE_UI_BASE_URL", "http://127.0.0.1:8012")
RAG_URL = os.getenv("KIOSK_CORE_UI_RAG_URL", "http://127.0.0.1:8020/api/v1/query")
TTS_URL = os.getenv("KIOSK_CORE_UI_TTS_URL", "http://127.0.0.1:8011/v1/audio/speech")
ANALYZER_URL = os.getenv("KIOSK_CORE_UI_ANALYZER_URL", "http://127.0.0.1:8010/v1/audio/transcriptions")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_UI_TIMEOUT_SECONDS", "120.0"))
POLL_INTERVAL_SECONDS = float(os.getenv("KIOSK_CORE_UI_POLL_INTERVAL_SECONDS", "0.35"))


STYLE = """
.gradio-container {
  background:
    radial-gradient(circle at top, #18344a 0%, #0d1822 42%, #071018 100%);
  color: #e8f0f7;
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
}

.kiosk-shell {
  max-width: 960px;
  margin: 0 auto;
}

.kiosk-hero {
  padding: 24px 28px 12px 28px;
  border: 1px solid rgba(163, 191, 214, 0.18);
  border-radius: 28px;
  background: linear-gradient(180deg, rgba(17, 34, 49, 0.88), rgba(8, 17, 25, 0.92));
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.28);
}

.kiosk-title h1 {
  margin: 0;
  font-size: 2.2rem;
  letter-spacing: -0.03em;
}

.kiosk-title p {
  margin: 10px 0 0 0;
  color: #a9bfd0;
  font-size: 1rem;
}

.assistant-orb-wrap {
  display: flex;
  justify-content: center;
  padding: 22px 0 10px 0;
}

.assistant-orb {
  width: 168px;
  height: 168px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(circle at 30% 30%, #67d7ff 0%, #2d8bb0 38%, #0f2f42 72%, #09131c 100%);
  box-shadow:
    0 0 0 10px rgba(95, 187, 222, 0.08),
    0 0 0 24px rgba(95, 187, 222, 0.04),
    0 18px 60px rgba(26, 159, 204, 0.35);
}

.assistant-mic {
  position: relative;
  width: 42px;
  height: 70px;
  border: 4px solid #e9f8ff;
  border-radius: 26px;
}

.assistant-mic::before {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -22px;
  width: 4px;
  height: 20px;
  transform: translateX(-50%);
  background: #e9f8ff;
}

.assistant-mic::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: -34px;
  width: 42px;
  height: 18px;
  transform: translateX(-50%);
  border: 4px solid #e9f8ff;
  border-top: none;
  border-radius: 0 0 28px 28px;
}

#kiosk-mic-input {
  border: 1px solid rgba(163, 191, 214, 0.18);
  border-radius: 24px;
  background: rgba(7, 16, 24, 0.55);
  padding: 14px;
}

#kiosk-mic-input button {
  min-height: 54px;
  border-radius: 999px;
}

.kiosk-panel {
  border: 1px solid rgba(163, 191, 214, 0.16);
  border-radius: 22px;
  background: rgba(8, 16, 24, 0.72);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.kiosk-status {
  padding: 12px 16px;
  border-radius: 16px;
  background: rgba(17, 36, 50, 0.9);
  color: #d8e9f5;
  border: 1px solid rgba(104, 188, 224, 0.18);
}

.kiosk-copy textarea,
.kiosk-copy .cm-content,
.kiosk-copy input {
  font-size: 1.03rem;
}

.kiosk-copy label {
  color: #dcecf6;
}
"""


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a 1-D int16 numpy array to raw WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def _open_stream_session(sample_rate: int) -> dict[str, Any]:
    """Open a browser stream session on kiosk-core."""
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.post(
            f"{KIOSK_CORE_URL}/api/v1/sessions/start-stream",
            json={
                "sample_rate": sample_rate,
                "chunk_seconds": kiosk_config.DEFAULT_CHUNK_SECONDS,
                "silence_timeout_seconds": 1.5,
                "max_session_seconds": 60.0,
                "silence_threshold": 900,
                "language": "en",
                "temperature": 0.0,
                "analyzer_url": ANALYZER_URL,
                "rag_url": RAG_URL,
                "tts_url": TTS_URL,
                "tts_model": "qwen-tts",
                "tts_language": "English",
            },
        )
    response.raise_for_status()
    return response.json()


def _push_chunk(session_id: str, wav_bytes: bytes) -> None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        client.post(
            f"{KIOSK_CORE_URL}/api/v1/sessions/{session_id}/audio",
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        ).raise_for_status()


def _end_stream(session_id: str) -> None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        client.post(f"{KIOSK_CORE_URL}/api/v1/sessions/{session_id}/audio/end").raise_for_status()


def _poll_session(session_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.get(f"{KIOSK_CORE_URL}/api/v1/sessions/{session_id}")
    response.raise_for_status()
    return response.json()


def _build_status(session: dict[str, Any] | None, phase: str) -> str:
    if phase == "idle":
        return "Ready. Tap the microphone, speak your question, then stop recording."
    if phase == "listening":
        return "Listening... finish speaking to submit your question."
    if session is None:
        return "Starting session..."

    tts_segments = len(session.get("tts_audio_segments", []))
    status = str(session.get("status", "unknown"))
    if status in {"running", "stopping"}:
        if tts_segments:
            return f"Speaking response... {tts_segments} sentence audio clip(s) ready."
        if session.get("response"):
            return "Generating response..."
        if session.get("transcript"):
            return "Transcription ready. Querying knowledge base..."
        return "Processing audio..."
    if status == "completed":
        return f"Done. {tts_segments} sentence audio clip(s) generated."
    error = session.get("error") or "Unknown failure"
    return f"Session failed: {error}"


def _latest_audio_update(session: dict[str, Any], previous_count: int) -> tuple[dict[str, Any], int]:
    tts_segments = session.get("tts_audio_segments", []) or []
    if len(tts_segments) > previous_count:
        return gr.update(value=tts_segments[-1]["audio_file"], autoplay=True), len(tts_segments)
    return gr.skip(), previous_count


# ── Streaming event handlers ──────────────────────────────────────────────────

# gr.State schema: {"session_id": str|None, "buffer": list[np.ndarray], "sample_rate": int}
_CHUNK_SECONDS = kiosk_config.DEFAULT_CHUNK_SECONDS  # pushed to kiosk-core and used as stream_every


def on_start_recording() -> tuple[dict, str, str, dict, str]:
    """Called when the user starts recording. Resets UI and stream state."""
    return (
        {"session_id": None, "buffer": [], "sample_rate": 16000},
        "",
        "",
        gr.update(value=None, autoplay=False),
        _build_status(None, "listening"),
    )


def on_stream_chunk(
    stream_state: dict,
    audio_chunk,   # (sample_rate: int, data: np.ndarray) from Gradio streaming
) -> tuple[dict, str, str, dict, str]:
    """Called every ~CHUNK_SECONDS while the mic is open.
    Accumulates audio and pushes to kiosk-core once we have a full chunk.
    """
    if audio_chunk is None:
        return stream_state, "", "", gr.skip(), _build_status(None, "listening")

    sample_rate, data = audio_chunk
    if data is None or len(data) == 0:
        return stream_state, "", "", gr.skip(), _build_status(None, "listening")

    # Flatten to mono int16
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.int16)

    state = dict(stream_state)
    state["sample_rate"] = sample_rate
    state["buffer"] = list(state.get("buffer", [])) + [data]

    # Open session on first chunk
    if state["session_id"] is None:
        try:
            started = _open_stream_session(sample_rate)
            state["session_id"] = started["session_id"]
        except Exception as exc:  # noqa: BLE001
            return state, "", "", gr.skip(), f"Failed to open session: {exc}"

    # Check if buffer has accumulated enough for a push
    buffer_samples = sum(len(b) for b in state["buffer"])
    if buffer_samples >= int(sample_rate * _CHUNK_SECONDS):
        audio = np.concatenate(state["buffer"], axis=0)
        state["buffer"] = []
        wav_bytes = _numpy_to_wav_bytes(audio, sample_rate)
        try:
            _push_chunk(state["session_id"], wav_bytes)
        except Exception:  # noqa: BLE001
            pass  # Drop the chunk — session will continue with next push

    # Poll for live transcript updates
    transcript = ""
    try:
        session = _poll_session(state["session_id"])
        transcript = str(session.get("transcript", "")).strip()
    except Exception:  # noqa: BLE001
        pass

    return state, transcript, "", gr.skip(), _build_status(None, "listening")


def on_stop_recording(
    stream_state: dict,
) -> Generator[tuple[dict, str, str, dict, str], None, None]:
    """Called when the user stops recording.
    Flushes remaining buffer, signals EOS, then polls until the session finishes.
    """
    state = dict(stream_state)
    session_id = state.get("session_id")
    sample_rate = state.get("sample_rate", 16000)

    if not session_id:
        yield state, "", "", gr.update(value=None, autoplay=False), "No audio was captured. Try again."
        return

    # Flush remaining buffer
    remaining = state.get("buffer", [])
    if remaining:
        audio = np.concatenate(remaining, axis=0)
        wav_bytes = _numpy_to_wav_bytes(audio, sample_rate)
        try:
            _push_chunk(session_id, wav_bytes)
        except Exception:  # noqa: BLE001
            pass

    # Signal end-of-stream
    try:
        _end_stream(session_id)
    except Exception as exc:  # noqa: BLE001
        yield state, "", "", gr.update(value=None, autoplay=False), f"Failed to finalise session: {exc}"
        return

    yield state, "", "", gr.update(value=None, autoplay=False), "Processing speech..."

    # Poll until done
    previous_audio_count = 0
    while True:
        try:
            session = _poll_session(session_id)
        except Exception as exc:  # noqa: BLE001
            yield state, "", "", gr.update(value=None, autoplay=False), f"Polling error: {exc}"
            return

        transcript = str(session.get("transcript", "")).strip()
        response_text = str(session.get("response", "")).strip()
        audio_update, previous_audio_count = _latest_audio_update(session, previous_audio_count)
        running = str(session.get("status", "")) in {"running", "stopping"}

        yield state, transcript, response_text, audio_update, _build_status(session, "processing")

        if not running:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Kiosk Core UI") as app:
        # Per-session streaming state: session_id, audio buffer, sample_rate
        stream_state = gr.State({"session_id": None, "buffer": [], "sample_rate": 16000})

        with gr.Column(elem_classes=["kiosk-shell"]):
            with gr.Column(elem_classes=["kiosk-hero"]):
                gr.HTML(
                    """
                    <div class="kiosk-title">
                      <h1>Kiosk Voice Assistant</h1>
                      <p>Speak a question — transcription and answer appear as you speak.</p>
                    </div>
                    <div class="assistant-orb-wrap">
                      <div class="assistant-orb">
                        <div class="assistant-mic"></div>
                      </div>
                    </div>
                    """
                )

                mic_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=True,
                    label="Tap the microphone and speak. Stop recording when done.",
                    elem_id="kiosk-mic-input",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                status_box = gr.Markdown(
                    value=_build_status(None, "idle"),
                    elem_classes=["kiosk-status"],
                )

            with gr.Row():
                transcript_box = gr.Textbox(
                    label="User transcription",
                    lines=5,
                    interactive=False,
                    elem_classes=["kiosk-panel", "kiosk-copy"],
                )
                response_box = gr.Textbox(
                    label="RAG response",
                    lines=8,
                    interactive=False,
                    elem_classes=["kiosk-panel", "kiosk-copy"],
                )

            tts_audio = gr.Audio(
                label="Assistant speech",
                interactive=False,
                autoplay=True,
                elem_classes=["kiosk-panel"],
                buttons=[],
            )

            _outputs = [stream_state, transcript_box, response_box, tts_audio, status_box]

            mic_input.start_recording(
                fn=on_start_recording,
                inputs=None,
                outputs=_outputs,
            )
            mic_input.stream(
                fn=on_stream_chunk,
                inputs=[stream_state, mic_input],
                outputs=_outputs,
                stream_every=_CHUNK_SECONDS,
            )
            mic_input.stop_recording(
                fn=on_stop_recording,
                inputs=[stream_state],
                outputs=_outputs,
            )

    return app


def launch_app() -> tuple[Any, str, str]:
    return create_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css=STYLE,
    )


if __name__ == "__main__":
    launch_app()