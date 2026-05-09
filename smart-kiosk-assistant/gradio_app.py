from __future__ import annotations

import os
import time
import wave
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import httpx


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


def _read_sample_rate(audio_path: str) -> int:
    with wave.open(audio_path, "rb") as wav_file:
        return int(wav_file.getframerate())


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


def _start_session(audio_path: str) -> dict[str, Any]:
    sample_rate = _read_sample_rate(audio_path)
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        with open(audio_path, "rb") as audio_file:
            response = client.post(
                f"{KIOSK_CORE_URL}/api/v1/sessions/start-file",
                files={"file": (Path(audio_path).name, audio_file, "audio/wav")},
                data={
                    "sample_rate": str(sample_rate),
                    "chunk_seconds": "4.0",
                    "silence_timeout_seconds": "1.5",
                    "max_session_seconds": "20.0",
                    "silence_threshold": "900",
                    "language": "en",
                    "temperature": "0.0",
                    "analyzer_url": ANALYZER_URL,
                    "rag_url": RAG_URL,
                    "tts_url": TTS_URL,
                    "tts_model": "qwen-tts",
                    "tts_language": "English",
                    "realtime_factor": "10.0",
                },
            )
    response.raise_for_status()
    return response.json()


def _poll_session(session_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, trust_env=False) as client:
        response = client.get(f"{KIOSK_CORE_URL}/api/v1/sessions/{session_id}")
    response.raise_for_status()
    return response.json()


def begin_recording() -> tuple[dict[str, Any], str, str, dict[str, Any], str]:
    return (
        gr.update(value=None, interactive=True),
        "",
        "",
        gr.update(value=None, autoplay=False),
        _build_status(None, "listening"),
    )


def process_turn(audio_path: str | None) -> Generator[tuple[dict[str, Any], str, str, dict[str, Any], str], None, None]:
    if not audio_path:
        yield (
            gr.update(interactive=True, value=None),
            "",
            "",
            gr.update(value=None, autoplay=False),
            "No microphone audio was captured. Try again.",
        )
        return

    yield (
        gr.update(interactive=False, value=audio_path),
        "",
        "",
        gr.update(value=None, autoplay=False),
        "Uploading audio and starting session...",
    )

    try:
        started = _start_session(audio_path)
        session_id = str(started["session_id"])
    except Exception as exc:  # noqa: BLE001
        yield (
            gr.update(interactive=True, value=None),
            "",
            "",
            gr.update(value=None, autoplay=False),
            f"Failed to start kiosk session: {exc}",
        )
        return

    previous_audio_count = 0
    while True:
        try:
            session = _poll_session(session_id)
        except Exception as exc:  # noqa: BLE001
            yield (
                gr.update(interactive=True, value=None),
                "",
                "",
                gr.update(value=None, autoplay=False),
                f"Failed to read session state: {exc}",
            )
            return

        transcript = str(session.get("transcript", "")).strip()
        response_text = str(session.get("response", "")).strip()
        audio_update, previous_audio_count = _latest_audio_update(session, previous_audio_count)
        status_text = _build_status(session, "processing")
        running = str(session.get("status", "")) in {"running", "stopping"}

        yield (
            gr.update(interactive=not running, value=None if not running else audio_path),
            transcript,
            response_text,
            audio_update,
            status_text,
        )

        if not running:
            break
        time.sleep(POLL_INTERVAL_SECONDS)


def create_app() -> gr.Blocks:
    with gr.Blocks(title="Kiosk Core UI") as app:
        with gr.Column(elem_classes=["kiosk-shell"]):
            with gr.Column(elem_classes=["kiosk-hero"]):
                gr.HTML(
                    """
                    <div class="kiosk-title">
                      <h1>Kiosk Voice Assistant</h1>
                      <p>Speak a question, watch the transcription appear, then follow the live answer and audio playback.</p>
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
                    type="filepath",
                    format="wav",
                    label="Tap the microphone, speak, then stop recording",
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

            mic_input.start_recording(
                fn=begin_recording,
                inputs=None,
                outputs=[mic_input, transcript_box, response_box, tts_audio, status_box],
            )
            mic_input.stop_recording(
                fn=process_turn,
                inputs=[mic_input],
                outputs=[mic_input, transcript_box, response_box, tts_audio, status_box],
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