from pathlib import Path

import httpx

from kiosk_core import config
from kiosk_core.speech_normalizer import for_speech


class TtsClient:
    def __init__(self, tts_url: str, timeout_seconds: float | None = None):
        self.tts_url = tts_url
        self.timeout_seconds = timeout_seconds or config.DEFAULT_HTTP_TIMEOUT_SECONDS

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        model: str,
        voice: str | None = None,
        language: str | None = None,
        instructions: str | None = None,
    ) -> None:
        # Applied here, not at each call site, so every path (streamed
        # clauses, the pre-synthesized opener) speaks prices/times instead
        # of reading raw symbols/digits — see speech_normalizer.py.
        if config.DEFAULT_TTS_SPEECH_NORMALIZE_ENABLED:
            text = for_speech(text)

        payload = {
            "model": model,
            "input": text,
            "response_format": "wav",
        }
        if voice:
            payload["voice"] = voice
        if language:
            payload["language"] = language
        if instructions:
            payload["instructions"] = instructions

        with httpx.Client(timeout=self.timeout_seconds, trust_env=False) as client:
            response = client.post(self.tts_url, json=payload)
            response.raise_for_status()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)