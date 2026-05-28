# How It Works

This page describes the architecture and the internal flow of a single
voice request through Smart Kiosk Assistant.

## Architecture

Smart Kiosk Assistant runs as five cooperating services on a single host.
The browser captures microphone audio and uploads it to `kiosk-core`, which
orchestrates speech-to-text, retrieval-augmented answer generation, and
speech synthesis through three model-hosting microservices.

![Smart Kiosk Assistant architecture](assets/architecture.png)

## Components

- **`kiosk-ui`** — Gradio browser interface. Captures microphone audio
  via the Web Audio API and posts it to `kiosk-core`. Polls the session
  endpoint until the answer text and generated audio are available, then
  plays the audio clips back in order.
- **`kiosk-core`** — FastAPI session orchestrator. Owns the per-session
  state machine, forwards audio to `audio-analyzer`, sends the
  transcription to `rag-service`, and streams the generated answer
  sentence-by-sentence to `text-to-speech`.
- **`audio-analyzer`** — OpenAI-compatible speech-to-text microservice
  built on Whisper and OpenVINO. Configurable device (`CPU` or `GPU`).
- **`rag-service`** — Local retrieval-augmented generation microservice.
  Hosts a Qwen LLM, a BGE embedding model, and a BGE reranker, all on
  OpenVINO. Each model can be pinned to `CPU` or `GPU` independently.
- **`text-to-speech`** — OpenVINO TTS microservice. Supports the SpeechT5
  family (`CPU`, `GPU`) and the Qwen-TTS family (`CPU`, `GPU`, `NPU`).

`kiosk-core` and `kiosk-ui` host no models and do not require a device
selection. All inference happens inside the three model-hosting services.

## Request Flow

1. **Capture** — The browser records a microphone utterance and uploads
   it to `kiosk-core` as a WAV file along with session parameters.
2. **Session start** — `kiosk-core` creates a session, returns the
   `session_id` immediately, and runs the rest of the pipeline in the
   background. The UI polls
   `GET /api/v1/sessions/{session_id}` to track progress.
3. **Speech-to-text** — `kiosk-core` chunks the upload at silence
   boundaries and forwards each chunk to `audio-analyzer`. The combined
   transcript is appended to the session snapshot.
4. **Retrieval-augmented answer** — When the user has finished speaking
   (silence timeout or max-duration reached), `kiosk-core` sends the
   transcript and recent conversation history to `rag-service`.
   `rag-service`:
   - embeds the question with the BGE embedding model,
   - retrieves candidate chunks from Chroma,
   - optionally reranks them with the BGE cross-encoder,
   - prompts the Qwen LLM with the retrieved context, and
   - streams the answer back token-by-token.
5. **Speech synthesis** — As `kiosk-core` receives the answer stream, it
   splits the text into sentences and posts each sentence to
   `text-to-speech`. The generated WAV files are written to the shared
   `generated_audio/` volume and recorded in the session snapshot.
6. **Playback** — The browser UI sees new `tts_audio_segments` in the
   session snapshot, downloads them from `kiosk-core`, and plays them
   sequentially.

## Per-Service Device Placement

Each model-hosting service decides where its models execute. The
selection is read from that service's pinned configuration:

| Service | Config file | Device fields |
|---|---|---|
| `audio-analyzer` | `configs/audio-analyzer/config.yaml` | `models.asr.device`, `sentiment.device` |
| `text-to-speech` | `configs/text-to-speech/config.yaml` | `models.tts.device` |
| `rag-service` | `rag-service/config.yaml` | `models.llm.device`, `models.embedding.device`, `retrieval.reranker.device` |

See [configuration.md](configuration.md#choosing-the-inference-device-cpu--gpu--npu)
for supported values and recommended combinations.

## Configuration Surface

- `kiosk-core` and `kiosk-ui` are configured purely through environment
  variables — see [configuration.md](configuration.md#environment-variables).
- The three model-hosting services read pinned YAML files mounted from
  `configs/` and `rag-service/`. See
  [configs/README.md](../configs/README.md) for why the indirection
  exists.

## Where to Go Next

- [Get Started](get-started.md) — recommended positive flow.
- [API Reference](api-reference.md) — `kiosk-core` HTTP surface.
- [Run With Docker Compose](run-container.md) and
  [Run On The Host](run-standalone.md) — deployment recipes.
