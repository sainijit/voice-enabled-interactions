# Overview

Smart Kiosk Assistant is a voice-first, retrieval-augmented kiosk stack for
retail, QSR, and other customer-facing deployments. A browser session
captures microphone audio, streams it through speech-to-text, retrieves a
grounded answer from a local knowledge base, and plays a synthesized spoken
response back to the user. The full stack runs locally on Intel CPU, GPU,
or NPU using OpenVINO-backed services.

## Use Cases

- In-store and QSR self-service kiosks that answer customer questions about
  menus, hours, products, returns, and store policies.
- Voice-enabled lobby or receptionist assistants in showrooms, hospitality,
  and healthcare front-desks.
- Internal employee help desks where a curated knowledge base needs to be
  answerable by voice.
- Demonstration and evaluation of the Edge AI Suite voice pipeline
  (audio-analyzer + RAG + text-to-speech) end to end on a single host.

## Key Capabilities

- Browser-based microphone capture (Web Audio API) with no host audio
  device passthrough required.
- End-to-end voice loop: speech-to-text → retrieval-augmented answer →
  synthesized speech playback.
- Local knowledge-base ingestion with semantic chunking, ANN retrieval, and
  optional cross-encoder reranking.
- Selectable inference device (CPU, GPU, or NPU) per microservice; see
  [configuration.md](configuration.md#choosing-the-inference-device-cpu--gpu--npu).
- Sample knowledge-base content for QSR and retail scenarios.
- Single Docker Compose stack that starts all five services with pinned
  release images.

## Services in the Stack

| Service | Port | Role |
|---|---|---|
| `audio-analyzer` | `8010` | Speech-to-text (Whisper, OpenVINO) |
| `text-to-speech` | `8011` | Speech synthesis (SpeechT5 or Qwen-TTS) |
| `rag-service` | `8020` | Knowledge-base retrieval and answer generation |
| `kiosk-core` | `8012` | FastAPI session orchestrator |
| `kiosk-ui` | `7860` | Gradio browser interface |

Only `audio-analyzer`, `text-to-speech`, and `rag-service` host inference
models. `kiosk-core` and `kiosk-ui` are I/O-only and do not need a device
selection.

## Where to Go Next

- New to the stack? Start with [get-started.md](get-started.md).
- Want to understand the request flow? See [how-it-works.md](how-it-works.md).
- Choosing a device per service: see
  [configuration.md](configuration.md#choosing-the-inference-device-cpu--gpu--npu).
- Hardware and OS prerequisites: [system-requirements.md](system-requirements.md).
- HTTP API of `kiosk-core`: [api-reference.md](api-reference.md).
