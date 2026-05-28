# Configuration

This page covers two configuration surfaces:

1. `kiosk-core` and `kiosk-ui` are configured purely through **environment
   variables** (see [Environment Variables](#environment-variables)).
2. The three model-hosting services (`audio-analyzer`, `text-to-speech`,
   `rag-service`) are configured through **YAML files** that the kiosk
   pins and mounts into the containers. The most common change here is
   the **inference device** — see
   [Choosing the Inference Device (CPU / GPU / NPU)](#choosing-the-inference-device-cpu--gpu--npu).

## Choosing the Inference Device (CPU / GPU / NPU)

Smart Kiosk Assistant hosts models in three of its five services. Each can
be pinned to a specific Intel inference device at deployment time.

### Where the Device Is Configured

| Service | File | Field(s) |
|---|---|---|
| `audio-analyzer` | [`configs/audio-analyzer/config.yaml`](../configs/audio-analyzer/config.yaml) | `models.asr.device`, `sentiment.device` |
| `text-to-speech` | [`configs/text-to-speech/config.yaml`](../configs/text-to-speech/config.yaml) | `models.tts.device` |
| `rag-service` | [`rag-service/config.yaml`](../rag-service/config.yaml) | `models.llm.device`, `models.embedding.device`, `retrieval.reranker.device` |

The pinned YAML for `audio-analyzer` and `text-to-speech` lives under
`configs/` and is mounted into the upstream container without forking the
service — see [configs/README.md](../configs/README.md). `rag-service`
reads its own `config.yaml` directly.

### Supported Devices per Service

| Service | Model | Supported devices | Notes |
|---|---|---|---|
| `audio-analyzer` | Whisper ASR | `CPU`, `GPU` | `GPU` requires `provider: openvino` in the same block. |
| `audio-analyzer` | Voice sentiment (optional) | `CPU`, `GPU` | `GPU` requires `provider: openvino`. Disabled by default. |
| `text-to-speech` | SpeechT5 (default) | `CPU`, `GPU` | `int4` on iGPU produces noise; prefer `fp16` or `int8` on GPU. |
| `text-to-speech` | Qwen-TTS variant | `CPU`, `GPU`, `NPU` | `NPU` only supported by this variant. |
| `rag-service` | LLM (`models.llm.device`) | `CPU`, `GPU` | `GPU` strongly recommended for acceptable latency. |
| `rag-service` | Embedding (`models.embedding.device`) | `CPU`, `GPU` | `CPU` is usually fast enough. |
| `rag-service` | Reranker (`retrieval.reranker.device`) | `CPU`, `GPU` | Optional; disable in config to skip. |

### Recommended Combinations

| Goal | audio-analyzer | text-to-speech | rag-service (LLM / embed / rerank) |
|---|---|---|---|
| Safe default, mixed Intel hardware | `CPU` | `CPU` | `GPU` / `CPU` / `GPU` |
| Maximum throughput on Core Ultra + iGPU | `GPU` | `GPU` | `GPU` / `CPU` / `GPU` |
| Free the GPU for the LLM | `CPU` | `CPU` | `GPU` / `CPU` / `CPU` |

### Steps to Change a Device

1. Edit the relevant field(s) in the table above. Use uppercase device
   names (`CPU`, `GPU`, `NPU`) without quotes for `audio-analyzer` and
   `text-to-speech`, and quoted strings (`"CPU"`, `"GPU"`) for
   `rag-service` to stay consistent with the rest of that file.
2. Restart only the affected service so it picks up the new device:

   ```bash
   docker compose up -d --build --force-recreate audio-analyzer
   docker compose up -d --build --force-recreate text-to-speech
   docker compose up -d --build --force-recreate rag-service
   ```

3. Confirm the device was actually used by tailing the logs:

   ```bash
   docker compose logs <service-name> | grep -i -E "device|compiling|GPU|NPU|CPU"
   ```

   OpenVINO prints a line such as `Compiling model on GPU` (or `NPU` /
   `CPU`) when the model is first loaded.

> **Note on NPU/GPU support.** NPU/GPU execution in this stack is delegated
> entirely to the OpenVINO backend used by each model-hosting service.
> Smart Kiosk Assistant exposes `NPU`/`GPU` as a valid device choice wherever
> the service config accepts it, but whether a specific model actually
> runs on the NPU/GPU — and how it performs — depends on the OpenVINO

## Environment Variables

kiosk-core has no config file. All settings are controlled through environment variables.

### kiosk-core API (`main:app`)

| Variable | Default | Description |
|---|---|---|
| `KIOSK_CORE_ANALYZER_URL` | `http://127.0.0.1:8010/v1/audio/transcriptions` | audio-analyzer transcription endpoint |
| `KIOSK_CORE_RAG_URL` | `http://127.0.0.1:8020/api/v1/query` | RAG query endpoint |
| `KIOSK_CORE_TTS_URL` | `http://127.0.0.1:8011/v1/audio/speech` | TTS speech synthesis endpoint |
| `KIOSK_CORE_TTS_MODEL` | `qwen-tts` | Model name sent to the TTS service |
| `KIOSK_CORE_TTS_VOICE` | *(unset)* | Voice name sent to the TTS service |
| `KIOSK_CORE_TTS_LANGUAGE` | `English` | Language sent to the TTS service |
| `KIOSK_CORE_TTS_INSTRUCTIONS` | *(unset)* | Optional style instructions for TTS |
| `KIOSK_CORE_SAMPLE_RATE` | `16000` | Default audio sample rate in Hz |
| `KIOSK_CORE_CHUNK_SECONDS` | `4.0` | Length of each audio chunk sent to audio-analyzer |
| `KIOSK_CORE_SILENCE_TIMEOUT_SECONDS` | `1.5` | Silence duration after speech that ends a session |
| `KIOSK_CORE_MAX_SESSION_SECONDS` | `20.0` | Hard cap on session duration |
| `KIOSK_CORE_SILENCE_THRESHOLD` | `900` | RMS threshold below which audio is treated as silence |
| `KIOSK_CORE_BLOCK_DURATION_SECONDS` | `0.1` | PortAudio capture block size |
| `KIOSK_CORE_PREROLL_SECONDS` | `0.3` | Audio buffered before speech starts |
| `KIOSK_CORE_HTTP_TIMEOUT_SECONDS` | `120.0` | HTTP client timeout for downstream calls |

### Gradio UI (`gradio_app.py`)

| Variable | Default | Description |
|---|---|---|
| `KIOSK_CORE_UI_BASE_URL` | `http://127.0.0.1:8012` | Base URL of the kiosk-core API |
| `KIOSK_CORE_UI_ANALYZER_URL` | `http://127.0.0.1:8010/v1/audio/transcriptions` | Passed to start-file sessions as `analyzer_url` |
| `KIOSK_CORE_UI_RAG_URL` | `http://127.0.0.1:8020/api/v1/query` | Passed to start-file sessions as `rag_url` |
| `KIOSK_CORE_UI_TTS_URL` | `http://127.0.0.1:8011/v1/audio/speech` | Passed to start-file sessions as `tts_url` |
| `KIOSK_CORE_UI_TIMEOUT_SECONDS` | `120.0` | HTTP client timeout in the UI |
| `KIOSK_CORE_UI_POLL_INTERVAL_SECONDS` | `0.35` | How often the UI polls for session state updates |

## Compose Defaults

When running with the top-level [docker-compose.yml](../docker-compose.yml), the defaults are wired to the internal Compose network:

- `KIOSK_CORE_ANALYZER_URL=http://audio-analyzer:8010/v1/audio/transcriptions`
- `KIOSK_CORE_RAG_URL=http://rag-service:8020/api/v1/query`
- `KIOSK_CORE_TTS_URL=http://text-to-speech:8011/v1/audio/speech`
- `KIOSK_CORE_UI_BASE_URL=http://kiosk-core:8012`

Most deployments should leave these values unchanged. Override them only when `kiosk-core` or `kiosk-ui` must call services outside the local Compose stack.

## Session Parameters

Session parameters (chunk duration, silence threshold, etc.) can also be provided per-request in the POST body for `/api/v1/sessions/start` and `/api/v1/sessions/start-file`. Per-request values take precedence over the environment variable defaults.
