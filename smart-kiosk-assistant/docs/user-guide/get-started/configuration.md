# Configuration

`kiosk-core` and `kiosk-ui` are configured through environment variables
(see [Environment Variables](#environment-variables)).

The three model-hosting services (`audio-analyzer`, `text-to-speech`,
`rag-service`) are configured through YAML files that the kiosk pins
and mounts into the containers. The most common changes are the
[model](#model-selection) and the [inference device](#inference-device).

## Model Selection

Each model-hosting service reads the model identifier from the same
pinned config file used for device selection:

| Service | File | Model fields |
|---|---|---|
| `audio-analyzer` | [`configs/audio-analyzer/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/audio-analyzer/config.yaml) | `models.asr.name` (e.g. `whisper-tiny`, `whisper-base`); `sentiment.model` (optional) |
| `text-to-speech` | [`configs/text-to-speech/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/text-to-speech/config.yaml) | `models.tts.name` (e.g. `microsoft/speecht5_tts`, Qwen-TTS variant); `model_variant` |
| `rag-service` | [`rag-service/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/rag-service/config.yaml) | `models.llm.hf_id`, `models.embedding.hf_id`, `retrieval.reranker.hf_id`; per-model `weight_format` (`int4`, `int8`, `fp16`) |

Use Hugging Face IDs where the field name is `hf_id`. Models are
downloaded and exported on first start into the per-service `models/`
directory; subsequent starts reuse the cache.

### Supported / validated models

The kiosk ships with the following defaults. These are the models the
stack has been validated with — they are the recommended starting point.
The **Devices** column lists the supported inference devices for each:

| Service | Field | Default (validated) | Other examples | Devices |
|---|---|---|---|---|
| `audio-analyzer` ASR | `models.asr.name` | `whisper-base` | `whisper-tiny`, `whisper-small`, `whisper-medium`, `whisper-large` | `CPU`, `GPU`, `NPU` (`GPU`/`NPU` require `provider: openvino`) |
| `audio-analyzer` sentiment | `sentiment.model` | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | other SpeechBrain emotion-recognition models | `CPU`, `GPU` (disabled by default) |
| `text-to-speech` | `models.tts.name` | `microsoft/speecht5_tts` (SpeechT5) | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` (Qwen-TTS) | `CPU`, `GPU` (`int4` on iGPU produces noise; use `fp16` or `int8` on GPU) |
| `rag-service` LLM | `models.llm.hf_id` | `Qwen/Qwen3-4B-Instruct-2507` | other OpenVINO-exportable instruct LLMs | `CPU`, `GPU` (`GPU` recommended for acceptable latency) |
| `rag-service` embedding | `models.embedding.hf_id` | `BAAI/bge-large-en-v1.5` | `BAAI/bge-base-en-v1.5`, `BAAI/bge-small-en-v1.5` | `CPU`, `GPU` (`CPU` is usually fast enough) |
| `rag-service` reranker | `retrieval.reranker.hf_id` | `BAAI/bge-reranker-base` | `BAAI/bge-reranker-large` | `CPU`, `GPU` (optional) |

> [!IMPORTANT]
> **Changing models is at your own discretion.** The defaults above are
> the only combinations validated with this stack. Configuring models,
> variants, devices, or precisions other than the defaults may negatively
> affect the functionality, accuracy, latency, or stability of the
> application. You are responsible for ensuring the configuration you
> choose is correct and works for your use case — make changes only if you
> understand the implications.
>
> In particular:
> - Some models do not function properly at aggressive quantization. If a
>   model produces garbled, empty, or low-quality output at `int4`, switch
>   that model's `weight_format`/`dtype` to `int8` or `fp16`.
> - A model must be exportable to OpenVINO IR for the OpenVINO backend; not
>   every Hugging Face model is supported.
> - Larger models increase first-run download/export time, memory use, and
>   per-request latency, and may not fit on the selected device.
> - After any change, restart the affected service and verify it loads and
>   responds correctly before relying on it.

## Inference Device

Each model-hosting service reads its device from a pinned config file:

| Service | File | Fields |
|---|---|---|
| `audio-analyzer` | [`configs/audio-analyzer/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/audio-analyzer/config.yaml) | `models.asr.device`, `sentiment.device` |
| `text-to-speech` | [`configs/text-to-speech/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/configs/text-to-speech/config.yaml) | `models.tts.device` |
| `rag-service` | [`rag-service/config.yaml`](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/rag-service/config.yaml) | `models.llm.device`, `models.embedding.device`, `retrieval.reranker.device` |

The supported devices for each model are listed in the
[Supported / validated models](#supported--validated-models) table above.

Use uppercase device names (`CPU`, `GPU`, `NPU`). `rag-service` expects
them as quoted strings; `audio-analyzer` and `text-to-speech` unquoted.

After editing, restart the affected service and confirm OpenVINO picked
the device:

```bash
docker compose up -d --build --force-recreate <service-name>
docker compose logs <service-name> | grep -i -E "device|compiling|GPU|CPU"
```

OpenVINO prints a `Compiling model on <DEVICE>` line on first load.

> GPU execution is delegated to the OpenVINO backend used by each
> service. Whether a given model actually runs on GPU and how it
> performs depends on the OpenVINO version and operator coverage for
> that model.

## Audio Analyzer ASR Provider/Device (`config.yaml`)

For ASR provider/device selection, use:

1. `configs/audio-analyzer/config.yaml`

This repository treats it as the single source of truth. Configure:

- `models.asr.provider`
- `models.asr.device`

There is exactly one checked-in Compose file in this project: `docker-compose.yml`.
No checked-in hardware-specific Compose override files are used (the Makefile may generate a temporary runtime override under `/tmp` to inject `/dev/accel` for the OpenVINO+NPU case).
`docker-compose.yml` provides container runtime access, while ASR
provider/device selection remains in `config.yaml` only. Provider/device
is validated before startup by `make check-env`; startup is rejected early
with actionable errors when configured hardware is unavailable.

The NPU device mapping is controlled by `ACCEL_MOUNT_PATH` in the Compose
environment. The checked-in Compose file defaults that mapping to
`/dev/null`, so CPU/GPU-only hosts can start cleanly. For OpenVINO + NPU,
`make up` auto-detects the host Intel NPU node, validates that OpenVINO can
see it inside the container, and exports that path for the Compose run.
For direct `docker compose up`, set `ACCEL_MOUNT_PATH` in `.env` or the
shell to the host NPU device node first.

Recommended workflow for OpenVINO + NPU:

```bash
make check-env
make up
```

When `models.asr.provider=openvino` and `models.asr.device=NPU`, `make`
automatically detects a host NPU node under `/dev/accel/accel*` (for
example `/dev/accel/accel0`) and passes it to Compose through
`ACCEL_MOUNT_PATH`.

Direct Compose workflow (no Makefile wrapper):

```bash
ACCEL_MOUNT_PATH=/dev/accel/accel0 docker compose up -d audio-analyzer
```

`/dev/accel/accel0` is a common host path on Meteor Lake systems, but the
host path may differ by platform. `ACCEL_MOUNT_PATH` always refers to the
host NPU device node and is mapped into the container as
`/dev/accel/accel0`.

### Enable OpenVINO Whisper on NPU

Edit `configs/audio-analyzer/config.yaml`:

```yaml
models:
  asr:
    provider: openvino
    device: NPU
```

Then restart `audio-analyzer`:

```bash
make check-env
make up
```

If you intentionally run Compose directly (without `make`), provide the NPU
host device path explicitly:

```bash
ACCEL_MOUNT_PATH=/dev/accel/accel0 docker compose up -d audio-analyzer
```

Verify effective environment and health:

```bash
docker exec audio-analyzer env | grep AUDIO_ANALYZER
docker logs audio-analyzer
curl http://localhost:8010/health
```

Also verify that compose is not injecting ASR provider/device overrides:

```bash
docker compose config | grep AUDIO_ANALYZER__MODELS__ASR__
```

The provider/device override keys should not be present.

Hardware checks happen before startup:

- `provider=openvino, device=CPU` does not require GPU/NPU nodes.
- `provider=openvino, device=GPU` requires Intel GPU detection.
- `provider=openvino, device=NPU` requires Intel NPU detection and
  container-level OpenVINO visibility.
- Missing requested hardware fails during `make check-env` (no fallback).

`ACCEL_MOUNT_PATH` drives NPU passthrough for `audio-analyzer`.
CPU/GPU paths do not require `/dev/accel`; the Compose default maps
`/dev/null` instead.

Look for startup log lines showing OpenVINO Whisper loaded on `NPU`
(for example `Loading Model: model name=whisper-base, device=NPU`) and
`Application startup complete.`

### Other Supported ASR Configurations

OpenAI + CPU:

```yaml
models:
  asr:
    provider: openai
    device: CPU
```

OpenVINO + GPU:

```yaml
models:
  asr:
    provider: openvino
    device: GPU
```

OpenVINO + CPU:

```yaml
models:
  asr:
    provider: openvino
    device: CPU
```

### Unsupported Combinations: OpenAI + GPU/NPU

`models.asr.provider=openai` supports `CPU` only in this stack.
The following are not supported by the current OpenAI/PyTorch Whisper backend:

- `models.asr.provider=openai` with `models.asr.device=GPU`
- `models.asr.provider=openai` with `models.asr.device=NPU`

- Use `openvino + NPU` for NPU execution.
- Use `openvino + GPU` for GPU execution.
- For OpenAI/PyTorch Whisper, use a supported device such as `CPU`.

### ASR Support Matrix

| Provider | CPU | GPU | NPU |
|---|---|---|---|
| `openai` | Yes | No | No |
| `whispercpp` | Yes | No | No |
| `openvino` | Yes | Yes (Intel GPU required) | Yes (Intel NPU required, and `ACCEL_MOUNT_PATH` must point at the host NPU device for Compose runs) |

If `GPU` or `NPU` is configured and unavailable on the host,
`make check-env` fails before any container startup. The stack does not
silently fall back to another device.

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

### Kiosk UI runtime mode {#kiosk_ui_mode}

The React kiosk UI (`kiosk-ui/`) ships as a single image that can serve
either of two screens, selected at container start — no rebuild:

| Variable | Default | Description |
|---|---|---|
| `KIOSK_UI_MODE` | `operator` | `operator` — chat transcript + performance dashboard (existing behaviour), served on port 7860. `customer` — single-view kiosk screen with a queue-aware menu, live cart, and a voice-only "Ask" button, intended for the physical kiosk touchscreen. |

The value is written to `/usr/share/nginx/html/config.js` by
`docker-entrypoint.sh` (installed as an nginx `docker-entrypoint.d`
script) and read by the SPA before the React bundle loads. In
`docker-compose.yml` the two screens are separate containers
(`kiosk-ui` and `kiosk-ui-customer`) built from the *same* image/context,
published on different host ports (`7860` and `7861` respectively) so
they can be shown on two separate monitors during a demo.

## Compose Defaults

When running with the top-level [docker-compose.yml](https://github.com/intel-retail/voice-enabled-interactions/blob/main/smart-kiosk-assistant/docker-compose.yml), the defaults are wired to the internal Compose network:

- `KIOSK_CORE_ANALYZER_URL=http://audio-analyzer:8010/v1/audio/transcriptions`
- `KIOSK_CORE_RAG_URL=http://rag-service:8020/api/v1/query`
- `KIOSK_CORE_TTS_URL=http://text-to-speech:8011/v1/audio/speech`
- `KIOSK_CORE_UI_BASE_URL=http://kiosk-core:8012`

Most deployments should leave these values unchanged. Override them only when `kiosk-core` or `kiosk-ui` must call services outside the local Compose stack.

## Session Parameters

Session parameters (chunk duration, silence threshold, etc.) can also be provided per-request in the POST body for `/api/v1/sessions/start` and `/api/v1/sessions/start-file`. Per-request values take precedence over the environment variable defaults.

---

## NPU Deployment Workflow

This section provides a complete step-by-step workflow to run the Smart Kiosk Assistant with Intel NPU acceleration for Whisper ASR.

> **Which services support NPU?**
> Only `audio-analyzer` supports NPU (`provider: openvino, device: NPU`). All other services use CPU (kiosk-core, rag-service, text-to-speech) or GPU (ovms-llm). Do not set NPU on other services — they will fail to start.

### 1 — System requirements

| Requirement | Details |
|---|---|
| Hardware | Intel Core Ultra (Meteor Lake or later) with integrated NPU |
| Host driver | Intel NPU driver (`intel-npu-driver`) installed and loaded |
| User-space runtime | `intel-level-zero-npu` package |
| Host device | `/dev/accel/accel0` (or similar) present and accessible |
| OpenVINO | Container image already bundles the correct runtime |

Verify the NPU device node is present before proceeding:
```bash
ls /dev/accel/
# Expected: accel0   accelmon0
```

### 2 — Install the Intel NPU driver (if not already installed)

Refer to the Intel NPU driver repository: <https://github.com/intel/linux-npu-driver/releases>. Installation varies by distribution. After installation:

```bash
# Verify kernel driver is loaded
lsmod | grep intel_vpu
# Verify device node exists
ls -la /dev/accel/accel0
```

### 3 — Set NPU device in audio-analyzer config

Edit `smart-kiosk-assistant/configs/audio-analyzer/config.yaml`:

```yaml
models:
  asr:
    provider: openvino
    device: NPU
    name: whisper-base  # whisper-tiny recommended for NPU latency targets
    weight_format: null  # NPU uses FP16 by default; INT8 is not required
```

> **Model recommendation for NPU:** Use `whisper-tiny` or `whisper-base`. Larger models increase NPU compiler warmup time on first inference.

### 4 — Start the stack

The recommended path is `make up`, which auto-detects the NPU device node and validates OpenVINO visibility before starting:

```bash
cd smart-kiosk-assistant
make check-env
make up
```

`make` automatically:
- Detects `/dev/accel/accel*` on the host
- Sets `ACCEL_MOUNT_PATH` to the detected device node
- Passes `ACCEL_MOUNT_PATH` into the Compose invocation

If you use `docker compose` directly, set `ACCEL_MOUNT_PATH` yourself:

```bash
ACCEL_MOUNT_PATH=/dev/accel/accel0 docker compose up -d
```

### 5 — Verify NPU is active

```bash
# Check audio-analyzer started healthy
docker ps --filter "name=audio-analyzer" --format "{{.Names}}\t{{.Status}}"

# Confirm NPU device is visible to OpenVINO inside the container
docker exec audio-analyzer python3 -c "import openvino as ov; print(ov.Core().available_devices)"
# Expected output includes: NPU

# Check the service is using the NPU provider
curl -s http://localhost:8010/v1/model-info | python3 -m json.tool
# Look for: "provider": "openvino", "device": "NPU"

# Check logs for successful NPU model load
docker logs audio-analyzer 2>&1 | grep -i "npu\|compile"
```

### 6 — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Container unhealthy, `NPU not in available_devices` | NPU driver not loaded or `/dev/accel/accel0` not mapped | Verify host driver and set `ACCEL_MOUNT_PATH` |
| `libopenvino_intel_npu_compiler_loader.so` missing | NPU compiler not in image | Rebuild the `audio-analyzer` image with NPU user-space packages |
| Slow first inference (20–60 s) | NPU compiler cache is empty (cold start) | Normal on first run; subsequent requests will be fast |
| Non-NPU containers unhealthy after NPU config change | NPU-unrelated services picking up wrong env | Only modify `configs/audio-analyzer/config.yaml`; do not add `ASR_DEVICE=NPU` to `.env` (affects all services) |

> **Cold-start note:** The OpenVINO NPU compiler caches compiled kernels inside the container under `/tmp/ov_cache/`. The first inference after a container restart takes significantly longer (20–60 s) while the cache warms up. This is expected behavior.

