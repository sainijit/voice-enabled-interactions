# Get Started

This page walks through the recommended positive-flow setup for Smart Kiosk
Assistant: clone the repository, start the full stack, and confirm a
working voice request end to end before exploring any advanced
configuration.

## Before You Begin

- Confirm your machine meets the
  [system requirements](system-requirements.md).
- Decide which inference device you want each model-hosting service to use
  (CPU is a safe default; GPU is recommended for the RAG LLM). You can
  change this later — see
  [configuration.md](configuration.md#choosing-the-inference-device-cpu--gpu--npu).

## Step 1: Clone and Prepare Sources

Clone the repository and populate the two upstream microservices the stack
needs from the `edge-ai-libraries` submodule:

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone \
  microservices/audio-analyzer microservices/text-to-speech
cd smart-kiosk-assistant
```

If the repository is already cloned, run the two `git` commands above from
the repository root.

## Step 2: (Optional) Pick a Device per Service

The stack ships with a safe CPU/GPU default. Skip this step on a first
run and come back once you have the positive flow working.

To pin a specific device, edit the corresponding YAML:

| Service | File | Field(s) |
|---|---|---|
| `audio-analyzer` | `configs/audio-analyzer/config.yaml` | `models.asr.device` (and `sentiment.device`) |
| `text-to-speech` | `configs/text-to-speech/config.yaml` | `models.tts.device` |
| `rag-service` | `rag-service/config.yaml` | `models.llm.device`, `models.embedding.device`, `retrieval.reranker.device` |

See [configuration.md](configuration.md#choosing-the-inference-device-cpu--gpu--npu)
for the supported values per service and the recommended combinations.

## Step 3: Start the Stack

From the `smart-kiosk-assistant/` directory:

```bash
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
docker compose build
docker compose up -d
```

Exporting `LOCAL_UID` and `LOCAL_GID` keeps bind-mounted files writable
from the host user that launched the stack.

Compose starts five containers:

| Container | Port | Purpose |
|---|---|---|
| `audio-analyzer` | `8010` | Speech-to-text |
| `text-to-speech` | `8011` | Speech synthesis |
| `rag-service` | `8020` | Knowledge-base retrieval |
| `kiosk-core` | `8012` | FastAPI session API |
| `kiosk-ui` | `7860` | Gradio voice UI |

First startup can take a few minutes because the model-hosting services
download and export model assets on demand.

## Step 4: Verify the Stack Is Healthy

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8010/health   # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health   # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health   # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health   # kiosk-core
```

Every response should be `{"status": "ok"}`.

To confirm the selected device was actually picked up, tail the service
logs and look for the OpenVINO `Compiling ... on <DEVICE>` line:

```bash
docker compose logs audio-analyzer | grep -i -E "device|compiling|GPU|NPU|CPU"
docker compose logs text-to-speech | grep -i -E "device|compiling|GPU|NPU|CPU"
docker compose logs rag-service    | grep -i -E "device|compiling|GPU|NPU|CPU"
```

## Step 5: Run the Positive Voice Flow

Open the Gradio UI in a browser:

```text
http://127.0.0.1:7860
```

Then:

1. When the browser prompts for microphone permission, click **Allow**.
2. Use the **knowledge base** panel to ingest the bundled sample content,
   or upload your own `.txt` / `.md` files. Sample content is in
   [knowledge-base-samples/](../knowledge-base-samples/).
3. Click the microphone button and ask a question that the knowledge base
   can answer (for example, "What are the store hours?").
4. Wait for the assistant card to update with:
   - the transcript of what you said,
   - the streamed answer text,
   - a spoken response that plays back automatically.

If all three appear, the positive flow is working end to end and you are
ready to explore the rest of the documentation.

## Next Steps

- [How It Works](how-it-works.md) — request flow and architecture diagram.
- [Configuration](configuration.md) — environment variables, per-service
  device selection, and per-request overrides.
- [Run With Docker Compose](run-container.md) — full container-mode
  reference including logs, restart, and teardown.
- [Run On The Host](run-standalone.md) — develop `kiosk-core` and the UI
  outside Docker while keeping the model-hosting services in containers.
- [Build From Source](build-from-source.md) — image build details.
- [API Reference](api-reference.md) — `kiosk-core` HTTP API.
- [Troubleshooting](troubleshooting.md) — common startup and runtime issues.
