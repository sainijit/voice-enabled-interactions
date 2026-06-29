# Get Started

Clone the repository, build the stack, and confirm a working voice
ordering session end to end.

Confirm your machine meets the
[System Requirements](./get-started/system-requirements.md) before starting.

## Step 1: Install Docker and Intel GPU Drivers

Skip if Docker is already installed (`docker --version` and
`docker compose version` both succeed).

```bash
# Docker Engine + Compose v2
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker
```

For Intel iGPU acceleration (recommended) install the GPU compute runtime:

```bash
sudo apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero
sudo usermod -aG render $USER && newgrp render
```

Verify the GPU device node exists: `ls /dev/dri/renderD*`.
If the packages are not found, follow the [Intel GPU driver guide](https://dgpu-docs.intel.com).
Skip this block and set `TARGET_DEVICE=CPU` in `.env` if no Intel GPU is available.

## Step 2: Clone the Repository

```bash
git clone https://github.com/sainijit/voice-enabled-interactions.git
cd voice-enabled-interactions/smart-kiosk-assistant
```

## Step 3: Clone `edge-ai-libraries` (Audio and TTS Source)

The `audio-analyzer` and `text-to-speech` images are built from the
[edge-ai-libraries](https://github.com/open-edge-platform/edge-ai-libraries)
monorepo. The compose file expects them at `../edge-ai-libraries/microservices/`,
so both repositories must sit side by side:

```bash
cd ..   # move to the parent directory that contains voice-enabled-interactions
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/open-edge-platform/edge-ai-libraries.git
git -C edge-ai-libraries sparse-checkout set \
    microservices/audio-analyzer microservices/text-to-speech
cd voice-enabled-interactions/smart-kiosk-assistant
```

## Step 4: Set Your HuggingFace Token

Open `.env` and set your HuggingFace token (free account at
https://huggingface.co/settings/tokens):

```bash
# .env
HF_TOKEN=hf_your_token_here
```

## Step 5: Download the LLM Model for OVMS

The ordering agent uses a Qwen3-4B model served by OVMS. Download it
before starting the stack:

```bash
# GPU (recommended, INT8 ~4 GB)
./setup_models.sh

# CPU only (no Intel GPU)
./setup_models.sh --device CPU

# INT4 model (smaller/faster, ~2 GB, slightly lower accuracy)
./setup_models.sh --int4
```

The script downloads the pre-converted OpenVINO model into `./models/`
and updates `OVMS_MODEL_NAME`, `TARGET_DEVICE`, and `RENDER_GID` in `.env`.
The download happens once — subsequent starts reuse the cached model.

## Step 6: Build Images and Start the Stack

```bash
docker compose build
docker compose up -d
```

`docker compose build` compiles all five service images from source.
First build takes 10–20 minutes. `ovms-llm` uses a pre-built upstream
image and is pulled automatically.

The stack starts seven containers:

| Container | Port | Role |
|---|---|---|
| `ovms-llm` | 8000 | Serves Qwen3-4B via OpenAI-compatible API |
| `metrics-collector` | 9000 | Hardware utilization metrics |
| `audio-analyzer` | 8010 | Whisper ASR + speaker diarization |
| `text-to-speech` | 8011 | SpeechT5 TTS synthesis |
| `rag-service` | 8020 | RAG pipeline + ordering agent |
| `kiosk-core` | 8012 | Session API + product ordering |
| `kiosk-ui` | 7860 | Voice kiosk React UI |

## Step 7: Verify the Stack Is Healthy

Services start in dependency order. Allow 2–5 minutes on first run for
model assets to download into Docker volumes.

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8000/v3/models   # ovms-llm
curl --noproxy '*' http://127.0.0.1:8010/health       # audio-analyzer
curl --noproxy '*' http://127.0.0.1:8011/health       # text-to-speech
curl --noproxy '*' http://127.0.0.1:8020/health       # rag-service
curl --noproxy '*' http://127.0.0.1:8012/health       # kiosk-core
```

Every health endpoint should return `{"status": "ok"}`. The OVMS endpoint
returns the active model name (`OpenVINO/Qwen3-4B-int8-ov`).

## Step 8: Open the Kiosk and Try Voice Ordering

Open a browser on the same machine:

```
http://127.0.0.1:7860
```

1. Click **Allow** when the browser asks for microphone permission.
   (Use `127.0.0.1`, not the machine hostname — browsers block microphone
   on non-HTTPS origins except `localhost`.)
2. Click the **🎤 microphone** button and speak. Try:
   - *"What's on the menu?"*
   - *"Show me your burgers"*
   - *"I'd like to order a Classic Chicken Burger"*
   - *"Confirm my order"*
3. Watch the **🍔 QSR** tab in the right panel — the **Menu** sub-tab
   shows the full catalogue; the **Cart** sub-tab shows your live order
   and the confirmed receipt after you say *"Confirm"*.

## Next Steps

- [How It Works](./how-it-works.md)
- [Configuration](./get-started/configuration.md)
- [Build From Source](./get-started/build-from-source.md)
- [API Reference](./api-reference.md)
- [Troubleshooting](./troubleshooting.md)

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements.md
./get-started/build-from-source.md
./get-started/run-container.md
./get-started/run-standalone.md
./get-started/configuration.md

:::
hide_directive-->
