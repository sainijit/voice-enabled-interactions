# Run With Docker Compose

`docker compose up` starts `audio-analyzer`, `text-to-speech`,
`rag-service`, `kiosk-core` (REST API), and `kiosk-ui` (Gradio interface)
as containers, using the prebuilt images published on Docker Hub.

Microphone audio is captured by the browser and uploaded to `kiosk-core`
as a WAV file. No host audio device is passed into the containers.

To rebuild the images from source instead of pulling, see
[build-from-source.md](build-from-source.md). To run `kiosk-core` and
the UI directly on the host, see [run-standalone.md](run-standalone.md).

## Clone and Prepare

```bash
git clone https://github.com/intel-retail/voice-enabled-interactions.git
cd voice-enabled-interactions
git submodule update --init --depth 1 edge-ai-libraries
git -C edge-ai-libraries sparse-checkout set --cone \
  microservices/audio-analyzer microservices/text-to-speech
cd smart-kiosk-assistant
```

If the repository is already cloned, run the two `git` commands above
from the repository root.

## Pull And Start

From `smart-kiosk-assistant/`:

```bash
docker compose pull
docker compose up -d
```

`docker compose pull` fetches all five images from Docker Hub:

- `intel/audio-analyzer:${RELEASE_TAG}`
- `intel/text-to-speech:${RELEASE_TAG}`
- `intel/rag-service:${RELEASE_TAG}`
- `intel/kiosk-core:${RELEASE_TAG}`
- `intel/kiosk-ui:${RELEASE_TAG}`

`REGISTRY` and `RELEASE_TAG` are read from [.env](../.env) (defaults
`REGISTRY=intel`, committed `RELEASE_TAG` pins the current release).

This starts five containers:

| Container | Port | Purpose |
|---|---|---|
| `audio-analyzer` | 8010 | Speech-to-text |
| `text-to-speech` | 8011 | Speech synthesis |
| `rag-service` | 8020 | Knowledge-base retrieval |
| `kiosk-core` | 8012 | FastAPI session API |
| `kiosk-ui` | 7860 | Gradio voice UI |

Containers run as non-root; the published images default to UID/GID
`1000:1000`. To match a different host user, export `LOCAL_UID` /
`LOCAL_GID` (those vars are only consumed during a local build, so the
pull flow uses the image defaults).

## Verify

```bash
docker compose ps
curl --noproxy '*' http://127.0.0.1:8012/health   # {"status":"ok"}
```

Open `http://127.0.0.1:7860` in a browser, click the microphone, and
speak your question.

## Logs

```bash
docker compose logs -f kiosk-core
docker compose logs -f kiosk-ui
```

## Restart / Stop

```bash
docker compose restart            # after env var change
docker compose pull && docker compose up -d   # after a new release tag
docker compose down               # teardown
```

## Notes

- The default Compose wiring connects `kiosk-core` and `kiosk-ui` to the
  internal `audio-analyzer`, `rag-service`, and `text-to-speech`
  containers. Override these URLs only when this stack must call
  services outside the local Compose network.
- See [configuration.md](configuration.md) for environment variables,
  model selection, and inference device, and
  [api-reference.md](api-reference.md) for endpoint details.
