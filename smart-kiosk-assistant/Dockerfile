# Dockerfile — kiosk-core / kiosk-ui
#
# Default CMD starts the kiosk-core FastAPI service on port 8012.
# The kiosk-ui service in docker-compose.yml overrides CMD to run gradio_app.py.
#
# Mic audio is NOT captured server-side in this image.
# The Gradio UI captures audio in the browser (Web Audio API) and uploads it
# as a file.  For host-side mic capture, run kiosk-core standalone instead —
# see docs/run-standalone.md.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libportaudio2 is required by the sounddevice wheel at import time even
# when no mic device is used (Gradio file-upload mode).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 8012

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
