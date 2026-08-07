"""FastAPI application for the queue-service HTTP API.

Exposes:
  GET /health                   — liveness probe
  GET /api/v1/queue/count       — latest queue count + status (JSON)
  GET /stream                   — MJPEG live video stream

The uvicorn server is started in a daemon thread from ``main.py`` before the
GLib main loop starts so both coexist in the same process.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

import frame_buffer
import queue_state

logger = logging.getLogger(__name__)

# Comment frame sent when the count has not changed. It keeps intermediaries
# (nginx has proxy_read_timeout 300s in front of this service) and the browser
# from reaping a connection that is simply idle because the queue is stable.
_HEARTBEAT_SECONDS = 15.0


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Publish the uvicorn event loop so the GStreamer thread can notify it."""
    queue_state.bind_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="queue-service", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/queue/count")
async def get_queue_count() -> JSONResponse:
    """Return the latest queue count with LOW / MEDIUM / HIGH status label.

    Retained alongside ``/api/v1/queue/events`` as the fallback for clients
    that cannot hold a streaming connection open.
    """
    return JSONResponse(content=queue_state.get())


@app.get("/api/v1/queue/events")
async def queue_events(request: Request) -> StreamingResponse:
    """Push the queue count to the browser as Server-Sent Events.

    Replaces the UI's 2-second poll, which showed a change up to two seconds
    late and spent most requests re-fetching an identical value. The current
    snapshot is sent immediately on connect so a freshly-opened tab is never
    blank, then one event per genuine change.

    ``EventSource`` reconnects on its own, so no retry logic is needed on the
    client; ``retry:`` sets how long it waits before doing so.
    """

    async def generator():
        queue = queue_state.subscribe()
        try:
            yield f"retry: {int(_HEARTBEAT_SECONDS * 1000)}\n\n".encode()
            yield f"event: queue\ndata: {json.dumps(queue_state.get())}\n\n".encode()

            while True:
                if await request.is_disconnected():
                    break
                try:
                    snapshot = await asyncio.wait_for(
                        queue.get(), timeout=_HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Count unchanged for a while — comment frame keeps the
                    # connection (and any proxy in front of it) alive.
                    yield b": ping\n\n"
                    continue
                yield f"event: queue\ndata: {json.dumps(snapshot)}\n\n".encode()
        except asyncio.CancelledError:
            raise
        finally:
            queue_state.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            # Belt and braces: nginx already sets proxy_buffering off for
            # /queue-svc/, but this makes the stream correct behind any proxy
            # that honours the header instead.
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/stream")
async def mjpeg_stream() -> StreamingResponse:
    """Deliver a multipart/x-mixed-replace MJPEG stream.

    The browser renders this with a plain ``<img src="/stream">``.  Each frame
    is sent as soon as it is available; when no new frame has arrived the
    server waits briefly before polling again so the connection stays alive
    without spinning.

    ``frame_buffer.get_jpeg()`` performs the JPEG encode lazily, so it is
    dispatched to a worker thread to keep the event loop responsive.
    """

    async def generator():
        last: bytes | None = None
        while True:
            frame = await asyncio.to_thread(frame_buffer.get_jpeg)
            if frame is None:
                # No frame yet — wait a little and retry
                await asyncio.sleep(0.1)
                continue
            if frame is not last:
                last = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            else:
                # Same frame — keep connection alive without flooding
                await asyncio.sleep(0.005)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
