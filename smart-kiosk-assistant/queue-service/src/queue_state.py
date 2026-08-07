"""Thread-safe singleton that stores the latest queue count and status.

Written by queue_counter.py (GLib/GStreamer thread) and read by api.py
(uvicorn/asyncio thread) without any shared coroutine state — a plain
threading.Lock is the correct primitive here.

This module also fans the count out to Server-Sent Events subscribers so the
UI is pushed a new value the moment it changes instead of polling for it.
Two details make that safe:

* ``set_count`` is invoked once per decoded frame (~9-30 fps), but the vast
  majority of those calls carry an identical count. Only a change in the
  ``(count, nearby, status)`` triple wakes subscribers, so an idle queue
  produces no traffic at all.
* The writer lives on the GStreamer thread while subscribers live on the
  uvicorn event loop. Handing the snapshot over with
  ``loop.call_soon_threadsafe`` is the only supported way to touch asyncio
  objects from a foreign thread; ``asyncio.Queue`` is NOT thread-safe.
"""
from __future__ import annotations

import asyncio
import threading
import time

_lock = threading.Lock()
_count: int = 0
_nearby: int = 0
_status: str = "unknown"
_timestamp: float = 0.0

# Bumped only when the meaningful payload changes, so a client can tell a
# genuine update apart from a heartbeat or a reconnect replay.
_version: int = 0

# Event loop owned by the uvicorn thread, published by api.py at startup.
_loop: asyncio.AbstractEventLoop | None = None

# Subscriber queues. Only ever mutated from the event loop thread, which is
# also the only thread that dispatches into them, so no lock is required.
_subscribers: set[asyncio.Queue] = set()

# Depth of each subscriber queue. A browser that stops reading must never be
# able to stall the GStreamer thread, and a stale queue count has no value, so
# the oldest entry is dropped once this many updates are outstanding.
_SUBSCRIBER_QUEUE_SIZE = 8


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the event loop that owns the SSE subscribers.

    Called once from the FastAPI lifespan handler. Until this is called,
    ``set_count`` simply skips the fan-out and the HTTP poll endpoint keeps
    working unchanged.

    Args:
        loop: The running uvicorn event loop.
    """
    global _loop  # noqa: PLW0603
    _loop = loop


def subscribe() -> asyncio.Queue:
    """Register a new SSE subscriber. Must be called from the loop thread.

    Returns:
        A queue that receives one snapshot dict per genuine change.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
    _subscribers.add(queue)
    return queue


def unsubscribe(queue: asyncio.Queue) -> None:
    """Remove a subscriber. Must be called from the loop thread.

    Args:
        queue: The queue previously returned by :func:`subscribe`.
    """
    _subscribers.discard(queue)


def subscriber_count() -> int:
    """Return the number of live SSE subscribers (diagnostics only)."""
    return len(_subscribers)


def _dispatch(snapshot: dict) -> None:
    """Push a snapshot to every subscriber. Runs on the event loop thread."""
    for queue in _subscribers:
        if queue.full():
            # Drop the oldest: a client that cannot keep up wants the newest
            # count, never a backlog of superseded ones.
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race with reader
                pass
        try:
            queue.put_nowait(snapshot)
        except asyncio.QueueFull:  # pragma: no cover - race with reader
            pass


def set_count(
    count: int,
    medium_threshold: int = 3,
    high_threshold: int = 7,
    nearby: int = 0,
) -> None:
    """Update the queue count (called from the GStreamer thread).

    ``count`` is the number of people standing inside the queue ROI;
    ``nearby`` is the number visible in frame but outside it.

    The timestamp is refreshed on every call so the poll endpoint can still be
    used as a liveness signal, but SSE subscribers are only woken when the
    count, nearby count or status actually changes.
    """
    global _count, _nearby, _status, _timestamp, _version  # noqa: PLW0603
    if count <= medium_threshold:
        status = "LOW"
    elif count <= high_threshold:
        status = "MEDIUM"
    else:
        status = "HIGH"

    with _lock:
        changed = (count, nearby, status) != (_count, _nearby, _status)
        _count = count
        _nearby = nearby
        _status = status
        _timestamp = time.time()
        if changed:
            _version += 1
        snapshot = {
            "count": _count,
            "nearby": _nearby,
            "status": _status,
            "timestamp": _timestamp,
            "version": _version,
        }

    if not changed:
        return

    loop = _loop
    if loop is None:
        return
    try:
        loop.call_soon_threadsafe(_dispatch, snapshot)
    except RuntimeError:
        # Loop already closed during shutdown — nothing left to notify.
        pass


def get() -> dict:
    """Return the latest snapshot (called from asyncio/uvicorn thread)."""
    with _lock:
        return {
            "count": _count,
            "nearby": _nearby,
            "status": _status,
            "timestamp": _timestamp,
            "version": _version,
        }
