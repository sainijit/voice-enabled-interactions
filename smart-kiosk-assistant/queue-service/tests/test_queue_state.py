"""Tests for the queue_state SSE fan-out.

Covers the two properties that make pushing the count to the UI safe:
the writer runs on the GStreamer thread and must not wake subscribers for
unchanged counts, and a subscriber that stops reading must never be able to
block that writer.
"""
from __future__ import annotations

import asyncio
import importlib
import threading

import pytest


@pytest.fixture()
def state():
    """A freshly imported queue_state so module globals never leak between tests."""
    import queue_state

    return importlib.reload(queue_state)


def test_only_real_changes_notify_subscribers(state):
    """set_count runs once per frame; identical counts must not emit events."""

    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        queue = state.subscribe()

        state.set_count(2)
        # Same count repeated, as happens on every subsequent decoded frame.
        for _ in range(30):
            state.set_count(2)
        await asyncio.sleep(0)

        assert queue.qsize() == 1
        assert queue.get_nowait()["count"] == 2

        state.set_count(3)
        await asyncio.sleep(0)
        assert queue.get_nowait()["count"] == 3

    asyncio.run(scenario())


def test_version_increments_only_on_change(state):
    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        state.set_count(1)
        first = state.get()["version"]
        state.set_count(1)
        assert state.get()["version"] == first
        state.set_count(2)
        assert state.get()["version"] == first + 1

    asyncio.run(scenario())


def test_nearby_change_alone_notifies(state):
    """`nearby` is part of the payload, so a change in it must be pushed."""

    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        queue = state.subscribe()
        state.set_count(1, nearby=0)
        await asyncio.sleep(0)
        queue.get_nowait()

        state.set_count(1, nearby=4)
        await asyncio.sleep(0)
        assert queue.get_nowait()["nearby"] == 4

    asyncio.run(scenario())


def test_slow_subscriber_never_blocks_the_writer(state):
    """A client that stops reading must lose old counts, not stall inference."""

    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        queue = state.subscribe()

        # Far more distinct counts than the queue can hold.
        for count in range(state._SUBSCRIBER_QUEUE_SIZE + 20):
            state.set_count(count)
        await asyncio.sleep(0)

        assert queue.qsize() == state._SUBSCRIBER_QUEUE_SIZE
        # The newest value must have survived the overflow.
        drained = [queue.get_nowait()["count"] for _ in range(queue.qsize())]
        assert drained[-1] == state._SUBSCRIBER_QUEUE_SIZE + 19

    asyncio.run(scenario())


def test_unsubscribe_stops_delivery(state):
    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        queue = state.subscribe()
        state.unsubscribe(queue)
        assert state.subscriber_count() == 0

        state.set_count(5)
        await asyncio.sleep(0)
        assert queue.qsize() == 0

    asyncio.run(scenario())


def test_set_count_works_before_a_loop_is_bound(state):
    """The GStreamer thread starts before uvicorn; this must not raise."""
    state.set_count(4)
    assert state.get()["count"] == 4


def test_cross_thread_notification(state):
    """The real topology: writer on another thread, subscriber on the loop."""

    async def scenario():
        state.bind_loop(asyncio.get_running_loop())
        queue = state.subscribe()

        threading.Thread(target=state.set_count, args=(7,)).start()

        snapshot = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert snapshot["count"] == 7
        assert snapshot["status"] == "MEDIUM"

    asyncio.run(scenario())


def test_status_thresholds(state):
    state.set_count(3)
    assert state.get()["status"] == "LOW"
    state.set_count(5)
    assert state.get()["status"] == "MEDIUM"
    state.set_count(9)
    assert state.get()["status"] == "HIGH"
