"""Tests for the single-slot frame buffer that backs the MJPEG endpoint.

The buffer gates frame production on demand and on the frame-rate cap so the
GStreamer streaming threads never pay for a full-resolution frame copy that
would be thrown away.
"""
from __future__ import annotations

import numpy as np
import pytest

frame_buffer = pytest.importorskip("frame_buffer")


@pytest.fixture(autouse=True)
def reset_buffer():
    """Restore module state so tests do not leak into one another."""
    frame_buffer._frame = None
    frame_buffer._frame_seq = 0
    frame_buffer._jpeg = None
    frame_buffer._jpeg_seq = -1
    frame_buffer._last_primary_ts = 0.0
    frame_buffer._last_demand_ts = 0.0
    frame_buffer._last_store_ts = 0.0
    frame_buffer.configure(jpeg_quality=80, max_height=0, max_fps=0)
    yield


def _frame(value: int = 7):
    return np.full((16, 32, 3), value, dtype=np.uint8)


def _demand():
    """Simulate a connected MJPEG client polling the buffer."""
    frame_buffer.get_jpeg()


# ── demand gating ───────────────────────────────────────────────────────────

def test_no_frame_accepted_before_any_client_polls():
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is False


def test_frame_accepted_once_a_client_has_polled():
    _demand()
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is True


def test_put_is_a_no_op_while_nobody_is_watching():
    frame_buffer.put(_frame(), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.get_jpeg() is None


def test_put_stores_the_frame_once_a_client_is_watching():
    _demand()
    frame_buffer.put(_frame(), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.get_jpeg() is not None


def test_demand_expires_after_the_timeout(monkeypatch):
    _demand()
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is True
    later = frame_buffer.time.monotonic() + frame_buffer._DEMAND_TIMEOUT_SECONDS + 1
    monkeypatch.setattr(frame_buffer.time, "monotonic", lambda: later)
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is False


# ── frame-rate cap applied at the producer ──────────────────────────────────

def test_rate_cap_rejects_a_second_frame_within_the_interval():
    frame_buffer.configure(jpeg_quality=80, max_height=0, max_fps=15)
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is False


def test_rate_cap_allows_a_frame_once_the_interval_elapses(monkeypatch):
    frame_buffer.configure(jpeg_quality=80, max_height=0, max_fps=15)
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    later = frame_buffer.time.monotonic() + 1.0
    monkeypatch.setattr(frame_buffer.time, "monotonic", lambda: later)
    assert frame_buffer.accepts(frame_buffer.SOURCE_COUNTER) is True


# ── primary / fallback producer arbitration ─────────────────────────────────

def test_fallback_producer_suppressed_while_primary_is_healthy():
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.accepts(frame_buffer.SOURCE_APPSINK) is False


def test_fallback_producer_takes_over_when_primary_goes_silent(monkeypatch):
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    later = frame_buffer.time.monotonic() + frame_buffer._PRIMARY_TIMEOUT_SECONDS + 0.5
    monkeypatch.setattr(frame_buffer.time, "monotonic", lambda: later)
    assert frame_buffer.accepts(frame_buffer.SOURCE_APPSINK) is True


def test_rate_capped_primary_still_counts_as_alive():
    """A primary throttled by the frame-rate cap must not look silent.

    Otherwise the fallback producer would start pushing duplicate frames of
    the same picture without the overlay.
    """
    frame_buffer.configure(jpeg_quality=80, max_height=0, max_fps=15)
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    # Rejected by the rate cap, but liveness was still recorded.
    frame_buffer.put(_frame(2), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.accepts(frame_buffer.SOURCE_APPSINK) is False


# ── encoding ────────────────────────────────────────────────────────────────

def test_jpeg_is_cached_until_a_new_frame_arrives():
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    first = frame_buffer.get_jpeg()
    assert frame_buffer.get_jpeg() is first


def test_new_frame_produces_a_new_jpeg():
    _demand()
    frame_buffer.put(_frame(1), frame_buffer.SOURCE_COUNTER)
    first = frame_buffer.get_jpeg()
    frame_buffer.put(_frame(200), frame_buffer.SOURCE_COUNTER)
    assert frame_buffer.get_jpeg() is not first


def test_downscale_respects_max_height():
    import cv2

    frame_buffer.configure(jpeg_quality=80, max_height=8, max_fps=0)
    _demand()
    frame_buffer.put(np.zeros((32, 64, 3), dtype=np.uint8), frame_buffer.SOURCE_COUNTER)
    data = frame_buffer.get_jpeg()
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[0] == 8
