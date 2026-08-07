"""Thread-safe single-slot buffer holding the latest video frame.

``put()`` is called from GStreamer threads (the ``gvapython`` overlay callback
and the appsink ``new-sample`` callback). ``get_jpeg()`` is polled by the MJPEG
streaming endpoint.

Encoding is **lazy**: ``put()`` only stores a reference to the frame, and the
JPEG is produced on first ``get_jpeg()`` for that frame. This matters because

* ``put()`` runs on the GStreamer streaming threads — encoding there stalls the
  pipeline and adds latency to the live feed;
* two producers push the same frame, so eager encoding did the work twice;
* with no MJPEG client connected, eager encoding burned CPU for nothing.

The encoded result is cached per frame sequence number, so repeated polls of an
unchanged frame do not re-encode. A plain ``threading.Lock`` avoids any
asyncio/GLib cross-loop interaction, and the encode itself runs *outside* the
lock so producers are never blocked by a slow encode.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

_lock = threading.Lock()

# Frame producers, in priority order. ``queue_counter`` draws the overlay and
# is the authoritative source; the appsink callback pushes the same frame and
# only acts as a fallback if the overlay path stops producing.
SOURCE_COUNTER = "counter"
SOURCE_APPSINK = "appsink"

# How long the fallback producer stays suppressed after the last primary frame.
_PRIMARY_TIMEOUT_SECONDS = 1.0
_last_primary_ts: float = 0.0

# How long after the last ``get_jpeg()`` call a client is assumed to still be
# watching. Producers skip their frame copy entirely outside this window.
#
# Storing a frame costs a full-resolution copy in the producer (at 1080p that
# is a 6 MB memcpy, ~180 MB/s at 30 fps) on the GStreamer streaming thread --
# the same thread that has to keep draining the RTP socket. Paying that when
# no browser is connected is pure waste, and paying it 30 times a second when
# the stream is capped at 15 fps is double the necessary cost. Both are now
# skipped before the copy happens rather than after it.
_DEMAND_TIMEOUT_SECONDS = 2.0
_last_demand_ts: float = 0.0

# When the most recent frame was accepted, used to enforce ``_min_interval``
# at the producer instead of only at the encoder.
_last_store_ts: float = 0.0

# Latest raw BGR frame and a monotonically increasing sequence number.
_frame: np.ndarray | None = None
_frame_seq: int = 0

# Cached JPEG for _jpeg_seq. -1 means "nothing encoded yet".
_jpeg: bytes | None = None
_jpeg_seq: int = -1

_quality: int = 80
_max_height: int = 0  # 0 disables downscaling

# Minimum seconds between two encoded JPEGs (0 disables the cap). The pipeline
# runs at 30 fps, which is far more than a queue-monitoring tile needs and cost
# ~34 Mbps of MJPEG — fine on localhost, but it stutters over any real network.
_min_interval: float = 0.0
_last_encode_ts: float = 0.0


def configure(
    jpeg_quality: int = 80,
    max_height: int = 0,
    max_fps: float = 0.0,
) -> None:
    """Set JPEG encoding quality, streaming height cap and frame-rate cap.

    Args:
        jpeg_quality: JPEG quality (1-100, higher = better quality/larger).
        max_height: Downscale frames taller than this before encoding, keeping
            aspect ratio. ``0`` disables downscaling.
        max_fps: Cap how often a *new* JPEG is produced. ``0`` disables the
            cap. Detection, tracking and the overlay are unaffected — this
            only limits what the browser is sent.
    """
    global _quality, _max_height, _min_interval  # noqa: PLW0603
    _quality = max(1, min(100, jpeg_quality))
    _max_height = max(0, int(max_height))
    _min_interval = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0


def _wanted(source: str, now: float) -> bool:
    """Whether a frame from ``source`` would be stored right now.

    Caller must hold ``_lock``. Checks, in increasing order of cost to the
    caller: is anyone watching, is the next frame due yet, and is this
    producer the active one.
    """
    if (now - _last_demand_ts) > _DEMAND_TIMEOUT_SECONDS:
        return False
    if _min_interval and (now - _last_store_ts) < _min_interval:
        return False
    if source == SOURCE_COUNTER:
        return True
    return (now - _last_primary_ts) >= _PRIMARY_TIMEOUT_SECONDS


def accepts(source: str) -> bool:
    """Report whether a frame from ``source`` would currently be stored.

    Lets producers skip an expensive full-resolution frame copy when no client
    is connected, when the frame-rate cap means the frame would be discarded
    anyway, or (for the fallback producer) while the primary overlay producer
    is healthy.
    """
    now = time.monotonic()
    with _lock:
        return _wanted(source, now)


def put(bgr_frame: np.ndarray, source: str = SOURCE_COUNTER) -> None:
    """Store the latest BGR frame, replacing any previous one.

    Called from GStreamer threads -- deliberately does no encoding so it
    returns immediately and never adds latency to the pipeline. The same
    gating as :func:`accepts` is re-applied here so a producer that does not
    check first is still cheap, and so the same frame is never stored (and
    later encoded) twice by both producers.
    """
    global _frame, _frame_seq, _last_primary_ts, _last_store_ts  # noqa: PLW0603
    now = time.monotonic()
    with _lock:
        wanted = _wanted(source, now)
        if source == SOURCE_COUNTER:
            # Track liveness of the primary producer even when its frame is
            # not stored, otherwise a rate-capped primary would look silent
            # and let the fallback producer start pushing duplicates.
            _last_primary_ts = now
        if not wanted:
            return
        _frame = bgr_frame
        _frame_seq += 1
        _last_store_ts = now


def _downscale(frame: np.ndarray) -> np.ndarray:
    """Shrink the frame to ``_max_height`` preserving aspect ratio."""
    if not _max_height:
        return frame
    height, width = frame.shape[:2]
    if height <= _max_height:
        return frame
    scale = _max_height / float(height)
    # Even dimensions keep the JPEG chroma subsampling happy.
    new_w = max(2, int(round(width * scale)) & ~1)
    return cv2.resize(frame, (new_w, _max_height), interpolation=cv2.INTER_AREA)


def get_jpeg() -> bytes | None:
    """Return the latest frame as JPEG bytes, encoding it on demand.

    Returns ``None`` until the first frame has been stored. This performs the
    JPEG encode, so callers on an asyncio event loop should dispatch it to a
    worker thread.
    """
    global _jpeg, _jpeg_seq, _last_encode_ts, _last_demand_ts  # noqa: PLW0603

    with _lock:
        # Record demand before anything else: producers skip their frame copy
        # unless a client has polled recently, so the very first poll is what
        # switches frame production on.
        _last_demand_ts = time.monotonic()
        if _jpeg_seq == _frame_seq:
            return _jpeg
        # Frame-rate cap: return the already-encoded frame until the interval
        # has elapsed. Callers compare identity, so handing back the same
        # object also stops it being re-sent, capping bandwidth as well as
        # encode CPU.
        if _min_interval and _jpeg is not None:
            if (time.monotonic() - _last_encode_ts) < _min_interval:
                return _jpeg
        frame = _frame
        seq = _frame_seq

    if frame is None:
        return None

    ok, buf = cv2.imencode(
        ".jpg", _downscale(frame), [cv2.IMWRITE_JPEG_QUALITY, _quality]
    )
    if not ok:
        with _lock:
            return _jpeg
    data = buf.tobytes()

    with _lock:
        # Another thread may have encoded a newer frame while we worked.
        if seq > _jpeg_seq:
            _jpeg = data
            _jpeg_seq = seq
            _last_encode_ts = time.monotonic()
    return data
