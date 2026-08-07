"""Tests for the streaming-branch shape of the GStreamer launch string.

Covers the queue insertion that decouples the pipeline's processing stages,
and guards the invariant that the BGRx capsfilter must never pin a resolution.
"""
from __future__ import annotations

import pytest

pipeline = pytest.importorskip("pipeline")

QueuePipeline = pipeline.QueuePipeline


def _new(**attrs):
    obj = QueuePipeline.__new__(QueuePipeline)
    obj._config = {}
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


# ── the BGRx capsfilter must not rescale ────────────────────────────────────

def test_bgrx_capsfilter_pins_format_only_never_a_resolution():
    """Scaling here drops gvadetect's ROI metadata and zeroes the queue count.

    The failure is invisible in the video (gvawatermark draws its boxes
    upstream of this element), so it must be prevented structurally.
    """
    caps = QueuePipeline._element_properties(_new(), "capsfilter")["caps"]
    assert caps == "video/x-raw,format=BGRx"
    assert "width" not in caps
    assert "height" not in caps


@pytest.mark.parametrize("va_memory", [True, False])
def test_bgrx_chain_contains_no_scaling_element(va_memory):
    chain = QueuePipeline._convert_to_bgrx(_new(_va_memory=va_memory))
    assert "videoscale" not in chain
    assert chain[-1] == "capsfilter"


# ── _insert_queues ──────────────────────────────────────────────────────────

def _insert(types):
    return QueuePipeline._insert_queues(_new(), types)


def test_queue_inserted_after_decode_and_track_and_before_sink():
    chain = _insert(
        ["urisourcebin", "decodebin", "gvadetect", "gvatrack", "gvapython", "appsink"]
    )
    assert chain == [
        "urisourcebin",
        "decodebin",
        "queue",
        "gvadetect",
        "gvatrack",
        "queue",
        "gvapython",
        "queue",
        "appsink",
    ]


@pytest.mark.parametrize("sink", ["appsink", "fakesink", "autovideosink"])
def test_every_sink_type_is_decoupled(sink):
    chain = _insert(["decodebin", sink])
    assert chain[-2:] == ["queue", sink]


def test_no_duplicate_queue_when_track_directly_precedes_sink():
    """gvatrack already appends a queue; the sink must not add a second one."""
    assert _insert(["gvatrack", "appsink"]) == ["gvatrack", "queue", "appsink"]


def test_insert_queues_does_not_mutate_input():
    types = ["decodebin", "appsink"]
    _insert(types)
    assert types == ["decodebin", "appsink"]


def test_queues_are_leaky_downstream_and_shallow():
    """The drop point must sit in the decoded domain, not the compressed one."""
    props = QueuePipeline._element_properties(_new(), pipeline._QUEUE_TYPE)
    assert props["leaky"] == "downstream"
    assert props["max-size-buffers"] == pipeline._QUEUE_MAX_BUFFERS
    # Byte/time limits must be off or they bound the queue before the buffer
    # count does on large frames.
    assert props["max-size-bytes"] == 0
    assert props["max-size-time"] == 0


def test_queue_element_token_maps_to_a_real_gstreamer_element():
    assert QueuePipeline._element_name(pipeline._QUEUE_TYPE) == "queue"
