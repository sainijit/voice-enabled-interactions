"""Tests for the YOLO26 detector path in the GStreamer element graph.

The decoder path exists because DLStreamer's `yolo_v10`/`yolo_v11` converters
are accepted without error but decode YOLO26 to zero detections. These tests
pin the resulting graph shape so nobody silently reverts to gvadetect.
"""
from __future__ import annotations

import pytest

pipeline = pytest.importorskip("pipeline")

QueuePipeline = pipeline.QueuePipeline


def _new(config=None, **attrs):
    obj = QueuePipeline.__new__(QueuePipeline)
    obj._config = config if config is not None else {}
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def _yolo26(**model_overrides):
    model = {
        "name": "yolo26n",
        "postproc": "decoder",
        "ir_path": "./models/yolo26n.xml",
        "device": "NPU",
        "threshold": 0.35,
    }
    model.update(model_overrides)
    return _new({"model": model})


def _retail():
    return _new(
        {
            "model": {
                "name": "person-detection-retail-0013",
                "ir_path": "./models/person-detection-retail-0013.xml",
                "device": "NPU",
                "threshold": 0.35,
            }
        }
    )


# ── mode selection ──────────────────────────────────────────────────────────

def test_yolo26_config_selects_the_decoder_path():
    assert _yolo26()._uses_decoder() is True


def test_missing_postproc_key_defaults_to_gvadetect():
    """The retail-0013 rollback path must stay byte-for-byte unchanged."""
    obj = _retail()
    assert obj._postproc_mode() == pipeline._POSTPROC_GVADETECT
    assert obj._uses_decoder() is False


def test_postproc_value_is_case_and_whitespace_insensitive():
    assert _yolo26(postproc="  Decoder ")._uses_decoder() is True


def test_unknown_postproc_value_falls_back_to_gvadetect():
    assert _yolo26(postproc="magic")._uses_decoder() is False


# ── element graph rewriting ─────────────────────────────────────────────────

def test_decoder_chain_replaces_gvadetect_with_inference_plus_decoder():
    types = ["decodebin", "videoconvert", "gvadetect", "gvatrack", "fakesink"]

    out = _yolo26()._decoder_element_chain(types)

    assert "gvadetect" not in out
    assert out == [
        "decodebin",
        "videoconvert",
        "gvainference",
        pipeline._DECODER_TYPE,
        "gvatrack",
        "fakesink",
    ]


def test_decoder_runs_before_gvatrack():
    """Tracking must only ever see decoded person regions."""
    out = _yolo26()._decoder_element_chain(["gvadetect", "gvatrack"])
    assert out.index(pipeline._DECODER_TYPE) < out.index("gvatrack")


def test_decoder_chain_does_not_mutate_its_input():
    types = ["gvadetect", "gvatrack"]
    _yolo26()._decoder_element_chain(types)
    assert types == ["gvadetect", "gvatrack"]


def test_graph_without_gvadetect_is_left_alone():
    """A hand-edited pipeline.yaml must not silently lose its detector."""
    types = ["decodebin", "videoconvert", "fakesink"]
    assert _yolo26()._decoder_element_chain(types) == types


# ── element properties ──────────────────────────────────────────────────────

def test_gvainference_carries_no_threshold_property():
    """Thresholding belongs to YoloDecoder; gvainference emits raw tensors."""
    props = _yolo26()._element_properties("gvainference")
    assert "threshold" not in props
    assert "model-proc" not in props
    assert props["device"] == "NPU"


def test_decoder_element_points_at_the_yolo_decoder_module():
    props = _yolo26()._element_properties(pipeline._DECODER_TYPE)
    assert props["module"].endswith("src/yolo_decoder.py")
    assert props["class"] == pipeline._DECODER_CLASS
    assert props["function"] == pipeline._DECODER_FUNCTION


def test_decoder_token_maps_to_a_real_gstreamer_element():
    assert QueuePipeline._element_name(pipeline._DECODER_TYPE) == "gvapython"


def test_gvadetect_still_carries_a_threshold_on_the_legacy_path():
    props = _retail()._element_properties("gvadetect")
    assert props["threshold"] == 0.35
