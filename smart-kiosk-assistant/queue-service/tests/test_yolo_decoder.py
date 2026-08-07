# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the YOLO26 tensor decoder.

These are pure numpy tests -- no GStreamer, no model, no camera -- so they run
in CI and pin down the coordinate maths that is otherwise only observable by
eyeballing bounding boxes on a live stream.
"""
import numpy as np
import pytest

from yolo_decoder import YoloDecoder


def _decoder(**kw):
    kw.setdefault("threshold", 0.5)
    kw.setdefault("keep_id", 0)
    kw.setdefault("keep_label", "person")
    kw.setdefault("input_size", 640)
    return YoloDecoder(**kw)


def _row(x1, y1, x2, y2, score, cls):
    return [x1, y1, x2, y2, score, cls]


def test_decodes_a_person_box_into_frame_coordinates():
    # Regression test against measured ground truth.
    #
    # A 1920x1080 frame letterboxed into 640x640 scales by 1/3 to 640x360,
    # leaving 280 rows of padding split symmetrically -> 140 top, 140 bottom.
    # These raw values were captured from the real pipeline; the expected
    # frame coordinates were computed independently from raw OpenVINO with an
    # explicit letterbox. The two must agree.
    dec = _decoder()
    raw = np.array([_row(276.6, 364.4, 495.9, 498.7, 0.86, 0)], dtype=np.float32)

    boxes = dec.decode(raw, 1920, 1080)

    assert len(boxes) == 1
    x, y, w, h, score = boxes[0]
    assert score == pytest.approx(0.86, abs=1e-4)
    # Ground truth (830, 673) - (1488, 1076), within a rounding pixel or two.
    assert x == pytest.approx(830, abs=2)
    assert y == pytest.approx(673, abs=2)
    assert x + w == pytest.approx(1488, abs=2)
    assert y + h == pytest.approx(1076, abs=2)


def test_padding_is_centered_not_top_left():
    """The single most important invariant in the decoder.

    Top-left padding would place this box 140 px too low. The failure is
    silent -- boxes merely shift, quietly changing ROI membership and the
    queue count -- so it is asserted explicitly.
    """
    dec = _decoder()
    # A box occupying the full letterboxed image area: y from 140 to 500.
    raw = np.array([_row(0.0, 140.0, 640.0, 500.0, 0.9, 0)], dtype=np.float32)

    x, y, w, h, _ = dec.decode(raw, 1920, 1080)[0]

    assert (x, y) == (0, 0)
    assert w == 1920
    assert h == pytest.approx(1080, abs=2)


def test_non_person_classes_are_dropped():
    dec = _decoder()
    raw = np.array(
        [
            _row(100, 200, 200, 300, 0.99, 69),  # high score, wrong class
            _row(100, 200, 200, 300, 0.99, 16),
        ],
        dtype=np.float32,
    )
    assert dec.decode(raw, 1920, 1080) == []


def test_low_confidence_rows_are_dropped():
    dec = _decoder(threshold=0.5)
    raw = np.array([_row(100, 200, 200, 300, 0.49, 0)], dtype=np.float32)
    assert dec.decode(raw, 1920, 1080) == []


def test_threshold_boundary_is_inclusive():
    dec = _decoder(threshold=0.5)
    raw = np.array([_row(100, 200, 200, 300, 0.5, 0)], dtype=np.float32)
    assert len(dec.decode(raw, 1920, 1080)) == 1


def test_accepts_flat_tensor_data():
    """gvapython hands back a flat buffer, not a shaped array."""
    dec = _decoder()
    flat = np.array(
        _row(276.6, 364.4, 495.9, 498.7, 0.86, 0) + _row(0, 0, 0, 0, 0.0, 0),
        dtype=np.float32,
    )
    assert len(dec.decode(flat, 1920, 1080)) == 1


def test_padded_all_zero_rows_produce_no_boxes():
    """YOLO26 emits a fixed 300 rows; unused slots are zeroed."""
    dec = _decoder()
    raw = np.zeros((300, 6), dtype=np.float32)
    assert dec.decode(raw, 1920, 1080) == []


def test_boxes_are_clamped_to_the_frame():
    dec = _decoder()
    # Deliberately outside the letterboxed region on both axes.
    raw = np.array([_row(-50.0, -50.0, 900.0, 900.0, 0.9, 0)], dtype=np.float32)

    x, y, w, h, _ = dec.decode(raw, 1920, 1080)[0]

    assert x >= 0 and y >= 0
    assert x + w <= 1920
    assert y + h <= 1080


def test_zero_area_boxes_are_discarded():
    dec = _decoder()
    # Entirely inside the top padding band -> clips to zero height.
    raw = np.array([_row(100.0, 0.0, 200.0, 100.0, 0.9, 0)], dtype=np.float32)
    assert dec.decode(raw, 1920, 1080) == []


def test_malformed_and_empty_input_is_survivable():
    dec = _decoder()
    assert dec.decode(None, 1920, 1080) == []
    assert dec.decode(np.array([], dtype=np.float32), 1920, 1080) == []
    # Fewer values than one full row.
    assert dec.decode(np.array([1.0, 2.0, 3.0], dtype=np.float32), 1920, 1080) == []
    # Trailing partial row must not raise.
    partial = np.array(_row(276.6, 364.4, 495.9, 498.7, 0.86, 0) + [1.0, 2.0],
                       dtype=np.float32)
    assert len(dec.decode(partial, 1920, 1080)) == 1


def test_invalid_frame_dimensions_return_no_boxes():
    dec = _decoder()
    raw = np.array([_row(276.6, 364.4, 495.9, 498.7, 0.86, 0)], dtype=np.float32)
    assert dec.decode(raw, 0, 1080) == []
    assert dec.decode(raw, 1920, 0) == []


def test_square_frame_needs_no_padding():
    dec = _decoder()
    raw = np.array([_row(160.0, 160.0, 480.0, 480.0, 0.9, 0)], dtype=np.float32)

    x, y, w, h, _ = dec.decode(raw, 1080, 1080)[0]

    assert x == pytest.approx(270, abs=2)
    assert y == pytest.approx(270, abs=2)
    assert w == pytest.approx(540, abs=2)
    assert h == pytest.approx(540, abs=2)


def test_multiple_people_are_all_returned():
    dec = _decoder()
    raw = np.array(
        [
            _row(100.0, 200.0, 200.0, 400.0, 0.9, 0),
            _row(300.0, 200.0, 400.0, 400.0, 0.8, 0),
            _row(500.0, 200.0, 600.0, 400.0, 0.7, 3),  # not a person
        ],
        dtype=np.float32,
    )
    boxes = dec.decode(raw, 1920, 1080)
    assert len(boxes) == 2
    assert [round(b[4], 1) for b in boxes] == [0.9, 0.8]
