"""Unit tests for tracker.py (BYTE association + NSA-Kalman + Hungarian)."""
from __future__ import annotations

import pytest

from tracker import ByteTracker, Detection, iou, _gated_assignment


# ── iou() ─────────────────────────────────────────────────────────────────

def test_iou_identical_boxes_is_one():
    box = (0.1, 0.1, 0.3, 0.3)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 0.1, 0.1)
    b = (0.5, 0.5, 0.6, 0.6)
    assert iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = (0.0, 0.0, 0.2, 0.2)
    b = (0.1, 0.0, 0.3, 0.2)
    # overlap width 0.1, height 0.2 -> inter=0.02; each area=0.04; union=0.06
    assert iou(a, b) == pytest.approx(0.02 / 0.06, rel=1e-6)


# ── _gated_assignment() ──────────────────────────────────────────────────

def test_gated_assignment_matches_close_boxes():
    tracks = [(0.10, 0.10, 0.30, 0.30)]
    dets = [(0.11, 0.11, 0.31, 0.31)]
    matches, un_t, un_d = _gated_assignment(tracks, dets, iou_threshold=0.3)
    assert matches == [(0, 0)]
    assert un_t == []
    assert un_d == []


def test_gated_assignment_rejects_far_boxes():
    tracks = [(0.0, 0.0, 0.1, 0.1)]
    dets = [(0.8, 0.8, 0.9, 0.9)]
    matches, un_t, un_d = _gated_assignment(tracks, dets, iou_threshold=0.3)
    assert matches == []
    assert un_t == [0]
    assert un_d == [0]


def test_gated_assignment_is_one_to_one_even_with_ties():
    """Two nearly identical detections for one track must not both match --
    this is the mechanism that replaces the old post-hoc IoU dedup pass."""
    tracks = [(0.10, 0.10, 0.30, 0.30)]
    dets = [(0.10, 0.10, 0.30, 0.30), (0.11, 0.11, 0.31, 0.31)]
    matches, un_t, un_d = _gated_assignment(tracks, dets, iou_threshold=0.3)
    assert len(matches) == 1
    assert un_t == []
    assert len(un_d) == 1  # exactly one detection is left unmatched


def test_gated_assignment_empty_inputs():
    assert _gated_assignment([], [(0.0, 0.0, 0.1, 0.1)], 0.3) == ([], [], [0])
    assert _gated_assignment([(0.0, 0.0, 0.1, 0.1)], [], 0.3) == ([], [0], [])


# ── ByteTracker lifecycle ────────────────────────────────────────────────

STATIONARY_BOX = (0.40, 0.40, 0.55, 0.80)


def _detections(box=STATIONARY_BOX, confidence=0.9):
    return [Detection(bbox=box, confidence=confidence)]


def test_track_not_confirmed_before_min_hits():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=3, max_age=30)
    result = tracker.update(_detections(), now=0.0)
    assert result == []  # 1st hit, min_hits=3 -> not yet confirmed
    result = tracker.update(_detections(), now=0.1)
    assert result == []  # 2nd hit


def test_track_confirmed_after_min_hits_and_keeps_same_id():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=3, max_age=30)
    tracker.update(_detections(), now=0.0)
    tracker.update(_detections(), now=0.1)
    result = tracker.update(_detections(), now=0.2)
    assert len(result) == 1
    confirmed_id = result[0].track_id

    result = tracker.update(_detections(), now=0.3)
    assert len(result) == 1
    assert result[0].track_id == confirmed_id  # identity persists


def test_low_confidence_detection_sustains_existing_track_without_new_id():
    """The core BYTE guarantee: a confirmed track that momentarily only gets
    a blurry/low-score detection is kept alive under the SAME id, instead of
    being dropped and re-spawned as a new one (the id-churn bug)."""
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=3, max_age=30)
    for i in range(3):
        result = tracker.update(_detections(confidence=0.9), now=float(i))
    confirmed_id = result[0].track_id

    # Same person, same position, but a blurry/marginal-confidence box.
    low_conf_result = tracker.update(_detections(confidence=0.25), now=3.0)
    assert len(low_conf_result) == 1
    assert low_conf_result[0].track_id == confirmed_id


def test_low_confidence_detection_never_spawns_a_new_track():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=30)
    result = tracker.update(_detections(confidence=0.2), now=0.0)
    assert result == []  # never confirmed, because it never even creates a track


def test_detection_below_low_thresh_is_ignored_entirely():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=30)
    result = tracker.update(_detections(confidence=0.02), now=0.0)
    assert result == []


def test_two_people_get_two_distinct_ids():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=30)
    dets = [
        Detection(bbox=(0.05, 0.1, 0.20, 0.6), confidence=0.9),
        Detection(bbox=(0.60, 0.1, 0.75, 0.6), confidence=0.9),
    ]
    result = tracker.update(dets, now=0.0)
    assert len(result) == 2
    assert result[0].track_id != result[1].track_id


def test_duplicate_boxes_in_one_frame_spawn_only_one_track():
    """Two near-identical detections in the very first frame (simulating a
    detector artifact) must not mint two identities for one person."""
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, iou_threshold=0.3, min_hits=1, max_age=30)
    dets = [
        Detection(bbox=(0.40, 0.40, 0.55, 0.80), confidence=0.91),
        Detection(bbox=(0.41, 0.41, 0.56, 0.81), confidence=0.85),
    ]
    result = tracker.update(dets, now=0.0)
    assert len(result) == 1


def test_track_coasts_through_brief_gap_then_is_evicted_after_max_age():
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=3)
    tracker.update(_detections(), now=0.0)  # confirmed immediately (min_hits=1)

    # 3 frames with no detections at all: still within max_age, track coasts.
    for i in range(1, 4):
        tracker.update([], now=float(i))
    assert len(tracker._tracks) == 1

    # One more empty frame pushes time_since_update past max_age -> evicted.
    tracker.update([], now=4.0)
    assert len(tracker._tracks) == 0


def test_kalman_prediction_tracks_a_moving_person():
    """A person walking at constant velocity should be re-associated across
    frames without an id switch, even though the box moves each frame."""
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, iou_threshold=0.2, min_hits=1, max_age=10)
    x = 0.10
    result = tracker.update([Detection(bbox=(x, 0.3, x + 0.15, 0.7), confidence=0.9)], now=0.0)
    track_id = result[0].track_id

    for step in range(1, 6):
        x += 0.03  # small, steady motion -- still within the IoU gate
        result = tracker.update(
            [Detection(bbox=(x, 0.3, x + 0.15, 0.7), confidence=0.9)], now=float(step)
        )
        assert len(result) == 1
        assert result[0].track_id == track_id


# ── furniture / stationary-object filter ─────────────────────────────────

def test_stationary_object_is_flagged_furniture_after_the_threshold():
    """A chair backrest or curtain fold that never moves must eventually be
    flagged as furniture, not counted as a person forever."""
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=30,
        stationary_seconds=5.0, stationary_drift=0.02,
    )
    box = (0.40, 0.40, 0.55, 0.80)  # never moves -- e.g. a fixed chair
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=0.0)
    assert result[0].is_furniture is False  # not yet -- just spawned

    for t in range(1, 11):
        result = tracker.update([Detection(bbox=box, confidence=0.9)], now=float(t))

    assert len(result) == 1
    assert result[0].is_furniture is True


def test_moving_person_is_never_flagged_furniture():
    """A real customer repositioning over time -- shifting stance, stepping
    forward as the queue moves -- must never be classified as furniture,
    because each net displacement replants the anchor and restarts the
    clock before `stationary_seconds` can elapse."""
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, iou_threshold=0.2, min_hits=1, max_age=30,
        stationary_seconds=5.0, stationary_drift=0.02,
    )
    x = 0.40
    result = tracker.update([Detection(bbox=(x, 0.4, x + 0.15, 0.8), confidence=0.9)], now=0.0)

    for t in range(1, 21):
        # Net one-directional creep every other frame -- unlike a
        # perfectly-symmetric oscillation, the Kalman filter cannot average
        # this back toward the anchor, so each step is real progress away
        # from wherever the anchor last reset to.
        if t % 2 == 0:
            x += 0.03
        result = tracker.update(
            [Detection(bbox=(x, 0.4, x + 0.15, 0.8), confidence=0.9)], now=float(t)
        )

    assert len(result) == 1
    assert result[0].is_furniture is False


def test_stationary_filter_disabled_when_seconds_is_zero():
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=30,
        stationary_seconds=0.0,
    )
    box = (0.40, 0.40, 0.55, 0.80)
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=0.0)
    for t in range(1, 100):
        result = tracker.update([Detection(bbox=box, confidence=0.9)], now=float(t))
    assert result[0].is_furniture is False


def test_stationary_clock_survives_track_id_churn_at_the_same_spot():
    """A chair/curtain whose track briefly gets evicted (a marginal-
    confidence gap that outlasts `max_age`) and immediately re-detected at
    the same spot must inherit its accumulated stationary clock instead of
    restarting -- otherwise the id churn a faintly-textured static object is
    especially prone to would let it dodge the furniture filter forever."""
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=2,
        stationary_seconds=5.0, stationary_drift=0.02,
    )
    box = (0.40, 0.40, 0.55, 0.80)  # never moves -- e.g. a fixed chair

    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=0.0)
    first_id = result[0].track_id

    # Sustain for 3s, then go undetected long enough to blow past max_age
    # and evict the track (2 missed frames).
    for t in range(1, 4):
        result = tracker.update([Detection(bbox=box, confidence=0.9)], now=float(t))
    for t in [3.3, 3.6, 3.9]:
        result = tracker.update([], now=t)
    assert result == []  # track evicted -- gone from live output

    # Re-detected at the exact same spot -- gets a new id (old one is gone)...
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=4.0)
    assert result[0].track_id != first_id

    # ...but only 1.1s more (total elapsed 4.1s) should be needed to clear
    # the 5.0s threshold, not another fresh 5.0s from the respawn.
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=4.1)
    assert result[0].is_furniture is False
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=5.1)
    assert result[0].is_furniture is True


def test_stationary_clock_does_not_carry_over_after_a_long_absence():
    """A real customer who later happens to stand exactly where a chair/
    curtain track once was must not inherit its stale stationary clock --
    the eviction memory expires after `_ANCHOR_MEMORY_SECONDS`."""
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=2,
        stationary_seconds=5.0, stationary_drift=0.02,
    )
    box = (0.40, 0.40, 0.55, 0.80)

    for t in range(0, 4):
        result = tracker.update([Detection(bbox=box, confidence=0.9)], now=float(t))
    for t in [3.3, 3.6, 3.9]:
        result = tracker.update([], now=t)
    assert result == []

    # A genuinely new occupant only shows up long after the eviction memory
    # (5s) has expired -- must start its clock fresh at `now`.
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=20.0)
    result = tracker.update([Detection(bbox=box, confidence=0.9)], now=24.0)
    assert result[0].is_furniture is False

