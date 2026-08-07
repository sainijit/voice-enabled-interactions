"""Unit tests for the QueueCounter <-> ByteTracker <-> ROIManager integration."""
from __future__ import annotations

from roi import ROIManager
from tracker import ByteTracker
from queue_counter import QueueCounter


class _Rect:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class _FakeRegion:
    def __init__(self, rect, confidence=0.9):
        self._rect = rect
        self._confidence = confidence

    def rect(self):
        return self._rect

    def confidence(self):
        return self._confidence

    def label(self):
        return ""

    def label_id(self):
        return 0


class _VideoInfo:
    def __init__(self, width, height):
        self.width, self.height = width, height


class _FakeFrame:
    def __init__(self, regions, width=1000, height=1000):
        self._regions = regions
        self._info = _VideoInfo(width, height)

    def regions(self):
        return self._regions

    def video_info(self):
        return self._info


def _make_counter(min_hits=1, iou_threshold=0.3, max_age=10):
    # Full-frame ROI so every point is "inside" by default; individual tests
    # override via a custom ROIManager where needed.
    roi = ROIManager(polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, iou_threshold=iou_threshold,
        max_age=max_age, min_hits=min_hits,
    )
    return QueueCounter(roi_manager=roi, class_label=None, tracker=tracker, log_mode="on_change")


def test_single_person_counted_once_across_frames():
    counter = _make_counter(min_hits=1)
    # 1000x1000 frame; person occupies (400,400)-(550,800) px.
    region = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)
    counter.process_frame(_FakeFrame([region]))
    assert counter._current_count() == 1
    assert counter._nearby_count() == 0


def test_duplicate_detection_in_one_frame_counts_as_one_person():
    """Regression test for the original bug: two near-identical boxes for
    one person must not inflate the count."""
    counter = _make_counter(min_hits=1)
    r1 = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.91)
    r2 = _FakeRegion(_Rect(405, 405, 150, 400), confidence=0.85)
    counter.process_frame(_FakeFrame([r1, r2]))
    assert counter._current_count() == 1


def test_person_outside_roi_counts_as_nearby_not_queued():
    # ROI restricted to the right half of the frame only.
    roi = ROIManager(polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=10)
    counter = QueueCounter(roi_manager=roi, class_label=None, tracker=tracker)

    # Person entirely in the left half (outside the ROI).
    region = _FakeRegion(_Rect(50, 400, 150, 400), confidence=0.9)
    counter.process_frame(_FakeFrame([region]))
    assert counter._current_count() == 0
    assert counter._nearby_count() == 1


def test_track_not_counted_until_min_hits_reached():
    counter = _make_counter(min_hits=3)
    region = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)
    frame = _FakeFrame([region])
    counter.process_frame(frame)
    assert counter._current_count() == 0  # 1st hit
    counter.process_frame(frame)
    assert counter._current_count() == 0  # 2nd hit
    counter.process_frame(frame)
    assert counter._current_count() == 1  # 3rd hit -> confirmed


def test_low_confidence_frame_does_not_drop_a_confirmed_person():
    """Simulates one blurry frame for an already-confirmed person: the
    person must still be counted, under the same track id."""
    counter = _make_counter(min_hits=2, iou_threshold=0.2)
    sharp = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)
    for _ in range(2):
        counter.process_frame(_FakeFrame([sharp]))
    assert counter._current_count() == 1
    confirmed_id = next(iter(counter._tracks))

    blurry = _FakeRegion(_Rect(402, 402, 150, 400), confidence=0.25)
    counter.process_frame(_FakeFrame([blurry]))
    assert counter._current_count() == 1
    assert next(iter(counter._tracks)) == confirmed_id


def test_no_detections_reports_zero_count():
    counter = _make_counter(min_hits=1)
    counter.process_frame(_FakeFrame([]))
    assert counter._current_count() == 0
    assert counter._nearby_count() == 0


# ── furniture / manual exclude zones ─────────────────────────────────────

def test_furniture_track_is_never_counted_inside_or_nearby(monkeypatch):
    """Regression test: a fixed chair backrest confirmed as a 'person' with
    high confidence must eventually stop being counted, once tracker.py
    flags it as furniture (stationary_duration >= stationary_seconds)."""
    roi = ROIManager(polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    tracker = ByteTracker(
        high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=100,
        stationary_seconds=5.0, stationary_drift=0.02,
    )
    counter = QueueCounter(roi_manager=roi, class_label=None, tracker=tracker)

    region = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)  # never moves
    fake_time = [0.0]
    monkeypatch.setattr("time.monotonic", lambda: fake_time[0])

    counter.process_frame(_FakeFrame([region]))
    assert counter._current_count() == 1  # not yet flagged -- just spawned

    for t in range(1, 11):
        fake_time[0] = float(t)
        counter.process_frame(_FakeFrame([region]))

    assert counter._current_count() == 0
    assert counter._nearby_count() == 0
    assert next(iter(counter._tracks.values())).is_furniture is True


def test_manual_exclude_zone_removes_track_from_both_counts():
    """Same effect as furniture, but instant and operator-configured via
    roi.exclude_zones, for a known fixed false-positive object."""
    roi = ROIManager(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        exclude_zones=[[(0.30, 0.30), (0.65, 0.30), (0.65, 0.90), (0.30, 0.90)]],
    )
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=10)
    counter = QueueCounter(roi_manager=roi, class_label=None, tracker=tracker)

    # Centroid at (0.475, 0.6) -- inside the exclude zone above.
    region = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)
    counter.process_frame(_FakeFrame([region]))

    assert counter._current_count() == 0
    assert counter._nearby_count() == 0
    assert next(iter(counter._tracks.values())).is_excluded is True


def test_person_outside_exclude_zone_still_counted():
    roi = ROIManager(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        exclude_zones=[[(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)]],
    )
    tracker = ByteTracker(high_thresh=0.6, low_thresh=0.1, min_hits=1, max_age=10)
    counter = QueueCounter(roi_manager=roi, class_label=None, tracker=tracker)

    region = _FakeRegion(_Rect(400, 400, 150, 400), confidence=0.9)
    counter.process_frame(_FakeFrame([region]))
    assert counter._current_count() == 1
