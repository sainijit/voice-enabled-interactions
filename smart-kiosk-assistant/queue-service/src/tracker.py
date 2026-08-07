"""Multi-object tracker for queue-service (Phase 1 architecture fix).

Replaces the previous "per-frame IoU dedup + moving-average smoothing"
approach with a proper per-track lifecycle:

  * BYTE two-stage association -- low-confidence detections are used to
    *sustain* an existing track (never to spawn a new identity), so a
    momentary confidence dip from blur/partial occlusion no longer forces
    ``gvatrack`` to retire the id and mint a new one on reacquisition.
  * NSA-Kalman filtering -- a constant-velocity Kalman filter per track
    whose measurement noise is scaled by ``(1 - confidence)``, so a weak
    detection nudges the track gently while a strong one corrects it hard.
  * Global one-to-one (Hungarian) assignment -- duplicates are prevented
    *at association time* because two detections can no longer both match
    the same track, and a track can no longer claim two detections. This
    replaces the old post-hoc IoU deduplication pass.

This module intentionally ignores ``gvatrack``'s own persistent object id:
short-term-imageless has no appearance model and was the direct cause of
the id churn described in ``queue-config.yaml``. ``QueueCounter`` now feeds
this tracker raw per-frame ``(bbox, confidence)`` detections and reads back
a stable track id from here instead.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # normalized (x_min, y_min, x_max, y_max)

# Kalman process/measurement noise tuning. State is [cx, cy, w, h, vx, vy, vw, vh]
# in normalized (0..1) coordinates. Position/size noise is kept small; velocity
# noise is comparatively larger because a queue-standing person's velocity is
# the least predictable part of the state.
_POS_PROCESS_VAR = 1e-4
_VEL_PROCESS_VAR = 1e-3
_BASE_MEASUREMENT_VAR = 5e-3
# Detections below this confidence never receive weight 0 in NSA scaling; a
# 0-confidence measurement would otherwise zero out R and make the filter
# trust total noise completely.
_MIN_NSA_WEIGHT = 0.05

# How long a track's stationary-anchor history is remembered after eviction,
# so a same-spot respawn (a marginal-confidence frame that briefly couldn't
# sustain the old track id, or eviction right at `max_age`) inherits the
# original clock instead of restarting `stationary_seconds` from zero. This
# is what makes furniture detection survive the id churn a static, faintly-
# textured object (a chair backrest, a curtain fold) is especially prone to.
# Kept short deliberately -- a real customer who later stands in the same
# spot after a genuine gap must not inherit stale furniture history.
_ANCHOR_MEMORY_SECONDS = 5.0


@dataclass(frozen=True)
class Detection:
    """One raw per-frame detection, prior to any association."""

    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class TrackResult:
    """A confirmed, currently-live track, as reported to ``QueueCounter``."""

    track_id: int
    bbox: BBox
    confidence: float
    hits: int
    age: int
    time_since_update: int
    is_furniture: bool = False


def iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two normalized boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_cost_matrix(track_boxes: list[BBox], det_boxes: list[BBox]) -> np.ndarray:
    cost = np.ones((len(track_boxes), len(det_boxes)), dtype=float)
    for i, t in enumerate(track_boxes):
        for j, d in enumerate(det_boxes):
            cost[i, j] = 1.0 - iou(t, d)
    return cost


def _gated_assignment(
    track_boxes: list[BBox], det_boxes: list[BBox], iou_threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """One-to-one Hungarian assignment gated by a minimum IoU.

    Returns ``(matches, unmatched_track_indices, unmatched_det_indices)``
    where ``matches`` is a list of ``(track_index, det_index)`` pairs, all
    indices relative to the input lists.

    A single global (Hungarian) solve -- rather than greedy nearest-match --
    is what prevents duplicate tracks from being created in the first place:
    two detections can never both be assigned to the same track, and a
    track can never claim two detections, even when both pairings look
    locally plausible.
    """
    n_tracks, n_dets = len(track_boxes), len(det_boxes)
    if n_tracks == 0 or n_dets == 0:
        return [], list(range(n_tracks)), list(range(n_dets))

    cost = _iou_cost_matrix(track_boxes, det_boxes)
    row_idx, col_idx = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_idx, col_idx):
        if cost[r, c] > (1.0 - iou_threshold):
            continue  # Hungarian still proposed a pair below the IoU gate.
        matches.append((r, c))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_rows]
    unmatched_dets = [j for j in range(n_dets) if j not in matched_cols]
    return matches, unmatched_tracks, unmatched_dets


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter over ``[cx, cy, w, h]`` with NSA scaling.

    NSA ("Noise Scale Adaptive", StrongSORT) scales the measurement noise
    covariance by the detector's confidence for that observation: a
    low-confidence (blurry/occluded) box is trusted less than the motion
    model's prediction, instead of being treated as equally reliable as a
    high-confidence box.
    """

    _next_id = 1

    def __init__(
        self,
        bbox: BBox,
        confidence: float,
        now: float,
        stationary_drift: float = 0.02,
        stationary_since: float | None = None,
    ) -> None:
        self.track_id = _KalmanBoxTracker._next_id
        _KalmanBoxTracker._next_id += 1

        cx, cy, w, h = self._bbox_to_z(bbox)
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.eye(8) * 10.0

        self.confidence = confidence
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.first_seen = now
        self.last_update = now

        # Furniture/fixture detection: a chair backrest or a hanging strip
        # curtain gets confirmed as a "person" just as easily as a real one
        # (same confidence, same box shape) -- the one thing that reliably
        # tells them apart is that a real customer eventually moves and
        # furniture never does. `_anchor_*` is the centroid the track has
        # not strayed from; it is re-planted (and the clock restarted)
        # whenever the track drifts more than `stationary_drift` (normalized
        # units) away from it, so genuine motion -- however slow -- always
        # resets the timer. See `stationary_duration`.
        #
        # `stationary_since` lets a caller (``ByteTracker``, via its recent-
        # eviction memory) carry over an inherited clock instead of always
        # starting fresh at ``now`` -- see `_ANCHOR_MEMORY_SECONDS`.
        self._stationary_drift = stationary_drift
        self._anchor_cx = cx
        self._anchor_cy = cy
        self._stationary_since = now if stationary_since is None else stationary_since

        self._F = np.eye(8)
        for i in range(4):
            self._F[i, i + 4] = 1.0  # cx += vx, cy += vy, w += vw, h += vh (dt=1 frame)

        self._Q = np.diag(
            [_POS_PROCESS_VAR] * 4 + [_VEL_PROCESS_VAR] * 4
        )
        self._H = np.zeros((4, 8))
        for i in range(4):
            self._H[i, i] = 1.0

    @staticmethod
    def _bbox_to_z(bbox: BBox) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> BBox:
        cx, cy, w, h = z
        w = max(w, 1e-6)
        h = max(h, 1e-6)
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self, now: float) -> BBox:
        """Advance the motion model by one frame and return the predicted bbox.

        Called for every live track once per frame, *before* association --
        matching happens against where the person should be now, not where
        they were last observed. Also re-evaluates the stationary anchor
        (see ``__init__``) so furniture is judged on total elapsed time, not
        frame count -- robust to FPS drops.
        """
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q
        self.age += 1
        self.time_since_update += 1

        cx, cy = float(self.x[0]), float(self.x[1])
        drift = math.hypot(cx - self._anchor_cx, cy - self._anchor_cy)
        if drift > self._stationary_drift:
            self._anchor_cx, self._anchor_cy = cx, cy
            self._stationary_since = now
        return self._z_to_bbox(self.x[:4])

    def stationary_duration(self, now: float) -> float:
        """Seconds this track has stayed within ``stationary_drift`` of its
        anchor centroid, uninterrupted. Reset to 0 by any real movement."""
        return now - self._stationary_since

    def update(self, bbox: BBox, confidence: float, now: float) -> None:
        """Correct the state with an observed detection (NSA-weighted)."""
        z = np.array(self._bbox_to_z(bbox))
        nsa_weight = max(confidence, _MIN_NSA_WEIGHT)
        # Confidence closer to 1.0 -> smaller R -> trust the measurement more.
        R = np.eye(4) * (_BASE_MEASUREMENT_VAR * (1.0 - nsa_weight) / nsa_weight + _BASE_MEASUREMENT_VAR)

        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + R
        K = self.P @ self._H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self._H) @ self.P

        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.last_update = now

    @property
    def bbox(self) -> BBox:
        return self._z_to_bbox(self.x[:4])


class ByteTracker:
    """BYTE two-stage association + NSA-Kalman + Hungarian assignment.

    Consumes raw per-frame detections (already class-filtered by the
    caller) and produces a stable set of confirmed tracks. Low-confidence
    detections (between ``low_thresh`` and ``high_thresh``) are only ever
    used to keep an existing track alive; they never spawn a new one, so a
    single blurry frame cannot mint a phantom customer.
    """

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 21,
        min_hits: int = 3,
        stationary_seconds: float = 900.0,
        stationary_drift: float = 0.02,
    ) -> None:
        self._high_thresh = high_thresh
        self._low_thresh = low_thresh
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._min_hits = min_hits
        # Furniture/fixture filter -- see _KalmanBoxTracker.stationary_duration.
        # 900s (15 min) is deliberately far beyond any realistic QSR queue
        # wait, so a customer standing still is never misclassified; only an
        # object motionless for its *entire* tracked lifetime (a chair
        # backrest, a curtain fold) gets excluded. Set to <= 0 to disable.
        self._stationary_seconds = stationary_seconds
        self._stationary_drift = stationary_drift
        self._tracks: dict[int, _KalmanBoxTracker] = {}
        # Recently-evicted (anchor_cx, anchor_cy, stationary_since, evicted_at)
        # tuples -- see `_ANCHOR_MEMORY_SECONDS` and Stage 3/4 below.
        self._recent_anchors: list[tuple[float, float, float, float]] = []

    def update(self, detections: list[Detection], now: float) -> list[TrackResult]:
        """Advance the tracker by one frame; returns confirmed tracks only."""
        for track in self._tracks.values():
            track.predict(now)

        high_dets = [d for d in detections if d.confidence >= self._high_thresh]
        low_dets = [
            d for d in detections if self._low_thresh <= d.confidence < self._high_thresh
        ]

        track_ids = list(self._tracks.keys())
        track_boxes = [self._tracks[tid].bbox for tid in track_ids]

        # Stage 1: high-confidence detections against every live track.
        matches1, un_track_idx, un_high_idx = _gated_assignment(
            track_boxes, [d.bbox for d in high_dets], self._iou_threshold
        )
        for ti, di in matches1:
            self._tracks[track_ids[ti]].update(high_dets[di].bbox, high_dets[di].confidence, now)

        # Stage 2: low-confidence detections against tracks stage 1 left
        # unmatched. This is the BYTE step that survives blur/occlusion --
        # a track that lost its high-score detection this frame is still
        # sustained by a weak one rather than being marked lost.
        remaining_ids = [track_ids[i] for i in un_track_idx]
        remaining_boxes = [track_boxes[i] for i in un_track_idx]
        matches2, un_track_idx2, _un_low_idx = _gated_assignment(
            remaining_boxes, [d.bbox for d in low_dets], self._iou_threshold
        )
        for ti, di in matches2:
            self._tracks[remaining_ids[ti]].update(low_dets[di].bbox, low_dets[di].confidence, now)

        # Stage 3: unmatched high-confidence detections spawn new tracks.
        # Low-confidence detections never do -- a blurry/marginal box must
        # not be allowed to mint a new identity. Detections that overlap each
        # other heavily are suppressed before spawning (keeping the highest
        # confidence one) -- this is the residual insurance against two
        # duplicate boxes for one person both minting a track in the same
        # frame, on top of Hungarian already preventing them from both
        # matching one *existing* track.
        #
        # Before spawning, drop stale entries from the recent-eviction memory
        # (see `_ANCHOR_MEMORY_SECONDS`) so a real customer who only later
        # happens to stand where a chair/curtain track once was does not
        # inherit its accumulated stationary clock.
        self._recent_anchors = [
            a for a in self._recent_anchors if now - a[3] <= _ANCHOR_MEMORY_SECONDS
        ]
        spawn_candidates = sorted(
            (high_dets[di] for di in un_high_idx), key=lambda d: d.confidence, reverse=True
        )
        spawned_boxes: list[BBox] = []
        for det in spawn_candidates:
            if any(iou(det.bbox, box) >= self._iou_threshold for box in spawned_boxes):
                continue
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            inherited_since: float | None = None
            for i, (anchor_cx, anchor_cy, since, _evicted_at) in enumerate(self._recent_anchors):
                if math.hypot(cx - anchor_cx, cy - anchor_cy) <= self._stationary_drift:
                    inherited_since = since
                    del self._recent_anchors[i]
                    break
            new_track = _KalmanBoxTracker(
                det.bbox,
                det.confidence,
                now,
                stationary_drift=self._stationary_drift,
                stationary_since=inherited_since,
            )
            self._tracks[new_track.track_id] = new_track
            spawned_boxes.append(det.bbox)

        # Stage 4: evict tracks that have coasted past the dwell timeout.
        # Their stationary anchor is remembered briefly (see Stage 3 above)
        # so a same-spot respawn is not treated as a brand-new object.
        dead = [tid for tid, t in self._tracks.items() if t.time_since_update > self._max_age]
        for tid in dead:
            t = self._tracks[tid]
            self._recent_anchors.append((t._anchor_cx, t._anchor_cy, t._stationary_since, now))
            del self._tracks[tid]

        return [
            TrackResult(
                track_id=tid,
                bbox=t.bbox,
                confidence=t.confidence,
                hits=t.hits,
                age=t.age,
                time_since_update=t.time_since_update,
                is_furniture=(
                    self._stationary_seconds > 0
                    and t.stationary_duration(now) >= self._stationary_seconds
                ),
            )
            for tid, t in self._tracks.items()
            if t.hits >= self._min_hits
        ]

    def reset(self) -> None:
        """Drop all live tracks (used by tests; not called in production)."""
        self._tracks.clear()
