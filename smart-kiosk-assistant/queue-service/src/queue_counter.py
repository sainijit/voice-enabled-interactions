"""Queue counting for queue-service.

Executed by the DLStreamer ``gvapython`` element. ``QueueCounter`` reads the
raw per-frame detection metadata from ``gvadetect``, runs its own BYTE +
NSA-Kalman + Hungarian tracker (see ``tracker.py``) to obtain a stable
identity per person, classifies each confirmed track against the queue ROI,
and logs the live queue count whenever it changes.

``gvatrack``'s own persistent object id is intentionally NOT used for
identity: ``short-term-imageless`` has no appearance model and was the
direct cause of the id churn documented in ``queue-config.yaml`` (up to 14
distinct ids for one stationary person). ``tracker.ByteTracker`` replaces
that with a proper track lifecycle driven by raw detections, so the
element stays in the pipeline but its id output is ignored.

The module avoids importing GStreamer at module scope so it stays
unit-testable; ``process_frame`` only relies on the ``gstgva.VideoFrame``
duck-typed API that DLStreamer passes in at runtime.
"""
import logging
import time
from dataclasses import dataclass

from roi import ROIManager
from tracker import ByteTracker, Detection

logger = logging.getLogger(__name__)


@dataclass
class _DisplayTrack:
    """A confirmed track augmented with ROI classification, for the overlay
    and the count aggregation. Sourced from ``tracker.TrackResult`` each
    frame -- ``QueueCounter`` holds no track state of its own anymore."""

    track_id: int
    bbox: tuple[float, float, float, float]  # normalized (x_min, y_min, x_max, y_max)
    centroid: tuple[float, float]             # normalized (x, y)
    confidence: float
    inside: bool                              # ROI status flag
    roi_status: str                           # "Inside" | "Outside"
    hits: int
    age: int
    is_furniture: bool = False                # motionless for its entire tracked lifetime
    is_excluded: bool = False                 # inside a manually configured roi.exclude_zones polygon


class QueueCounter:
    """gvapython callback that counts tracked people inside the queue ROI."""

    def __init__(
        self,
        roi_manager: ROIManager | None = None,
        dwell_timeout_seconds: float | None = None,
        log_mode: str | None = None,
        log_interval_seconds: float | None = None,
        class_label: str | None = None,
        tracker: ByteTracker | None = None,
    ) -> None:
        counter = self._counter_config()
        tracker_cfg = self._tracker_config()

        self._roi = roi_manager if roi_manager is not None else ROIManager()
        self._log_mode = log_mode or counter.get("log_mode", "on_change")
        self._log_interval = float(
            log_interval_seconds
            if log_interval_seconds is not None
            else counter.get("log_interval_seconds", 5.0)
        )
        # Generic class filter: only detections whose label matches are
        # tracked/counted. Configured via model.class_label so switching
        # detectors needs no code change. Empty/unset => accept every detection
        # (person-only models like retail emit no other classes).
        if class_label is None:
            class_label = self._model_config().get("class_label") or None
        self._class_label = class_label

        # BYTE + NSA-Kalman + Hungarian tracker (src/tracker.py). Owns the
        # full track lifecycle (association, coasting, eviction, hit
        # confirmation) that used to be spread across `_evict`,
        # `_distinct_people`, `_smoothed_counts` and `_apply_hysteresis`.
        if tracker is not None:
            self._tracker = tracker
        else:
            dwell_timeout = float(
                dwell_timeout_seconds
                if dwell_timeout_seconds is not None
                else tracker_cfg.get("dwell_timeout_seconds", 0.7)
            )
            assumed_fps = float(tracker_cfg.get("assumed_fps", 15.0))
            max_age = max(1, round(dwell_timeout * assumed_fps))
            self._tracker = ByteTracker(
                high_thresh=float(tracker_cfg.get("high_thresh", 0.6)),
                low_thresh=float(tracker_cfg.get("low_thresh", 0.1)),
                iou_threshold=float(tracker_cfg.get("iou_threshold", 0.3)),
                max_age=max_age,
                min_hits=max(1, int(tracker_cfg.get("min_hits", 3))),
                stationary_seconds=float(tracker_cfg.get("stationary_seconds", 900.0)),
                stationary_drift=float(tracker_cfg.get("stationary_drift_max", 0.02)),
            )

        # Confirmed tracks from the most recent frame, augmented with ROI
        # classification. Rebuilt every frame from the tracker's output --
        # QueueCounter holds no track state of its own.
        self._tracks: dict[int, _DisplayTrack] = {}
        self._display_count = 0
        self._display_nearby = 0
        self._last_count: int | None = None
        self._last_log_time = 0.0
        self._frame_w = 0
        self._frame_h = 0
        self._frame_number = 0

        # Debug visualization (overlay drawn on top of gvawatermark output).
        debug = self._debug_config()
        self._debug = bool(debug.get("visualization", False))
        self._overlay_enabled = self._debug or bool(self._api_config().get("enabled", False))
        self._medium_threshold = int(counter.get("medium_threshold", 3))
        self._high_threshold = int(counter.get("high_threshold", 7))
        self._fps = 0.0
        self._last_frame_time = 0.0

        logger.info(
            "QueueCounter ready (high_thresh=%.2f, low_thresh=%.2f, "
            "iou_threshold=%.2f, min_hits=%d, max_age=%d frames, "
            "log_mode=%s, class_label=%s)",
            self._tracker._high_thresh, self._tracker._low_thresh,
            self._tracker._iou_threshold, self._tracker._min_hits,
            self._tracker._max_age, self._log_mode, self._class_label or "<all>",
        )

    # ── configuration ────────────────────────────────────────────────────────

    @staticmethod
    def _counter_config() -> dict:
        try:
            from config_loader import config

            counter = getattr(config, "counter", None)
            if counter is None:
                return {}
            return vars(counter) if hasattr(counter, "__dict__") else dict(counter)
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    @staticmethod
    def _tracker_config() -> dict:
        try:
            from config_loader import config

            tracker_cfg = getattr(config, "tracker", None)
            if tracker_cfg is None:
                return {}
            return vars(tracker_cfg) if hasattr(tracker_cfg, "__dict__") else dict(tracker_cfg)
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    @staticmethod
    def _debug_config() -> dict:
        try:
            from config_loader import config

            debug = getattr(config, "debug", None)
            if debug is None:
                return {}
            return vars(debug) if hasattr(debug, "__dict__") else dict(debug)
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    @staticmethod
    def _model_config() -> dict:
        try:
            from config_loader import config

            model = getattr(config, "model", None)
            if model is None:
                return {}
            return vars(model) if hasattr(model, "__dict__") else dict(model)
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    @staticmethod
    def _api_config() -> dict:
        try:
            from config_loader import config

            api = getattr(config, "api", None)
            if api is None:
                return {}
            return vars(api) if hasattr(api, "__dict__") else dict(api)
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    # ── gvapython entry point ────────────────────────────────────────────────

    def process_frame(self, frame) -> bool:
        """Process one frame of detection+tracking metadata.

        Returns ``True`` so the buffer continues downstream (gvapython
        contract).
        """
        self._frame_number += 1
        now = time.monotonic()
        width, height = self._frame_size(frame)
        regions = self._regions(frame)

        detections: list[Detection] = []
        for region in regions:
            if not self._accept(region):
                continue
            bbox = self._normalized_bbox(region, width, height)
            detections.append(Detection(bbox=bbox, confidence=self._confidence(region)))

        # gvatrack's own object id is intentionally not read here -- identity
        # comes entirely from our own BYTE/Kalman tracker, driven by the raw
        # per-frame detections above. See module docstring.
        tracked = self._tracker.update(detections, now)

        self._tracks = {}
        for result in tracked:
            centroid = self._roi.point_from_bbox(result.bbox)
            inside = self._roi.is_inside_roi(centroid)
            excluded = self._roi.is_excluded(centroid)
            self._tracks[result.track_id] = _DisplayTrack(
                track_id=result.track_id,
                bbox=result.bbox,
                centroid=centroid,
                confidence=result.confidence,
                inside=inside,
                roi_status="Inside" if inside else "Outside",
                hits=result.hits,
                age=result.age,
                is_furniture=result.is_furniture,
                is_excluded=excluded,
            )
            logger.debug(
                "frame=%d track_id=%d conf=%.3f bbox=(%.3f,%.3f,%.3f,%.3f) "
                "centroid=(%.3f,%.3f) roi=%s hits=%d furniture=%s",
                self._frame_number, result.track_id, result.confidence,
                result.bbox[0], result.bbox[1], result.bbox[2], result.bbox[3],
                centroid[0], centroid[1],
                self._tracks[result.track_id].roi_status, result.hits, result.is_furniture,
            )

        if not detections:
            logger.debug(
                "frame=%d detections=0 queue_count=%d",
                self._frame_number,
                self._current_count(),
            )

        self._update_count(now)
        if self._overlay_enabled:
            self._draw_overlay(frame, width, height, now)
        return True

    # ── metadata helpers (defensive against gstgva version differences) ──────

    @staticmethod
    def _regions(frame):
        try:
            return list(frame.regions())
        except Exception:  # noqa: BLE001
            return []

    def _frame_size(self, frame) -> tuple[int, int]:
        if self._frame_w and self._frame_h:
            return self._frame_w, self._frame_h
        try:
            info = frame.video_info()
            self._frame_w = int(info.width)
            self._frame_h = int(info.height)
        except Exception:  # noqa: BLE001
            logger.debug("Could not read frame video_info; treating rects as normalized")
        return self._frame_w, self._frame_h

    @staticmethod
    def _safe_label(region) -> str | None:
        """Safely read region.label(), return None if unavailable."""
        try:
            return region.label()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _safe_class_id(region) -> int | None:
        """Safely read region class_id, trying label_id() then class_id()."""
        for attr in ("label_id", "class_id"):
            method = getattr(region, attr, None)
            if method is None:
                continue
            try:
                return int(method())
            except Exception:  # noqa: BLE001
                continue
        return None

    def _accept(self, region) -> bool:
        """Accept a region whose class matches the configured label or class_id.

        person-detection-retail-0013 and other OMZ models run without a
        model-proc file emit regions with an *empty-string* label and a
        numeric class_id of 0.  The original label-only check therefore
        rejected every detection.  A class_id fallback (the same pattern used
        by PersonFilter for YOLO) handles both cases without any config change.
        """
        if self._class_label is None:
            return True

        # Primary: non-empty string label (set when a model-proc file is used).
        try:
            label = region.label()
            if label and label == self._class_label:
                return True
        except Exception:  # noqa: BLE001
            pass

        # Fallback: numeric class_id (set by OMZ models without model-proc).
        model_cfg = self._model_config()
        model_class_id = model_cfg.get("class_id")
        if model_class_id is None:
            return False
        try:
            model_class_id = int(model_class_id)
        except (TypeError, ValueError):
            return False
        for attr in ("label_id", "class_id"):
            method = getattr(region, attr, None)
            if method is None:
                continue
            try:
                if int(method()) == model_class_id:
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    @staticmethod
    def _normalized_bbox(region, width: int, height: int):
        rect = region.rect()
        x, y, w, h = rect.x, rect.y, rect.w, rect.h
        if width > 0 and height > 0:
            return (x / width, y / height, (x + w) / width, (y + h) / height)
        return (float(x), float(y), float(x + w), float(y + h))

    @staticmethod
    def _confidence(region) -> float:
        try:
            return float(region.confidence())
        except Exception:  # noqa: BLE001 - confidence accessor differs across gstgva versions
            return 0.0

    # ── track table / counting ───────────────────────────────────────────────
    #
    # Track lifecycle (association, coasting, min-hits confirmation, eviction)
    # and duplicate prevention now live entirely in tracker.ByteTracker, which
    # runs a global Hungarian assignment each frame -- two detections can
    # never both match the same track, so duplicates are prevented rather
    # than merged after the fact. `self._tracks` is rebuilt every frame
    # directly from its output (see process_frame); the counts below are a
    # pure ROI classification over that already-correct set, with no
    # additional smoothing or hysteresis.

    def _current_count(self) -> int:
        """Number of distinct people standing inside the queue ROI.

        Tracks flagged ``is_furniture`` (motionless for their entire tracked
        lifetime -- a chair backrest, a fold in a hanging curtain) never
        count, even if their centroid sits inside the ROI. Tracks inside a
        manually configured ``roi.exclude_zones`` polygon are excluded too.
        """
        return sum(
            1 for state in self._tracks.values()
            if state.inside and not state.is_furniture and not state.is_excluded
        )

    def _nearby_count(self) -> int:
        """Number of distinct people visible but *outside* the queue ROI.

        These are customers in the vicinity who have not joined the queue.
        Tracking them separately makes it possible to distinguish "nobody is
        around" from "people are around but not queuing", which is the more
        useful signal for staffing and for upsell prompts. Furniture and
        manually excluded tracks are excluded here too -- see
        ``_current_count``.
        """
        return sum(
            1 for state in self._tracks.values()
            if not state.inside and not state.is_furniture and not state.is_excluded
        )

    def _update_count(self, now: float) -> None:
        count = self._current_count()
        nearby = self._nearby_count()
        self._display_count = count
        self._display_nearby = nearby
        # Always push the latest count into the shared queue_state so the API
        # endpoint always returns a fresh value regardless of log_mode.
        try:
            import queue_state
            queue_state.set_count(
                count, self._medium_threshold, self._high_threshold, nearby=nearby
            )
        except Exception:  # noqa: BLE001
            pass

        if self._log_mode == "interval":
            if now - self._last_log_time >= self._log_interval:
                logger.info(
                    "queue_count=%d nearby=%d (confirmed_tracks=%d)",
                    count, nearby, len(self._tracks),
                )
                self._last_log_time = now
                self._last_count = count
        elif count != self._last_count:
            logger.info(
                "queue_count=%d nearby=%d (confirmed_tracks=%d)",
                count, nearby, len(self._tracks),
            )
            self._last_count = count

    # ── debug visualization ──────────────────────────────────────────────────

    def _status(self, count: int) -> str:
        if count <= self._medium_threshold:
            return "LOW"
        if count <= self._high_threshold:
            return "MEDIUM"
        return "HIGH"

    def _draw_overlay(self, frame, width: int, height: int, now: float) -> None:
        """Draw ROI polygon, per-track ROI state, queue count, status and FPS.

        Failures are non-fatal -- counting/logging is never affected.
        """
        if width <= 0 or height <= 0:
            return
        if self._last_frame_time:
            dt = now - self._last_frame_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
        self._last_frame_time = now

        try:
            import cv2
            import numpy as np

            count = self._display_count
            status = self._status(count)
            with frame.data() as mat:
                poly = self._roi.polygon
                if len(poly) >= 3:
                    pts = np.array(
                        [[int(x * width), int(y * height)] for x, y in poly],
                        dtype=np.int32,
                    )
                    cv2.polylines(mat, [pts], True, (255, 0, 0), 2)

                for zone in self._roi.exclude_zones:
                    if len(zone) >= 3:
                        zone_pts = np.array(
                            [[int(x * width), int(y * height)] for x, y in zone],
                            dtype=np.int32,
                        )
                        cv2.polylines(mat, [zone_pts], True, (0, 255, 255), 2)

                for state in self._tracks.values():
                    # self._tracks only ever contains tracks the tracker has
                    # already confirmed (hits >= min_hits); furniture tracks
                    # are still drawn (grey, labelled) so an operator can see
                    # *why* something isn't being counted, rather than the
                    # box just silently vanishing.
                    x_min, y_min, x_max, y_max = state.bbox
                    x1 = max(0, min(width - 1, int(x_min * width)))
                    y1 = max(0, min(height - 1, int(y_min * height)))
                    x2 = max(0, min(width - 1, int(x_max * width)))
                    y2 = max(0, min(height - 1, int(y_max * height)))
                    if state.is_furniture:
                        color = (128, 128, 128)
                    elif state.is_excluded:
                        color = (128, 128, 128)
                    else:
                        color = (0, 255, 0) if state.inside else (0, 0, 255)
                    logger.debug(
                        "Track %s | Inside=%s | ROI Status=%s | Furniture=%s | Color=%s",
                        state.track_id,
                        state.inside,
                        state.roi_status,
                        state.is_furniture,
                        color,
                    )

                    cv2.rectangle(mat, (x1, y1), (x2, y2), color, 3)

                    cx, cy = state.centroid
                    cx_px = max(0, min(width - 1, int(cx * width)))
                    cy_px = max(0, min(height - 1, int(cy * height)))
                    cv2.circle(mat, (cx_px, cy_px), 4, color, -1)

                    status_label = "Static" if state.is_furniture else ("Excluded" if state.is_excluded else state.roi_status)
                    label = f"ID:{state.track_id} | {status_label}"
                    label_y = y1 - 8 if y1 > 18 else y1 + 18
                    cv2.putText(
                        mat,
                        label,
                        (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                cv2.putText(mat, f"Queue Count: {count}  Status: {status}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(mat, f"Nearby (not in queue): {self._display_nearby}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(mat, f"FPS: {self._fps:.0f}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Push the annotated frame into the shared buffer so the
                # HTTP MJPEG streaming endpoint can serve it. accepts() is
                # checked first because mat.copy() is a full-resolution
                # memcpy on the GStreamer streaming thread; it is skipped
                # when nobody is watching or the frame-rate cap means the
                # copy would be discarded anyway.
                try:
                    import frame_buffer
                    if frame_buffer.accepts(frame_buffer.SOURCE_COUNTER):
                        frame_buffer.put(mat.copy(), frame_buffer.SOURCE_COUNTER)
                except Exception:  # noqa: BLE001 - frame push is best-effort
                    pass
        except Exception:  # noqa: BLE001 - overlay is best-effort
            logger.debug("Debug overlay skipped", exc_info=True)
