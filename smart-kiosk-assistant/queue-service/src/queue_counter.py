"""Queue counting for queue-service.

Executed by the DLStreamer ``gvapython`` element. ``QueueCounter`` reads the
detection metadata from ``gvadetect`` and the persistent object IDs from
``gvatrack``, filters each tracked person against the queue ROI, maintains a
table of active tracks (with dwell-based eviction), and logs the live queue
count whenever it changes.

The module avoids importing GStreamer at module scope so it stays
unit-testable; ``process_frame`` only relies on the ``gstgva.VideoFrame``
duck-typed API that DLStreamer passes in at runtime.
"""
import logging
import time
from dataclasses import dataclass

from roi import ROIManager

logger = logging.getLogger(__name__)

# gvatrack assigns IDs starting at 1; 0 means "no persistent track".
_UNTRACKED_ID = 0


@dataclass
class TrackState:
    """State stored per active tracked person."""

    track_id: int
    bbox: tuple[float, float, float, float]  # normalized (x_min, y_min, x_max, y_max)
    centroid: tuple[float, float]             # normalized (x, y)
    confidence: float                         # detector score, best-effort
    timestamp: float                          # monotonic seconds, last seen
    inside: bool                              # ROI status flag
    roi_status: str                           # "Inside" | "Outside"


class QueueCounter:
    """gvapython callback that counts tracked people inside the queue ROI."""

    def __init__(
        self,
        roi_manager: ROIManager | None = None,
        dwell_timeout_seconds: float | None = None,
        log_mode: str | None = None,
        log_interval_seconds: float | None = None,
        class_label: str | None = None,
    ) -> None:
        counter = self._counter_config()

        self._roi = roi_manager if roi_manager is not None else ROIManager()
        self._dwell_timeout = float(
            dwell_timeout_seconds
            if dwell_timeout_seconds is not None
            else counter.get("dwell_timeout_seconds", 2.0)
        )
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

        self._tracks: dict[int, TrackState] = {}
        self._last_count: int | None = None
        self._last_log_time = 0.0
        self._frame_w = 0
        self._frame_h = 0

        # Debug visualization (overlay drawn on top of gvawatermark output).
        debug = self._debug_config()
        self._debug = bool(debug.get("visualization", False))
        self._overlay_enabled = self._debug or bool(self._api_config().get("enabled", False))
        self._medium_threshold = int(counter.get("medium_threshold", 3))
        self._high_threshold = int(counter.get("high_threshold", 7))
        self._fps = 0.0
        self._last_frame_time = 0.0

        logger.info(
            "QueueCounter ready (dwell_timeout=%.1fs, log_mode=%s, class_label=%s)",
            self._dwell_timeout, self._log_mode, self._class_label or "<all>",
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
        now = time.monotonic()
        width, height = self._frame_size(frame)

        for region in self._regions(frame):
            track_id = self._object_id(region)
            if track_id == _UNTRACKED_ID:
                continue
            if not self._accept(region):
                continue
            bbox = self._normalized_bbox(region, width, height)
            centroid = self._roi.point_from_bbox(bbox)
            inside = self._roi.is_inside_roi(centroid)
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=bbox,
                centroid=centroid,
                confidence=self._confidence(region),
                timestamp=now,
                inside=inside,
                roi_status="Inside" if inside else "Outside",
            )

        self._evict(now)
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
    def _object_id(region) -> int:
        try:
            return int(region.object_id())
        except Exception:  # noqa: BLE001
            return _UNTRACKED_ID

    def _accept(self, region) -> bool:
        if self._class_label is None:
            return True
        try:
            return region.label() == self._class_label
        except Exception:  # noqa: BLE001
            return True

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

    def _evict(self, now: float) -> None:
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if now - state.timestamp > self._dwell_timeout
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _current_count(self) -> int:
        return sum(1 for state in self._tracks.values() if state.inside)

    def _update_count(self, now: float) -> None:
        count = self._current_count()
        # Always push the latest count into the shared queue_state so the API
        # endpoint always returns a fresh value regardless of log_mode.
        try:
            import queue_state
            queue_state.set_count(count, self._medium_threshold, self._high_threshold)
        except Exception:  # noqa: BLE001
            pass

        if self._log_mode == "interval":
            if now - self._last_log_time >= self._log_interval:
                logger.info("queue_count=%d", count)
                self._last_log_time = now
                self._last_count = count
        elif count != self._last_count:
            logger.info("queue_count=%d", count)
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

            count = self._current_count()
            status = self._status(count)
            with frame.data() as mat:
                poly = self._roi.polygon
                if len(poly) >= 3:
                    pts = np.array(
                        [[int(x * width), int(y * height)] for x, y in poly],
                        dtype=np.int32,
                    )
                    cv2.polylines(mat, [pts], True, (255, 0, 0), 2)

                for state in self._tracks.values():
                    x_min, y_min, x_max, y_max = state.bbox
                    x1 = max(0, min(width - 1, int(x_min * width)))
                    y1 = max(0, min(height - 1, int(y_min * height)))
                    x2 = max(0, min(width - 1, int(x_max * width)))
                    y2 = max(0, min(height - 1, int(y_max * height)))
                    color = (0, 255, 0) if state.inside else (0, 0, 255)
                    logger.debug(
                        "Track %s | Inside=%s | ROI Status=%s | Color=%s",
                        state.track_id,
                        state.inside,
                        state.roi_status,
                        color,
                    )

                    cv2.rectangle(mat, (x1, y1), (x2, y2), color, 3)

                    cx, cy = state.centroid
                    cx_px = max(0, min(width - 1, int(cx * width)))
                    cy_px = max(0, min(height - 1, int(cy * height)))
                    cv2.circle(mat, (cx_px, cy_px), 4, color, -1)

                    label = f"ID:{state.track_id} | {state.roi_status}"
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
                cv2.putText(mat, f"FPS: {self._fps:.0f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Push the annotated frame into the shared buffer so the
                # HTTP MJPEG streaming endpoint can serve it.
                try:
                    import frame_buffer
                    frame_buffer.put(mat.copy())
                except Exception:  # noqa: BLE001 - frame push is best-effort
                    pass
        except Exception:  # noqa: BLE001 - overlay is best-effort
            logger.debug("Debug overlay skipped", exc_info=True)
