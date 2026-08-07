"""Unit tests for roi.py -- queue polygon and manual exclude zones."""
from __future__ import annotations

from roi import ROIManager


def test_point_inside_polygon_is_counted_inside():
    roi = ROIManager(polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert roi.is_inside_roi((0.5, 0.5)) is True


def test_point_outside_polygon_is_not_inside():
    roi = ROIManager(polygon=[(0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)])
    assert roi.is_inside_roi((0.8, 0.8)) is False


def test_no_exclude_zones_by_default():
    roi = ROIManager(polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert roi.exclude_zones == []
    assert roi.is_excluded((0.5, 0.5)) is False


def test_point_inside_exclude_zone_is_flagged():
    roi = ROIManager(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        exclude_zones=[[(0.30, 0.40), (0.55, 0.40), (0.55, 0.85), (0.30, 0.85)]],
    )
    assert roi.is_excluded((0.40, 0.60)) is True  # inside the chair exclude zone


def test_point_outside_exclude_zone_is_not_flagged():
    roi = ROIManager(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        exclude_zones=[[(0.30, 0.40), (0.55, 0.40), (0.55, 0.85), (0.30, 0.85)]],
    )
    assert roi.is_excluded((0.90, 0.90)) is False


def test_multiple_exclude_zones_are_all_checked():
    roi = ROIManager(
        polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        exclude_zones=[
            [(0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)],
            [(0.8, 0.8), (0.9, 0.8), (0.9, 0.9), (0.8, 0.9)],
        ],
    )
    assert roi.is_excluded((0.05, 0.05)) is True
    assert roi.is_excluded((0.85, 0.85)) is True
    assert roi.is_excluded((0.5, 0.5)) is False
