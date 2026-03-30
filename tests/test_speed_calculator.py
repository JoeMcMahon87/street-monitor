import math

import pytest

from street_monitor.config import CalibrationConfig, SpeedConfig
from street_monitor.speed_calculator import SpeedCalculator
from street_monitor.tracker import Track


def _calc(fps=30.0):
    calib = CalibrationConfig(
        image_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        world_points=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    )
    speed_cfg = SpeedConfig(
        min_track_frames=3,
        smoothing_window=4,
        min_speed_kmh=3.0,
        max_speed_kmh=200.0,
    )
    return SpeedCalculator(calib, speed_cfg, fps)


def test_pixel_to_world_corner():
    calc = _calc()
    # Top-left pixel maps to world origin
    wx, wy = calc.pixel_to_world(0, 0)
    assert abs(wx) < 0.01 and abs(wy) < 0.01


def test_pixel_to_world_centre():
    calc = _calc()
    wx, wy = calc.pixel_to_world(50, 50)
    assert abs(wx - 5.0) < 0.1 and abs(wy - 5.0) < 0.1


def test_compute_speed_stationary_returns_none():
    calc = _calc()
    track = Track(track_id=0, class_name="car")
    # Same position repeated
    for _ in range(6):
        track.world_positions.append((5.0, 5.0))
    assert calc.compute_speed(track) is None  # below min_speed_kmh


def test_compute_speed_reasonable_vehicle():
    calc = _calc(fps=30.0)
    track = Track(track_id=1, class_name="car")
    # Vehicle moving ~14 m/s = ~50 km/h; advance 14/30 metres per frame
    step = 14.0 / 30.0
    for i in range(6):
        track.world_positions.append((1.8, i * step))
    speed = calc.compute_speed(track)
    assert speed is not None
    assert 45.0 < speed < 55.0


def test_compute_speed_too_fast_returns_none():
    calc = _calc(fps=30.0)
    track = Track(track_id=2, class_name="car")
    # Unrealistically fast: 300 km/h -> ~83 m/s -> ~2.8 m per frame
    for i in range(6):
        track.world_positions.append((0.0, i * 2.8))
    assert calc.compute_speed(track) is None
