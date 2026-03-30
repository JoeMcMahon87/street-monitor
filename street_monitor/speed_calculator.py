from __future__ import annotations

import math

import cv2
import numpy as np

from .config import CalibrationConfig, SpeedConfig
from .tracker import Track


class SpeedCalculator:
    def __init__(self, calib: CalibrationConfig, speed_cfg: SpeedConfig, fps: float):
        self._speed_cfg = speed_cfg
        self._fps = fps

        src = np.float32(calib.image_points)   # shape (4, 2)
        dst = np.float32(calib.world_points)   # shape (4, 2)
        self._H = cv2.getPerspectiveTransform(src, dst)

    def pixel_to_world(self, px: int, py: int) -> tuple[float, float]:
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._H)
        return float(world[0][0][0]), float(world[0][0][1])

    def compute_speed(self, track: Track) -> float | None:
        cfg = self._speed_cfg
        positions = list(track.world_positions)
        window = min(cfg.smoothing_window, len(positions))
        if window < 2:
            return None
        recent = positions[-window:]
        total_dist = 0.0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]
            total_dist += math.sqrt(dx * dx + dy * dy)
        elapsed = (len(recent) - 1) / self._fps
        if elapsed <= 0:
            return None
        speed_ms = total_dist / elapsed
        speed_kmh = speed_ms * 3.6
        if speed_kmh < cfg.min_speed_kmh or speed_kmh > cfg.max_speed_kmh:
            return None
        return speed_kmh

    def is_stationary(self, track: Track) -> bool:
        """Return True when the track has enough data and is moving below min_speed_kmh."""
        positions = list(track.world_positions)
        window = min(self._speed_cfg.smoothing_window, len(positions))
        if window < 2:
            return False  # not enough data yet to decide
        recent = positions[-window:]
        total_dist = sum(
            math.sqrt((recent[i][0] - recent[i - 1][0]) ** 2 +
                      (recent[i][1] - recent[i - 1][1]) ** 2)
            for i in range(1, len(recent))
        )
        elapsed = (len(recent) - 1) / self._fps
        if elapsed <= 0:
            return False
        return (total_dist / elapsed) * 3.6 < self._speed_cfg.min_speed_kmh

