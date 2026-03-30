from __future__ import annotations

import cv2
import numpy as np

from .config import DisplayConfig, LinesConfig
from .pipeline import CompletedDetection, PipelineResult
from .tracker import Track

_GREEN = (0, 200, 0)
_ORANGE = (0, 140, 255)
_GREY = (150, 150, 150)
_BLUE = (255, 100, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 220, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


class Renderer:
    def __init__(self, config: DisplayConfig, lines_cfg: LinesConfig):
        self._cfg = config
        self._lines = lines_cfg
        # track_id -> remaining flash frames
        self._flash: dict[int, int] = {}

    def draw(self, frame: np.ndarray, result: PipelineResult, fps: float = 0.0) -> np.ndarray:
        out = frame.copy()

        # Register new completions for flash effect
        for c in result.completed:
            self._flash[c.track_id] = 20

        if self._cfg.show_bboxes or self._cfg.show_tracks or self._cfg.show_speed:
            for tid, track in result.tracks.items():
                if not track.bboxes or track.stationary:
                    continue
                self._draw_track(out, track)

        # Tick down flash counters
        expired = [tid for tid, n in self._flash.items() if n <= 0]
        for tid in expired:
            del self._flash[tid]
        for tid in self._flash:
            self._flash[tid] -= 1

        # Entry / exit lines
        self._draw_measurement_lines(out)

        # Stats panel
        self._draw_stats(out, len(result.completed), fps)

        return out

    def _draw_track(self, frame: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = list(track.bboxes)[-1]
        cx, cy = list(track.centroids)[-1]

        flashing = track.track_id in self._flash
        if flashing:
            color = _YELLOW
            thickness = 3
        elif not track.confirmed:
            color = _GREY
            thickness = 1
        elif track.class_name == "truck":
            color = _ORANGE
            thickness = 2
        else:
            color = _GREEN
            thickness = 2

        if self._cfg.show_bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"#{track.track_id}"
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, self._cfg.font_scale * 0.7,
                    color, 1, cv2.LINE_AA)

        if self._cfg.show_speed and track.speed_kmh is not None:
            spd = track.speed_kmh
            spd_color = _RED if spd > self._cfg.speed_limit_kmh else _WHITE
            if flashing:
                spd_color = _YELLOW
            spd_text = f"{spd:.0f} km/h"
            cv2.putText(frame, spd_text, (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, self._cfg.font_scale,
                        spd_color, 2, cv2.LINE_AA)

        if self._cfg.show_tracks and len(track.centroids) > 1:
            pts = list(track.centroids)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    def _draw_measurement_lines(self, frame: np.ndarray) -> None:
        h = frame.shape[0]
        self._draw_dashed_vline(frame, self._lines.entry_x, h, _BLUE, label="entry")
        self._draw_dashed_vline(frame, self._lines.exit_x, h, _BLUE, label="exit")

    @staticmethod
    def _draw_dashed_vline(frame: np.ndarray, x: int, height: int,
                            color: tuple, label: str = "") -> None:
        dash, gap = 20, 10
        y = 0
        while y < height:
            y2 = min(y + dash, height)
            cv2.line(frame, (x, y), (x, y2), color, 1)
            y += dash + gap
        if label:
            cv2.putText(frame, label, (x + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _draw_stats(self, frame: np.ndarray, completed_this_frame: int, fps: float) -> None:
        lines = [f"FPS: {fps:.1f}"]
        cv2.rectangle(frame, (0, 0), (140, 30), _BLACK, -1)
        cv2.putText(frame, lines[0], (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA)
