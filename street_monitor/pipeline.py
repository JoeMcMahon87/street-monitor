from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .config import Config
from .detector import Detection, VehicleDetector
from .speed_calculator import SpeedCalculator
from .tracker import CentroidTracker, Track


@dataclass
class CompletedDetection:
    track_id: int
    class_name: str
    avg_speed_kmh: float
    min_speed_kmh: float
    max_speed_kmh: float
    direction: str
    timestamp: datetime
    frame_number: int


@dataclass
class PipelineResult:
    detections: list[Detection]
    tracks: dict[int, Track]
    completed: list[CompletedDetection]


class Pipeline:
    def __init__(self, config: Config, fps: float, debug: bool = False):
        self._config = config
        self._debug = debug
        self._detector = VehicleDetector(config.detection)
        self._tracker = CentroidTracker(config.tracking, debug=debug)
        self._speed_calc = SpeedCalculator(config.calibration, config.speed, fps)
        # Track which IDs have already emitted a completed detection
        self._emitted: set[int] = set()
        # Track which IDs have been consumed by fragment stitching
        self._stitched: set[int] = set()
        # Track which IDs have already logged a stationary message (suppress repeats)
        self._reported_stationary: set[int] = set()

    def process_frame(self, frame: np.ndarray, frame_number: int) -> PipelineResult:
        detections = self._detector.detect(frame)
        tracks = self._tracker.update(detections, frame_number)

        for tid in self._tracker.recently_expired:
            self._emitted.discard(tid)
            self._stitched.discard(tid)
            self._reported_stationary.discard(tid)

        completed: list[CompletedDetection] = []
        entry_x = self._config.lines.entry_x
        exit_x = self._config.lines.exit_x
        min_frames = self._config.speed.min_track_frames

        for tid, track in tracks.items():
            # Track centroid extents from the very first frame so we capture
            # positions before confirmation (e.g. vehicle enters left of entry_x
            # but is only confirmed after it has already crossed entry_x).
            cx, cy = list(track.centroids)[-1]
            track.min_cx = cx if track.min_cx is None else min(track.min_cx, cx)
            track.max_cx = cx if track.max_cx is None else max(track.max_cx, cx)

            if not track.confirmed:
                continue

            if track.frames_seen < min_frames:
                continue

            # Update world position for the latest centroid
            wx, wy = self._speed_calc.pixel_to_world(cx, cy)
            track.world_positions.append((wx, wy))

            # Update speed estimate and stationary flag
            track.stationary = self._speed_calc.is_stationary(track)
            track.speed_kmh = self._speed_calc.compute_speed(track)
            if track.speed_kmh is not None:
                track.speed_samples.append(track.speed_kmh)

            if tid in self._emitted or track.stationary:
                if self._debug and track.stationary and tid not in self._reported_stationary:
                    print(f"[DBG f{frame_number}] STATIONARY #{tid} {track.class_name} "
                          f"speed={track.speed_kmh}")
                    self._reported_stationary.add(tid)
                continue

            # Full traversal: centroid has historically been on both sides of the window.
            # min_cx < entry_x means it was ever left of the left line.
            # max_cx > exit_x means it was ever right of the right line.
            samples = track.speed_samples
            if self._debug and frame_number % 15 == 0:
                left_ok = track.min_cx is not None and track.min_cx < entry_x
                right_ok = track.max_cx is not None and track.max_cx > exit_x
                print(f"[DBG f{frame_number}] CHECK #{tid} {track.class_name}: "
                      f"min_cx={track.min_cx}{'<' if left_ok else '>='}{entry_x}{'✓' if left_ok else '✗'} "
                      f"max_cx={track.max_cx}{'>=' if right_ok else '<'}{exit_x}{'✓' if right_ok else '✗'} "
                      f"samples={len(samples)} speed={track.speed_kmh}")
            if (track.min_cx < entry_x and track.max_cx > exit_x and samples):
                direction = self._infer_direction(track)
                completed.append(CompletedDetection(
                    track_id=tid,
                    class_name=track.class_name,
                    avg_speed_kmh=round(sum(samples) / len(samples), 1),
                    min_speed_kmh=round(min(samples), 1),
                    max_speed_kmh=round(max(samples), 1),
                    direction=direction,
                    timestamp=datetime.now(),
                    frame_number=frame_number,
                ))
                self._emitted.add(tid)

        # Fragment stitching: two tracks from the same car can fail to re-identify
        # when the occlusion gap is shorter than max_disappeared (so the first track
        # is still active when the car re-emerges).  Once both fragments are in the
        # dormant pool, check if any same-class pair collectively covers the full
        # traversal window (combined min_cx < entry_x AND max_cx > exit_x).
        dormant = self._tracker.dormant_tracks
        dormant_ids = [
            tid for tid in dormant
            if tid not in self._emitted and tid not in self._stitched
        ]
        newly_stitched: set[int] = set()
        for i, tid_a in enumerate(dormant_ids):
            if tid_a in newly_stitched:
                continue
            track_a = dormant[tid_a]
            if track_a.min_cx is None or track_a.max_cx is None:
                continue
            for tid_b in dormant_ids[i + 1:]:
                if tid_b in newly_stitched:
                    continue
                track_b = dormant[tid_b]
                if track_b.class_name != track_a.class_name:
                    continue
                if track_b.min_cx is None or track_b.max_cx is None:
                    continue
                # Ensure temporal adjacency: the later fragment must have started
                # within max_dormant_frames of the earlier fragment's last sighting.
                first, second = (
                    (track_a, track_b) if track_a.first_seen_frame <= track_b.first_seen_frame
                    else (track_b, track_a)
                )
                gap = second.first_seen_frame - first.last_seen_frame
                if gap < 0 or gap > self._config.tracking.max_dormant_frames:
                    continue
                combined_min = min(track_a.min_cx, track_b.min_cx)
                combined_max = max(track_a.max_cx, track_b.max_cx)
                combined_samples = track_a.speed_samples + track_b.speed_samples
                if combined_min < entry_x and combined_max > exit_x and combined_samples:
                    # Direction: whichever fragment started on the exit side came first
                    direction = (
                        "right-to-left" if first.max_cx > exit_x else "left-to-right"
                    )
                    if self._debug:
                        print(f"[DBG f{frame_number}] STITCHED #{tid_a}+#{tid_b} "
                              f"{track_a.class_name} combined "
                              f"min_cx={combined_min} max_cx={combined_max} "
                              f"samples={len(combined_samples)} gap={gap}f")
                    completed.append(CompletedDetection(
                        track_id=first.track_id,
                        class_name=track_a.class_name,
                        avg_speed_kmh=round(sum(combined_samples) / len(combined_samples), 1),
                        min_speed_kmh=round(min(combined_samples), 1),
                        max_speed_kmh=round(max(combined_samples), 1),
                        direction=direction,
                        timestamp=datetime.now(),
                        frame_number=frame_number,
                    ))
                    newly_stitched.add(tid_a)
                    newly_stitched.add(tid_b)
                    break  # each fragment stitches at most once
        self._stitched.update(newly_stitched)

        return PipelineResult(detections=detections, tracks=tracks, completed=completed)

    def _infer_direction(self, track: Track) -> str:
        centroids = list(track.centroids)
        if len(centroids) < 2:
            return "unknown"
        dx = centroids[-1][0] - centroids[0][0]
        return "left-to-right" if dx >= 0 else "right-to-left"
