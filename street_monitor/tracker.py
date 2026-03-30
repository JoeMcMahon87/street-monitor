from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackingConfig
from .detector import Detection
from .utils import euclidean_distance

_MAX_HISTORY = 60   # how many centroids / world positions to keep per track


@dataclass
class Track:
    track_id: int
    class_name: str
    centroids: deque = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY))
    bboxes: deque = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY))
    world_positions: deque = field(default_factory=lambda: deque(maxlen=_MAX_HISTORY))
    frames_seen: int = 0
    frames_disappeared: int = 0
    confirmed: bool = False
    stationary: bool = False
    speed_kmh: float | None = None
    speed_samples: list = field(default_factory=list)
    min_cx: int | None = None
    max_cx: int | None = None
    first_seen_frame: int = 0
    last_seen_frame: int = 0


class CentroidTracker:
    def __init__(self, config: TrackingConfig, debug: bool = False):
        self._config = config
        self._debug = debug
        self._tracks: dict[int, Track] = {}
        # Tracks that disappeared and are held for possible re-identification.
        # Maps track_id -> (track, expiry_frame).
        self._dormant: dict[int, tuple[Track, int]] = {}
        self._next_id = 0

    def _predict_centroid(self, track: Track) -> tuple[float, float]:
        """Return last centroid + one-step velocity, or last centroid if only one point."""
        centroids = list(track.centroids)
        if len(centroids) >= 2:
            dx = centroids[-1][0] - centroids[-2][0]
            dy = centroids[-1][1] - centroids[-2][1]
            return (centroids[-1][0] + dx, centroids[-1][1] + dy)
        return centroids[-1]

    def _adaptive_max_distance(self, track: Track) -> float:
        """Scale the distance threshold by the track's current pixel speed."""
        centroids = list(track.centroids)
        if len(centroids) >= 2:
            dx = centroids[-1][0] - centroids[-2][0]
            dy = centroids[-1][1] - centroids[-2][1]
            speed_px = math.sqrt(dx * dx + dy * dy)
            return max(self._config.max_distance, speed_px * self._config.velocity_scale)
        return self._config.max_distance

    def update(self, detections: list[Detection], frame_number: int) -> dict[int, Track]:
        # Expire dormant tracks that have been waiting too long.
        expired = [
            tid for tid, (_, exp) in self._dormant.items()
            if frame_number > exp
        ]
        for tid in expired:
            track, _ = self._dormant.pop(tid)
            if self._debug:
                print(f"[DBG f{frame_number}] EXPIRED dormant #{tid} {track.class_name} "
                      f"(min_cx={track.min_cx} max_cx={track.max_cx} "
                      f"samples={len(track.speed_samples)})")

        if not detections:
            for track in self._tracks.values():
                track.frames_disappeared += 1
            self._retire_lost_tracks(frame_number)
            return dict(self._tracks)

        if not self._tracks:
            for det in detections:
                self._register_or_reidentify(det, frame_number)
            return dict(self._tracks)

        track_ids = list(self._tracks.keys())
        track_centroids = [self._predict_centroid(self._tracks[tid]) for tid in track_ids]
        det_centroids = [d.centroid for d in detections]

        cost = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for r, tc in enumerate(track_centroids):
            for c, dc in enumerate(det_centroids):
                cost[r, c] = euclidean_distance(tc, dc)

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > self._adaptive_max_distance(self._tracks[track_ids[r]]):
                continue
            tid = track_ids[r]
            track = self._tracks[tid]
            det = detections[c]
            track.centroids.append(det.centroid)
            track.bboxes.append(det.bbox)
            track.frames_seen += 1
            track.frames_disappeared = 0
            track.last_seen_frame = frame_number
            if not track.confirmed and track.frames_seen >= 3:
                track.confirmed = True
                if self._debug:
                    print(f"[DBG f{frame_number}] CONFIRMED #{tid} {track.class_name} "
                          f"at cx={det.centroid[0]}")
            matched_tracks.add(r)
            matched_dets.add(c)

        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                t = self._tracks[tid]
                t.frames_disappeared += 1
                if self._debug and t.confirmed:
                    print(f"[DBG f{frame_number}] UNMATCHED #{tid} {t.class_name} "
                          f"disappeared={t.frames_disappeared}/{self._config.max_disappeared}")

        self._retire_lost_tracks(frame_number)

        for c, det in enumerate(detections):
            if c not in matched_dets:
                self._register_or_reidentify(det, frame_number)

        return dict(self._tracks)

    def _retire_lost_tracks(self, frame_number: int) -> None:
        to_retire = [
            tid for tid, t in self._tracks.items()
            if t.frames_disappeared > self._config.max_disappeared
        ]
        for tid in to_retire:
            track = self._tracks.pop(tid)
            expiry = frame_number + self._config.max_dormant_frames
            self._dormant[tid] = (track, expiry)
            if self._debug:
                print(f"[DBG f{frame_number}] LOST #{tid} {track.class_name} "
                      f"→ dormant until f{expiry} "
                      f"(min_cx={track.min_cx} max_cx={track.max_cx} "
                      f"samples={len(track.speed_samples)})")

    def _register_or_reidentify(self, det: Detection, frame_number: int) -> None:
        best_tid = None
        best_dist = self._config.reidentification_distance

        for tid, (track, _) in self._dormant.items():
            if track.class_name != det.class_name:
                continue
            last_centroid = list(track.centroids)[-1]
            dist = euclidean_distance(last_centroid, det.centroid)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None:
            track, _ = self._dormant.pop(best_tid)
            track.centroids.append(det.centroid)
            track.bboxes.append(det.bbox)
            track.frames_seen += 1
            track.frames_disappeared = 0
            track.last_seen_frame = frame_number
            self._tracks[best_tid] = track
            if self._debug:
                print(f"[DBG f{frame_number}] RE-ID #{best_tid} {track.class_name} "
                      f"dist={best_dist:.0f}px cx={det.centroid[0]} "
                      f"(min_cx={track.min_cx} max_cx={track.max_cx} "
                      f"samples={len(track.speed_samples)})")
        else:
            if self._debug:
                dormant_count = len(self._dormant)
                print(f"[DBG f{frame_number}] NEW #{self._next_id} {det.class_name} "
                      f"cx={det.centroid[0]} "
                      f"(no dormant match, pool size={dormant_count})")
            self._register(det, frame_number)

    @property
    def dormant_tracks(self) -> dict[int, "Track"]:
        """Tracks that have disappeared and are awaiting re-identification."""
        return {tid: track for tid, (track, _) in self._dormant.items()}

    def _register(self, det: Detection, frame_number: int) -> None:
        track = Track(
            track_id=self._next_id,
            class_name=det.class_name,
            first_seen_frame=frame_number,
            last_seen_frame=frame_number,
        )
        track.centroids.append(det.centroid)
        track.bboxes.append(det.bbox)
        track.frames_seen = 1
        self._tracks[self._next_id] = track
        self._next_id += 1
