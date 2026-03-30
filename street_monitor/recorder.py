from __future__ import annotations

import csv
import os
from datetime import datetime

import numpy as np

from .config import RecordingConfig
from .pipeline import CompletedDetection


class Recorder:
    def __init__(self, config: RecordingConfig):
        self._cfg = config
        self._csv_fh = None
        self._csv_writer = None

        if not config.enabled:
            return

        os.makedirs(config.output_dir, exist_ok=True)
        if config.save_csv:
            write_header = not os.path.exists(config.csv_path)
            self._csv_fh = open(config.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_fh,
                fieldnames=["timestamp", "track_id", "class_name",
                            "avg_speed_kmh", "min_speed_kmh", "max_speed_kmh",
                            "direction", "frame_number"],
            )
            if write_header:
                self._csv_writer.writeheader()

    def record_detection(self, detection: CompletedDetection,
                          frame: np.ndarray | None = None) -> None:
        if not self._cfg.enabled:
            return
        if self._csv_writer:
            self._csv_writer.writerow({
                "timestamp": detection.timestamp.isoformat(),
                "track_id": detection.track_id,
                "class_name": detection.class_name,
                "avg_speed_kmh": detection.avg_speed_kmh,
                "min_speed_kmh": detection.min_speed_kmh,
                "max_speed_kmh": detection.max_speed_kmh,
                "direction": detection.direction,
                "frame_number": detection.frame_number,
            })
            self._csv_fh.flush()

    def close(self) -> None:
        if self._csv_fh:
            self._csv_fh.close()
            self._csv_fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
