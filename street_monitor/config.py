from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class CameraConfig:
    device_id: int | str = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0


@dataclass
class CalibrationConfig:
    image_points: list[list[float]] = field(default_factory=lambda: [
        [312, 280], [968, 280], [1100, 580], [180, 580]
    ])
    world_points: list[list[float]] = field(default_factory=lambda: [
        [0.0, 0.0], [3.65, 0.0], [3.65, 15.0], [0.0, 15.0]
    ])


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    confidence: float = 0.4
    classes: list[int] = field(default_factory=lambda: [2, 7])
    device: str = "cpu"


@dataclass
class TrackingConfig:
    max_disappeared: int = 15
    max_distance: float = 80.0
    max_dormant_frames: int = 45          # frames to hold a lost track for re-ID (1.5 s at 30 fps)
    reidentification_distance: float = 200.0  # max pixel distance for re-ID match
    velocity_scale: float = 2.0           # adaptive threshold = max(max_distance, speed_px * velocity_scale)


@dataclass
class SpeedConfig:
    min_track_frames: int = 8
    smoothing_window: int = 6
    min_speed_kmh: float = 3.0
    max_speed_kmh: float = 200.0


@dataclass
class LinesConfig:
    entry_x: int = 200
    exit_x: int = 1080


@dataclass
class DisplayConfig:
    show_tracks: bool = True
    show_bboxes: bool = True
    show_speed: bool = True
    show_calibration_grid: bool = False
    font_scale: float = 0.7
    speed_limit_kmh: float = 50.0


@dataclass
class RecordingConfig:
    enabled: bool = True
    output_dir: str = "data/recordings"
    save_video: bool = False
    save_csv: bool = True
    csv_path: str = "data/recordings/detections.csv"


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    lines: LinesConfig = field(default_factory=LinesConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)


def _from_dict(cls, data: dict):
    """Recursively populate a dataclass from a dict."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return data
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        # Resolve string annotations
        ft = f.type
        if isinstance(ft, str):
            ft = eval(ft)  # noqa: S307 - controlled internal use only
        if dataclasses.is_dataclass(ft) and isinstance(val, dict):
            kwargs[f.name] = _from_dict(ft, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def load_config(path: str = "config.yaml") -> Config:
    if not os.path.exists(path):
        return Config()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(Config, raw)


def save_config(config: Config, path: str = "config.yaml") -> None:
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return obj

    with open(path, "w") as fh:
        yaml.dump(to_dict(config), fh, default_flow_style=False, sort_keys=False)
