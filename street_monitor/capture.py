from __future__ import annotations

import cv2
import numpy as np

from .config import CameraConfig


class VideoCapture:
    def __init__(self, config: CameraConfig):
        self._config = config
        self._cap = cv2.VideoCapture(config.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera/video: {config.device_id!r}")
        # Only set resolution/fps for live cameras (integer device IDs).
        if isinstance(config.device_id, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            self._cap.set(cv2.CAP_PROP_FPS, config.fps)

    def read(self) -> tuple[bool, np.ndarray]:
        return self._cap.read()

    def get_fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return fps
        return float(self._config.fps)

    def release(self) -> None:
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
