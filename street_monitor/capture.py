from __future__ import annotations

import queue
import threading

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

        # Small queue: we only buffer 2 frames so the main thread always gets
        # the most recent frame rather than a stale backlog.
        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                # Signal EOF to the consumer and exit.
                self._queue.put((False, None))
                break
            self._put_latest((True, frame))

    def _put_latest(self, item: tuple) -> None:
        """Put a frame in the queue, evicting the oldest entry if full."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()  # discard stale frame
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass  # lost the race; drop this frame rather than blocking

    def read(self) -> tuple[bool, np.ndarray]:
        try:
            return self._queue.get(timeout=1.0)
        except queue.Empty:
            # Capture thread died unexpectedly; signal end-of-stream.
            return False, None

    def get_fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            return fps
        return float(self._config.fps)

    def release(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
