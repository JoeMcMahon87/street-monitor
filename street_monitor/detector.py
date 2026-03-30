from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectionConfig
from .utils import centroid_of_bbox


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    centroid: tuple[int, int]


_COCO_NAMES = {2: "car", 7: "truck"}


class VehicleDetector:
    def __init__(self, config: DetectionConfig):
        from ultralytics import YOLO
        self._config = config
        self._model = YOLO(config.model)
        self._model.to(config.device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(
            frame,
            conf=self._config.confidence,
            classes=self._config.classes,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    class_id=cls_id,
                    class_name=_COCO_NAMES.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                    centroid=centroid_of_bbox(x1, y1, x2, y2),
                ))
        return detections
