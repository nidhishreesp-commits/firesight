from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..types import Box, Detection
from .base import BaseDetector, normalize_label

log = logging.getLogger(__name__)


class YoloDetector(BaseDetector):
    def __init__(
        self,
        model_path: str | Path,
        imgsz: int = 640,
        iou: float = 0.45,
        max_det: int = 100,
        device: str = "auto",
        conf_floor: float = 0.01,
        class_map: dict[int, str] | None = None,
        name: str | None = None,
        license: str = "unknown",
        family: str = "yolo",
    ):
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det
        self.device = self._resolve_device(device)
        self.conf_floor = conf_floor
        self.class_map = class_map or {}
        self.name = name or self.model_path.stem
        self.license = license
        self.family = family
        self.model = YOLO(str(self.model_path), task="detect")
        self.names = {int(k): str(v) for k, v in (self.model.names or {}).items()}
        self._sha256: str | None = None
        log.info("loaded detector %s device=%s classes=%s", self.name, self.device, self.names)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def model_size_mb(self) -> float:
        return self.model_path.stat().st_size / (1024 * 1024)

    @property
    def sha256(self) -> str:
        if self._sha256 is None:
            h = hashlib.sha256()
            with self.model_path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            self._sha256 = h.hexdigest()
        return self._sha256

    def _to_input(self, image: str | Path | np.ndarray) -> str | np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        return str(image)

    def _postprocess(self, result: Any) -> list[Detection]:
        detections: list[Detection] = []
        if result is None or result.boxes is None:
            return detections
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, "detach") else np.asarray(boxes.xyxy)
        confs = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, "detach") else np.asarray(boxes.conf)
        classes = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, "detach") else np.asarray(boxes.cls)
        ids = None
        if getattr(boxes, "id", None) is not None:
            ids = boxes.id.detach().cpu().numpy() if hasattr(boxes.id, "detach") else np.asarray(boxes.id)
        for idx, (coords, conf, cls_id) in enumerate(zip(xyxy, confs, classes)):
            cls_id = int(cls_id)
            label = normalize_label(self.names.get(cls_id, str(cls_id)))
            if label is None:
                label = normalize_label(self.class_map.get(cls_id, ""))
            if label is None:
                continue
            track_id = int(ids[idx]) if ids is not None else None
            detections.append(Detection(
                label=label,
                confidence=float(conf),
                box=Box(*[float(v) for v in coords[:4]]),
                track_id=track_id,
            ))
        return detections

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        results = self.model.predict(
            self._to_input(image),
            imgsz=self.imgsz,
            conf=self.conf_floor,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        return self._postprocess(results[0] if results else None)

    def predict_batch(self, images: Sequence[str | Path | np.ndarray]) -> list[list[Detection]]:
        inputs = [self._to_input(image) for image in images]
        if self.model_path.suffix.lower() in {".onnx", ".engine"} or len(inputs) == 1:
            return [self.predict(image) for image in inputs]
        results = self.model.predict(
            inputs,
            imgsz=self.imgsz,
            conf=self.conf_floor,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        return [self._postprocess(result) for result in results]

    def predict_tracked(self, frame: np.ndarray, persist: bool = True) -> list[Detection]:
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_floor,
            iou=self.iou,
            max_det=self.max_det,
            device=self.device,
            persist=persist,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return self._postprocess(results[0] if results else None)
