from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import Detection


class BaseDetector(ABC):
    name: str = "detector"
    license: str = "unknown"
    family: str = "unknown"

    @abstractmethod
    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        raise NotImplementedError

    def predict_batch(self, images: Sequence[str | Path | np.ndarray]) -> list[list[Detection]]:
        return [self.predict(image) for image in images]

    def close(self) -> None:
        return None

    @property
    def model_size_mb(self) -> float:
        return 0.0


LABEL_ALIASES = {
    "fire": "fire",
    "flame": "fire",
    "fires": "fire",
    "smoke": "smoke",
    "smokes": "smoke",
    "smoke plume": "smoke",
}


def normalize_label(label: str) -> str | None:
    return LABEL_ALIASES.get(str(label).strip().lower())


def load_detector(backend: str, model_path: str | Path, **kwargs) -> BaseDetector:
    backend = backend.lower()
    if backend in {"yolo", "onnx", "torch", "ultralytics", "rtdetr"}:
        from .yolo import YoloDetector
        return YoloDetector(model_path, **kwargs)
    if backend in {"grounding_dino", "grounding-dino", "gdino"}:
        from .grounding_dino import GroundingDinoDetector
        return GroundingDinoDetector(model_path, **kwargs)
    if backend in {"rtdetr_v2", "rtdetrv2", "rtdetr2"}:
        from .rtdetr_v2 import RTDetrV2Detector
        return RTDetrV2Detector(model_path, **kwargs)
    raise ValueError(f"Unknown detector backend: {backend}")
