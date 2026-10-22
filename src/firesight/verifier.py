from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .types import Box, Detection


class CropVerifier:
    def __init__(self, model_path: str | Path, classes: Sequence[str] = ("fire", "lookalike", "smoke"),
                 input_size: int = 128, threshold: float = 0.6):
        import onnxruntime as ort
        self.model_path = Path(model_path)
        self.classes = list(classes)
        self.input_size = input_size
        self.threshold = threshold
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        crop = cv2.resize(crop, (self.input_size, self.input_size))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        return np.transpose(crop, (2, 0, 1))

    def predict_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        batch = np.stack([self._preprocess(c) for c in crops]).astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        logits = np.asarray(logits)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def verify(self, frame: np.ndarray, detections: Sequence[Detection]) -> tuple[list[Detection], list[dict]]:
        import cv2
        h, w = frame.shape[:2]
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det.box.to_list()
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * 0.15, bh * 0.15
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
            else:
                crops.append(frame[y1:y2, x1:x2])
        probs = self.predict_crops(crops)
        kept: list[Detection] = []
        rows: list[dict] = []
        for det, prob in zip(detections, probs):
            scores = {c: float(p) for c, p in zip(self.classes, prob)}
            row = {"label": det.label, "confidence": round(det.confidence, 4), **{k: round(v, 4) for k, v in scores.items()}}
            rows.append(row)
            det.meta["verifier"] = {k: round(v, 4) for k, v in scores.items()}
            if scores.get("lookalike", 0.0) >= self.threshold:
                continue
            det.meta["verifier_kept"] = True
            kept.append(det)
        return kept, rows


def load_verifier(path: str | Path, classes: Sequence[str], threshold: float) -> CropVerifier:
    return CropVerifier(path, classes=classes, threshold=threshold)
