from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..types import Box, Detection
from .base import BaseDetector, normalize_label

log = logging.getLogger(__name__)

LABEL_MAP = {
    "smoke_visible": "smoke",
    "flame_visible": "fire",
    "fire": "fire",
    "smoke": "smoke",
    "flame": "fire",
}


class RTDetrV2Detector(BaseDetector):
    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.20,
        preprocess_mode: str = "hf",
        device: str = "auto",
        name: str | None = None,
        license: str = "unknown",
        family: str = "RT-DETRv2",
        **kwargs,
    ):
        self.preprocess_mode = preprocess_mode
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self.model_id = str(model_path)
        self.threshold = threshold
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.name = name or "rtdetr_v2"
        self.license = license
        self.family = family
        self.processor = RTDetrImageProcessor.from_pretrained(self.model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @property
    def model_size_mb(self) -> float:
        try:
            from huggingface_hub import scan_cache_dir
            for repo in scan_cache_dir().repos:
                if repo.repo_id == self.model_id:
                    return repo.size_on_disk / (1024 * 1024)
        except Exception:
            pass
        return 0.0

    def _preprocess(self, image: Image.Image):
        if self.preprocess_mode == "hf":
            return self.processor(images=image, return_tensors="pt")
        w, h = image.size
        if self.preprocess_mode == "squash":
            prepared = image.resize((768, 768))
        elif self.preprocess_mode == "centercrop":
            side = min(w, h)
            prepared = image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2)).resize((768, 768))
        else:
            raise ValueError(self.preprocess_mode)
        return self.processor(images=prepared, return_tensors="pt", do_resize=False, do_pad=False, do_normalize=False)

    def _predict_pil(self, image: Image.Image) -> list[Detection]:
        import torch
        inputs = self._preprocess(image).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[image.size[::-1]], threshold=self.threshold,
        )[0]
        detections: list[Detection] = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            raw = self.id2label.get(int(label_id), str(int(label_id)))
            label = LABEL_MAP.get(raw.lower()) or normalize_label(raw)
            if label is None:
                continue
            coords = box.detach().cpu().numpy()
            detections.append(Detection(label=label, confidence=float(score), box=Box(*[float(v) for v in coords[:4]])))
        return detections

    def predict(self, image: str | Path | np.ndarray | Image.Image) -> list[Detection]:
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            import cv2
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil = Image.open(image).convert("RGB")
        return self._predict_pil(pil)

    def predict_batch(self, images: Sequence[str | Path | np.ndarray]) -> list[list[Detection]]:
        return [self.predict(image) for image in images]
