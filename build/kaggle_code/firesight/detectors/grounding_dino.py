from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from ..types import Box, Detection
from .base import BaseDetector, normalize_label

log = logging.getLogger(__name__)


class GroundingDinoDetector(BaseDetector):
    def __init__(
        self,
        model_path: str | Path = "IDEA-Research/grounding-dino-tiny",
        prompt: str = "fire. smoke.",
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
        device: str = "auto",
        name: str | None = None,
        license: str = "Apache-2.0",
        family: str = "GroundingDINO",
        **kwargs,
    ):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.model_id = str(model_path)
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        self.name = name or "grounding_dino"
        self.license = license
        self.family = family
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self._size_mb = 0.0

    @property
    def model_size_mb(self) -> float:
        if self._size_mb:
            return self._size_mb
        try:
            from huggingface_hub import scan_cache_dir
            for repo in scan_cache_dir().repos:
                if repo.repo_id == self.model_id:
                    self._size_mb = repo.size_on_disk / (1024 * 1024)
                    break
        except Exception:
            pass
        return self._size_mb

    def _predict_pil(self, image: Image.Image) -> list[Detection]:
        import torch
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prompt_labels = [[p.strip().rstrip(".") for p in self.prompt.split(".") if p.strip()]]
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
            text_labels=prompt_labels,
        )[0]
        detections: list[Detection] = []
        for box, score, text_label in zip(results.get("boxes", []), results.get("scores", []), results.get("labels", [])):
            label = normalize_label(str(text_label))
            if label is None:
                continue
            coords = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
            detections.append(Detection(label=label, confidence=float(score), box=Box(*[float(v) for v in coords[:4]])))
        return detections

    def predict(self, image: str | Path | np.ndarray) -> list[Detection]:
        if isinstance(image, np.ndarray):
            import cv2
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil = Image.open(image).convert("RGB")
        return self._predict_pil(pil)

    def predict_batch(self, images: Sequence[str | Path | np.ndarray]) -> list[list[Detection]]:
        return [self.predict(image) for image in images]
