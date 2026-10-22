from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from .types import Detection

log = logging.getLogger(__name__)

FLAME_TYPES = ("uncontrolled", "controlled", "spark", "screen", "unknown")


class FireTypePolicy:
    """Runtime include/exclude policy, independent of detector retraining.

    Detections without a trained flame-type classifier are `unknown`. The default
    includes `uncontrolled` + `unknown` and excludes controlled flames/sparks/
    screen fire, but `unknown` inclusion means nothing is silently dropped until
    the Phase B/D classifier exists.
    """

    def __init__(self, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None):
        self.include = set(include or ("uncontrolled", "unknown"))
        self.exclude = set(exclude or ("controlled", "spark", "screen"))

    def allows(self, detection: Detection) -> bool:
        if detection.label != "fire":
            return True
        ftype = detection.flame_type or detection.meta.get("flame_type") or "unknown"
        return ftype in self.include and ftype not in self.exclude

    def partition(self, detections: Sequence[Detection]) -> tuple[list[Detection], list[Detection]]:
        kept, suppressed = [], []
        for det in detections:
            if self.allows(det):
                kept.append(det)
            else:
                det.meta["suppressed_by_policy"] = True
                det.meta["policy_flame_type"] = det.flame_type or det.meta.get("flame_type") or "unknown"
                suppressed.append(det)
        return kept, suppressed

    def to_dict(self) -> dict:
        return {"include": sorted(self.include), "exclude": sorted(self.exclude)}


class FlameTypeClassifier:
    """Optional ONNX crop classifier for flame_type.

    If the model is absent, detections remain `unknown`; this is deliberate so the
    runtime policy can be configured before Phase B/D training.
    """

    def __init__(self, model_path: str | Path | None, input_size: int = 128):
        self.model_path = Path(model_path) if model_path else None
        self.input_size = input_size
        self.session = None
        self.classes = list(FLAME_TYPES)
        if self.model_path and self.model_path.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
                log.info("flame-type classifier loaded: %s", self.model_path)
            except Exception as exc:
                log.warning("flame-type classifier unavailable: %s", exc)

    def annotate(self, frame, detections: Sequence[Detection]) -> list[Detection]:
        if self.session is None:
            for det in detections:
                det.flame_type = det.flame_type or "unknown"
            return list(detections)
        import cv2
        import numpy as np
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det.box.to_list())
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            crops.append(cv2.resize(crop, (self.input_size, self.input_size)) if crop.size else np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
        batch = np.stack([(cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0) for c in crops])
        input_name = self.session.get_inputs()[0].name
        logits = self.session.run(None, {input_name: np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        for det, prob in zip(detections, probs):
            idx = int(prob.argmax())
            det.flame_type = self.classes[idx] if idx < len(self.classes) else "unknown"
            det.meta["flame_type_confidence"] = round(float(prob[idx]), 4)
        return list(detections)
