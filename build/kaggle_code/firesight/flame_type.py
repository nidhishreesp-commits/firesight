from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from .types import Detection

log = logging.getLogger(__name__)

FLAME_TYPES = ("uncontrolled", "controlled", "spark_benign", "electrical_arc", "screen", "unknown")

LEGACY_ALIASES = {"spark": "spark_benign"}

DEFAULT_INCLUDE = ("uncontrolled", "unknown")
DEFAULT_EXCLUDE = ("controlled", "spark_benign", "screen")
DEFAULT_WARNING = ("electrical_arc",)
POLICY_FILE = "configs/fire_type_policy.json"

TIER_ALARM = "alarm"
TIER_WARNING = "warning"
TIER_LOG = "log"
TIERS = (TIER_ALARM, TIER_WARNING, TIER_LOG)


def canonical_flame_type(value: str | None) -> str:
    """Map a flame type onto the 6-class taxonomy, resolving legacy aliases.

    The locked bench and older labels still say `spark`; it means `spark_benign`
    (welding, grinding, cutting, fireworks). Unknown/custom values are preserved
    so a policy can still name them explicitly.
    """
    if not value:
        return "unknown"
    key = str(value).strip().lower()
    key = LEGACY_ALIASES.get(key, key)
    return key or "unknown"


class FireTypePolicy:
    """Runtime alarm/warning/log policy, independent of detector retraining.

    Three tiers (plan patch):
      * `alarm`   - uncontrolled fire/smoke: raise an alarm (default: uncontrolled, unknown)
      * `warning` - electrical arc: surface a warning, never an alarm (default: electrical_arc)
      * `log`     - controlled flame, benign sparks, screen fire: record only

    The policy applies to fire AND smoke detections alike: an uncontrolled-fire
    plume must alarm, while controlled smoke (cooking, exhaust, steam) must not.

    Detections without a trained flame-type classifier are `unknown`. The default
    includes `unknown` and excludes controlled/spark/screen, but `unknown`
    inclusion means nothing is silently dropped until the classifier exists. The
    policy is persisted to `configs/fire_type_policy.json` so a restart does not
    revert it to the yaml defaults.
    """

    def __init__(self, include: Sequence[str] | None = None, exclude: Sequence[str] | None = None,
                 warning: Sequence[str] | None = None):
        self.include = {canonical_flame_type(t) for t in (DEFAULT_INCLUDE if include is None else include)}
        self.exclude = {canonical_flame_type(t) for t in (DEFAULT_EXCLUDE if exclude is None else exclude)}
        self.warning = {canonical_flame_type(t) for t in (DEFAULT_WARNING if warning is None else warning)}

    @classmethod
    def load(cls, path: str | Path, default_include: Sequence[str] | None = None,
             default_exclude: Sequence[str] | None = None,
             default_warning: Sequence[str] | None = None) -> "FireTypePolicy":
        p = Path(path)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                warning = data.get("warning")
                if warning is None:
                    warning = DEFAULT_WARNING if default_warning is None else default_warning
                return cls(data.get("include"), data.get("exclude"), warning)
            except Exception as exc:
                log.warning("fire-type policy at %s unreadable, using defaults: %s", p, exc)
        return cls(default_include, default_exclude, default_warning)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _type_of(detection: Detection) -> str:
        return canonical_flame_type(detection.flame_type or detection.meta.get("flame_type"))

    def tier(self, detection: Detection) -> str:
        """Return `alarm`, `warning` or `log` for a detection.

        Precedence: explicit exclusion (log) > inclusion (alarm) > warning > log.
        That keeps the historical include/exclude contract intact while adding a
        middle tier that is visible but never alarms.
        """
        ftype = self._type_of(detection)
        if ftype in self.exclude:
            return TIER_LOG
        if ftype in self.include:
            return TIER_ALARM
        if ftype in self.warning:
            return TIER_WARNING
        return TIER_LOG

    def allows(self, detection: Detection) -> bool:
        """True when the detection is alarm-worthy (the `alarm` tier)."""
        return self.tier(detection) == TIER_ALARM

    def partition(self, detections: Sequence[Detection]) -> tuple[list[Detection], list[Detection]]:
        kept, suppressed = [], []
        for det in detections:
            tier = self.tier(det)
            det.meta["policy_tier"] = tier
            if tier == TIER_ALARM:
                kept.append(det)
                continue
            if tier == TIER_LOG:
                det.meta["suppressed_by_policy"] = True
            else:
                det.meta["warning_by_policy"] = True
            det.meta["policy_flame_type"] = self._type_of(det)
            suppressed.append(det)
        return kept, suppressed

    def to_dict(self) -> dict:
        return {"include": sorted(self.include), "exclude": sorted(self.exclude),
                "warning": sorted(self.warning)}


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
                det.flame_type = canonical_flame_type(det.flame_type)
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
