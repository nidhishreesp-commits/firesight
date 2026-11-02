from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from prometheus_client import Counter as PCounter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except Exception:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False


if PROMETHEUS_AVAILABLE:
    DETECTIONS_TOTAL = PCounter("firesight_detections_total", "Detections emitted", ["camera", "label"])
    ALARMS_TOTAL = PCounter("firesight_alarms_total", "Alarm events", ["camera", "level"])
    FEEDBACK_TOTAL = PCounter("firesight_feedback_total", "Operator feedback", ["verdict"])
    INFERENCE_LATENCY = Histogram("firesight_inference_latency_ms", "Per-frame inference latency", ["model"])
    STREAM_FPS = Gauge("firesight_stream_fps", "Measured stream FPS", ["camera"])
    ACTIVE_ALARMS = Gauge("firesight_active_alarms", "Active alarms", ["camera"])


def metrics_payload() -> tuple[bytes, str]:
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


class DriftMonitor:
    def __init__(self, reference_path: str | Path, window: int = 500):
        self.reference_path = Path(reference_path)
        self.window = window
        self.history: list[dict] = []
        self.reference = json.loads(self.reference_path.read_text(encoding="utf-8")) if self.reference_path.exists() else None

    def update(self, detections: list[dict], fps: float) -> dict:
        row = {
            "count": len(detections),
            "labels": Counter(d["label"] for d in detections),
            "mean_conf": sum(d["confidence"] for d in detections) / len(detections) if detections else 0.0,
            "fps": fps,
        }
        self.history.append(row)
        self.history = self.history[-self.window:]
        if self.reference is None and len(self.history) >= self.window:
            self.reference = {
                "mean_detections": sum(r["count"] for r in self.history) / len(self.history),
                "mean_conf": sum(r["mean_conf"] for r in self.history) / len(self.history),
            }
        drift = None
        if self.reference:
            mean_det = sum(r["count"] for r in self.history) / max(len(self.history), 1)
            drift = {
                "reference_mean_detections": self.reference.get("mean_detections"),
                "current_mean_detections": round(mean_det, 4),
                "relative_change": round((mean_det - self.reference.get("mean_detections", 0)) / max(self.reference.get("mean_detections", 1e-6), 1e-6), 4),
            }
        return drift or {}
