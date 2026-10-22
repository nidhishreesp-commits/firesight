from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .config import AlarmConfig, DetectorConfig, StreamConfig, TemporalConfig
from .detectors.base import BaseDetector
from .temporal import AlarmEvent, AlarmManager, TemporalConfirmer
from .types import Detection

log = logging.getLogger(__name__)


def point_in_polygon(x: float, y: float, poly: Sequence[float]) -> bool:
    if len(poly) < 6:
        if len(poly) == 4:
            x1, y1, x2, y2 = poly
            return x1 <= x <= x2 and y1 <= y <= y2
        return True
    inside = False
    j = len(poly) - 2
    for i in range(0, len(poly), 2):
        xi, yi = poly[i], poly[i + 1]
        xj, yj = poly[j], poly[j + 1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def apply_roi(detections: Sequence[Detection], roi: list[list[float]] | None, width: int, height: int) -> list[Detection]:
    if not roi:
        return list(detections)
    out = []
    for det in detections:
        cx, cy = det.box.center
        nx, ny = cx / max(width, 1), cy / max(height, 1)
        if any(point_in_polygon(nx, ny, poly) for poly in roi):
            out.append(det)
    return out


def filter_confidence(detections: Sequence[Detection], conf: dict[str, float]) -> list[Detection]:
    return [d for d in detections if d.confidence >= conf.get(d.label, 1.0)]


@dataclass
class VideoRunResult:
    path: str
    duration_s: float
    frames_native: int
    frames_processed: int
    elapsed_s: float
    inference_fps: float
    detections: int
    confirmed_detections: int
    first_confirmed: dict[str, float] = field(default_factory=dict)
    alarm_events: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    video_width: int = 0
    video_height: int = 0

    @property
    def first_alarm_s(self) -> float | None:
        values = [v for v in self.first_confirmed.values()]
        return min(values) if values else None

    def to_dict(self) -> dict:
        d = {
            "path": self.path, "duration_s": round(self.duration_s, 3),
            "frames_native": self.frames_native, "frames_processed": self.frames_processed,
            "elapsed_s": round(self.elapsed_s, 3), "inference_fps": round(self.inference_fps, 3),
            "detections": self.detections, "confirmed_detections": self.confirmed_detections,
            "first_confirmed_s": {k: round(v, 3) for k, v in self.first_confirmed.items()},
            "first_alarm_s": round(self.first_alarm_s, 3) if self.first_alarm_s is not None else None,
            "alarm_events": self.alarm_events,
            "video_width": self.video_width,
            "video_height": self.video_height,
        }
        return d


def run_video(
    detector: BaseDetector,
    path: str | Path,
    det_cfg: DetectorConfig,
    temporal_cfg: TemporalConfig,
    alarm_cfg: AlarmConfig,
    sample_fps: float = 5.0,
    max_seconds: float | None = None,
    roi: list[list[float]] | None = None,
    camera_id: str = "clip",
    collect_frames: bool = False,
    frame_stride: int | None = None,
    verifier=None,
) -> VideoRunResult:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames_native / native_fps if frames_native else 0.0
    stride = frame_stride or max(1, int(round(native_fps / max(sample_fps, 0.1))))
    confirmer = TemporalConfirmer(temporal_cfg)
    events: list[AlarmEvent] = []

    def _on_event(event: AlarmEvent):
        events.append(event)

    alarm = AlarmManager(alarm_cfg, on_event=_on_event)
    first_confirmed: dict[str, float] = {}
    frames_out: list[dict] = []
    det_count = confirmed_count = 0
    frame_idx = 0
    processed = 0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start = time.perf_counter()
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            ts = frame_idx / native_fps
            if max_seconds is not None and ts > max_seconds:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            h, w = frame.shape[:2]
            dets = detector.predict(frame)
            dets = apply_roi(dets, roi, w, h)
            dets = filter_confidence(dets, det_cfg.conf)
            from .flame import relabel_flame_evidence
            dets = relabel_flame_evidence(frame, dets)
            if verifier is not None and dets:
                dets, _ = verifier.verify(frame, dets)
            det_count += len(dets)
            temp = confirmer.update(processed, ts, dets, size=(h, w))
            confirmed_count += len(temp.confirmed)
            for det in temp.confirmed:
                first_confirmed.setdefault(det.label, ts)
            alarm_state = alarm.update(camera_id, ts, temp.confirmed)
            if collect_frames:
                frames_out.append({
                    "frame": frame_idx, "t": round(ts, 3),
                    "detections": [d.to_dict() for d in temp.detections],
                    "confirmed": [d.to_dict() for d in temp.confirmed],
                    "alarm": alarm_state,
                })
            processed += 1
            frame_idx += 1
    finally:
        cap.release()
    elapsed = time.perf_counter() - start
    return VideoRunResult(
        path=str(path), duration_s=duration, frames_native=frames_native,
        frames_processed=processed, elapsed_s=elapsed,
        inference_fps=processed / elapsed if elapsed else 0.0,
        detections=det_count, confirmed_detections=confirmed_count,
        first_confirmed=first_confirmed,
        alarm_events=[e.to_dict() for e in events],
        frames=frames_out,
        video_width=video_width,
        video_height=video_height,
    )


def load_clips(path: str | Path) -> list[dict]:
    clips = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            clips.append(json.loads(line))
    return clips
