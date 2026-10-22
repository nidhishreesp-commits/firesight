from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .config import TemporalConfig, AlarmConfig
from .types import Box, Detection

log = logging.getLogger(__name__)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Detection]:
        assigned: set[int] = set()
        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for track_id, state in self.tracks.items():
                if track_id in assigned or state["label"] != det.label:
                    continue
                iou = _iou(state["box"], det.box)
                if iou >= best_iou:
                    best_id, best_iou = track_id, iou
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            assigned.add(best_id)
            missing = self.tracks.get(best_id, {}).get("missing", 0)
            self.tracks[best_id] = {"box": det.box, "label": det.label, "missing": 0}
            det.track_id = best_id
            det.meta["track_source"] = "iou"
        for track_id, state in list(self.tracks.items()):
            if track_id in assigned:
                continue
            state["missing"] += 1
            if state["missing"] > self.max_missing:
                del self.tracks[track_id]
        return list(detections)


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class ByteTrackAdapter:
    def __init__(self, frame_rate: int = 5):
        try:
            import torch
            from ultralytics.engine.results import Boxes
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
            self.torch = torch
            self.Boxes = Boxes
            args = IterableSimpleNamespace(
                track_high_thresh=0.30, track_low_thresh=0.10, new_track_thresh=0.35,
                track_buffer=60, match_thresh=0.80, fuse_score=True,
            )
            self.tracker = BYTETracker(args)
            self.enabled = True
        except Exception as exc:
            log.warning("ByteTrack unavailable (%s); using IoU fallback", exc)
            self.tracker = None
            self.enabled = False

    def update(self, detections: Sequence[Detection], size: tuple[int, int] | None = None) -> list[Detection]:
        if not self.enabled or not detections:
            return list(detections)
        data = np.asarray([
            [*d.box.to_list(), d.confidence, 0 if d.label == "fire" else 1]
            for d in detections
        ], dtype=np.float32)
        boxes = self.Boxes(self.torch.from_numpy(data), orig_shape=size or (0, 0))
        try:
            tracks = self.tracker.update(boxes)
        except Exception as exc:
            log.warning("ByteTrack update failed (%s); keeping untracked detections", exc)
            return list(detections)
        for row in tracks:
            idx = int(row[7])
            if 0 <= idx < len(detections):
                detections[idx].track_id = int(row[4])
                detections[idx].meta["track_source"] = "bytetrack"
        return list(detections)


@dataclass
class TemporalResult:
    detections: list[Detection]
    confirmed: list[Detection]
    alarm_labels: set[str] = field(default_factory=set)
    diagnostics: dict = field(default_factory=dict)


class TemporalConfirmer:
    def __init__(self, cfg: TemporalConfig):
        self.cfg = cfg
        self.states: dict[int, deque] = defaultdict(lambda: deque(maxlen=max(cfg.n_of_m.get("fire", [3, 5])[1], cfg.n_of_m.get("smoke", [4, 6])[1], cfg.smoke_growth_window)))
        self.tracker = ByteTrackAdapter(frame_rate=5) if cfg.tracker.lower() == "bytetrack" else IoUTracker(cfg.track_iou)
        self._last_ts = 0.0

    def reset(self) -> None:
        self.states.clear()
        if self.cfg.tracker.lower() == "bytetrack":
            self.tracker = ByteTrackAdapter(frame_rate=5)
        else:
            self.tracker = IoUTracker(self.cfg.track_iou)

    def update(self, frame_idx: int, timestamp: float, detections: Sequence[Detection],
               size: tuple[int, int] | None = None) -> TemporalResult:
        if not self.cfg.enabled:
            return TemporalResult(list(detections), list(detections), {d.label for d in detections})
        try:
            tracked = self.tracker.update(detections, size=size)
        except TypeError:
            tracked = self.tracker.update(detections)
        confirmed: list[Detection] = []
        diagnostics: dict = {}
        for det in tracked:
            if det.track_id is None:
                continue
            m = self.cfg.n_of_m.get(det.label, [3, 5])
            n, window = int(m[0]), int(m[1])
            state = self.states[int(det.track_id)]
            state.append({
                "frame": frame_idx, "ts": timestamp, "conf": det.confidence,
                "area": det.box.area, "box": det.box.to_list(),
            })
            hits = [s for s in state if timestamp - s["ts"] <= self.cfg.max_gap_s * window]
            diagnostics[int(det.track_id)] = {"hits": len(hits), "window": window, "n": n}
            if len(hits) < n:
                continue
            reason = f"{n}-of-{window} within {self.cfg.max_gap_s * window:.1f}s"
            if det.label == "smoke" and self.cfg.smoke_growth_min_ratio > 0:
                areas = [s["area"] for s in hits[-min(len(hits), self.cfg.smoke_growth_window):]]
                if len(areas) >= 3:
                    growth = (areas[-1] - areas[0]) / max(areas[0], 1.0)
                    diagnostics[int(det.track_id)]["growth"] = round(growth, 4)
                    if growth < self.cfg.smoke_growth_min_ratio and det.confidence < 0.60:
                        continue
                    reason += f", growth={growth:.2f}"
            det.meta["confirmed"] = True
            det.meta["confirm_reason"] = reason
            confirmed.append(det)
        return TemporalResult(
            detections=tracked,
            confirmed=confirmed,
            alarm_labels={d.label for d in confirmed},
            diagnostics=diagnostics,
        )


@dataclass
class AlarmEvent:
    id: str
    camera_id: str
    labels: list[str]
    started_at: float
    peak_confidence: float
    boxes: list[list[float]]
    level: int = 1

    def to_dict(self) -> dict:
        return {
            "id": self.id, "camera_id": self.camera_id, "labels": self.labels,
            "started_at": self.started_at, "peak_confidence": round(self.peak_confidence, 4),
            "boxes": self.boxes, "level": self.level,
        }


class AlarmManager:
    def __init__(self, cfg: AlarmConfig, clear_after_s: float = 10.0, on_event: Callable[[AlarmEvent], None] | None = None):
        self.cfg = cfg
        self.clear_after_s = clear_after_s
        self.on_event = on_event
        self._active: dict[str, dict] = {}
        self._cooldowns: dict[str, float] = {}

    def update(self, camera_id: str, timestamp: float, confirmed: Sequence[Detection]) -> dict:
        state = self._active.get(camera_id)
        labels = sorted({d.label for d in confirmed})
        if confirmed:
            if state is None:
                if self._cooldown_expired(camera_id, timestamp):
                    event = AlarmEvent(
                        id=uuid.uuid4().hex[:12], camera_id=camera_id, labels=labels,
                        started_at=timestamp,
                        peak_confidence=max(d.confidence for d in confirmed),
                        boxes=[d.box.to_list() for d in confirmed],
                    )
                    self._active[camera_id] = {"event": event, "last_seen": timestamp, "level": 1}
                    self._cooldowns[camera_id] = timestamp
                    if self.on_event:
                        self.on_event(event)
                    state = self._active[camera_id]
                else:
                    return {"active": False, "labels": labels, "reason": "cooldown"}
            state["last_seen"] = timestamp
            state["event"].peak_confidence = max(state["event"].peak_confidence, max(d.confidence for d in confirmed))
            duration = timestamp - state["event"].started_at
            state["event"].level = self._level(duration)
            return {
                "active": True, "labels": labels, "duration_s": round(duration, 2),
                "level": state["event"].level, "event_id": state["event"].id,
            }
        if state is not None:
            if timestamp - state["last_seen"] > self.clear_after_s:
                event = state["event"]
                del self._active[camera_id]
                return {"active": False, "labels": sorted(event.labels), "cleared_event": event.to_dict()}
            return {"active": True, "labels": sorted(state["event"].labels),
                    "duration_s": round(timestamp - state["event"].started_at, 2), "level": state["event"].level,
                    "stale": True, "event_id": state["event"].id}
        return {"active": False, "labels": []}

    def _level(self, duration: float) -> int:
        level = 1
        for row in sorted(self.cfg.escalation, key=lambda r: r.get("after_s", 0)):
            if duration >= row.get("after_s", 0):
                level = int(row.get("level", level))
        return level

    def _cooldown_expired(self, camera_id: str, timestamp: float) -> bool:
        last = self._cooldowns.get(camera_id, -1e9)
        return (timestamp - last) >= self.cfg.cooldown_s


class EventLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: AlarmEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict()) + "\n")


def draw_detections(frame: np.ndarray, detections: Sequence[Detection], confirmed_ids: set[int] | None = None,
                    color_confirmed=(0, 0, 255), color_candidate=(0, 165, 255), roi: list[list[float]] | None = None):
    import cv2
    out = frame.copy()
    h, w = out.shape[:2]
    if roi:
        for poly in roi:
            pts = np.asarray([[int(x * w), int(y * h)] for x, y in zip(poly[0::2], poly[1::2])], dtype=np.int32)
            if pts.shape[0] >= 3:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (255, 255, 0), 2)
    for det in detections:
        is_confirmed = det.meta.get("confirmed") or (confirmed_ids and det.track_id in confirmed_ids)
        color = color_confirmed if is_confirmed else color_candidate
        x1, y1, x2, y2 = map(int, det.box.to_list())
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.confidence:.2f}"
        if det.track_id is not None:
            text += f" #{det.track_id}"
        cv2.putText(out, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out
