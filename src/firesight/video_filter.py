"""Video false-alarm layer: flicker/growth requirement, static-scene memory, dismissal suppression.

A detection that persists in the same place *without* growth or flicker (sun, clouds, lamps,
sunset glow) is down-weighted and routed to review instead of alarming. Reviewer dismissals are
remembered per camera (region + cheap colour-histogram appearance) and suppress repeats.
"""
from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .types import Detection


def _hist(frame, box, bins: int = 12) -> np.ndarray:
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return np.zeros(bins * 3, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).ravel()
    hs = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).ravel()
    hv = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).ravel()
    v = np.concatenate([hh, hs, hv]).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


@dataclass
class FilterConfig:
    window: int = 10                 # frames of track history
    min_frames: int = 6              # frames before a static verdict
    growth_min_ratio: float = 0.15   # relative area growth that counts as "not static"
    flicker_min_std: float = 0.04    # confidence std that counts as flicker
    static_conf_penalty: float = 0.5 # multiply confidence for static detections
    static_route_review: bool = True # static detections go to review, not alarm
    suppression_iou: float = 0.6
    suppression_cos: float = 0.90
    suppression_ttl_s: float = 3600.0
    min_dismissals: int = 2          # dismissals before a region is suppressed


@dataclass
class _Track:
    hist: deque = field(default_factory=lambda: deque(maxlen=64))
    areas: deque = field(default_factory=lambda: deque(maxlen=64))
    confs: deque = field(default_factory=lambda: deque(maxlen=64))
    first_ts: float = 0.0
    last_ts: float = 0.0


class VideoFalseAlarmFilter:
    def __init__(self, cfg: FilterConfig | None = None, log_path: str | Path | None = None):
        self.cfg = cfg or FilterConfig()
        self.tracks: dict[tuple[str, int], _Track] = {}
        self.dismissed: dict[str, list[dict]] = defaultdict(list)
        self.log_path = Path(log_path) if log_path else None
        self.events: list[dict] = []

    # ---- reviewer dismissals -------------------------------------------------
    def record_dismissal(self, camera_id: str, frame, box, ts: float | None = None) -> None:
        row = {"camera_id": camera_id, "ts": ts or time.time(), "box": list(box),
               "hist": _hist(frame, box).tolist()}
        self.dismissed[camera_id].append(row)
        self._log({"event": "dismissal", "camera_id": camera_id, "box": list(box)})

    def _suppressed_by_memory(self, camera_id: str, frame, det: Detection, ts: float) -> bool:
        hist = _hist(frame, det.box.to_list())
        for row in self.dismissed.get(camera_id, []):
            if ts - row["ts"] > self.cfg.suppression_ttl_s:
                continue
            if _iou(det.box.to_list(), row["box"]) >= self.cfg.suppression_iou and \
               _cos(hist, np.asarray(row["hist"], dtype=np.float32)) >= self.cfg.suppression_cos:
                return True
        return False

    # ---- static / flicker ----------------------------------------------------
    def update(self, camera_id: str, ts: float, frame, detections: Sequence[Detection]):
        kept, review, suppressed = [], [], []
        for det in detections:
            if det.track_id is None:
                kept.append(det)
                continue
            key = (camera_id, int(det.track_id))
            tr = self.tracks.get(key)
            if tr is None:
                tr = _Track(first_ts=ts)
                self.tracks[key] = tr
            tr.areas.append(det.box.area)
            tr.confs.append(det.confidence)
            tr.hist.append(_hist(frame, det.box.to_list()))
            tr.last_ts = ts

            if self._suppressed_by_memory(camera_id, frame, det, ts):
                det.meta["video_suppressed"] = "reviewer_dismissal"
                suppressed.append(det)
                continue

            n = len(tr.areas)
            static = False
            if n >= self.cfg.min_frames:
                growth = (tr.areas[-1] - tr.areas[0]) / max(tr.areas[0], 1.0)
                flicker = float(np.std(np.asarray(tr.confs, dtype=np.float32)))
                static = abs(growth) < self.cfg.growth_min_ratio and flicker < self.cfg.flicker_min_std
                det.meta["video_growth"] = round(growth, 4)
                det.meta["video_flicker"] = round(flicker, 4)
            if static:
                det.confidence = det.confidence * self.cfg.static_conf_penalty
                det.meta["static_scene"] = True
                if self.cfg.static_route_review:
                    det.meta["review_queue"] = True
                    review.append(det)
                    self._log({"event": "static_review", "camera_id": camera_id,
                               "track": det.track_id, "conf": det.confidence})
                    continue
            kept.append(det)
        return kept, review, suppressed

    def _log(self, row: dict) -> None:
        row["ts"] = row.get("ts", time.time())
        self.events.append(row)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0
