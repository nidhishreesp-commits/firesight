from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config, load_config
from ..detectors.base import BaseDetector, load_detector
from ..temporal import AlarmManager, TemporalConfirmer, draw_detections
from ..verifier import CropVerifier
from ..flame_type import POLICY_FILE, FireTypePolicy, FlameTypeClassifier

log = logging.getLogger(__name__)


@dataclass
class CameraState:
    camera_id: str
    name: str
    uri: str
    enabled: bool
    roi: list[list[float]] = field(default_factory=list)
    conf: dict[str, float] = field(default_factory=dict)
    n_of_m: list[int] = field(default_factory=lambda: [3, 5])
    alarm_cfg: dict = field(default_factory=dict)
    status: str = "idle"
    fps: float = 0.0
    last_frame_jpeg: bytes | None = None
    last_detections: list[dict] = field(default_factory=list)
    alarm: dict = field(default_factory=dict)
    error: str | None = None
    frames: int = 0


class CameraWorker(threading.Thread):
    def __init__(self, state: CameraState, detector: BaseDetector, cfg: Config,
                 verifier: CropVerifier | None = None, inference_lock: threading.Lock | None = None,
                 fire_type_policy: FireTypePolicy | None = None,
                 flame_type_classifier: FlameTypeClassifier | None = None):
        super().__init__(daemon=True)
        self.state = state
        self.detector = detector
        self.cfg = cfg
        self.verifier = verifier
        self.inference_lock = inference_lock or threading.Lock()
        self.fire_type_policy = fire_type_policy
        self.flame_type_classifier = flame_type_classifier
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._source: Any = None
        self.confirmer = TemporalConfirmer(self._temporal_cfg())
        self.alarm = AlarmManager(self._alarm_cfg(), clear_after_s=self._clear_after_s())

    def _temporal_cfg(self):
        n, m = (int(v) for v in (self.state.n_of_m or [0, 0]))
        if n <= 0 or m <= 0:
            return self.cfg.temporal
        return replace(self.cfg.temporal, n_of_m={
            "fire": [n, m],
            "smoke": self.cfg.temporal.n_of_m.get("smoke", [4, 6]),
        })

    def _alarm_cfg(self):
        over = {k: float(v) for k, v in (self.state.alarm_cfg or {}).items()
                if k in ("min_duration_s", "cooldown_s") and v is not None}
        return replace(self.cfg.alarm, **over) if over else self.cfg.alarm

    def _clear_after_s(self) -> float:
        return max(3.0, self._alarm_cfg().cooldown_s / 4)

    def apply_config(self) -> None:
        """Rebuild the temporal confirmer and alarm manager from the current camera config."""
        self.confirmer = TemporalConfirmer(self._temporal_cfg())
        self.alarm = AlarmManager(self._alarm_cfg(), clear_after_s=self._clear_after_s())

    def stop(self) -> None:
        self._stop.set()

    def _open(self):
        import cv2
        uri = self.state.uri
        if uri.isdigit():
            return cv2.VideoCapture(int(uri))
        return cv2.VideoCapture(uri)

    def run(self) -> None:
        import cv2
        cap = self._open()
        if not cap.isOpened():
            self.state.status = "error"
            self.state.error = f"cannot open {self.state.uri}"
            log.error("camera %s: %s", self.state.camera_id, self.state.error)
            return
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(native_fps / max(self.cfg.stream.sample_fps, 0.1))))
        self.state.status = "running"
        frame_idx = processed = 0
        t0 = time.time()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    if self.state.uri.isdigit() or str(self.state.uri).startswith(("rtsp", "http")):
                        time.sleep(self.cfg.stream.reconnect_s)
                        cap.release()
                        cap = self._open()
                        continue
                    break
                if frame_idx % stride:
                    frame_idx += 1
                    continue
                ts = time.time() - t0
                h, w = frame.shape[:2]
                with self.inference_lock:
                    dets = self.detector.predict(frame)
                from ..video import apply_roi, filter_confidence
                from ..flame import relabel_flame_evidence
                dets = filter_confidence(apply_roi(dets, self.state.roi, w, h), self.state.conf or self.cfg.detector.conf)
                dets = relabel_flame_evidence(frame, dets)
                if self.verifier is not None and dets:
                    with self.inference_lock:
                        dets, _ = self.verifier.verify(frame, dets)
                if self.flame_type_classifier is not None and dets:
                    dets = self.flame_type_classifier.annotate(frame, dets)
                displayed = list(dets)
                if self.fire_type_policy is not None:
                    dets, suppressed = self.fire_type_policy.partition(dets)
                    displayed = dets + suppressed
                temp = self.confirmer.update(processed, ts, dets, size=(h, w))
                alarm_state = self.alarm.update(self.state.camera_id, ts, temp.confirmed)
                annotated = draw_detections(frame, temp.detections, roi=self.state.roi)
                ok_jpg, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.stream.jpeg_quality])
                with self._lock:
                    if ok_jpg:
                        self.state.last_frame_jpeg = buf.tobytes()
                    self.state.last_detections = [d.to_dict() for d in displayed]
                    self.state.alarm = alarm_state
                    self.state.frames = processed
                processed += 1
                frame_idx += 1
                if processed % 20 == 0:
                    self.state.fps = processed / max(time.time() - t0, 1e-6)
        finally:
            cap.release()
            self.state.status = "stopped"

    def snapshot(self) -> tuple[bytes | None, list[dict], dict]:
        with self._lock:
            return self.state.last_frame_jpeg, list(self.state.last_detections), dict(self.state.alarm)


class AppState:
    def __init__(self, config: Config | None = None, detector: BaseDetector | None = None,
                 cameras_path: str | Path | None = None, audit_path: str | Path | None = None):
        self.config = config or load_config()
        self.detector = detector or load_detector(
            self.config.detector.backend, self.config.path(self.config.detector.model_path),
            imgsz=self.config.detector.imgsz, iou=self.config.detector.iou,
            max_det=self.config.detector.max_det, device=self.config.detector.device,
            name=Path(self.config.detector.model_path).stem,
            license="proprietary (SmartSight)",
            family="YOLOv12n",
        )
        self.verifier: CropVerifier | None = None
        if self.config.verifier.enabled and self.config.path(self.config.verifier.model_path).exists():
            self.verifier = CropVerifier(self.config.path(self.config.verifier.model_path),
                                         classes=self.config.verifier.classes, threshold=self.config.verifier.threshold)
        self.cameras: dict[str, CameraState] = {}
        self.workers: dict[str, CameraWorker] = {}
        self.feedback_path = self.config.path("data/feedback/queue.jsonl")
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.cameras_path = Path(cameras_path) if cameras_path else self.config.path("configs/cameras.yaml")
        self.audit_path = Path(audit_path) if audit_path else self.config.path("data/audit/camera_config.jsonl")
        self.started_at = time.time()
        self.inference_lock = threading.Lock()
        self.fire_type_policy = FireTypePolicy.load(
            self.config.path(POLICY_FILE),
            default_include=self.config.alarm.fire_type_include,
            default_exclude=self.config.alarm.fire_type_exclude,
            default_warning=self.config.alarm.fire_type_warning)
        self.flame_type_classifier = FlameTypeClassifier(self.config.path("models/flame_type.onnx"))

    def load_cameras(self, path: str | Path | None = None) -> None:
        import yaml
        p = Path(path) if path else self.cameras_path
        if not p.exists():
            return
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for cam in data.get("cameras", []):
            temporal = cam.get("temporal") or {}
            state = CameraState(
                camera_id=cam["id"], name=cam.get("name", cam["id"]), uri=str(cam["uri"]),
                enabled=bool(cam.get("enabled", False)), roi=cam.get("roi", []),
                conf=cam.get("conf", {}),
                n_of_m=[int(temporal.get("n", 3)), int(temporal.get("m", 5))],
                alarm_cfg=dict(cam.get("alarm") or {}),
            )
            self.cameras[state.camera_id] = state

    def _camera_public(self, cam: CameraState) -> dict:
        return {
            "id": cam.camera_id, "name": cam.name, "uri": cam.uri, "enabled": cam.enabled,
            "roi": cam.roi, "conf": cam.conf,
            "temporal": {"n": cam.n_of_m[0], "m": cam.n_of_m[1]},
            "alarm": {"min_duration_s": cam.alarm_cfg.get("min_duration_s"),
                      "cooldown_s": cam.alarm_cfg.get("cooldown_s")},
        }

    def camera_config(self, camera_id: str) -> dict:
        cam = self.cameras.get(camera_id)
        if cam is None:
            raise KeyError(camera_id)
        return self._camera_public(cam)

    def save_cameras(self) -> Path:
        import yaml
        data = {"cameras": [self._camera_public(c) for c in self.cameras.values()]}
        self.cameras_path.parent.mkdir(parents=True, exist_ok=True)
        self.cameras_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return self.cameras_path

    def _audit(self, actor: str, camera_id: str, before: dict, after: dict) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "actor": actor or "unknown",
               "camera_id": camera_id, "before": before, "after": after,
               "changed": sorted(k for k in after if before.get(k) != after.get(k))}
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def update_camera_config(self, camera_id: str, updates: dict, actor: str) -> dict:
        cam = self.cameras.get(camera_id)
        if cam is None:
            raise KeyError(camera_id)
        before = self._camera_public(cam)

        if updates.get("roi") is not None:
            roi = updates["roi"]
            if not isinstance(roi, list) or any(len(poly) != 4 for poly in roi):
                raise ValueError("roi must be a list of [x1, y1, x2, y2] polygons")
            cam.roi = [[float(v) for v in poly] for poly in roi]
        if updates.get("conf") is not None:
            conf = updates["conf"]
            if not isinstance(conf, dict) or any(float(v) <= 0 or float(v) > 1 for v in conf.values()):
                raise ValueError("conf must map class -> (0, 1]")
            cam.conf = {str(k): float(v) for k, v in conf.items()}
        n, m = updates.get("temporal_n"), updates.get("temporal_m")
        if n is not None or m is not None:
            n = cam.n_of_m[0] if n is None else int(n)
            m = cam.n_of_m[1] if m is None else int(m)
            if n < 1 or m < 1 or n > m:
                raise ValueError("temporal thresholds require 1 <= n <= m")
            cam.n_of_m = [n, m]
        for key, field_name in (("alarm_min_duration_s", "min_duration_s"),
                                ("alarm_cooldown_s", "cooldown_s")):
            value = updates.get(key)
            if value is not None:
                if float(value) < 0:
                    raise ValueError(f"{key} must be >= 0")
                cam.alarm_cfg[field_name] = float(value)

        after = self._camera_public(cam)
        worker = self.workers.get(camera_id)
        if worker is not None and worker.is_alive():
            worker.apply_config()
        self._audit(actor, camera_id, before, after)
        self.save_cameras()
        return after

    def start_camera(self, camera_id: str) -> dict:
        state = self.cameras[camera_id]
        if camera_id in self.workers and self.workers[camera_id].is_alive():
            return {"status": "already_running"}
        worker = CameraWorker(state, self.detector, self.config, self.verifier,
                              inference_lock=self.inference_lock,
                              fire_type_policy=self.fire_type_policy,
                              flame_type_classifier=self.flame_type_classifier)
        self.workers[camera_id] = worker
        worker.start()
        return {"status": "started", "camera_id": camera_id}

    def stop_camera(self, camera_id: str) -> dict:
        worker = self.workers.pop(camera_id, None)
        if worker:
            worker.stop()
        return {"status": "stopped", "camera_id": camera_id}

    def add_feedback(self, payload: dict) -> dict:
        payload = {**payload, "ts": time.time()}
        with self.feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        return payload

    def read_feedback(self, limit: int = 200) -> list[dict]:
        if not self.feedback_path.exists():
            return []
        rows = [json.loads(line) for line in self.feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return rows[-limit:]

    def shutdown(self) -> None:
        for worker in list(self.workers.values()):
            worker.stop()
