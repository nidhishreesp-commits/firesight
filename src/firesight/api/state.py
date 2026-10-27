from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
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
        self.confirmer = TemporalConfirmer(cfg.temporal)
        self.alarm = AlarmManager(cfg.alarm, clear_after_s=max(3.0, cfg.alarm.cooldown_s / 4))
        self._source: Any = None

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
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.detector = load_detector(
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
        self.started_at = time.time()
        self.inference_lock = threading.Lock()
        self.fire_type_policy = FireTypePolicy.load(
            self.config.path(POLICY_FILE),
            default_include=self.config.alarm.fire_type_include,
            default_exclude=self.config.alarm.fire_type_exclude,
            default_warning=self.config.alarm.fire_type_warning)
        self.flame_type_classifier = FlameTypeClassifier(self.config.path("models/flame_type.onnx"))

    def load_cameras(self, path: str | Path = "configs/cameras.yaml") -> None:
        import yaml
        p = self.config.path(path)
        if not p.exists():
            return
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for cam in data.get("cameras", []):
            state = CameraState(
                camera_id=cam["id"], name=cam.get("name", cam["id"]), uri=str(cam["uri"]),
                enabled=bool(cam.get("enabled", False)), roi=cam.get("roi", []),
                conf=cam.get("conf", {}), n_of_m=[cam.get("temporal", {}).get("n", 3), cam.get("temporal", {}).get("m", 5)],
            )
            self.cameras[state.camera_id] = state

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
