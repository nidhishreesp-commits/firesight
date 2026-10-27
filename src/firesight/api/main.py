from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import PROJECT_ROOT, load_config
from ..data.manifest import load_manifest, manifest_stats
from ..eval.metrics import _greedy_match
from ..eval.runner import load_predictions
from ..flame_type import POLICY_FILE
from ..logging_utils import setup_logging
from ..monitoring import ACTIVE_ALARMS, ALARMS_TOTAL, DETECTIONS_TOTAL, FEEDBACK_TOTAL, INFERENCE_LATENCY, metrics_payload
from ..temporal import draw_detections
from ..types import Box, Detection
from .state import AppState

log = logging.getLogger("firesight.api")

MANIFESTS = {
    "test_bench": "data/splits/test_bench.jsonl",
    "test_mini": "data/splits/test_mini.jsonl",
    "val_bench": "data/splits/val_bench.jsonl",
    "external_cctv_nc": "data/splits/external_cctv_nc.jsonl",
    "testbench100": "testbench100/manifest.jsonl",
}
PREDICTION_CACHES = {
    "test_bench": "artifacts/benchmark_stage1/predictions/ssprod_best_fire_v3_test.jsonl",
    "test_mini": "artifacts/benchmark_mini/predictions/ssprod_best_fire_v3_test.jsonl",
    "external_cctv_nc": "artifacts/external_cctv_nc/predictions/ssprod_best_fire_v3.jsonl",
    "testbench100": "bench_results/predictions/v3_bench.jsonl",
}


class FeedbackIn(BaseModel):
    record_id: str
    verdict: str
    note: str | None = None
    model: str | None = None
    label: str | None = None


class FireTypePolicyIn(BaseModel):
    include: list[str]
    exclude: list[str]
    warning: list[str] | None = None


class CameraConfigIn(BaseModel):
    actor: str
    roi: list[list[float]] | None = None
    conf: dict[str, float] | None = None
    temporal_n: int | None = None
    temporal_m: int | None = None
    alarm_min_duration_s: float | None = None
    alarm_cooldown_s: float | None = None


class BenchRReviewIn(BaseModel):
    id: str
    has_fire: int
    has_smoke: int
    flame_type: str | None = None
    category: str
    difficulty: str
    ambiguous: int = 0
    notes: str = ""
    reviewer: str
    boxes_fire: list[list[float]] | None = None
    boxes_smoke: list[list[float]] | None = None
    width: int | None = None
    height: int | None = None
    accepted_as_proposed: bool = False
    edits_made: int = 0
    time_spent_s: float | None = None
    blind: bool = False


class BenchRSpotIn(BaseModel):
    id: str
    has_fire: int
    has_smoke: int
    flame_type: str | None = None
    reviewer: str
    notes: str = ""


class BenchReviewIn(BaseModel):
    id: str
    has_fire: int
    has_smoke: int
    category: str
    difficulty: str
    ambiguous: int = 0
    notes: str = ""
    reviewer: str
    boxes_fire: list[list[float]] | None = None
    boxes_smoke: list[list[float]] | None = None
    width: int | None = None
    height: int | None = None


BENCH_DIR = PROJECT_ROOT / "testbench100"
BENCH_INDEX = BENCH_DIR / "index.json"
BENCH_REVIEWS = BENCH_DIR / "reviews.json"
BENCH_LOCK = BENCH_DIR / "LOCKED.json"

BENCHR_DIR = PROJECT_ROOT / "bench_R"
BENCHR_INDEX = BENCHR_DIR / "index.json"
BENCHR_REVIEWS = BENCHR_DIR / "reviews.json"
BENCHR_SPOT = BENCHR_DIR / "spot_checks.json"
BENCHR_LOCK = BENCHR_DIR / "LOCKED.json"


def _benchR_rows() -> list[dict]:
    if not BENCHR_INDEX.exists():
        return []
    rows = json.loads(BENCHR_INDEX.read_text(encoding="utf-8"))
    reviews = json.loads(BENCHR_REVIEWS.read_text(encoding="utf-8")) if BENCHR_REVIEWS.exists() else {}
    for row in rows:
        review = reviews.get(row["id"])
        if review:
            row.update({k: review[k] for k in ("has_fire", "has_smoke", "flame_type", "category",
                                               "difficulty", "ambiguous", "notes", "reviewer", "reviewed_at",
                                               "review_status", "boxes_fire", "boxes_smoke", "width", "height")
                        if k in review})
    return rows


def _benchR_comp(rows: list[dict]) -> dict:
    fire_only = [r for r in rows if r["has_fire"] and not r["has_smoke"]]
    smoke_only = [r for r in rows if r["has_smoke"] and not r["has_fire"]]
    both = [r for r in rows if r["has_fire"] and r["has_smoke"]]
    negatives = [r for r in rows if not r["has_fire"] and not r["has_smoke"]]
    types = Counter(r.get("flame_type") or "unknown" for r in fire_only)
    families = Counter(r.get("family", "other") for r in negatives)
    return {"fire_only": len(fire_only), "smoke_only": len(smoke_only), "both": len(both),
            "negative": len(negatives), "fire_only_types": dict(types), "negative_families": dict(families),
            "reviewed": sum(1 for r in rows if r.get("review_status") == "reviewed")}


def _write_benchR_labels(rows: list[dict], reviewer: str) -> dict:
    import csv
    import hashlib
    from datetime import datetime, timezone
    reviewed_at = datetime.now(timezone.utc).isoformat()
    labels_csv = BENCHR_DIR / "labels.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "file", "has_fire", "has_smoke", "flame_type", "category", "difficulty",
                         "source", "license", "boxes_fire", "boxes_smoke", "ambiguous", "reviewer",
                         "reviewed_at", "review_status", "notes"])
        for row in rows:
            writer.writerow([row["id"], row["file"], row["has_fire"], row["has_smoke"], row.get("flame_type") or "",
                             row.get("category", row.get("family")), row.get("difficulty", "medium"), row["source"],
                             row["license"], json.dumps(row.get("boxes_fire", [])), json.dumps(row.get("boxes_smoke", [])),
                             row.get("ambiguous", 0), row.get("reviewer", ""), row.get("reviewed_at", ""),
                             "reviewed", row.get("notes", "")])
    yolo_dir = BENCHR_DIR / "labels"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for row in rows:
        w, h = row.get("width"), row.get("height")
        if not w or not h:
            try:
                with Image.open(PROJECT_ROOT / row["file"]) as im:
                    w, h = im.size
            except Exception:
                w, h = 1, 1
        lines = []
        if row["has_fire"]:
            for box in row.get("boxes_fire", []):
                x1, y1, x2, y2 = box
                lines.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        if row["has_smoke"]:
            for box in row.get("boxes_smoke", []):
                x1, y1, x2, y2 = box
                lines.append(f"1 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        (yolo_dir / (Path(row["file"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    image_hashes = sorted(f"{row['sha256']}  images/{row['file']}" for row in rows)
    manifest_hash = hashlib.sha256("\n".join(image_hashes).encode("utf-8")).hexdigest()
    (BENCHR_DIR / "MANIFEST.sha256").write_text("\n".join(image_hashes) + "\n", encoding="utf-8")
    lock = {"locked": True, "reviewer": reviewer, "locked_at": reviewed_at, "manifest_sha256": manifest_hash,
            "images": len(rows), "composition": _benchR_comp(rows),
            "labels_csv_sha256": hashlib.sha256(labels_csv.read_bytes()).hexdigest()}
    BENCHR_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    # Stratified DEV/SEALED split, 300/700, after lock.
    import random
    rng = random.Random(20261022)
    groups = defaultdict(list)
    for row in rows:
        if row["has_fire"] and not row["has_smoke"]:
            key = f"fire:{row.get('flame_type') or 'unknown'}"
        elif row["has_smoke"] and not row["has_fire"]:
            key = "smoke_only"
        elif row["has_fire"] and row["has_smoke"]:
            key = "both"
        else:
            key = f"negative:{row.get('family', 'other')}"
        groups[key].append(row)
    dev, sealed = [], []
    for key, members in groups.items():
        rng.shuffle(members)
        n_dev = max(1, round(300 * len(members) / len(rows)))
        dev += members[:n_dev]
        sealed += members[n_dev:]
    # correct size exactly 300/700
    rng.shuffle(dev)
    while len(dev) > 300:
        sealed.append(dev.pop())
    while len(dev) < 300:
        sealed.sort(key=lambda r: r["id"])
        dev.append(sealed.pop(0))
    (BENCHR_DIR / "DEV.jsonl").write_text("\n".join(json.dumps(r) for r in dev) + "\n", encoding="utf-8")
    (BENCHR_DIR / "SEALED.jsonl").write_text("\n".join(json.dumps(r) for r in sealed) + "\n", encoding="utf-8")
    (BENCHR_DIR / "SPLIT.lock.json").write_text(json.dumps({"dev": len(dev), "sealed": len(sealed),
                                                             "seed": 20261022, "rule": "stratified random after lock"}, indent=2), encoding="utf-8")
    return lock

CATEGORY_FAMILY = {
    "sunset_sunrise": "light_colour", "street_head_lights": "light_colour",
    "lamps_neon_brake": "light_colour", "chandelier_festival": "light_colour",
    "reflection_glare": "light_colour", "orange_clothing_ppe": "light_colour",
    "cones_machinery": "light_colour", "flowers_leaves": "light_colour",
    "fog_mist_haze": "smoke_like", "clouds": "smoke_like", "steam_cooking": "smoke_like",
    "exhaust_chimney": "smoke_like", "dust_snow": "smoke_like", "overexposed_white": "smoke_like",
    "candle": "fire_adjacent", "match_lighter": "fire_adjacent", "gas_stove": "fire_adjacent",
    "campfire": "fire_adjacent", "welding": "fire_adjacent", "fireworks": "fire_adjacent",
    "fireplace": "fire_adjacent", "screen_fire": "fire_adjacent",
    "blur_noise_compression": "camera_artifact", "rain_on_lens": "camera_artifact",
    "infrared_night": "camera_artifact",
}


def _bench_rows() -> list[dict]:
    if not BENCH_INDEX.exists():
        return []
    rows = json.loads(BENCH_INDEX.read_text(encoding="utf-8"))
    reviews = json.loads(BENCH_REVIEWS.read_text(encoding="utf-8")) if BENCH_REVIEWS.exists() else {}
    for row in rows:
        review = reviews.get(row["id"])
        if review:
            row.update({k: review[k] for k in ("has_fire", "has_smoke", "category", "difficulty",
                                               "ambiguous", "notes", "reviewer", "reviewed_at", "review_status",
                                               "boxes_fire", "boxes_smoke", "width", "height") if k in review})
    return rows


def _bench_composition(rows: list[dict]) -> dict:
    fire = sum(1 for r in rows if r["has_fire"] and not r["has_smoke"])
    smoke = sum(1 for r in rows if r["has_smoke"] and not r["has_fire"])
    both = sum(1 for r in rows if r["has_fire"] and r["has_smoke"])
    neg = sum(1 for r in rows if not r["has_fire"] and not r["has_smoke"])
    fam = {}
    for r in rows:
        if not r["has_fire"] and not r["has_smoke"]:
            family = r.get("family") or CATEGORY_FAMILY.get(r.get("category", ""), "other")
            fam[family] = fam.get(family, 0) + 1
    return {"fire_only": fire, "smoke_only": smoke, "both": both, "negative": neg,
            "negative_families": fam, "reviewed": sum(1 for r in rows if r["review_status"] == "reviewed")}


def _write_bench_labels(rows: list[dict], reviewer: str, composition_override: bool = False,
                        expected_composition: dict | None = None) -> dict:
    import csv
    import hashlib
    from datetime import datetime, timezone
    reviewed_at = datetime.now(timezone.utc).isoformat()
    labels_csv = BENCH_DIR / "labels.csv"
    with labels_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "file", "has_fire", "has_smoke", "category", "difficulty", "source", "license",
                         "boxes_fire", "boxes_smoke", "ambiguous", "reviewer", "reviewed_at", "review_status", "notes"])
        for row in rows:
            writer.writerow([row["id"], row["file"], row["has_fire"], row["has_smoke"], row["category"], row["difficulty"],
                             row["source"], row["license"], json.dumps(row.get("boxes_fire", [])),
                             json.dumps(row.get("boxes_smoke", [])), row.get("ambiguous", 0), row.get("reviewer", ""),
                             row.get("reviewed_at", ""), "reviewed", row.get("notes", "")])
    yolo_dir = BENCH_DIR / "labels"
    from PIL import Image
    for row in rows:
        lines = []
        w, h = row.get("width"), row.get("height")
        if (not w or not h):
            try:
                with Image.open(BENCH_DIR / "images" / row["file"]) as im:
                    w, h = im.size
            except Exception:
                w, h = 1, 1
        if row["has_fire"]:
            for box in row.get("boxes_fire", []):
                x1, y1, x2, y2 = box
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        if row["has_smoke"]:
            for box in row.get("boxes_smoke", []):
                x1, y1, x2, y2 = box
                lines.append(f"1 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (yolo_dir / (Path(row["file"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    image_hashes = sorted(f"{row['sha256']}  images/{row['file']}" for row in rows)
    manifest_hash = hashlib.sha256("\n".join(image_hashes).encode("utf-8")).hexdigest()
    (BENCH_DIR / "MANIFEST.sha256").write_text("\n".join(image_hashes) + "\n", encoding="utf-8")
    lock = {"locked": True, "reviewer": reviewer, "locked_at": reviewed_at, "manifest_sha256": manifest_hash,
            "images": len(rows), "composition": _bench_composition(rows),
            "composition_override": composition_override,
            "expected_composition": expected_composition,
            "labels_csv_sha256": hashlib.sha256(labels_csv.read_bytes()).hexdigest()}
    BENCH_LOCK.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    return lock


def _record_status(rec, dets: list[Detection], threshold: float) -> dict:
    dets = [d for d in dets if d.confidence >= threshold]
    matched_pred, matched_gt = _greedy_match(dets, rec.labels, 0.5)
    fp = [d for d, m in zip(dets, matched_pred) if not m]
    fn = [g for g, m in zip(rec.labels, matched_gt) if not m]
    if not rec.labels:
        status = "false_positive" if dets else "true_negative"
    elif fp or fn:
        status = "false_positive" if not any(matched_pred) else "partial"
    else:
        status = "true_positive"
    return {
        "id": rec.id, "path": rec.path, "source": rec.source, "scene": rec.scene,
        "split": rec.split, "category": rec.category, "width": rec.width, "height": rec.height,
        "ground_truth": [g.to_dict() for g in rec.labels],
        "detections": [d.to_dict() for d in dets],
        "status": status, "false_positives": len(fp), "false_negatives": len(fn),
    }


def create_app(config_path: str | None = None) -> FastAPI:
    cfg = load_config(config_path)
    setup_logging(cfg.app.log_level)
    state = AppState(cfg)
    state.load_cameras()

    app = FastAPI(title="FireSight API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.firesight = state

    @app.on_event("shutdown")
    def _shutdown() -> None:
        state.shutdown()

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - state.started_at, 2),
            "model": state.config.detector.model_path,
            "verifier": state.config.verifier.model_path if state.verifier else None,
            "cameras": {cid: cam.status for cid, cam in state.cameras.items()},
        }

    @app.get("/metrics")
    def metrics() -> Response:
        payload, content_type = metrics_payload()
        return Response(content=payload, media_type=content_type)

    @app.get("/api/policy/fire-types")
    def get_fire_type_policy() -> dict:
        return state.fire_type_policy.to_dict()

    @app.post("/api/policy/fire-types")
    def set_fire_type_policy(payload: FireTypePolicyIn) -> dict:
        from ..flame_type import canonical_flame_type
        state.fire_type_policy.include = {canonical_flame_type(t) for t in payload.include}
        state.fire_type_policy.exclude = {canonical_flame_type(t) for t in payload.exclude}
        if payload.warning is not None:
            state.fire_type_policy.warning = {canonical_flame_type(t) for t in payload.warning}
        for worker in state.workers.values():
            worker.fire_type_policy = state.fire_type_policy
        state.fire_type_policy.save(state.config.path(POLICY_FILE))
        return state.fire_type_policy.to_dict()

    @app.get("/api/config")
    def get_config() -> dict:
        return {
            "detector": cfg.detector.__dict__,
            "temporal": cfg.temporal.__dict__,
            "verifier": cfg.verifier.__dict__,
            "alarm": cfg.alarm.__dict__,
        }

    @app.get("/api/models")
    def models() -> dict:
        import yaml
        registry_path = cfg.path("models/registry.yaml")
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {"models": []}
        return registry

    @app.post("/api/detect/image")
    async def detect_image(file: UploadFile = File(...), threshold: float = Form(0.25), annotate: bool = Form(True)) -> dict:
        raw = await file.read()
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "invalid image")
        t0 = time.perf_counter()
        with state.inference_lock:
            dets = state.detector.predict(frame)
        dets = [d for d in dets if d.confidence >= threshold]
        from ..flame import relabel_flame_evidence
        dets = relabel_flame_evidence(frame, dets)
        if state.verifier is not None and dets:
            with state.inference_lock:
                dets, _ = state.verifier.verify(frame, dets)
        dets = state.flame_type_classifier.annotate(frame, dets)
        kept, suppressed = state.fire_type_policy.partition(dets)
        displayed = kept + suppressed
        latency = (time.perf_counter() - t0) * 1000
        INFERENCE_LATENCY.labels(model=Path(cfg.detector.model_path).stem).observe(latency)
        for d in displayed:
            DETECTIONS_TOTAL.labels(camera="upload", label=d.label).inc()
        response: dict[str, Any] = {
            "detections": [d.to_dict() for d in displayed],
            "count": len(displayed),
            "latency_ms": round(latency, 2),
            "alarm": bool(kept),
            "fire_type_policy": state.fire_type_policy.to_dict(),
        }
        if annotate:
            annotated = draw_detections(frame, dets)
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                response["annotated_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return response

    @app.post("/api/detect/video")
    async def detect_video(file: UploadFile = File(...), max_seconds: float = Form(120.0), sample_fps: float = Form(5.0)) -> dict:
        from ..video import run_video
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "empty upload")
        if len(raw) > cfg.server.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"file exceeds {cfg.server.max_upload_mb} MB")
        suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        log.info("video upload name=%s bytes=%d max_seconds=%s sample_fps=%s", file.filename, len(raw), max_seconds, sample_fps)
        try:
            result = run_video(state.detector, tmp_path, cfg.detector, cfg.temporal, cfg.alarm,
                               sample_fps=sample_fps, max_seconds=max_seconds, camera_id="upload",
                               collect_frames=True, verifier=state.verifier,
                               fire_type_policy=state.fire_type_policy,
                               flame_type_classifier=state.flame_type_classifier)
            payload = result.to_dict()
            payload["frames"] = result.frames[:2000]
            payload["upload_name"] = file.filename
            payload["analysed_seconds"] = min(result.duration_s, max_seconds)
            for event in result.alarm_events:
                ALARMS_TOTAL.labels(camera="upload", level=str(event.get("level", 1))).inc()
            log.info("video done name=%s frames=%d detections=%d alarms=%d elapsed=%.1fs",
                     file.filename, result.frames_processed, result.detections, len(result.alarm_events), result.elapsed_s)
            return payload
        except Exception as exc:
            log.exception("video inference failed")
            raise HTTPException(422, f"could not process video ({type(exc).__name__}): {exc}")
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/api/bench100")
    def bench100() -> dict:
        rows = _bench_rows()
        locked = json.loads(BENCH_LOCK.read_text(encoding="utf-8")) if BENCH_LOCK.exists() else {"locked": False}
        return {"locked": locked.get("locked", False), "lock": locked, "composition": _bench_composition(rows),
                "images": rows}

    @app.get("/api/bench100/image/{image_id}")
    def bench100_image(image_id: str) -> FileResponse:
        for row in _bench_rows():
            if row["id"] == image_id:
                p = BENCH_DIR / "images" / row["file"]
                if p.exists():
                    return FileResponse(p)
        raise HTTPException(404, "bench image not found")

    @app.post("/api/bench100/review")
    def bench100_review(payload: BenchReviewIn) -> dict:
        if BENCH_LOCK.exists():
            raise HTTPException(409, "bench is locked")
        rows = _bench_rows()
        if not any(r["id"] == payload.id for r in rows):
            raise HTTPException(404, "unknown bench image")
        if not payload.reviewer.strip():
            raise HTTPException(422, "reviewer name is required")
        if payload.difficulty not in {"easy", "medium", "hard"}:
            raise HTTPException(422, "difficulty must be easy/medium/hard")
        reviews = json.loads(BENCH_REVIEWS.read_text(encoding="utf-8")) if BENCH_REVIEWS.exists() else {}
        existing = next(r for r in rows if r["id"] == payload.id)
        from datetime import datetime, timezone
        reviews[payload.id] = {
            "has_fire": int(payload.has_fire), "has_smoke": int(payload.has_smoke),
            "category": payload.category, "difficulty": payload.difficulty,
            "ambiguous": int(payload.ambiguous), "notes": payload.notes,
            "reviewer": payload.reviewer.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_status": "reviewed",
            "boxes_fire": payload.boxes_fire if payload.boxes_fire is not None else existing.get("boxes_fire", []),
            "boxes_smoke": payload.boxes_smoke if payload.boxes_smoke is not None else existing.get("boxes_smoke", []),
            "width": payload.width or existing.get("width"),
            "height": payload.height or existing.get("height"),
        }
        BENCH_REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
        return {"saved": payload.id, "reviewed": len(reviews), "total": len(rows)}

    @app.get("/api/bench100/export")
    def bench100_export() -> FileResponse:
        p = BENCH_DIR / "labels.csv"
        if not p.exists():
            raise HTTPException(404, "bench not locked yet")
        return FileResponse(p, filename="testbench100_labels.csv")

    @app.post("/api/bench100/lock")
    def bench100_lock(reviewer: str = Query(...), allow_composition_override: bool = Query(False)) -> dict:
        if BENCH_LOCK.exists():
            raise HTTPException(409, "bench already locked")
        rows = _bench_rows()
        if len(rows) != 100:
            raise HTTPException(422, f"expected 100 images, found {len(rows)}")
        unreviewed = [r["id"] for r in rows if r.get("review_status") != "reviewed"]
        if unreviewed:
            raise HTTPException(422, f"{len(unreviewed)} images still pending review")
        missing_boxes = [
            r["id"] for r in rows
            if not r.get("ambiguous") and (
                (r["has_fire"] and not r.get("boxes_fire")) or (r["has_smoke"] and not r.get("boxes_smoke"))
            )
        ]
        if missing_boxes:
            raise HTTPException(422, {
                "message": "Some positives have no boxes. Draw boxes in Bench review, or mark them ambiguous.",
                "missing_boxes": missing_boxes[:20],
                "missing_boxes_count": len(missing_boxes),
            })
        comp = _bench_composition(rows)
        expected = {"fire_only": 25, "smoke_only": 15, "both": 10, "negative": 50}
        mismatch = {k: {"actual": comp[k], "expected": v} for k, v in expected.items() if comp[k] != v}
        family_short = {f: comp["negative_families"].get(f, 0) for f in
                        ("light_colour", "smoke_like", "fire_adjacent", "camera_artifact")
                        if comp["negative_families"].get(f, 0) < 3}
        if (mismatch or family_short) and not allow_composition_override:
            raise HTTPException(422, {
                "message": "Reviewed labels no longer match the frozen composition. Lock as reviewed to record an explicit override, or edit the labels back.",
                "composition": comp,
                "mismatch": mismatch,
                "family_short": family_short,
                "can_override": True,
            })
        return _write_bench_labels(rows, reviewer, composition_override=bool(mismatch or family_short),
                                   expected_composition=expected)

    @app.get("/api/benchR")
    def benchR() -> dict:
        rows = _benchR_rows()
        locked = json.loads(BENCHR_LOCK.read_text(encoding="utf-8")) if BENCHR_LOCK.exists() else {"locked": False}
        spot = json.loads(BENCHR_SPOT.read_text(encoding="utf-8")) if BENCHR_SPOT.exists() else {}
        agree = sum(1 for v in spot.values() if v.get("agrees"))
        return {"locked": locked.get("locked", False), "lock": locked, "composition": _benchR_comp(rows),
                "spot_check": {"reviewed": len(spot), "required": 100,
                               "agreement_rate": round(agree / len(spot), 4) if spot else None},
                "images": rows}

    @app.get("/api/benchR/image/{image_id}")
    def benchR_image(image_id: str) -> FileResponse:
        for row in _benchR_rows():
            if row["id"] == image_id:
                p = PROJECT_ROOT / row["file"]
                if p.exists():
                    return FileResponse(p)
        raise HTTPException(404, "bench_R image not found")

    @app.post("/api/benchR/review")
    def benchR_review(payload: BenchRReviewIn) -> dict:
        if BENCHR_LOCK.exists():
            raise HTTPException(409, "bench_R is locked")
        rows = _benchR_rows()
        existing = next((r for r in rows if r["id"] == payload.id), None)
        if existing is None:
            raise HTTPException(404, "unknown bench_R image")
        if not payload.reviewer.strip():
            raise HTTPException(422, "reviewer name is required")
        reviews = json.loads(BENCHR_REVIEWS.read_text(encoding="utf-8")) if BENCHR_REVIEWS.exists() else {}
        from datetime import datetime, timezone
        reviews[payload.id] = {
            "has_fire": int(payload.has_fire), "has_smoke": int(payload.has_smoke),
            "flame_type": payload.flame_type, "category": payload.category, "difficulty": payload.difficulty,
            "ambiguous": int(payload.ambiguous), "notes": payload.notes,
            "reviewer": payload.reviewer.strip(), "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "review_status": "reviewed",
            "boxes_fire": payload.boxes_fire if payload.boxes_fire is not None else existing.get("boxes_fire", []),
            "boxes_smoke": payload.boxes_smoke if payload.boxes_smoke is not None else existing.get("boxes_smoke", []),
            "width": payload.width or existing.get("width"), "height": payload.height or existing.get("height"),
            "accepted_as_proposed": bool(payload.accepted_as_proposed),
            "edits_made": int(payload.edits_made), "time_spent_s": payload.time_spent_s,
            "blind": bool(payload.blind),
        }
        BENCHR_REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
        return {"saved": payload.id, "reviewed": len(reviews), "total": len(rows)}

    @app.get("/api/benchR/proposals")
    def benchR_proposals() -> dict:
        p = BENCHR_DIR / "proposals.json"
        if not p.exists():
            raise HTTPException(404, "no proposals yet")
        data = json.loads(p.read_text(encoding="utf-8"))
        blind = set(data.get("meta", {}).get("blind_ids", []))
        for image_id, row in data.get("images", {}).items():
            if image_id in blind:
                row["blind"] = True
                row["proposals"] = []
                row["suggested_has_fire"] = 0
                row["suggested_has_smoke"] = 0
                row["suggested_flame_type"] = None
        return data

    @app.get("/api/benchR/spotcheck")
    def benchR_spotcheck() -> dict:
        rows = _benchR_rows()
        if any(r.get("review_status") != "reviewed" for r in rows):
            raise HTTPException(422, "review all 1000 images before spot-checking")
        import random
        rng = random.Random(20261022)
        prop_path = BENCHR_DIR / "proposals.json"
        blind = set()
        if prop_path.exists():
            blind = set(json.loads(prop_path.read_text(encoding="utf-8")).get("meta", {}).get("blind_ids", []))
        pool = [r for r in rows if r["id"] not in blind]
        sample = rng.sample(pool, 100)
        spot = json.loads(BENCHR_SPOT.read_text(encoding="utf-8")) if BENCHR_SPOT.exists() else {}
        return {"required": 100, "reviewed": len(spot), "items": [{**r, "spot_done": r["id"] in spot} for r in sample]}

    @app.post("/api/benchR/spotcheck")
    def benchR_spotcheck_save(payload: BenchRSpotIn) -> dict:
        if BENCHR_LOCK.exists():
            raise HTTPException(409, "bench_R is locked")
        if not payload.reviewer.strip():
            raise HTTPException(422, "reviewer name is required")
        reviews = json.loads(BENCHR_REVIEWS.read_text(encoding="utf-8")) if BENCHR_REVIEWS.exists() else {}
        first = reviews.get(payload.id)
        if not first:
            raise HTTPException(422, "first-pass review missing")
        from datetime import datetime, timezone
        spot = json.loads(BENCHR_SPOT.read_text(encoding="utf-8")) if BENCHR_SPOT.exists() else {}
        agrees = (int(payload.has_fire) == int(first["has_fire"]) and int(payload.has_smoke) == int(first["has_smoke"])
                  and (payload.flame_type or "unknown") == (first.get("flame_type") or "unknown"))
        spot[payload.id] = {"has_fire": int(payload.has_fire), "has_smoke": int(payload.has_smoke),
                            "flame_type": payload.flame_type, "reviewer": payload.reviewer.strip(),
                            "reviewed_at": datetime.now(timezone.utc).isoformat(),
                            "first_pass": {k: first.get(k) for k in ("has_fire", "has_smoke", "flame_type")},
                            "agrees": agrees, "notes": payload.notes}
        BENCHR_SPOT.write_text(json.dumps(spot, indent=2), encoding="utf-8")
        agree = sum(1 for v in spot.values() if v.get("agrees"))
        return {"saved": payload.id, "spot_reviewed": len(spot), "required": 100,
                "agreement_rate": round(agree / len(spot), 4)}

    @app.get("/api/benchR/export")
    def benchR_export() -> FileResponse:
        p = BENCHR_DIR / "labels.csv"
        if not p.exists():
            raise HTTPException(404, "bench_R not locked yet")
        return FileResponse(p, filename="bench_R_labels.csv")

    @app.post("/api/benchR/lock")
    def benchR_lock(reviewer: str = Query(...)) -> dict:
        if BENCHR_LOCK.exists():
            raise HTTPException(409, "bench_R already locked")
        rows = _benchR_rows()
        if len(rows) != 1000:
            raise HTTPException(422, f"expected 1000 images, found {len(rows)}")
        unreviewed = [r["id"] for r in rows if r.get("review_status") != "reviewed"]
        if unreviewed:
            raise HTTPException(422, f"{len(unreviewed)} images still pending review")
        spot = json.loads(BENCHR_SPOT.read_text(encoding="utf-8")) if BENCHR_SPOT.exists() else {}
        if len(spot) < 100:
            raise HTTPException(422, f"second-pass spot check incomplete ({len(spot)}/100)")
        comp = _benchR_comp(rows)
        expected = {"fire_only": 96, "smoke_only": 106, "both": 82, "negative": 716}
        if any(comp[k] != v for k, v in expected.items()):
            raise HTTPException(422, f"composition mismatch: {comp}; expected {expected}")
        types = comp["fire_only_types"]
        if types.get("uncontrolled", 0) != 12 or types.get("controlled", 0) != 52 or \
           types.get("spark", 0) + types.get("spark_benign", 0) + types.get("screen", 0) != 31 or \
           types.get("unknown", 0) != 1:
            raise HTTPException(422, f"fire_only flame_type composition mismatch: {types}; expected 12 uncontrolled, 52 controlled, 31 spark/screen, 1 unknown")
        short = {k: v for k, v in comp["negative_families"].items() if v < 3}
        if short or sum(comp["negative_families"].values()) != 716:
            raise HTTPException(422, f"negative family coverage invalid: {comp['negative_families']}")
        return _write_benchR_labels(rows, reviewer)

    @app.get("/api/testsets")
    def testsets() -> dict:
        out = {}
        for name, rel in MANIFESTS.items():
            p = PROJECT_ROOT / rel
            if not p.exists():
                continue
            records = load_manifest(p, root=PROJECT_ROOT)
            out[name] = {"manifest": rel, "stats": manifest_stats(records)}
        return out

    @app.get("/api/testsets/{name}/images")
    def testset_images(name: str, filter: str = "all", limit: int = 60, offset: int = 0, threshold: float = 0.5) -> dict:
        if name not in MANIFESTS:
            raise HTTPException(404, "unknown testset")
        records = load_manifest(PROJECT_ROOT / MANIFESTS[name], root=PROJECT_ROOT)
        predictions: dict = {}
        cache_rel = PREDICTION_CACHES.get(name)
        if cache_rel and (PROJECT_ROOT / cache_rel).exists():
            predictions, _ = load_predictions(PROJECT_ROOT / cache_rel)
        rows = []
        for rec in records:
            row = _record_status(rec, predictions.get(rec.id, []), threshold)
            if filter != "all" and row["status"] != filter:
                continue
            rows.append(row)
        return {"total": len(rows), "offset": offset, "limit": limit, "items": rows[offset:offset + limit],
                "predictions_available": bool(predictions)}

    @app.get("/api/image/{record_id}")
    def image(record_id: str, manifest: str = "test_bench") -> FileResponse:
        records = load_manifest(PROJECT_ROOT / MANIFESTS[manifest], root=PROJECT_ROOT)
        for rec in records:
            if rec.id == record_id:
                p = Path(rec.path)
                if not p.exists():
                    raise HTTPException(404, "file missing")
                return FileResponse(p)
        raise HTTPException(404, "record not found")

    @app.post("/api/feedback")
    def feedback(payload: FeedbackIn) -> dict:
        FEEDBACK_TOTAL.labels(verdict=payload.verdict).inc()
        return state.add_feedback(payload.model_dump())

    @app.get("/api/review/queue")
    def review_queue(limit: int = 200) -> dict:
        rows = state.read_feedback(limit=limit)
        return {"count": len(rows), "items": rows}

    @app.get("/api/benchmark/latest")
    def benchmark_latest() -> dict:
        out: dict[str, Any] = {}
        for tag, rel in {
            "stage1_full_test_bench": "artifacts/benchmark_stage1/comparison.json",
            "mini": "artifacts/benchmark_mini/comparison.json",
            "zero_shot_grounding_dino": "artifacts/benchmark_zeroshot/comparison.json",
            "rtdetr_fireviewer": "artifacts/benchmark_rtdetr/comparison.json",
            "external_cctv_nc": "artifacts/external_cctv_nc/summary.json",
            "video_ablation": "artifacts/video_ablation",
        }.items():
            p = PROJECT_ROOT / rel
            if p.is_file():
                out[tag] = json.loads(p.read_text(encoding="utf-8"))
            elif p.is_dir():
                out[tag] = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in p.glob("*_video_metrics.json")}
        return out

    @app.get("/api/benchmark/benchR")
    def benchmark_benchR() -> dict:
        latest: dict[str, dict] = {}
        failures: dict = {}
        results_dir = PROJECT_ROOT / "bench_results" / "benchR"
        for p in sorted(results_dir.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("bench") == "bench_R" and "splits" in d:
                latest[d["model"]] = d
            elif p.name == "failures.json":
                failures = d
        models = []
        for model_id, d in sorted(latest.items()):
            splits = {}
            for name, s in d["splits"].items():
                splits[name] = {
                    "n": s["n"], "positives": s["positives"], "combined": s["combined"],
                    "fire": s["fire"], "smoke": s["smoke"],
                    "box_ap": s["box_ap"], "by_flame_type": s["by_flame_type"],
                    "policy_uncontrolled_plus_smoke": s.get("policy_uncontrolled_plus_smoke"),
                    "policy_strict": s.get("policy_strict"),
                }
            models.append({
                "model": model_id, "family": d.get("family"), "license": d.get("license"),
                "thresholds": d.get("thresholds"), "target_fpr": d.get("target_fpr"),
                "bench_locked_sha256": d.get("bench_locked_sha256"),
                "generated_utc": d.get("generated_utc"), "splits": splits,
            })
        return {
            "bench": "bench_R",
            "bench_locked_sha256": json.loads((PROJECT_ROOT / "bench_R" / "LOCKED.json").read_text(encoding="utf-8"))["manifest_sha256"],
            "splits": {"dev": 300, "sealed": 700, "all": 1000},
            "calibrated_on": "data/splits/val_mini.jsonl + data/splits/val_flames.jsonl",
            "models": models,
            "not_run": failures,
        }

    @app.get("/api/artifacts/{rel:path}")
    def artifact(rel: str) -> JSONResponse:
        p = (PROJECT_ROOT / "artifacts" / rel).resolve()
        artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
        if not str(p).startswith(str(artifacts_root)) or not p.is_file():
            raise HTTPException(404, "artifact not found")
        if p.suffix == ".json":
            return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
        return Response(content=p.read_bytes())

    @app.get("/api/live/cameras")
    def cameras() -> dict:
        return {"cameras": [
            {"id": c.camera_id, "name": c.name, "uri": c.uri, "enabled": c.enabled, "status": c.status,
             "fps": round(c.fps, 2), "frames": c.frames, "alarm": c.alarm}
            for c in state.cameras.values()
        ]}

    @app.get("/api/cameras/{camera_id}/config")
    def get_camera_config(camera_id: str) -> dict:
        try:
            return state.camera_config(camera_id)
        except KeyError:
            raise HTTPException(404, "unknown camera")

    @app.post("/api/cameras/{camera_id}/config")
    def set_camera_config(camera_id: str, payload: CameraConfigIn) -> dict:
        updates = {k: v for k, v in payload.model_dump().items() if k != "actor" and v is not None}
        if not updates:
            raise HTTPException(422, "no fields to update")
        try:
            return state.update_camera_config(camera_id, updates, actor=payload.actor)
        except KeyError:
            raise HTTPException(404, "unknown camera")
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.post("/api/live/{camera_id}/start")
    def start_camera(camera_id: str) -> dict:
        if camera_id not in state.cameras:
            raise HTTPException(404, "unknown camera")
        return state.start_camera(camera_id)

    @app.post("/api/live/{camera_id}/stop")
    def stop_camera(camera_id: str) -> dict:
        return state.stop_camera(camera_id)

    @app.get("/api/live/{camera_id}/mjpeg")
    def mjpeg(camera_id: str):
        if camera_id not in state.cameras:
            raise HTTPException(404, "unknown camera")
        if camera_id not in state.workers or not state.workers[camera_id].is_alive():
            state.start_camera(camera_id)
        worker = state.workers[camera_id]

        async def stream():
            boundary = b"--frame\r\n"
            while True:
                jpeg, dets, alarm = worker.snapshot()
                if jpeg:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1.0 / max(cfg.stream.sample_fps, 1))
        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/api/live/{camera_id}/ws")
    async def ws_live(websocket: WebSocket, camera_id: str):
        await websocket.accept()
        if camera_id not in state.cameras:
            await websocket.send_json({"error": "unknown camera"})
            await websocket.close()
            return
        if camera_id not in state.workers or not state.workers[camera_id].is_alive():
            state.start_camera(camera_id)
        worker = state.workers[camera_id]
        try:
            while True:
                _, dets, alarm = worker.snapshot()
                await websocket.send_json({"camera_id": camera_id, "detections": dets, "alarm": alarm, "ts": time.time()})
                await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
