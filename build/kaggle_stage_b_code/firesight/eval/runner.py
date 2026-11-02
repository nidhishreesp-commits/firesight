from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from ..detectors.base import BaseDetector
from ..types import Box, Detection, ImageRecord

log = logging.getLogger(__name__)


def _det_to_dict(det: Detection) -> dict:
    return {
        "label": det.label,
        "confidence": float(det.confidence),
        "box": [float(v) for v in det.box.to_list()],
        "track_id": det.track_id,
    }


def _det_from_dict(data: dict) -> Detection:
    return Detection(
        label=data["label"], confidence=float(data["confidence"]),
        box=Box(*data["box"]), track_id=data.get("track_id"),
    )


def cache_predictions(
    detector: BaseDetector,
    records: Sequence[ImageRecord],
    cache_path: str | Path,
    batch_size: int = 8,
    overwrite: bool = False,
    progress_every: int = 200,
) -> dict:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not overwrite:
        rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = {row["id"] for row in rows if row.get("id") != "__meta__"}
        expected = {rec.id for rec in records}
        if expected.issubset(ids):
            return {"rows": len(rows), "cached": True, "path": str(cache_path)}

    rows = []
    total_infer_s = 0.0
    total_images = 0
    start_all = time.perf_counter()
    with cache_path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            t0 = time.perf_counter()
            results = detector.predict_batch([r.path for r in batch])
            elapsed = time.perf_counter() - t0
            total_infer_s += elapsed
            total_images += len(batch)
            per_image_ms = elapsed * 1000.0 / max(len(batch), 1)
            for rec, dets in zip(batch, results):
                row = {
                    "id": rec.id,
                    "latency_ms": round(per_image_ms, 3),
                    "detections": [_det_to_dict(d) for d in dets],
                }
                fh.write(json.dumps(row) + "\n")
                rows.append(row)
            if progress_every and (start + len(batch)) % progress_every == 0:
                log.info("inference %d/%d (%.1f ms/image)", start + len(batch), len(records), per_image_ms)
        fh.write(json.dumps({
            "id": "__meta__",
            "model": detector.name,
            "model_path": str(getattr(detector, "model_path", "")),
            "model_size_mb": round(detector.model_size_mb, 3),
            "license": detector.license,
            "family": detector.family,
            "device": getattr(detector, "device", "unknown"),
            "images": total_images,
            "inference_seconds": round(total_infer_s, 3),
            "ms_per_image": round(total_infer_s * 1000.0 / max(total_images, 1), 3),
            "fps": round(total_images / total_infer_s, 3) if total_infer_s else 0.0,
            "wall_seconds": round(time.perf_counter() - start_all, 3),
        }) + "\n")
    return {"rows": len(rows), "cached": False, "path": str(cache_path)}


def load_predictions(cache_path: str | Path) -> tuple[dict[str, list[Detection]], dict]:
    predictions: dict[str, list[Detection]] = {}
    meta: dict = {}
    with open(cache_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] == "__meta__":
                meta = row
                continue
            predictions[row["id"]] = [_det_from_dict(d) for d in row["detections"]]
    return predictions, meta


def bootstrap_ci(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    thresholds: dict[str, float],
    iou_thr: float = 0.5,
    n: int = 500,
    seed: int = 0,
) -> dict:
    from .metrics import _greedy_match

    positives = [r for r in records if r.labels]
    negatives = [r for r in records if not r.labels]
    if not positives and not negatives:
        return {}
    rng = np.random.default_rng(seed)
    recalls, fprs = [], []
    for _ in range(n):
        p_sample = [positives[i] for i in rng.integers(0, len(positives), size=len(positives))] if positives else []
        n_sample = [negatives[i] for i in rng.integers(0, len(negatives), size=len(negatives))] if negatives else []
        hits = 0
        for rec in p_sample:
            preds = [d for d in predictions.get(rec.id, []) if d.confidence >= thresholds.get(d.label, 1.0)]
            matched, _ = _greedy_match(preds, rec.labels, iou_thr)
            hits += int(any(matched))
        fa = 0
        for rec in n_sample:
            preds = [d for d in predictions.get(rec.id, []) if d.confidence >= thresholds.get(d.label, 1.0)]
            fa += int(bool(preds))
        recalls.append(hits / len(p_sample) if p_sample else float("nan"))
        fprs.append(fa / len(n_sample) if n_sample else float("nan"))
    def ci(values):
        arr = np.asarray([v for v in values if not np.isnan(v)])
        if arr.size == 0:
            return [None, None]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return {"recall_95ci": ci(recalls), "fpr_95ci": ci(fprs), "bootstrap_n": n}


def per_source_metrics(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    thresholds: dict[str, float],
    iou_thr: float = 0.5,
) -> dict:
    from .metrics import _greedy_match

    by_source = defaultdict(list)
    for rec in records:
        by_source[rec.source].append(rec)
    out = {}
    for source, recs in sorted(by_source.items()):
        pos = [r for r in recs if r.labels]
        neg = [r for r in recs if not r.labels]
        hits = fa = 0
        for rec in pos:
            preds = [d for d in predictions.get(rec.id, []) if d.confidence >= thresholds.get(d.label, 1.0)]
            matched, _ = _greedy_match(preds, rec.labels, iou_thr)
            hits += int(any(matched))
        for rec in neg:
            preds = [d for d in predictions.get(rec.id, []) if d.confidence >= thresholds.get(d.label, 1.0)]
            fa += int(bool(preds))
        out[source] = {
            "images": len(recs),
            "positive_images": len(pos),
            "negative_images": len(neg),
            "image_recall": round(hits / len(pos), 6) if pos else None,
            "fpr": round(fa / len(neg), 6) if neg else None,
        }
    return out


def plot_pr(model_name: str, curve: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not curve:
        return
    x = [row["fpr"] * 100 for row in curve]
    y = [row["recall"] * 100 for row in curve]
    ordered = sorted(zip(x, y))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot([p[0] for p in ordered], [p[1] for p in ordered], linewidth=2)
    ax.set_xlabel("False positive rate on negative images (%)")
    ax.set_ylabel("Image-level recall (%)")
    ax.set_title(f"{model_name}: recall vs false-alarm rate")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(10.0, min(100.0, max(x) if x else 10.0)))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
