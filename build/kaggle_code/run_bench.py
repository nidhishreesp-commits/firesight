from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.data.manifest import load_manifest
from firesight.detectors.base import load_detector
from firesight.eval.metrics import box_iou, compute_ap
from firesight.eval.runner import cache_predictions, load_predictions
from firesight.flame import relabel_flame_evidence
from firesight.flame_type import canonical_flame_type
from firesight.types import Box, Detection, GroundTruth, ImageRecord

BENCH_DIR = PROJECT_ROOT / "testbench100"
BENCH_LOCK = BENCH_DIR / "LOCKED.json"
VAL_MINI = PROJECT_ROOT / "data/splits/val_mini.jsonl"
VAL_FLAMES = PROJECT_ROOT / "data/splits/val_flames.jsonl"
RUNS = PROJECT_ROOT / "bench_results"

MODELS = {
    "v3": {"kind": "yolo", "path": "models/best_fire_v3.onnx", "imgsz": 640, "iou": 0.45,
           "license": "proprietary (SmartSight)", "family": "YOLOv12n"},
    "v4": {"kind": "yolo", "path": "../smartsight-models/best-fire-v4/model.onnx", "imgsz": 640, "iou": 0.45,
           "license": "proprietary (SmartSight)", "family": "YOLOv12n"},
    "fire_pt": {"kind": "yolo", "path": "../iSight-python-licenseclean/isight-inference/models/fire.pt", "imgsz": 640,
                "iou": 0.45, "license": "proprietary (SmartSight)", "family": "YOLO11s"},
    "finetune_v2_1ep": {"kind": "yolo", "path": "runs/train/finetune_v2_1ep/weights/best.pt", "imgsz": 512,
                        "iou": 0.45, "license": "internal fine-tune", "family": "YOLOv12n"},
    "pyronear_sensitive": {"kind": "pyronear", "imgsz": 1024, "iou": 0.01,
                           "license": "Apache-2.0", "family": "YOLO11s (Pyronear)"},
    "fireviewer_rtdetrv2": {"kind": "rtdetr_v2", "path": "fireviewer/rtdetr-v2-r50-fire-smoke",
                            "threshold": 0.005, "preprocess_mode": "squash",
                            "license": "FireViewer Model Use License v1.0 (non-redistributable)",
                            "family": "RT-DETRv2-R50"},
    "cascade_v3": {"kind": "cascade", "base": "v3", "verifier": "models/verifier.onnx", "flame": True,
                   "license": "proprietary (SmartSight)", "family": "v3 + verifier + flame"},
    "cascade_union_v3_firept": {"kind": "cascade", "base_members": ["v3", "fire_pt"], "verifier": "models/verifier.onnx",
                                "flame": True, "license": "proprietary (SmartSight)",
                                "family": "v3+fire.pt union + verifier + flame"},
    "ensemble_v3_v4_union": {"kind": "ensemble", "members": ["v3", "v4"], "mode": "union",
                             "license": "proprietary (SmartSight)", "family": "v3+v4 union"},
    "ensemble_v3_v4_intersection": {"kind": "ensemble", "members": ["v3", "v4"], "mode": "intersection",
                                    "license": "proprietary (SmartSight)", "family": "v3+v4 intersection"},
    "ensemble_v3_firept_union": {"kind": "ensemble", "members": ["v3", "fire_pt"], "mode": "union",
                                 "license": "proprietary (SmartSight)", "family": "v3+fire.pt union"},
    "ensemble_v3_v4_firept_union": {"kind": "ensemble", "members": ["v3", "v4", "fire_pt"], "mode": "union",
                                    "license": "proprietary (SmartSight)", "family": "v3+v4+fire.pt union"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bench() -> list[ImageRecord]:
    if not BENCH_LOCK.exists():
        raise SystemExit("testbench100 is not locked; refusing to benchmark")
    index = {r["id"]: r for r in json.loads((BENCH_DIR / "index.json").read_text(encoding="utf-8"))}
    attr_path = PROJECT_ROOT / "bench_attributes.csv"
    flame_types = defaultdict(set)
    if attr_path.exists():
        with attr_path.open("r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                flame_types[row["image_id"]].add(canonical_flame_type(row["flame_type"]))
    records = []
    with (BENCH_DIR / "labels.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            meta = index.get(row["image_id"], {})
            boxes_fire = json.loads(row["boxes_fire"] or "[]")
            boxes_smoke = json.loads(row["boxes_smoke"] or "[]")
            w = meta.get("width")
            h = meta.get("height")
            img_path = BENCH_DIR / "images" / row["file"]
            if not w or not h:
                im = cv2.imread(str(img_path))
                if im is not None:
                    h, w = im.shape[:2]
            labels = [GroundTruth("fire", Box(*b)) for b in boxes_fire] + [GroundTruth("smoke", Box(*b)) for b in boxes_smoke]
            records.append(ImageRecord(
                id=row["image_id"], path=str(img_path), width=int(w or 0), height=int(h or 0),
                source=row["source"], scene=row["category"], split="testbench",
                labels=labels, category=row["category"],
                meta={"family": meta.get("family"), "difficulty": row["difficulty"],
                      "has_fire": int(row["has_fire"]), "has_smoke": int(row["has_smoke"]),
                      "ambiguous": int(row["ambiguous"]), "license": row["license"],
                      "flame_types": sorted(flame_types.get(row["image_id"], {"unknown"}))},
            ))
    return records


def load_val() -> list[ImageRecord]:
    return load_manifest(VAL_MINI, root=PROJECT_ROOT) + load_manifest(VAL_FLAMES, root=PROJECT_ROOT)


def calibrate_image_level(records: list[ImageRecord], preds: dict, target_fpr: float = 0.01) -> dict:
    def flags(rec):
        return (bool(rec.meta.get("has_fire") or any(g.label == "fire" for g in rec.labels)),
                bool(rec.meta.get("has_smoke") or any(g.label == "smoke" for g in rec.labels)))

    positives = [r for r in records if any(flags(r))]
    negatives = [r for r in records if not any(flags(r))]
    candidates = np.round(np.arange(0.01, 1.0, 0.02), 3)
    best = None
    for tf in candidates:
        for ts in candidates:
            fpr = 0
            for rec in negatives:
                if any(d.confidence >= (tf if d.label == "fire" else ts) for d in preds.get(rec.id, [])):
                    fpr += 1
            fpr /= max(len(negatives), 1)
            if fpr > target_fpr:
                continue
            hits = 0
            for rec in positives:
                hf, hs = flags(rec)
                if (hf and any(d.label == "fire" and d.confidence >= tf for d in preds.get(rec.id, []))) or \
                   (hs and any(d.label == "smoke" and d.confidence >= ts for d in preds.get(rec.id, []))):
                    hits += 1
            recall = hits / max(len(positives), 1)
            score = (recall, -fpr, tf + ts, tf, ts)
            if best is None or score > best:
                best = score
    if best is None:
        return {"fire": 0.99, "smoke": 0.99}
    return {"fire": float(best[3]), "smoke": float(best[4])}


def model_detector(model_id: str, cfg, device: str = "cpu"):
    spec = MODELS[model_id]
    if spec["kind"] == "pyronear":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("pyronear/yolo11s_sensitive-detector", "best.pt")
        return load_detector("yolo", path, imgsz=spec["imgsz"], iou=spec["iou"], device=device,
                             name=model_id, license=spec["license"], family=spec["family"],
                             class_map={0: "smoke"})
    if spec["kind"] == "rtdetr_v2":
        return load_detector("rtdetr_v2", spec["path"], device=device, name=model_id,
                             license=spec["license"], family=spec["family"],
                             threshold=spec.get("threshold", 0.005),
                             preprocess_mode=spec.get("preprocess_mode", "squash"))
    path = PROJECT_ROOT / spec["path"] if not Path(spec["path"]).is_absolute() else Path(spec["path"])
    return load_detector(spec.get("kind", "yolo"), path, imgsz=spec.get("imgsz", cfg.detector.imgsz),
                         iou=spec.get("iou", cfg.detector.iou), device=device,
                         name=model_id, license=spec["license"], family=spec["family"],
                         class_map=spec.get("class_map"))


def base_predictions(model_id: str, records: list[ImageRecord], split: str, cfg, overwrite: bool = False):
    cache = RUNS / "predictions" / f"{model_id}_{split}.jsonl"
    if cache.exists() and not overwrite:
        preds, _ = load_predictions(cache)
        if len(preds) == len(records):
            return preds
    det = model_detector(model_id, cfg)
    try:
        cache_predictions(det, records, cache, batch_size=1 if model_id == "pyronear_sensitive" else 8, overwrite=overwrite)
    finally:
        det.close()
    preds, _ = load_predictions(cache)
    return preds


def cascade_predictions(model_id: str, records: list[ImageRecord], split: str, cfg, overwrite: bool = False):
    base_ids = MODELS[model_id].get("base_members") or [MODELS[model_id]["base"]]
    bases = [base_predictions(b, records, split, cfg, overwrite=overwrite) for b in base_ids]
    base = {}
    for rec in records:
        dets = []
        for b in bases:
            for d in b.get(rec.id, []):
                if not any(d.label == x.label and box_iou(d.box, x.box) >= 0.5 for x in dets):
                    dets.append(d)
        base[rec.id] = dets
    cache = RUNS / "predictions" / f"{model_id}_{split}.jsonl"
    if cache.exists() and not overwrite:
        preds, _ = load_predictions(cache)
        if len(preds) == len(records):
            return preds
    from firesight.verifier import CropVerifier
    verifier = CropVerifier(PROJECT_ROOT / MODELS[model_id]["verifier"], classes=cfg.verifier.classes,
                            threshold=cfg.verifier.threshold)
    out = {}
    for i, rec in enumerate(records):
        img = cv2.imread(rec.path)
        dets = base.get(rec.id, [])
        if img is not None and dets:
            if MODELS[model_id].get("flame"):
                dets = relabel_flame_evidence(img, dets)
            dets, _ = verifier.verify(img, dets)
        out[rec.id] = dets
        if i % 50 == 0:
            print(f"cascade {split} {i}/{len(records)}", flush=True)
    with cache.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps({"id": rec.id, "latency_ms": 0, "detections": [
                {"label": d.label, "confidence": d.confidence, "box": list(d.box.to_list())} for d in out.get(rec.id, [])
            ]}) + "\n")
    return out


def ensemble_predictions(model_id: str, records: list[ImageRecord], split: str, cfg, overwrite: bool = False):
    spec = MODELS[model_id]
    members = [base_predictions(m, records, split, cfg, overwrite=overwrite) for m in spec["members"]]
    out = {}
    for rec in records:
        lists = [m.get(rec.id, []) for m in members]
        if spec["mode"] == "union":
            dets = []
            for dets_i in lists:
                for d in dets_i:
                    if not any(d.label == x.label and box_iou(d.box, x.box) >= 0.5 for x in dets):
                        dets.append(d)
        else:
            dets = []
            for d in lists[0]:
                matched = True
                best_conf = d.confidence
                for other in lists[1:]:
                    match = max((x for x in other if x.label == d.label), key=lambda x: box_iou(d.box, x.box), default=None)
                    if match is None or box_iou(d.box, match.box) < 0.5:
                        matched = False
                        break
                    best_conf = max(best_conf, match.confidence)
                if matched:
                    dets.append(Detection(d.label, best_conf, d.box))
        out[rec.id] = dets
    return out


def get_predictions(model_id: str, records: list[ImageRecord], split: str, cfg, overwrite: bool = False):
    spec = MODELS[model_id]
    if spec["kind"] == "cascade":
        return cascade_predictions(model_id, records, split, cfg, overwrite)
    if spec["kind"] == "ensemble":
        return ensemble_predictions(model_id, records, split, cfg, overwrite)
    return base_predictions(model_id, records, split, cfg, overwrite)


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rule_of_three(n: int) -> float | None:
    return 3.0 / n if n > 0 else None


def decision(rec: ImageRecord, preds: list[Detection], thresholds: dict[str, float], cls: str | None = None) -> bool:
    if cls is None:
        return any(d.confidence >= thresholds.get(d.label, 1.0) for d in preds)
    return any(d.label == cls and d.confidence >= thresholds.get(cls, 1.0) for d in preds)


def binary_metrics(records: list[ImageRecord], preds: dict, thresholds: dict[str, float], cls: str | None) -> dict:
    tp = fp = tn = fn = 0
    for rec in records:
        positive = bool(rec.meta.get("has_fire") if cls == "fire" else rec.meta.get("has_smoke") if cls == "smoke"
                        else (rec.meta.get("has_fire") or rec.meta.get("has_smoke")))
        alarm = decision(rec, preds.get(rec.id, []), thresholds, cls)
        if positive and alarm:
            tp += 1
        elif positive:
            fn += 1
        elif alarm:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round((tp + tn) / n, 6) if n else None,
        "accuracy_95ci": wilson(tp + tn, n),
        "precision": round(precision, 6) if precision is not None else None,
        "precision_95ci": wilson(tp, tp + fp),
        "recall": round(recall, 6) if recall is not None else None,
        "recall_95ci": wilson(tp, tp + fn),
        "specificity": round(specificity, 6) if specificity is not None else None,
        "specificity_95ci": wilson(tn, tn + fp),
        "f1": round(f1, 6) if f1 is not None else None,
        "zero_event_upper_95": rule_of_three(tn + fp) if (tn + fp) and fp == 0 else None,
    }


def box_ap(records: list[ImageRecord], preds: dict, cls: str) -> dict:
    aps = [compute_ap(records, preds, cls, t) for t in (0.5, 0.75)] + [
        compute_ap(records, preds, cls, t) for t in (0.5 + 0.05 * i for i in range(10))
    ]
    gt_boxes = sum(1 for r in records for g in r.labels if g.label == cls)
    return {"gt_boxes": gt_boxes, "ap50": round(aps[0], 6) if gt_boxes else None,
            "ap75": round(aps[1], 6) if gt_boxes else None,
            "ap50_95": round(float(np.nanmean(aps[2:])), 6) if gt_boxes else None}


def mean_iou(records: list[ImageRecord], preds: dict, cls: str, threshold: float) -> dict:
    ious = []
    matched = 0
    for rec in records:
        gts = [g for g in rec.labels if g.label == cls]
        dets = [d for d in preds.get(rec.id, []) if d.label == cls and d.confidence >= threshold]
        for gt in gts:
            best = max((box_iou(gt.box, d.box) for d in dets), default=0.0)
            ious.append(best)
            if best >= 0.5:
                matched += 1
    return {"gt_boxes": len(ious), "matched_iou50": matched,
            "recall_iou50": round(matched / len(ious), 6) if ious else None,
            "mean_iou_all_gt": round(float(np.mean(ious)), 6) if ious else None,
            "mean_iou_matched": round(float(np.mean([x for x in ious if x >= 0.5])), 6) if matched else None}


def threshold_sweep(records: list[ImageRecord], preds: dict) -> list[dict]:
    rows = []
    for t in np.arange(0.05, 1.0, 0.05):
        th = {"fire": float(t), "smoke": float(t)}
        fire = binary_metrics(records, preds, th, "fire")
        smoke = binary_metrics(records, preds, th, "smoke")
        combined = binary_metrics(records, preds, th, None)
        rows.append({"threshold": round(float(t), 2), "combined_accuracy": combined["accuracy"],
                     "combined_recall": combined["recall"], "combined_fpr": round(combined["fp"] / combined["n"], 6) if combined["n"] else None,
                     "fire_recall": fire["recall"], "smoke_recall": smoke["recall"]})
    return rows


def by_flame_type(records: list[ImageRecord], preds: dict, thresholds: dict[str, float]) -> dict:
    """Image-level alert recall per ground-truth flame type, fire AND smoke alike.

    Alert-level decision (any class over its own threshold), so these numbers are
    directly comparable to `policy_strict`: an image whose flame_type is
    `uncontrolled` counts here exactly as it does in the strict policy view.
    Every positive image contributes to each of its `flame_type` buckets;
    smoke-only images additionally keep the historical `smoke_only` bucket.
    """
    groups = defaultdict(lambda: {"n": 0, "detected": 0})
    for rec in records:
        hf = bool(rec.meta.get("has_fire"))
        hs = bool(rec.meta.get("has_smoke"))
        if not (hf or hs):
            continue
        types = rec.meta.get("flame_types") or ["unknown"]
        detected = int(decision(rec, preds.get(rec.id, []), thresholds))
        for ftype in types:
            groups[ftype]["n"] += 1
            groups[ftype]["detected"] += detected
        if hs and not hf:
            groups["smoke_only"]["n"] += 1
            groups["smoke_only"]["detected"] += detected
    out = {}
    for key, value in sorted(groups.items()):
        out[key] = {**value, "recall": round(value["detected"] / value["n"], 6) if value["n"] else None,
                    "recall_95ci": wilson(value["detected"], value["n"])}
    return out


EXCLUDED_TYPES = ("controlled", "spark_benign", "electrical_arc", "screen", "spark")


def _policy_view(records: list[ImageRecord], preds: dict, thresholds: dict[str, float],
                 expected, suppressed_key: str) -> dict:
    """Confusion matrix under a ground-truth-driven alarm policy.

    `expected(types)` returns True when an image is expected to alarm under the
    policy and False when the policy suppresses it. Suppressed positives leave the
    confusion matrix entirely (the policy is behaving correctly by staying quiet),
    and are reported separately so the suppression is never invisible.
    """
    tp = fp = tn = fn = suppressed = suppressed_alarm = 0
    for rec in records:
        positive = bool(rec.meta.get("has_fire") or rec.meta.get("has_smoke"))
        types = set(rec.meta.get("flame_types") or [])
        if positive and not expected(types):
            suppressed += 1
            suppressed_alarm += int(decision(rec, preds.get(rec.id, []), thresholds))
            continue
        alarm = decision(rec, preds.get(rec.id, []), thresholds)
        if positive and alarm:
            tp += 1
        elif positive:
            fn += 1
        elif alarm:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / n, 6) if n else None,
            "accuracy_95ci": wilson(tp + tn, n),
            "recall": round(tp / (tp + fn), 6) if tp + fn else None,
            "recall_95ci": wilson(tp, tp + fn),
            "specificity": round(tn / (tn + fp), 6) if tn + fp else None,
            "specificity_95ci": wilson(tn, tn + fp),
            suppressed_key: suppressed,
            "suppressed_images_that_alarmed": suppressed_alarm}


def policy_metrics(records: list[ImageRecord], preds: dict, thresholds: dict[str, float],
                   excluded_types=EXCLUDED_TYPES) -> dict:
    """Policy view A: alarm unless every ground-truth flame type is excluded.

    Historical key `policy_uncontrolled_plus_smoke`. The exclusion now applies to
    smoke-only images too — smoke no longer bypasses the policy.
    """
    excluded = set(excluded_types)
    return _policy_view(records, preds, thresholds,
                        lambda types: not (bool(types) and types.issubset(excluded)),
                        suppressed_key="suppressed_controlled_or_spark")


def policy_strict_metrics(records: list[ImageRecord], preds: dict, thresholds: dict[str, float]) -> dict:
    """Policy view B: alarm iff ground-truth flame_type == 'uncontrolled'.

    Applies to fire AND smoke: uncontrolled smoke alarms, controlled smoke /
    spark_benign / electrical_arc / screen / unknown do not.
    """
    return _policy_view(records, preds, thresholds,
                        lambda types: "uncontrolled" in types,
                        suppressed_key="suppressed_non_uncontrolled")


def group_metrics(records: list[ImageRecord], preds: dict, thresholds: dict[str, float]) -> dict:
    out = defaultdict(lambda: {"n": 0, "correct": 0})
    for rec in records:
        key = rec.meta.get("family") or rec.category
        if rec.meta.get("has_fire") or rec.meta.get("has_smoke"):
            key = f"positive:{rec.meta.get('difficulty')}"
        correct = decision(rec, preds.get(rec.id, []), thresholds)
        positive = bool(rec.meta.get("has_fire") or rec.meta.get("has_smoke"))
        out[key]["n"] += 1
        out[key]["correct"] += int(correct == positive)
    return {k: {**v, "accuracy": round(v["correct"] / v["n"], 6)} for k, v in sorted(out.items())}


def draw_error(rec: ImageRecord, preds: list[Detection], thresholds: dict[str, float], out_path: Path) -> None:
    img = cv2.imread(rec.path)
    if img is None:
        return
    for gt in rec.labels:
        x1, y1, x2, y2 = map(int, gt.box.to_list())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(img, f"GT {gt.label}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    for d in preds:
        if d.confidence < min(thresholds.values()):
            continue
        x1, y1, x2, y2 = map(int, d.box.to_list())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def contact_sheet(paths: list[Path], out: Path, title: str) -> None:
    tiles = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.resize(img, (320, 220))
        tiles.append(img)
    if not tiles:
        return
    while len(tiles) % 5:
        tiles.append(np.zeros((220, 320, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
    cv2.putText(sheet, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a model/pipeline on the locked testbench100. Thresholds come from val_mini only.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if not BENCH_LOCK.exists():
        raise SystemExit("testbench100 is not locked; refusing to benchmark")
    cfg = load_config(args.config)
    bench = load_bench()
    val = load_val()
    print(f"bench={len(bench)} val={len(val)} model={args.model}", flush=True)

    val_preds = get_predictions(args.model, val, "valall", cfg, overwrite=args.overwrite_cache)
    bench_preds = get_predictions(args.model, bench, "bench", cfg, overwrite=args.overwrite_cache)
    thresholds = calibrate_image_level(val, val_preds, target_fpr=args.target_fpr)
    print("thresholds from val_mini+val_flames:", thresholds, flush=True)

    metrics = {
        "model": args.model,
        "family": MODELS[args.model]["family"],
        "license": MODELS[args.model]["license"],
        "bench_locked_sha256": json.loads(BENCH_LOCK.read_text(encoding="utf-8"))["manifest_sha256"],
        "calibrated_on": "data/splits/val_mini.jsonl + data/splits/val_flames.jsonl",
        "target_fpr": args.target_fpr,
        "thresholds": thresholds,
        "bench_size": len(bench),
        "fire": binary_metrics(bench, bench_preds, thresholds, "fire"),
        "smoke": binary_metrics(bench, bench_preds, thresholds, "smoke"),
        "combined": binary_metrics(bench, bench_preds, thresholds, None),
        "box_ap": {"fire": box_ap(bench, bench_preds, "fire"), "smoke": box_ap(bench, bench_preds, "smoke")},
        "mean_iou": {"fire": mean_iou(bench, bench_preds, "fire", thresholds.get("fire", 1)),
                     "smoke": mean_iou(bench, bench_preds, "smoke", thresholds.get("smoke", 1))},
        "by_group": group_metrics(bench, bench_preds, thresholds),
        "by_flame_type": by_flame_type(bench, bench_preds, thresholds),
        "policy_uncontrolled_plus_smoke": policy_metrics(bench, bench_preds, thresholds),
        "policy_strict": policy_strict_metrics(bench, bench_preds, thresholds),
        "threshold_sweep": threshold_sweep(bench, bench_preds),
        "errors": [],
    }
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + args.model
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fp_paths, fn_paths = [], []
    for rec in bench:
        dets = [d for d in bench_preds.get(rec.id, []) if d.confidence >= min(thresholds.values())]
        positive = bool(rec.meta.get("has_fire") or rec.meta.get("has_smoke"))
        alarm = decision(rec, dets, thresholds)
        if positive == alarm:
            continue
        kind = "false_positive" if not positive else "false_negative"
        path = run_dir / "errors" / kind / f"{rec.id.replace(':','_')}.jpg"
        draw_error(rec, dets, thresholds, path)
        (fp_paths if kind == "false_positive" else fn_paths).append(path)
        metrics["errors"].append({
            "kind": kind, "id": rec.id, "family": rec.meta.get("family"), "category": rec.category,
            "difficulty": rec.meta.get("difficulty"), "ground_truth": [g.to_dict() for g in rec.labels],
            "predictions": [d.to_dict() for d in dets if d.confidence >= min(thresholds.values())],
            "files": [str(path.relative_to(PROJECT_ROOT))],
        })
    contact_sheet(fp_paths, run_dir / "false_positives.jpg", f"{args.model} false positives")
    contact_sheet(fn_paths, run_dir / "false_negatives.jpg", f"{args.model} false negatives")
    result = {
        **metrics,
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "notes": {
            "threshold_policy": "thresholds selected on val_mini + val_flames only; bench was not used for tuning",
            "composition_override": json.loads(BENCH_LOCK.read_text(encoding="utf-8")).get("composition_override"),
            "external_models": {
                "pyronear_temporal": "sequence model; not applicable to single-image bench; evaluate on video clips",
                "select_ai_fire_smoke": "no public weights found",
                "e1250_safety_detection": "no declared licence; not downloaded without approval",
            },
        },
    }
    out = RUNS / f"{run_id}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("model", "thresholds", "fire", "smoke", "combined")}, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
