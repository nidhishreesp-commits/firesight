from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from ..types import Box, Detection, GroundTruth, ImageRecord

CLASSES = ("fire", "smoke")
IOU_RANGE = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def box_iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(
    preds: Sequence[Detection],
    gts: Sequence[GroundTruth],
    iou_thr: float,
) -> tuple[list[bool], list[bool]]:
    matched_gt = [False] * len(gts)
    matched_pred = [False] * len(preds)
    order = sorted(range(len(preds)), key=lambda i: preds[i].confidence, reverse=True)
    for pi in order:
        pred = preds[pi]
        best_idx, best_iou = -1, iou_thr
        for gi, gt in enumerate(gts):
            if matched_gt[gi] or gt.label != pred.label:
                continue
            iou = box_iou(pred.box, gt.box)
            if iou >= best_iou:
                best_idx, best_iou = gi, iou
        if best_idx >= 0:
            matched_gt[best_idx] = True
            matched_pred[pi] = True
    return matched_pred, matched_gt


def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if tp.size == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(fp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([0.0], precision))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(idx, mrec, mpre)))


def compute_ap(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    class_name: str,
    iou_thr: float,
) -> float:
    entries: list[tuple[float, int]] = []
    n_gt = 0
    for rec in records:
        gts = [gt for gt in rec.labels if gt.label == class_name]
        n_gt += len(gts)
        preds = [d for d in predictions.get(rec.id, []) if d.label == class_name]
        matched_pred, _ = _greedy_match(preds, gts, iou_thr)
        matched_map = {id(d): m for d, m in zip(preds, matched_pred)}
        for det in preds:
            entries.append((det.confidence, 1 if matched_map[id(det)] else 0))
    entries.sort(key=lambda x: x[0], reverse=True)
    tp = np.array([e[1] for e in entries], dtype=float)
    fp = np.array([1 - e[1] for e in entries], dtype=float)
    return _ap_from_pr(tp, fp, n_gt)


@dataclass
class OperatingPoint:
    thresholds: dict[str, float]
    image_recall: float
    image_precision: float
    fpr: float
    box_recall: dict[str, float]
    box_precision: dict[str, float]
    per_class: dict[str, dict[str, float]]
    counts: dict[str, int]
    negative_images: int
    positive_images: int

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "image_recall": self.image_recall,
            "image_precision": self.image_precision,
            "fpr": self.fpr,
            "box_recall": self.box_recall,
            "box_precision": self.box_precision,
            "per_class": self.per_class,
            "counts": self.counts,
            "negative_images": self.negative_images,
            "positive_images": self.positive_images,
        }


def _filter(preds: list[Detection], thresholds: dict[str, float]) -> list[Detection]:
    return [d for d in preds if d.confidence >= thresholds.get(d.label, 1.0)]


def _counts_at(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    thresholds: dict[str, float],
    iou_thr: float,
) -> dict[str, float]:
    tp = fp = fn = 0
    img_tp = 0
    img_fp = 0
    positives = 0
    negatives = 0
    for rec in records:
        preds = _filter(predictions.get(rec.id, []), thresholds)
        gts = rec.labels
        if gts:
            positives += 1
            matched_pred, matched_gt = _greedy_match(preds, gts, iou_thr)
            tp += sum(matched_pred)
            fp += len(preds) - sum(matched_pred)
            fn += len(gts) - sum(matched_gt)
            if any(matched_pred):
                img_tp += 1
        else:
            negatives += 1
            if preds:
                img_fp += 1
            fp += len(preds)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "image_tp": img_tp, "image_fp": img_fp,
        "positives": positives, "negatives": negatives,
    }


def _image_level_arrays(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    unc, neg = [], []
    for rec in records:
        preds = predictions.get(rec.id, [])
        if rec.labels:
            matched_pred, _ = _greedy_match(preds, rec.labels, iou_thr)
            unc.append(max((d.confidence for d, m in zip(preds, matched_pred) if m), default=0.0))
        else:
            neg.append(max((d.confidence for d in preds), default=0.0))
    return np.asarray(unc, dtype=float), np.asarray(neg, dtype=float)


def image_level_curve(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    iou_thr: float = 0.5,
    steps: int = 101,
) -> list[dict[str, float]]:
    unc, neg = _image_level_arrays(records, predictions, iou_thr)
    if unc.size == 0:
        return []
    candidates = np.unique(np.concatenate(([0.01], np.quantile(unc, np.linspace(0, 1, steps)), neg))) if neg.size else np.quantile(unc, np.linspace(0, 1, steps))
    rows = []
    for t in candidates:
        recall = float((unc >= t).mean())
        fpr = float((neg >= t).mean()) if neg.size else 0.0
        rows.append({"threshold": round(float(t), 4), "recall": round(recall, 6), "fpr": round(fpr, 6)})
    return rows


def _operating_arrays(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_fire, pos_smoke, neg_fire, neg_smoke = [], [], [], []
    for rec in records:
        preds = predictions.get(rec.id, [])
        fire = [d for d in preds if d.label == "fire"]
        smoke = [d for d in preds if d.label == "smoke"]
        if rec.labels:
            matched_pred, _ = _greedy_match(preds, rec.labels, iou_thr)
            pos_fire.append(max((d.confidence for d, m in zip(preds, matched_pred) if m and d.label == "fire"), default=0.0))
            pos_smoke.append(max((d.confidence for d, m in zip(preds, matched_pred) if m and d.label == "smoke"), default=0.0))
        else:
            neg_fire.append(max((d.confidence for d in fire), default=0.0))
            neg_smoke.append(max((d.confidence for d in smoke), default=0.0))
    return (np.asarray(pos_fire), np.asarray(pos_smoke), np.asarray(neg_fire), np.asarray(neg_smoke))


def select_operating_point(
    val_records: Sequence[ImageRecord],
    val_predictions: dict[str, list[Detection]],
    test_records: Sequence[ImageRecord],
    test_predictions: dict[str, list[Detection]],
    target_fpr: float = 0.001,
    iou_thr: float = 0.5,
) -> OperatingPoint:
    candidates = np.round(np.arange(0.01, 1.0, 0.02), 3)
    pf, ps, nf, ns = _operating_arrays(val_records, val_predictions, iou_thr)
    best: tuple[float, float, float, float] | None = None
    if pf.size and nf.size:
        tf = candidates[:, None, None]
        ts = candidates[None, :, None]
        recall_grid = (((pf[None, None, :] >= tf) | (ps[None, None, :] >= ts)).mean(axis=2))
        fpr_grid = (((nf[None, None, :] >= tf) | (ns[None, None, :] >= ts)).mean(axis=2))
        valid = fpr_grid <= target_fpr
        if valid.any():
            masked_recall = np.where(valid, recall_grid, -1.0)
            best_flat = int(np.argmax(masked_recall))
            fi, si = np.unravel_index(best_flat, recall_grid.shape)
            best = (float(recall_grid[fi, si]), float(fpr_grid[fi, si]), float(candidates[fi]), float(candidates[si]))
    if best is None:
        thresholds = {"fire": 0.99, "smoke": 0.99}
    else:
        thresholds = {"fire": best[2], "smoke": best[3]}

    test_counts = _counts_at(test_records, test_predictions, thresholds, iou_thr)
    per_class = {}
    box_recall, box_precision = {}, {}
    for cls in CLASSES:
        gt_count = 0
        pred_count = 0
        tp = 0
        pos = 0
        img_hits = 0
        for rec in test_records:
            gts = [g for g in rec.labels if g.label == cls]
            preds = [d for d in test_predictions.get(rec.id, []) if d.label == cls and d.confidence >= thresholds[cls]]
            gt_count += len(gts)
            pred_count += len(preds)
            if gts:
                pos += 1
                matched_pred, _ = _greedy_match(preds, gts, iou_thr)
                tp += sum(matched_pred)
                img_hits += int(any(matched_pred))
        box_recall[cls] = tp / gt_count if gt_count else float("nan")
        box_precision[cls] = tp / pred_count if pred_count else float("nan")
        per_class[cls] = {
            "gt_boxes": gt_count,
            "pred_boxes": pred_count,
            "tp": tp,
            "box_recall": round(box_recall[cls], 6) if gt_count else 0.0,
            "box_precision": round(box_precision[cls], 6) if pred_count else 0.0,
            "image_recall": round(img_hits / pos, 6) if pos else 0.0,
            "positive_images": pos,
            "negative_images": len(test_records) - pos,
        }
    positives = test_counts["positives"]
    negatives = test_counts["negatives"]
    img_recall = test_counts["image_tp"] / positives if positives else 0.0
    img_precision = test_counts["image_tp"] / max(test_counts["image_tp"] + test_counts["image_fp"], 1)
    fpr = test_counts["image_fp"] / negatives if negatives else 0.0
    return OperatingPoint(
        thresholds=thresholds,
        image_recall=round(img_recall, 6),
        image_precision=round(img_precision, 6),
        fpr=round(fpr, 6),
        box_recall={k: round(v, 6) for k, v in box_recall.items()},
        box_precision={k: round(v, 6) for k, v in box_precision.items()},
        per_class=per_class,
        counts={k: int(v) for k, v in test_counts.items()},
        negative_images=negatives,
        positive_images=positives,
    )


def recall_at_fpr(curve: Sequence[dict], target_fpr: float) -> dict:
    if not curve:
        return {"fpr": target_fpr, "recall": None, "threshold": None}
    rows = sorted(curve, key=lambda r: r["fpr"])
    xs = np.asarray([r["fpr"] for r in rows], dtype=float)
    ys = np.asarray([r["recall"] for r in rows], dtype=float)
    if target_fpr < xs[0]:
        return {"fpr": float(xs[0]), "recall": float(ys[0]), "threshold": float(rows[0]["threshold"])}
    if target_fpr > xs[-1]:
        idx = int(np.argmax(ys))
        return {"fpr": target_fpr, "recall": float(ys[idx]), "threshold": float(rows[idx]["threshold"])}
    # Pick the lowest threshold whose measured FPR is >= target (conservative evaluation point).
    idx = int(np.searchsorted(xs, target_fpr, side="left"))
    return {"fpr": float(xs[idx]), "recall": float(ys[idx]), "threshold": float(rows[idx]["threshold"])}


def evaluate_predictions(
    records: Sequence[ImageRecord],
    predictions: dict[str, list[Detection]],
    operating: OperatingPoint,
    iou_thr: float = 0.5,
) -> dict:
    curve = image_level_curve(records, predictions, iou_thr)
    metrics: dict = {
        "operating_point": operating.to_dict(),
        "per_class": {},
        "map50": {},
        "map50_95": {},
        "pr_curve": curve,
        "recall_at_fpr": {str(t): recall_at_fpr(curve, t) for t in (0.0005, 0.001, 0.005, 0.01, 0.05, 0.10)},
        "failures": [],
    }
    for cls in CLASSES:
        metrics["per_class"][cls] = {
            "ap50": round(compute_ap(records, predictions, cls, 0.5), 6),
            "ap50_95": round(float(np.nanmean([compute_ap(records, predictions, cls, t) for t in IOU_RANGE])), 6),
            "gt_boxes": sum(1 for r in records for g in r.labels if g.label == cls),
            "gt_images": sum(1 for r in records if any(g.label == cls for g in r.labels)),
        }
        metrics["map50"][cls] = metrics["per_class"][cls]["ap50"]
        metrics["map50_95"][cls] = metrics["per_class"][cls]["ap50_95"]

    thresholds = operating.thresholds
    for rec in records:
        preds = _filter(predictions.get(rec.id, []), thresholds)
        matched_pred, matched_gt = _greedy_match(preds, rec.labels, iou_thr)
        fn = [g for g, m in zip(rec.labels, matched_gt) if not m]
        fp = [d for d, m in zip(preds, matched_pred) if not m]
        if fn or fp:
            metrics["failures"].append({
                "id": rec.id,
                "path": rec.path,
                "source": rec.source,
                "scene": rec.scene,
                "negative": rec.is_negative,
                "false_negatives": [g.to_dict() for g in fn],
                "false_positives": [d.to_dict() for d in fp],
            })
    metrics["failure_count"] = len(metrics["failures"])
    return metrics
