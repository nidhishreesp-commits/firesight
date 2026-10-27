"""Score a pilot checkpoint on the val.jsonl slices and emit bench_R-format metrics.

Headline metric: image-level recall at FPR <= 2% **per negative image**, with Wilson
95% CIs. Thresholds are calibrated on `lists/val_cal.txt` (a val.jsonl slice) and
scored on `lists/val_report.txt` (a disjoint val.jsonl slice). bench_R, TEST and
testbench100 are not present in the dataset by construction.

Usage:
    python pilot_eval.py --dataset /kaggle/input/firesight-pilot/dataset \
        --preds /kaggle/working/preds_P10.jsonl --model P10 --target-fpr 0.02 \
        --out /kaggle/working/metrics_P10.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # Kaggle: code/ holds run_bench.py + firesight/
sys.path.insert(0, str(HERE.parent / "scripts"))   # local: scripts/run_bench.py
sys.path.insert(0, str(HERE.parent / "src"))       # local: src/firesight

from run_bench import binary_metrics, box_ap, calibrate_image_level, decision, mean_iou, wilson  # noqa: E402

from firesight.types import Box, Detection, GroundTruth, ImageRecord  # noqa: E402

NAMES = {0: "fire", 1: "smoke"}


def load_list(path: Path, dataset: Path) -> list[Path]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = Path(line.strip())
        if not p.is_absolute():
            p = dataset / p
        if not p.exists():
            fallback = dataset / "images" / Path(line).name
            if fallback.exists():
                p = fallback
        out.append(p)
    return out


def read_preds(path: Path) -> dict[str, list[Detection]]:
    preds: dict[str, list[Detection]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            preds[row["id"]] = [Detection(d["label"], float(d["confidence"]), Box(*d["box"]))
                                for d in row.get("detections", [])]
    return preds


def image_id(path: Path) -> str:
    # lists use <id with ':' replaced by '_'>.jpg
    return path.stem


def resolve_label(image: Path, dataset: Path) -> Path | None:
    """Package layout first (images/ + labels/), then the processed-labels tree."""
    cand = dataset / "labels" / (image.stem + ".txt")
    if cand.exists():
        return cand
    text = str(image).replace("\\", "/")
    if "/images/" in text:
        cand = Path(text.replace("/images/", "/labels/")).with_suffix(".txt")
        if cand.exists():
            return cand
    root = Path(__file__).resolve().parents[1] / "data" / "processed" / "labels"
    if root.exists():
        matches = list(root.rglob(image.stem + ".txt"))
        if matches:
            return matches[0]
    return None


def load_records(paths: list[Path], dataset: Path) -> list[ImageRecord]:
    from PIL import Image
    records: list[ImageRecord] = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        lbl = resolve_label(p, dataset)
        labels: list[GroundTruth] = []
        if lbl is not None and lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                labels.append(GroundTruth(NAMES.get(cls, "fire"), Box(x1, y1, x2, y2)))
        records.append(ImageRecord(
            id=p.stem, path=str(p), width=w, height=h, source="val_jsonl", scene="val",
            split="val", labels=labels, category="none",
            meta={"has_fire": int(any(g.label == "fire" for g in labels)),
                  "has_smoke": int(any(g.label == "smoke" for g in labels)),
                  "flame_types": ["unknown"] if labels else []},
        ))
    return records


def fpr_per_negative(records: list[ImageRecord], preds: dict, thresholds: dict) -> dict:
    negatives = [r for r in records if not (r.meta.get("has_fire") or r.meta.get("has_smoke"))]
    fp = sum(1 for r in negatives if decision(r, preds.get(r.id, []), thresholds))
    n = len(negatives)
    return {"n_negatives": n, "fp": fp, "fpr": round(fp / n, 6) if n else None,
            "fpr_95ci": wilson(fp, n) if n else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--model", default="pilot")
    parser.add_argument("--target-fpr", type=float, default=0.02)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    cal_paths = load_list(dataset / "lists" / "val_cal.txt", dataset)
    report_paths = load_list(dataset / "lists" / "val_report.txt", dataset)
    cal = load_records(cal_paths, dataset)
    report = load_records(report_paths, dataset)
    preds = read_preds(Path(args.preds))

    thresholds = calibrate_image_level(cal, preds, target_fpr=args.target_fpr)
    combined = binary_metrics(report, preds, thresholds, None)
    out = {
        "bench": "val_pilot",
        "model": args.model,
        "selection_data": "data/splits/val.jsonl slices (cal + report); bench_R/TEST never present",
        "calibrated_on": f"lists/val_cal.txt ({len(cal)} images), target FPR {args.target_fpr} per negative image",
        "target_fpr": args.target_fpr,
        "thresholds": thresholds,
        "splits": {
            "val": {
                "n": len(report),
                "positives": sum(1 for r in report if r.meta.get("has_fire") or r.meta.get("has_smoke")),
                "combined": combined,
                "fire": binary_metrics(report, preds, thresholds, "fire"),
                "smoke": binary_metrics(report, preds, thresholds, "smoke"),
                "box_ap": {"fire": box_ap(report, preds, "fire"), "smoke": box_ap(report, preds, "smoke")},
                "mean_iou": {"fire": mean_iou(report, preds, "fire", thresholds.get("fire", 1)),
                             "smoke": mean_iou(report, preds, "smoke", thresholds.get("smoke", 1))},
                "fpr_per_negative_image": fpr_per_negative(report, preds, thresholds),
            }
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "thresholds": thresholds,
                      "recall_at_fpr": combined["recall"], "recall_95ci": combined["recall_95ci"],
                      "fpr_per_negative": out["splits"]["val"]["fpr_per_negative_image"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
