"""Evaluate an external fire/smoke model on bench_R DEV only (no SEALED).

Calibrates one image-level max-confidence threshold on val_cal at FPR<=2%, then reports
DEV recall/FPR. Class names are mapped by name, so models with smoke=0/fire=1 work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from run_benchR import load_benchR


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def class_map(names, model_name):
    if not names:
        return {0: "fire", 1: "smoke"} if "P50" in model_name else {}
    out = {}
    for k, v in names.items():
        low = str(v).lower()
        if "fire" in low:
            out[int(k)] = "fire"
        elif "smoke" in low:
            out[int(k)] = "smoke"
    return out


def predict(model, paths, imgsz, batch, device, chunk=32):
    scores = {}
    for start in range(0, len(paths), chunk):
        part = [str(p) for p in paths[start:start + chunk]]
        for r in model.predict(part, imgsz=imgsz, conf=0.001, batch=batch,
                               device=device, verbose=False, stream=True):
            best = 0.0
            for b in r.boxes:
                best = max(best, float(b.conf.item()))
            scores[Path(r.path).stem] = best
    return scores


def calibrate(scores, labels, target):
    pos = [i for i, y in labels.items() if y]
    neg = [i for i, y in labels.items() if not y]
    best = None
    for thr in sorted(set(scores.values()), reverse=True):
        tp = sum(1 for i in pos if scores.get(i, 0) >= thr)
        fp = sum(1 for i in neg if scores.get(i, 0) >= thr)
        fpr = fp / max(len(neg), 1)
        rec = tp / max(len(pos), 1)
        if fpr <= target and (best is None or rec > best[0]):
            best = (rec, fpr, float(thr))
    return best


def metrics(scores, labels, thr):
    pos = [i for i, y in labels.items() if y]
    neg = [i for i, y in labels.items() if not y]
    tp = sum(1 for i in pos if scores.get(i, 0) >= thr)
    fp = sum(1 for i in neg if scores.get(i, 0) >= thr)
    return {"tp": tp, "fp": fp, "n_pos": len(pos), "n_neg": len(neg),
            "recall": tp / len(pos) if pos else None, "recall_95ci": wilson(tp, len(pos)) if pos else None,
            "fpr": fp / len(neg) if neg else None, "fpr_95ci": wilson(fp, len(neg)) if neg else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="external")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-fpr", type=float, default=0.02)
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    cmap = class_map(getattr(model, "names", None), args.name)
    print("class map", cmap, "names", getattr(model, "names", None), flush=True)

    # val_cal
    cal_rows = json.loads((PROJECT_ROOT / "build/verifier/sample_val_cal.json").read_text(encoding="utf-8"))
    cal_paths = [PROJECT_ROOT / r["path"] for r in cal_rows]
    cal_labels = {r["id"]: int(bool(r["labels"])) for r in cal_rows}
    # DEV
    records, member = load_benchR()
    dev = [r for r in records if member[r.id] == "dev"]
    dev_labels = {Path(r.path).stem: int(bool(r.meta.get("has_fire") or r.meta.get("has_smoke"))) for r in dev}
    dev_paths = [Path(r.path) for r in dev]

    cal_scores = predict(model, cal_paths, args.imgsz, args.batch, args.device)
    dev_scores = predict(model, dev_paths, args.imgsz, args.batch, args.device)
    best = calibrate(cal_scores, cal_labels, args.target_fpr)
    if best is None:
        raise SystemExit("no calibration point")
    rec_cal, fpr_cal, thr = best
    result = {
        "model": args.name, "path": args.model, "class_map": cmap,
        "calibrated_on": "val_cal (2,000), image-level max confidence",
        "target_val_cal_fpr": args.target_fpr, "threshold": thr,
        "val_cal": {"recall": rec_cal, "fpr": fpr_cal},
        "bench_R_DEV": metrics(dev_scores, dev_labels, thr),
        "bench_R_DEV_n": len(dev),
        "note": "DEV only; SEALED not read. One DEV run per candidate.",
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / f"build/external_eval/{args.name}_dev.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
