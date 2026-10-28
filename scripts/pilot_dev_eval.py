"""bench_R DEV-only confirmation for pilot checkpoints (SEALED untouched).

Thresholds are calibrated on val_mini + val_flames (as for every bench_R number),
then each checkpoint is scored ONCE on bench_R DEV (300 images). DEV is used only
for this GO-gate confirmation and error analysis; the run is logged in
bench_results/BENCH_USAGE.md.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_bench import binary_metrics, calibrate_image_level, decision, load_val, wilson
from run_benchR import load_benchR
from firesight.config import PROJECT_ROOT

MODELS = {
    "P10": ("build/kaggle_weights/runs/P10/weights/best.onnx", 8),
    "P25": ("build/kaggle_weights/runs/P25/weights/best.onnx", 8),
    "P50": ("build/kaggle_weights/runs/P50/weights/best.onnx", 8),
    "BASE": ("models/best_fire_v3.onnx", 1),
}


def predict(onnx: Path, records, batch: int, tmp: Path) -> dict:
    from ultralytics import YOLO
    lst = tmp / f"paths_{onnx.stem}_{len(records)}.txt"
    lst.write_text("\n".join(str(PROJECT_ROOT / r.path) for r in records) + "\n", encoding="utf-8")
    # predictions must be keyed by the manifest id, not the file stem
    path_to_id = {str((PROJECT_ROOT / r.path).resolve()): r.id for r in records}
    model = YOLO(str(PROJECT_ROOT / onnx), task="detect")
    out = {}
    for res in model.predict(str(lst), imgsz=640, conf=0.01, batch=batch, device="cpu",
                             verbose=False, stream=True):
        from firesight.types import Box, Detection
        dets = [Detection("fire" if int(b.cls.item()) == 0 else "smoke",
                          float(b.conf.item()), Box(*[float(v) for v in b.xyxy[0].tolist()]))
                for b in res.boxes]
        out[path_to_id.get(str(Path(res.path).resolve()), Path(res.path).stem)] = dets
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/pilot_dev/dev_metrics.json")
    ap.add_argument("--target-fpr", type=float, default=0.02)
    args = ap.parse_args()
    tmp = PROJECT_ROOT / "build/pilot_dev"; tmp.mkdir(parents=True, exist_ok=True)

    records, member = load_benchR()
    dev = [r for r in records if member[r.id] == "dev"]
    assert len(dev) == 300, len(dev)
    cal = load_val()
    print("dev", len(dev), "cal", len(cal), flush=True)

    results = {}
    for name, (onnx, batch) in MODELS.items():
        if not (PROJECT_ROOT / onnx).exists():
            print("missing", onnx); continue
        cal_preds = predict(Path(onnx), cal, batch, tmp)
        thresholds = calibrate_image_level(cal, cal_preds, target_fpr=args.target_fpr)
        dev_preds = predict(Path(onnx), dev, batch, tmp)
        combined = binary_metrics(dev, dev_preds, thresholds, None)
        negatives = sum(1 for r in dev if not (r.meta.get("has_fire") or r.meta.get("has_smoke")))
        fp = sum(1 for r in dev if not (r.meta.get("has_fire") or r.meta.get("has_smoke"))
                 and decision(r, dev_preds.get(r.id, []), thresholds))
        results[name] = {
            "thresholds": thresholds,
            "recall": combined["recall"], "recall_95ci": combined["recall_95ci"],
            "precision": combined["precision"], "accuracy": combined["accuracy"],
            "fpr_per_negative": round(fp / negatives, 6) if negatives else None,
            "fpr_95ci": wilson(fp, negatives) if negatives else None,
            "n": len(dev), "negatives": negatives,
        }
        print(name, json.dumps(results[name]), flush=True)

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"bench": "bench_R", "split": "dev", "sealed_untouched": True,
                               "calibrated_on": "val_mini + val_flames",
                               "target_fpr": args.target_fpr, "models": results}, indent=2), encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
