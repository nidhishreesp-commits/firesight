from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.data.manifest import load_manifest
from firesight.eval.metrics import evaluate_predictions, select_operating_point
from firesight.eval.runner import load_predictions
from firesight.verifier import CropVerifier


def annotate(records, predictions, verifier: CropVerifier):
    out = {}
    for i, rec in enumerate(records, 1):
        img = cv2.imread(str(PROJECT_ROOT / rec.path))
        dets = predictions.get(rec.id, [])
        if img is None or not dets:
            out[rec.id] = dets
            continue
        kept, rows = verifier.verify(img, dets)
        for det, row in zip(dets, rows):
            det.meta["verifier"] = {k: row[k] for k in verifier.classes}
        out[rec.id] = dets
        if i % 500 == 0:
            print(f"verified {i}/{len(records)}", flush=True)
    return out


def filter_threshold(predictions, threshold: float):
    out = {}
    for rec_id, dets in predictions.items():
        out[rec_id] = [d for d in dets if d.meta.get("verifier", {}).get("lookalike", 0.0) < threshold]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate second-stage crop verifier on cached detector proposals.")
    parser.add_argument("--val-manifest", default="data/splits/val_bench.jsonl")
    parser.add_argument("--test-manifest", default="data/splits/test_bench.jsonl")
    parser.add_argument("--val-predictions", default="artifacts/benchmark_stage1/predictions/ssprod_best_fire_v3_val.jsonl")
    parser.add_argument("--test-predictions", default="artifacts/benchmark_stage1/predictions/ssprod_best_fire_v3_test.jsonl")
    parser.add_argument("--verifier", default="models/verifier.onnx")
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--out", default="artifacts/verifier_ablation")
    args = parser.parse_args()
    cfg = load_config()
    val = load_manifest(PROJECT_ROOT / args.val_manifest, root=PROJECT_ROOT)
    test = load_manifest(PROJECT_ROOT / args.test_manifest, root=PROJECT_ROOT)
    val_preds, _ = load_predictions(PROJECT_ROOT / args.val_predictions)
    test_preds, _ = load_predictions(PROJECT_ROOT / args.test_predictions)
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    base_op = select_operating_point(val, val_preds, test, test_preds, target_fpr=args.target_fpr)
    base_metrics = evaluate_predictions(test, test_preds, base_op)
    rows.append({"verifier_threshold": None, **base_metrics["operating_point"],
                 "recall_at_fpr": base_metrics["recall_at_fpr"], "per_class": base_metrics["per_class"]})

    verifier = CropVerifier(PROJECT_ROOT / args.verifier, classes=cfg.verifier.classes, threshold=1.1)
    print("scoring val proposals", flush=True)
    v_val = annotate(val, val_preds, verifier)
    print("scoring test proposals", flush=True)
    v_test = annotate(test, test_preds, verifier)

    for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        f_val = filter_threshold(v_val, threshold)
        f_test = filter_threshold(v_test, threshold)
        op = select_operating_point(val, f_val, test, f_test, target_fpr=args.target_fpr)
        metrics = evaluate_predictions(test, f_test, op)
        row = {"verifier_threshold": threshold, **metrics["operating_point"],
               "recall_at_fpr": metrics["recall_at_fpr"], "per_class": metrics["per_class"]}
        rows.append(row)
        (out_dir / f"verifier_{threshold:.2f}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k: row[k] for k in ("verifier_threshold", "image_recall", "fpr", "thresholds")}), flush=True)

    (out_dir / "ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["# Second-stage verifier ablation", "",
          f"Base detector: ssprod_best_fire_v3. Val target FPR: {args.target_fpr}. Metrics on frozen test_bench (never used for training).", "",
          "| Verifier lookalike threshold | Image recall | FPR | Fire box R | Smoke box R | Recall @ FPR 0.1% | Recall @ FPR 1% |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        raf = r["recall_at_fpr"]
        md.append(f"| {r['verifier_threshold'] if r['verifier_threshold'] is not None else 'none (base)'} | "
                  f"{r['image_recall']:.4f} | {r['fpr']:.4f} | {r['box_recall'].get('fire', 0):.4f} | "
                  f"{r['box_recall'].get('smoke', 0):.4f} | {raf.get('0.001', {}).get('recall', 0):.4f} | "
                  f"{raf.get('0.01', {}).get('recall', 0):.4f} |")
    (out_dir / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
