from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.data.manifest import load_manifest
from firesight.detectors.base import load_detector
from firesight.eval.metrics import evaluate_predictions, select_operating_point
from firesight.eval.runner import (
    bootstrap_ci,
    cache_predictions,
    load_predictions,
    per_source_metrics,
    plot_pr,
)
from firesight.logging_utils import setup_logging

log = logging.getLogger("benchmark")


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate
    return (PROJECT_ROOT / p).resolve()


def _load_registry(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {entry["id"]: entry for entry in data.get("models", [])}


def _make_detector(entry: dict, cfg, device: str):
    backend = entry.get("backend", "yolo")
    path = entry["path"] if backend in {"grounding_dino", "grounding-dino", "gdino", "rtdetr_v2", "rtdetrv2", "rtdetr2"} else _resolve(entry["path"])
    return load_detector(
        backend,
        path,
        imgsz=entry.get("imgsz", cfg.detector.imgsz),
        iou=cfg.detector.iou,
        max_det=cfg.detector.max_det,
        device=device,
        name=entry["id"],
        license=entry.get("license", "unknown"),
        family=entry.get("family", entry.get("backend", "unknown")),
        **({"preprocess_mode": entry["preprocess_mode"]} if entry.get("preprocess_mode") else {}),
        **({"threshold": entry["threshold"]} if entry.get("threshold") is not None else {}),
    )


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark fire/smoke detectors on the frozen test split.")
    parser.add_argument("--registry", default="models/registry.yaml")
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: all)")
    parser.add_argument("--manifest", default=None, help="Test manifest (default: config benchmark.test_manifest)")
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--out", default="artifacts/benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Deterministic prefix limit for quick runs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-fpr", type=float, default=None)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.app.log_level)
    registry = _load_registry(_resolve(args.registry))
    selected = list(registry) if not args.models else [m.strip() for m in args.models.split(",") if m.strip()]
    missing = [m for m in selected if m not in registry]
    if missing:
        raise SystemExit(f"Unknown model ids: {missing}")

    test_path = _resolve(args.manifest or cfg.benchmark.test_manifest)
    val_path = _resolve(args.val_manifest or cfg.benchmark.val_manifest)
    test_records = load_manifest(test_path, root=PROJECT_ROOT)
    val_records = load_manifest(val_path, root=PROJECT_ROOT)
    if args.limit:
        test_records = test_records[:args.limit]
    target_fpr = args.target_fpr if args.target_fpr is not None else cfg.benchmark.target_fpr
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps({
        "test_manifest": str(test_path), "val_manifest": str(val_path),
        "test_images": len(test_records), "val_images": len(val_records),
        "target_fpr": target_fpr, "limit": args.limit, "device": args.device,
    }, indent=2), encoding="utf-8")

    results = []
    for model_id in selected:
        entry = registry[model_id]
        cache_dir = out_dir / "predictions"
        test_cache = cache_dir / f"{model_id}_test.jsonl"
        val_cache = cache_dir / f"{model_id}_val.jsonl"
        log.info("=== %s (%s) ===", model_id, entry.get("path"))
        detector = _make_detector(entry, cfg, args.device)
        try:
            cache_predictions(detector, test_records, test_cache, batch_size=args.batch_size, overwrite=args.overwrite_cache)
            cache_predictions(detector, val_records, val_cache, batch_size=args.batch_size, overwrite=args.overwrite_cache)
        finally:
            detector.close()

        test_preds, test_meta = load_predictions(test_cache)
        val_preds, _ = load_predictions(val_cache)
        op = select_operating_point(val_records, val_preds, test_records, test_preds,
                                    target_fpr=target_fpr, iou_thr=cfg.benchmark.iou_threshold)
        metrics = evaluate_predictions(test_records, test_preds, op, iou_thr=cfg.benchmark.iou_threshold)
        metrics["sources"] = per_source_metrics(test_records, test_preds, op.thresholds, cfg.benchmark.iou_threshold)
        metrics["bootstrap"] = bootstrap_ci(test_records, test_preds, op.thresholds,
                                            iou_thr=cfg.benchmark.iou_threshold,
                                            n=cfg.benchmark.bootstrap_samples)
        metrics["model"] = {
            "id": model_id,
            "path": entry.get("path"),
            "family": entry.get("family"),
            "license": entry.get("license"),
            "trained_on": entry.get("trained_on"),
            "notes": entry.get("notes"),
            "model_size_mb": test_meta.get("model_size_mb"),
            "device": test_meta.get("device"),
            "ms_per_image": test_meta.get("ms_per_image"),
            "fps": test_meta.get("fps"),
        }
        metrics["benchmark_setup"] = {
            "test_manifest": str(test_path),
            "val_manifest": str(val_path),
            "target_fpr": target_fpr,
            "contamination_risk": entry.get("trained_on", ""),
        }
        model_out = out_dir / model_id
        model_out.mkdir(parents=True, exist_ok=True)
        (model_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (model_out / "failures.json").write_text(json.dumps(metrics["failures"], indent=2), encoding="utf-8")
        plot_pr(model_id, metrics["pr_curve"], model_out / "pr_curve.png")
        results.append(metrics)
        log.info("%s recall=%.4f fpr=%.4f thresholds=%s fps=%.2f",
                 model_id, op.image_recall, op.fpr, op.thresholds, test_meta.get("fps", 0))

    table = []
    for m in results:
        table.append({
            "model": m["model"]["id"],
            "family": m["model"]["family"],
            "license": m["model"]["license"],
            "size_mb": m["model"]["model_size_mb"],
            "fps_cpu": m["model"]["fps"],
            "thresholds": m["operating_point"]["thresholds"],
            "recall_at_fpr": m["operating_point"]["image_recall"],
            "fpr": m["operating_point"]["fpr"],
            "image_precision": m["operating_point"]["image_precision"],
            "fire_ap50": m["per_class"]["fire"]["ap50"],
            "fire_ap50_95": m["per_class"]["fire"]["ap50_95"],
            "smoke_ap50": m["per_class"]["smoke"]["ap50"],
            "smoke_ap50_95": m["per_class"]["smoke"]["ap50_95"],
            "fire_box_recall": m["operating_point"]["box_recall"].get("fire"),
            "smoke_box_recall": m["operating_point"]["box_recall"].get("smoke"),
            "recall_95ci": m["bootstrap"].get("recall_95ci"),
            "trained_on": m["model"]["trained_on"],
        })
    table.sort(key=lambda r: (r["recall_at_fpr"] or 0, -(r["fpr"] or 0)), reverse=True)
    (out_dir / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    with (out_dir / "comparison.csv").open("w", encoding="utf-8") as fh:
        import csv
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()) if table else [])
        writer.writeheader()
        writer.writerows(table)
    md = ["# Benchmark comparison", "",
          f"Frozen test manifest: `{test_path}` ({len(test_records)} images). "
          f"Thresholds calibrated on val at target image-level FPR <= {target_fpr}.", "",
          "| Model | Family | License | Size MB | CPU FPS | Fire AP50 | Smoke AP50 | Fire AP50-95 | Smoke AP50-95 | Recall @ FPR | FPR | Fire box R | Smoke box R |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['model']} | {r['family']} | {r['license']} | {_fmt(r['size_mb'], 1)} | {_fmt(r['fps_cpu'], 1)} | "
                  f"{_fmt(r['fire_ap50'])} | {_fmt(r['smoke_ap50'])} | {_fmt(r['fire_ap50_95'])} | {_fmt(r['smoke_ap50_95'])} | "
                  f"{_fmt(r['recall_at_fpr'])} | {_fmt(r['fpr'], 6)} | {_fmt(r['fire_box_recall'])} | {_fmt(r['smoke_box_recall'])} |")
    md += ["", "Contamination note: models marked `unknown` may have trained on D-Fire/FASDD or their source images. "
           "Those rows are not a clean generalization estimate."]
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
