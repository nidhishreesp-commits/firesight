from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.data.manifest import load_manifest
from firesight.detectors.base import load_detector
from firesight.eval.metrics import _greedy_match, compute_ap
from firesight.eval.runner import cache_predictions, load_predictions
from firesight.logging_utils import setup_logging


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate models on an external (possibly non-permissive) dataset.")
    parser.add_argument("--manifest", default="data/splits/external_cctv_nc.jsonl")
    parser.add_argument("--registry", default="models/registry.yaml")
    parser.add_argument("--models", required=True)
    parser.add_argument("--out", default="artifacts/external_cctv_nc")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config()
    setup_logging("INFO")
    records = load_manifest(PROJECT_ROOT / args.manifest, root=PROJECT_ROOT)
    registry = {e["id"]: e for e in (yaml.safe_load(resolve(args.registry).read_text(encoding="utf-8")) or {}).get("models", [])}
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
        entry = registry[model_id]
        backend = entry.get("backend", "yolo")
        ref = entry["path"] if backend in {"grounding_dino", "grounding-dino", "gdino", "rtdetr_v2", "rtdetrv2", "rtdetr2"} else resolve(entry["path"])
        detector = load_detector(backend, ref, imgsz=cfg.detector.imgsz, device=args.device, name=model_id,
                                 license=entry.get("license", "unknown"), family=entry.get("family", backend),
                                 **({"preprocess_mode": entry["preprocess_mode"]} if entry.get("preprocess_mode") else {}),
                                 **({"threshold": entry["threshold"]} if entry.get("threshold") is not None else {}))
        cache = out / "predictions" / f"{model_id}.jsonl"
        try:
            cache_predictions(detector, records, cache, batch_size=1 if backend in {"rtdetr_v2", "rtdetrv2", "grounding_dino"} else 8)
        finally:
            detector.close()
        preds, meta = load_predictions(cache)
        row = {"model": model_id, "family": entry.get("family"), "license": entry.get("license"),
               "trained_on": entry.get("trained_on"), "ms_per_image": meta.get("ms_per_image"), "fps": meta.get("fps")}
        for cls in ("fire", "smoke"):
            row[f"{cls}_ap50"] = round(compute_ap(records, preds, cls, 0.5), 4)
            row[f"{cls}_ap50_95"] = round(sum(compute_ap(records, preds, cls, t) for t in [0.5 + 0.05 * i for i in range(10)]) / 10, 4)
            for thr in (0.25, 0.5):
                hits = pos = 0
                for rec in records:
                    gts = [g for g in rec.labels if g.label == cls]
                    if not gts:
                        continue
                    pos += 1
                    dets = [d for d in preds.get(rec.id, []) if d.label == cls and d.confidence >= thr]
                    mp, _ = _greedy_match(dets, gts, 0.5)
                    hits += int(any(mp))
                row[f"{cls}_image_recall@{thr}"] = round(hits / pos, 4) if pos else None
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
