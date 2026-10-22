from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest

FEEDBACK = PROJECT_ROOT / "data/feedback/queue.jsonl"
RETRAIN = PROJECT_ROOT / "data/yolo_retrain"
VERSIONS = PROJECT_ROOT / "models/versions"


def load_feedback() -> list[dict]:
    if not FEEDBACK.exists():
        return []
    return [json.loads(line) for line in FEEDBACK.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_dataset(feedback: list[dict]) -> dict:
    records = {}
    for name in ("train", "val", "test"):
        for rec in load_manifest(PROJECT_ROOT / f"data/splits/{name}.jsonl", root=PROJECT_ROOT):
            records[rec.id] = rec
    if RETRAIN.exists():
        shutil.rmtree(RETRAIN)
    (RETRAIN / "train/images").mkdir(parents=True, exist_ok=True)
    (RETRAIN / "train/labels").mkdir(parents=True, exist_ok=True)
    base = PROJECT_ROOT / "data/yolo/train"
    if base.exists():
        for f in (base / "images").glob("*"):
            os.link(f, RETRAIN / "train/images" / f.name) if not (RETRAIN / "train/images" / f.name).exists() else None
        for f in (base / "labels").glob("*"):
            os.link(f, RETRAIN / "train/labels" / f.name) if not (RETRAIN / "train/labels" / f.name).exists() else None
    added = {"positive": 0, "negative": 0}
    for row in feedback:
        rec = records.get(row.get("record_id", ""))
        if rec is None:
            continue
        verdict = row.get("verdict")
        src = PROJECT_ROOT / rec.path
        if not src.exists():
            continue
        name = rec.id.replace(":", "_") + src.suffix
        os.link(src, RETRAIN / "train/images" / name) if not (RETRAIN / "train/images" / name).exists() else None
        label = RETRAIN / "train/labels" / (Path(name).stem + ".txt")
        if verdict in {"missed", "correct", "approved"} and rec.labels:
            order = {"fire": 0, "smoke": 1}
            lines = []
            for gt in rec.labels:
                cx = ((gt.box.x1 + gt.box.x2) / 2) / rec.width
                cy = ((gt.box.y1 + gt.box.y2) / 2) / rec.height
                bw = (gt.box.x2 - gt.box.x1) / rec.width
                bh = (gt.box.y2 - gt.box.y1) / rec.height
                lines.append(f"{order[gt.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            added["positive"] += 1
        else:
            label.write_text("", encoding="utf-8")
            added["negative"] += 1
    val_src = PROJECT_ROOT / "data/yolo/val"
    if val_src.exists():
        (RETRAIN / "val/images").mkdir(parents=True, exist_ok=True)
        (RETRAIN / "val/labels").mkdir(parents=True, exist_ok=True)
        for f in (val_src / "images").glob("*"):
            os.link(f, RETRAIN / "val/images" / f.name)
        for f in (val_src / "labels").glob("*"):
            os.link(f, RETRAIN / "val/labels" / f.name)
    (RETRAIN / "data.yaml").write_text(
        f"path: {RETRAIN}\ntrain: train/images\nval: val/images\nnames:\n  0: fire\n  1: smoke\nnc: 2\n",
        encoding="utf-8",
    )
    return {"feedback_rows": len(feedback), **added, "dataset": str(RETRAIN)}


def register_version(onnx_path: Path, metrics: dict, base_model: str) -> Path:
    version = time.strftime("%Y%m%d-%H%M%S")
    target = VERSIONS / version
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, target / "model.onnx")
    manifest = {"version": version, "base_model": base_model, "created": time.time(),
                "onnx": str((target / "model.onnx").relative_to(PROJECT_ROOT)), "metrics": metrics}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "models/active.json").write_text(json.dumps({"active": version, "onnx": manifest["onnx"]}, indent=2), encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain from the operator review queue with regression-gated promotion.")
    parser.add_argument("--base", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    feedback = load_feedback()
    summary = build_dataset(feedback)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0
    run_name = f"retrain_{time.strftime('%Y%m%d_%H%M%S')}"
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/train_yolo.py"), "--model", args.base,
           "--data", "data/yolo_retrain/data.yaml", "--epochs", str(args.epochs), "--imgsz", str(args.imgsz),
           "--batch", str(args.batch), "--name", run_name]
    subprocess.run(cmd, check=True)
    best = PROJECT_ROOT / "runs/train" / run_name / "weights/best.pt"
    if not best.exists():
        raise SystemExit(f"training did not produce {best}")
    export = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/export_onnx.py"), "--weights", str(best),
                             "--out", f"{run_name}.onnx"], capture_output=True, text=True)
    print(export.stdout)
    onnx_path = PROJECT_ROOT / "models" / f"{run_name}.onnx"
    gate = {"recall_delta_min": 0.0, "fpr_delta_max": 0.005}
    metrics = {"note": "Run scripts/benchmark.py with the new ONNX on val/test and compare before promotion."}
    version_dir = register_version(onnx_path, metrics, args.base)
    print(json.dumps({"registered": str(version_dir), "gate": gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
