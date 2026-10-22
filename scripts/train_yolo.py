from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune / train a YOLO or RT-DETR detector on the unified dataset.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base weights (yolo11n.pt, yolo11s.pt, yolo12n.pt, rtdetr-l.pt, ...)")
    parser.add_argument("--data", default="data/yolo/data.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO, RTDETR

    name = args.name or Path(args.model).stem
    model_cls = RTDETR if "rtdetr" in args.model.lower() or "rt-detr" in args.model.lower() else YOLO
    model = model_cls(args.model)
    results = model.train(
        data=str(PROJECT_ROOT / args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_ROOT / args.project),
        name=name,
        workers=args.workers,
        patience=args.patience,
        seed=0,
        deterministic=True,
        amp=False,
        plots=False,
        exist_ok=True,
        resume=args.resume,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    summary = {"model": args.model, "data": args.data, "epochs": args.epochs, "imgsz": args.imgsz,
               "batch": args.batch, "best": str(best), "run_dir": str(results.save_dir)}
    (Path(results.save_dir) / "firesight_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
