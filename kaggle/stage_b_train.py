"""E0/E1 paired fresh training with recall-at-FPR checkpoint selection.

Differences from pilot_train.py that matter for this experiment:
  * fresh `yolo11s.pt` (not P50), standard fresh-training LR (lr0=0.01, cosine);
  * 20 epochs, patience 10 on **recall at FPR <= 2% on val_cal**, not val loss;
  * every epoch is evaluated on val_cal and the best recall checkpoint is saved as
    `weights/best_recall.pt`;
  * training/validation lists are explicit, so E0 and E1 differ only in the train list.

Usage (GPU):
    python stage_b_train.py --dataset /kaggle/input/firesight-pilot/dataset \
        --train-list /kaggle/working/lists/e0_train.txt \
        --val-list /kaggle/input/firesight-pilot/dataset/lists/val_early.txt \
        --val-cal-list /kaggle/input/firesight-pilot/dataset/lists/val_cal.txt \
        --out /kaggle/working/runs/E0 --epochs 20 --patience 10 --device 0 --amp
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def resolve(line: str, dataset: Path) -> Path:
    p = Path(line.strip())
    if not p.is_absolute():
        p = dataset / p
    if not p.exists():
        fallback = dataset / "images" / p.name
        if fallback.exists():
            p = fallback
    return p.resolve()


def resolve_list(src: Path, dataset: Path, dst: Path) -> list[Path]:
    paths = [resolve(l, dataset) for l in src.read_text(encoding="utf-8").splitlines() if l.strip().lstrip('\ufeff')]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from {src}; first: {missing[:3]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    return paths


def label_for(image: Path) -> Path | None:
    text = str(image).replace("\\", "/")
    if "/images/" in text:
        cand = Path(text.replace("/images/", "/labels/")).with_suffix(".txt")
        if cand.exists():
            return cand
    return None


def load_binary_labels(paths: list[Path]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for p in paths:
        lbl = label_for(p)
        positive = False
        if lbl is not None and lbl.exists():
            positive = any(line.strip() for line in lbl.read_text(encoding="utf-8").splitlines())
        labels[p.stem] = int(positive)
    return labels


def predict_scores(model, paths: list[Path], imgsz: int, batch: int, device: str,
                   half: bool, tmp_list: Path) -> dict[str, float]:
    tmp_list.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    scores: dict[str, float] = {}
    for result in model.predict(str(tmp_list), imgsz=imgsz, conf=0.001, batch=batch,
                                device=device, half=half, verbose=False, stream=True):
        scores[Path(result.path).stem] = max(
            (float(b.conf.item()) for b in result.boxes), default=0.0)
    return scores


def recall_at_fpr(scores: dict[str, float], labels: dict[str, int], target: float) -> tuple[float, float, float | None]:
    """Maximum image-level recall with FPR per negative <= target."""
    n_pos = sum(labels.values())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0, 0.0, None
    best_rec, best_fpr, best_thr = 0.0, 1.0, None
    for thr in sorted(set(scores.values()), reverse=True):
        tp = sum(1 for k, v in scores.items() if v >= thr and labels.get(k, 0) == 1)
        fp = sum(1 for k, v in scores.items() if v >= thr and labels.get(k, 0) == 0)
        rec = tp / n_pos
        fpr = fp / n_neg
        if fpr <= target and (rec > best_rec or (rec == best_rec and fpr < best_fpr)):
            best_rec, best_fpr, best_thr = rec, fpr, thr
    return best_rec, best_fpr, best_thr


def throughput(results_csv: Path, n_train: int) -> dict:
    if not results_csv.exists():
        return {}
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    times = []
    for row in rows:
        try:
            times.append(float(row.get("time") or 0))
        except (TypeError, ValueError):
            pass
    deltas = [b - a for a, b in zip(times, times[1:]) if b - a >= 1.0]
    med = sorted(deltas)[len(deltas) // 2] if deltas else None
    return {
        "epochs_recorded": len(times),
        "median_s_per_epoch": round(med, 2) if med else None,
        "images_per_s": round(n_train / med, 2) if med else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--train-list", required=True)
    ap.add_argument("--val-list", required=True, help="internal training val (val_early)")
    ap.add_argument("--val-cal-list", required=True, help="checkpoint-selection list (val_cal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--eval-batch", type=int, default=32)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lists").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    train_paths = resolve_list(Path(args.train_list), dataset, out / "lists/train.txt")
    val_paths = resolve_list(Path(args.val_list), dataset, out / "lists/val.txt")
    cal_paths = resolve_list(Path(args.val_cal_list), dataset, out / "lists/val_cal.txt")
    cal_labels = load_binary_labels(cal_paths)
    print(json.dumps({"n_train": len(train_paths), "n_val": len(val_paths),
                      "n_val_cal": len(cal_paths),
                      "val_cal_positives": sum(cal_labels.values())}), flush=True)

    data_yaml = out / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(dataset.resolve()).replace("\\", "/"),
        "train": str((out / "lists/train.txt").resolve()).replace("\\", "/"),
        "val": str((out / "lists/val.txt").resolve()).replace("\\", "/"),
        "names": {0: "fire", 1: "smoke"},
        "nc": 2,
    }, sort_keys=False), encoding="utf-8")

    from ultralytics import YOLO

    last = out / "weights/last.pt"
    resumed = args.resume and last.exists()
    model = YOLO(str(last)) if resumed else YOLO(args.model)
    print(json.dumps({"resumed": resumed, "model": args.model, "epochs": args.epochs,
                      "patience": args.patience, "device": args.device, "amp": args.amp}), flush=True)

    best = {"recall": -1.0, "fpr": 1.0, "epoch": -1, "threshold": None}
    stale = 0
    metrics_path = out / "selection_metrics.jsonl"
    tmp_list = out / "lists/val_cal_eval.txt"

    def select(trainer):
        nonlocal best, stale
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        # Ultralytics fires on_fit_epoch_end once more for the final best.pt validation;
        # do not let that extra call count as a training epoch.
        if epoch > args.epochs:
            return
        last_pt = Path(getattr(trainer, "last", out / "weights/last.pt"))
        if not last_pt.exists():
            return
        eval_model = YOLO(str(last_pt))
        scores = predict_scores(eval_model, cal_paths, args.imgsz, args.eval_batch,
                                args.device, bool(args.amp and args.device != "cpu"), tmp_list)
        rec, fpr, thr = recall_at_fpr(scores, cal_labels, args.target_fpr)
        row = {"epoch": epoch, "recall_at_fpr": round(rec, 6), "fpr": round(fpr, 6),
               "threshold": None if thr is None else round(float(thr), 6),
               "n_pos": int(sum(cal_labels.values())), "n_neg": len(cal_labels) - int(sum(cal_labels.values())),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        improved = rec > best["recall"] or (rec == best["recall"] and fpr < best["fpr"])
        if improved:
            best = {"recall": rec, "fpr": fpr, "epoch": epoch, "threshold": thr}
            shutil.copy2(last_pt, out / "weights/best_recall.pt")
            (out / "best_recall.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            stale = 0
        else:
            stale += 1
        print(json.dumps({"epoch": epoch, "recall_at_fpr": round(rec, 4),
                          "fpr": round(fpr, 4), "best_epoch": best["epoch"],
                          "stale": stale}), flush=True)
        del eval_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch,
                              "best_epoch": best["epoch"]}), flush=True)
            trainer.stop = True

    model.add_callback("on_fit_epoch_end", select)
    t0 = time.time()
    try:
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, workers=args.workers, patience=1000,
            lr0=args.lr0, lrf=args.lrf, cos_lr=True, warmup_epochs=3.0,
            seed=args.seed, deterministic=True, amp=args.amp,
            mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
            degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4,
            plots=False, save=True, save_period=1, exist_ok=True, resume=resumed,
            project=str(out.parent.resolve()), name=args.name or out.name,
        )
    finally:
        elapsed = time.time() - t0
        run_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", out))
        summary = {
            "n_train": len(train_paths), "n_val": len(val_paths), "n_val_cal": len(cal_paths),
            "val_cal_positives": int(sum(cal_labels.values())),
            "model": args.model, "epochs_requested": args.epochs, "patience": args.patience,
            "lr0": args.lr0, "lrf": args.lrf, "cos_lr": True, "warmup_epochs": 3.0,
            "seed": args.seed, "imgsz": args.imgsz, "batch": args.batch, "amp": args.amp,
            "selection": "max recall at FPR<=2% on val_cal (image-level)",
            "best": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
            "wall_time_s": round(elapsed, 1), "throughput": throughput(run_dir / "results.csv", len(train_paths)),
            "best_recall_pt": str(out / "weights/best_recall.pt"),
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
