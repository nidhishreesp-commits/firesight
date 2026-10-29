"""Train one point of the Phase 3b learning curve (also used by the local CPU dry run).

Properties required by the plan:
  * fixed epoch budget, early-stop patience 5 on the internal val loss;
  * a checkpoint after **every** epoch (`save_period=1`) and `--resume` across
    Kaggle's 12 h sessions;
  * measured throughput (images/s, s/epoch) written to `run_summary.json`;
  * ONNX export of the best checkpoint;
  * selection data is `lists/val_early.txt` (a slice of `data/splits/val.jsonl`);
    bench_R / TEST are never present in the dataset.

Usage (Kaggle):
    python pilot_train.py --dataset /kaggle/input/firesight-pilot/dataset \
        --fraction 10 --model yolo11s.pt --epochs 15 --device 0 --batch 64 --out /kaggle/working/runs/P10
Resume after a session restart:
    python pilot_train.py ... --out /kaggle/working/runs/P10 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml


def resolve_list(dataset: Path, name: str, out: Path) -> Path:
    """Rewrite a list to absolute paths that exist on this machine.

    Package lists are written relative to the dataset root (`images/<file>`), which is
    what Kaggle needs; ultralytics wants resolvable paths, so we materialise an
    absolute copy under the run directory instead of mutating the read-only input.
    """
    src = dataset / "lists" / name
    resolved = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = Path(line.strip())
        if not p.is_absolute():
            p = dataset / p
        if not p.exists():
            fallback = dataset / "images" / Path(line).name
            if fallback.exists():
                p = fallback
        resolved.append(str(p.resolve()).replace("\\", "/"))
    dst = out / "lists" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    return dst


def build_data_yaml(dataset: Path, fraction: str, run_dir: Path) -> Path:
    """Write <run_dir>/data.yaml with absolute train/val list paths."""
    base = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    base["path"] = str(dataset.resolve()).replace("\\", "/")
    base["train"] = str(resolve_list(dataset, f"train_{fraction}.txt", run_dir).resolve())
    base["val"] = str(resolve_list(dataset, "val_early.txt", run_dir).resolve())
    yaml_path = run_dir / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return yaml_path


def count_images(list_path: Path) -> int:
    return sum(1 for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip())


def throughput_from_results(results_csv: Path, n_train: int) -> dict:
    if not results_csv.exists():
        return {}
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    times = []
    for row in rows:
        try:
            times.append(float(row.get("time") or 0))
        except (TypeError, ValueError):
            continue
    if len(times) < 2:
        return {"epochs_recorded": len(times)}
    # `time` is cumulative within a session and resets on --resume, so only keep
    # plausible intra-session deltas (>= 1 s) and use the median.
    deltas = [b - a for a, b in zip(times, times[1:]) if b - a >= 1.0]
    mean_s = sorted(deltas)[len(deltas) // 2] if deltas else None
    return {
        "epochs_recorded": len(times),
        "total_train_s": round(times[-1], 2),
        "median_s_per_epoch": round(mean_s, 2) if mean_s else None,
        "images_per_s": round(n_train / mean_s, 2) if mean_s else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="packaged dataset dir (contains data.yaml + lists/)")
    parser.add_argument("--fraction", default="10", choices=["10", "25", "50", "100"])
    parser.add_argument("--model", default="yolo11s.pt", help="base weights or a .yaml to build from scratch")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr0", type=float, default=0.01, help="initial LR (use ~0.001 for fine-tunes)")
    parser.add_argument("--data-yaml", default=None, help="use an existing data.yaml (custom positives+negatives)")
    parser.add_argument("--n-train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="enable AMP (GPU); leave off for CPU dry runs")
    parser.add_argument("--out", required=True, help="run directory (persist this under /kaggle/working)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="dry-run only: abort after N epochs so a later --resume is exercised")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out / "weights" / "last.pt"
    best = out / "weights" / "best.pt"
    if args.data_yaml:
        data_yaml = Path(args.data_yaml)
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        n_train = args.n_train
        if not n_train:
            train_ref = cfg.get("train")
            tp = Path(train_ref) if train_ref else None
            if tp is not None:
                if not tp.is_absolute():
                    tp = Path(cfg.get("path") or data_yaml.parent) / tp
                n_train = count_images(tp)
    else:
        data_yaml = build_data_yaml(dataset, args.fraction, out)
        n_train = count_images(out / "lists" / f"train_{args.fraction}.txt")

    from ultralytics import YOLO

    resumed = args.resume and last.exists()
    model = YOLO(str(last)) if resumed else YOLO(args.model)
    print(json.dumps({"fraction": args.fraction, "n_train": n_train, "resumed_from": str(last) if resumed else None,
                      "model": args.model, "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
                      "device": args.device}), flush=True)

    if args.stop_after_epoch:
        class _Stop(Exception):
            pass

        def _stop(trainer):
            if trainer.epoch + 1 >= args.stop_after_epoch:
                raise _Stop()

        # on_fit_epoch_end runs after validation and save_model(), so last.pt exists when we abort.
        model.add_callback("on_fit_epoch_end", _stop)

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            lr0=args.lr0,
            seed=args.seed,
            deterministic=True,
            amp=args.amp,
            plots=False,
            save=True,
            save_period=1,
            exist_ok=True,
            resume=resumed,
            project=str(out.parent.resolve()),
            name=args.name or out.name,
        )
    except Exception as exc:
        if type(exc).__name__ != "_Stop":
            raise
        results = None
        print(json.dumps({"interrupted_after_epoch": args.stop_after_epoch}), flush=True)
    run_dir = Path(results.save_dir) if results is not None else out

    trainer = getattr(model, "trainer", None)
    last_epoch_s = getattr(trainer, "epoch_time", None) if trainer is not None else None
    summary = {
        "fraction": args.fraction,
        "n_train": n_train,
        "model": args.model,
        "epochs_requested": args.epochs,
        "resumed": resumed,
        "run_dir": str(run_dir),
        "best_pt": str(run_dir / "weights" / "best.pt"),
        "last_pt": str(run_dir / "weights" / "last.pt"),
        "throughput": throughput_from_results(run_dir / "results.csv", n_train),
        "last_epoch_time_s": round(last_epoch_s, 2) if last_epoch_s else None,
        "last_epoch_images_per_s": round(n_train / last_epoch_s, 2) if last_epoch_s else None,
        "selection_data": "lists/val_early.txt (slice of data/splits/val.jsonl)",
    }
    if args.export_onnx and (run_dir / "weights" / "best.pt").exists():
        export_model = YOLO(str(run_dir / "weights" / "best.pt"))
        onnx_path = export_model.export(format="onnx", imgsz=args.imgsz, opset=17, dynamic=False, simplify=False)
        summary["onnx"] = str(onnx_path)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
