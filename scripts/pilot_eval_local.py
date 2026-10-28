"""Evaluate pilot checkpoints locally on CPU (ONNX), avoiding Kaggle GPU OOM.

Exports best.pt -> ONNX with NMS baked in, predicts the val_cal+val_report lists,
then runs the same pilot_eval.py metric code used for bench_R.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

def read_list(dataset: Path, name: str) -> list[Path]:
    out = []
    for line in (dataset / "lists" / name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = Path(line.strip())
            out.append(p if p.is_absolute() else dataset / p)
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="build/kaggle_dataset")
    ap.add_argument("--weights-dir", default=str(PROJECT_ROOT / "build/kaggle_weights/runs"))
    ap.add_argument("--out-dir", default="build/pilot_eval")
    ap.add_argument("--fractions", default="P10,P25,P50")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    from ultralytics import YOLO

    dataset = (PROJECT_ROOT / args.dataset).resolve()
    out = (PROJECT_ROOT / args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    paths = read_list(dataset, "val_cal.txt") + read_list(dataset, "val_report.txt")
    print("val images", len(paths), flush=True)

    for frac in [f.strip() for f in args.fractions.split(",") if f.strip()]:
        pt = Path(args.weights_dir) / frac / "weights" / "best.pt"
        onnx = pt.with_suffix(".onnx")
        if not onnx.exists():
            t = time.time()
            YOLO(str(pt)).export(format="onnx", imgsz=args.imgsz, opset=17, nms=False,
                                 simplify=True, dynamic=True)
            print("exported", onnx, round(time.time() - t, 1), "s", flush=True)
        model = YOLO(str(onnx), task="detect")
        rows, t0 = [], time.time()
        # IMPORTANT: pass a .txt list file, not a Python list. A list makes ultralytics
        # stack every image into one (N,640,640,3) array (8.15 GiB for 7,119 images).
        list_file = out / "val_paths.txt"
        list_file.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        for res in model.predict(str(list_file), imgsz=args.imgsz, conf=0.01,
                                 batch=args.batch, device="cpu", verbose=False, stream=True):
            dets = [{"label": "fire" if int(b.cls.item()) == 0 else "smoke",
                     "confidence": round(float(b.conf.item()), 6),
                     "box": [float(v) for v in b.xyxy[0].tolist()]} for b in res.boxes]
            rows.append({"id": Path(res.path).stem, "detections": dets})
        preds = out / f"preds_{frac}.jsonl"
        preds.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print(frac, "preds", len(rows), "in", round(time.time() - t0, 1), "s",
              "->", round(len(rows) / max(time.time() - t0, 1e-6), 1), "img/s", flush=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "kaggle/pilot_eval.py"),
                        "--dataset", str(dataset), "--preds", str(preds), "--model", frac,
                        "--target-fpr", "0.02", "--out", str(out / f"metrics_{frac}.json")], check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
