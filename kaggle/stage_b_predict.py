"""Run any YOLO checkpoint on a list of images and write image-level detections JSONL.

Used by Stage B for val_cal, val_report and the independent negative set.  The model can be a
fresh E0/E1 checkpoint; the output format is the same as pilot_eval.py expects.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    paths = [l.strip().lstrip('\ufeff') for l in Path(args.list).read_text(encoding="utf-8").splitlines() if l.strip().lstrip('\ufeff')]
    model = YOLO(args.model, task="detect")
    names = {int(k): str(v).lower() for k, v in (model.names or {}).items()}
    t0 = time.time()
    n = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for start in range(0, len(paths), args.chunk):
            chunk = paths[start:start + args.chunk]
            for result in model.predict(chunk, imgsz=args.imgsz, conf=args.conf, batch=args.batch,
                                        device=args.device, half=args.half, verbose=False, stream=True):
                dets = []
                for b in result.boxes:
                    name = names.get(int(b.cls.item()), "")
                    label = "fire" if "fire" in name else "smoke" if "smoke" in name else None
                    if label is None:
                        continue
                    dets.append({"label": label, "confidence": round(float(b.conf.item()), 6),
                                 "box": [float(v) for v in b.xyxy[0].tolist()]})
                fh.write(json.dumps({"id": Path(result.path).stem, "detections": dets}) + "\n")
                n += 1
                if n % 1000 == 0:
                    print(n, "/", len(paths), round(n / max(time.time() - t0, 1e-6), 2), "img/s", flush=True)
    print(json.dumps({"images": n, "seconds": round(time.time() - t0, 1), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
