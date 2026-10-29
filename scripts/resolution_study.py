"""Resolution / tiling study on P50's remaining misses (CPU, frozen model).

Modes: 640 (baseline), 960, 1280, tiled-2x2@640 (20% overlap, merged with NMS).
Reports per-mode recall on the target misses, a control set of detected positives, and
FPR on a random negative control; measures throughput (img/s).
"""
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from firesight.config import PROJECT_ROOT
from pilot_eval import load_list, load_records, read_preds
from run_bench import decision, wilson

ONNX = PROJECT_ROOT/"build/kaggle_weights/runs/P50/weights/best.onnx"
OUT = PROJECT_ROOT/"build/resolution_study"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def nms(dets, thr=0.5):
    dets = sorted(dets, key=lambda d: d[1], reverse=True)
    keep = []
    for d in dets:
        if all(iou(d[2], k[2]) < thr or d[0] != k[0] for k in keep):
            keep.append(d)
    return keep


def run_mode(paths, mode):
    from ultralytics import YOLO
    model = YOLO(str(ONNX), task="detect")
    t0 = time.time()
    out = {}
    if mode != "tiled":
        imgsz = int(mode)
        lst = OUT/f"paths_{mode}.txt"
        lst.write_text("\n".join(str(p) for p in paths)+"\n", encoding="utf-8")
        for res in model.predict(str(lst), imgsz=imgsz, conf=0.01, batch=4, device="cpu", verbose=False, stream=True):
            out[Path(res.path).stem] = [(("fire" if int(b.cls.item())==0 else "smoke"), float(b.conf.item()),
                                         [float(v) for v in b.xyxy[0].tolist()]) for b in res.boxes]
        return out, len(paths)/max(time.time()-t0, 1e-6)
    # tiled 2x2 with 20% overlap
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            out[Path(p).stem] = []
            continue
        W, H = im.size
        ow, oh = int(W*0.2), int(H*0.2)
        xs = [0, max(0, W//2 - ow//2)]
        ys = [0, max(0, H//2 - oh//2)]
        tiles, offsets = [], []
        for y in ys:
            for x in xs:
                box = (x, y, min(W, x + W//2 + ow), min(H, y + H//2 + oh))
                tiles.append(im.crop(box)); offsets.append((box[0], box[1]))
        dets = []
        for tile, (ox, oy) in zip(tiles, offsets):
            arr = np.asarray(tile)
            r = model.predict(arr, imgsz=640, conf=0.01, device="cpu", verbose=False)[0]
            for b in r.boxes:
                xy = [float(v) for v in b.xyxy[0].tolist()]
                dets.append((("fire" if int(b.cls.item())==0 else "smoke"), float(b.conf.item()),
                             [xy[0]+ox, xy[1]+oy, xy[2]+ox, xy[3]+oy]))
        out[Path(p).stem] = nms(dets)
    return out, len(paths)/max(time.time()-t0, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=300)
    ap.add_argument("--positives-control", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ds = PROJECT_ROOT/"build/kaggle_dataset"
    records = load_records(load_list(ds/"lists"/"val_report.txt", ds), ds)
    preds = read_preds(PROJECT_ROOT/"build/pilot_eval/preds_P50.jsonl")
    thr = json.loads((PROJECT_ROOT/"build/pilot_eval/metrics_P50.json").read_text())["thresholds"]
    low = {"fire": 0.10, "smoke": 0.10}
    pos = [r for r in records if (r.meta.get("has_fire") or r.meta.get("has_smoke"))]
    neg = [r for r in records if not (r.meta.get("has_fire") or r.meta.get("has_smoke"))]
    missed_at_010 = [r for r in pos if not decision(r, preds.get(r.id, []), low)]
    detected = [r for r in pos if decision(r, preds.get(r.id, []), thr)]
    import random
    rng = random.Random(args.seed)
    ctrl_pos = rng.sample(detected, min(args.positives_control, len(detected)))
    ctrl_neg = rng.sample(neg, min(args.negatives, len(neg)))
    print("missed at 0.10:", len(missed_at_010), "| control positives:", len(ctrl_pos), "| control negatives:", len(ctrl_neg), flush=True)

    # vantage from captions (symmetric audit index)
    cap = {}
    idx = PROJECT_ROOT/"build/audit_symmetric/index.csv"
    if idx.exists():
        for row in csv.DictReader(idx.open(encoding="utf-8")):
            cap[row["id"]] = row.get("caption", "")
    AERIAL = ("aerial", "from the top", "high vantage", "vantage point", "drone", "bird", "from a high", "hilltop", "mountain top")
    def vantage(rec):
        c = cap.get(rec.id, "").lower()
        return "aerial_high" if any(k in c for k in AERIAL) else "ground_cctv"

    groups = {"missed53": missed_at_010, "ctrl_pos": ctrl_pos, "ctrl_neg": ctrl_neg}
    results = {}
    for mode in ("640", "960", "1280", "tiled"):
        preds_mode = {}
        speed = None
        for gname, recs in groups.items():
            paths = [PROJECT_ROOT/r.path for r in recs]
            out, speed = run_mode(paths, mode)
            preds_mode[gname] = out
        row = {"mode": mode, "images_per_s": round(speed, 2)}
        for gname, recs in groups.items():
            hits = 0
            for r in recs:
                dets = [{"label": l, "confidence": c} for l, c, _ in preds_mode[gname].get(Path(r.path).stem, [])]
                if any(d["confidence"] >= thr.get(d["label"], 1.0) for d in dets):
                    hits += 1
            row[gname] = {"n": len(recs), "hits": hits, "rate": round(hits/len(recs), 4) if recs else None,
                          "ci": wilson(hits, len(recs))}
        # vantage breakdown on missed
        vt = {}
        for v in ("aerial_high", "ground_cctv"):
            sub = [r for r in missed_at_010 if vantage(r) == v]
            hits = sum(1 for r in sub if any(c >= thr.get(l, 1.0) for l, c, _ in preds_mode["missed53"].get(Path(r.path).stem, [])))
            vt[v] = {"n": len(sub), "hits": hits, "rate": round(hits/len(sub), 4) if sub else None}
        row["missed_vantage"] = vt
        results[mode] = row
        print(json.dumps(row), flush=True)
    (OUT/"results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
