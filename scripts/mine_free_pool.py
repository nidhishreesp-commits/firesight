"""Stage A step 1 - free hard-negative pool from P50's own out-of-sample training images.

Runs frozen P50 at a low confidence on the 29,492 images that are in train_100 but NOT in
train_50 (the exact images P50 never saw), caches detections, and reports false detections on
empty-label negatives.  No external download, no TEST/val/bench use.

Resumable: detections are appended to a cache keyed by image id.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def key_of(row: dict) -> str:
    return row["id"].replace(":", "_") + Path(row["path"]).suffix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="build/kaggle_weights/runs/P50/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--label-filter", choices=["neg", "all", "pos"], default="all")
    ap.add_argument("--out", default="build/stage_a")
    args = ap.parse_args()

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / f"free_pool_dets_conf{args.conf:.3f}.jsonl"

    pkg = PROJECT_ROOT / "build/kaggle_dataset/lists"
    t50 = {Path(l.strip()).name for l in (pkg / "train_50.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    t100 = {Path(l.strip()).name for l in (pkg / "train_100.txt").read_text(encoding="utf-8").splitlines() if l.strip()}

    rows = [json.loads(l) for l in (PROJECT_ROOT / "data/splits/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    oos = [r for r in rows if key_of(r) in t100 and key_of(r) not in t50]
    if args.label_filter == "neg":
        oos = [r for r in oos if not r["labels"]]
    elif args.label_filter == "pos":
        oos = [r for r in oos if r["labels"]]
    if args.limit:
        oos = oos[: args.limit]
    print(f"out-of-sample rows: {len(oos)} (negatives={sum(1 for r in oos if not r['labels'])}, "
          f"positives={sum(1 for r in oos if r['labels'])})", flush=True)

    done: set[str] = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [r for r in oos if r["id"] not in done]
    print(f"cached={len(done)} todo={len(todo)}", flush=True)

    if todo:
        from ultralytics import YOLO

        model = YOLO(str(PROJECT_ROOT / args.model), task="detect")
        lm = out / "free_pool_todo.txt"
        lm.write_text("\n".join(str(PROJECT_ROOT / r["path"]) for r in todo) + "\n", encoding="utf-8")
        p2r = {str((PROJECT_ROOT / r["path"]).resolve()): r for r in todo}
        t0 = time.time()
        n = 0
        with cache_path.open("a", encoding="utf-8") as fh:
            for res in model.predict(str(lm), imgsz=args.imgsz, conf=args.conf, batch=args.batch,
                                     device="cpu", verbose=False, stream=True):
                r = p2r.get(str(Path(res.path).resolve()))
                if r is None:
                    continue
                dets = []
                for b in res.boxes:
                    dets.append({
                        "label": "fire" if int(b.cls.item()) == 0 else "smoke",
                        "conf": round(float(b.conf.item()), 6),
                        "box": [round(float(v), 2) for v in b.xyxy[0].tolist()],
                    })
                mc = max((d["conf"] for d in dets), default=0.0)
                fh.write(json.dumps({
                    "id": r["id"], "path": r["path"], "source": r["source"],
                    "negative": not bool(r["labels"]), "width": r.get("width"), "height": r.get("height"),
                    "max_conf": round(mc, 6), "dets": dets,
                }) + "\n")
                fh.flush()
                n += 1
                if n % 250 == 0:
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} {n/el:.2f} img/s eta {(len(todo)-n)/max(n/el,1e-6)/60:.1f} min", flush=True)
        print(f"inference done in {(time.time()-t0)/60:.1f} min", flush=True)

    # summaries from cache
    allrows = [json.loads(l) for l in cache_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    neg = [r for r in allrows if r["negative"]]
    false = [r for r in neg if r["max_conf"] >= 0.05]
    hard = [r for r in neg if r["max_conf"] >= 0.10]
    very = [r for r in neg if r["max_conf"] >= 0.30]
    quasi = [r for r in neg if r["max_conf"] >= 0.50]
    (out / "free_pool_false.jsonl").write_text(
        "\n".join(json.dumps(r) for r in false) + ("\n" if false else ""), encoding="utf-8")
    summary = {
        "out_of_sample_rows": len(oos), "cached_rows": len(allrows),
        "negatives_scored": len(neg), "positives_scored": len(allrows) - len(neg),
        "false_detections_conf_ge_0.05": len(false),
        "hard_conf_ge_0.10": len(hard), "very_hard_conf_ge_0.30": len(very),
        "quarantine_conf_ge_0.50": len(quasi),
        "by_source": dict(Counter(r["source"] for r in false)),
        "by_source_all_negatives": dict(Counter(r["source"] for r in neg)),
        "measured": "P50 ONNX CPU inference at conf=0.01; false = empty-label image with max_conf>=0.05",
    }
    (out / "free_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
