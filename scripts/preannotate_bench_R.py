from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

PROMPTS = ["fire", "flame", "candle flame", "smoke", "sparks"]
LABEL_MAP = {"fire": "fire", "flame": "fire", "candle flame": "fire", "smoke": "smoke", "sparks": "fire"}


def load_index():
    return json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["id"])
    return out


def run_owlv2(rows, out_path: Path, limit=0, offset=0):
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    mid = "google/owlv2-base-patch16-ensemble"
    processor = Owlv2Processor.from_pretrained(mid)
    model = Owlv2ForObjectDetection.from_pretrained(mid).eval()
    done = done_ids(out_path)
    rows = [r for r in rows if r["id"] not in done][offset:]
    if limit:
        rows = rows[:limit]
    t_all = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            try:
                im = Image.open(PROJECT_ROOT / row["file"]).convert("RGB")
            except Exception:
                continue
            t = time.time()
            inputs = processor(text=[PROMPTS], images=im, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            res = processor.post_process_grounded_object_detection(outputs, target_sizes=[im.size[::-1]], threshold=0.01)[0]
            dets = []
            for box, score, lab in zip(res["boxes"], res["scores"], res["labels"]):
                prompt = PROMPTS[int(lab)] if int(lab) < len(PROMPTS) else "fire"
                dets.append({"label": LABEL_MAP.get(prompt, "fire"), "prompt": prompt,
                             "score": round(float(score), 4),
                             "box": [float(v) for v in box.tolist()]})
            fh.write(json.dumps({"id": row["id"], "model": "owlv2", "elapsed_s": round(time.time() - t, 3),
                                 "detections": dets}) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"owlv2 {i}/{len(rows)} elapsed={time.time()-t_all:.0f}s", flush=True)


def run_dino(rows, out_path: Path, limit=0, offset=0):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
    done = done_ids(out_path)
    rows = [r for r in rows if r["id"] not in done][offset:]
    if limit:
        rows = rows[:limit]
    t_all = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            try:
                im = Image.open(PROJECT_ROOT / row["file"]).convert("RGB")
            except Exception:
                continue
            t = time.time()
            inputs = processor(images=im, text=". ".join(PROMPTS) + ".", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            res = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=0.01, text_threshold=0.01,
                target_sizes=[im.size[::-1]], text_labels=[PROMPTS],
            )[0]
            dets = []
            for box, score, lab in zip(res["boxes"], res["scores"], res["labels"]):
                prompt = str(lab).lower()
                dets.append({"label": LABEL_MAP.get(prompt, "fire"), "prompt": prompt,
                             "score": round(float(score), 4), "box": [float(v) for v in box.tolist()]})
            fh.write(json.dumps({"id": row["id"], "model": "grounding_dino_tiny", "elapsed_s": round(time.time() - t, 3),
                                 "detections": dets}) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"dino {i}/{len(rows)} elapsed={time.time()-t_all:.0f}s", flush=True)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_raw(index, raw_rows, blind_ids):
    by_id = defaultdict(list)
    for r in raw_rows:
        by_id[r["id"]].append(r)
    images = {}
    for row in index:
        dets = []
        for r in by_id.get(row["id"], []):
            for d in r["detections"]:
                dets.append({**d, "source": r["model"]})
        merged = []
        for cls in ("fire", "smoke"):
            cls_dets = sorted([d for d in dets if d["label"] == cls], key=lambda d: d["score"], reverse=True)
            clusters = []
            for d in cls_dets:
                placed = False
                for cluster in clusters:
                    if iou(d["box"], cluster["box"]) >= 0.5:
                        w = max(cluster["score"], 1e-6)
                        cluster["box"] = [round((cluster["box"][i] * w + d["box"][i] * d["score"]) / (w + d["score"]), 2) for i in range(4)]
                        cluster["score"] = round(max(cluster["score"], d["score"]), 4)
                        cluster["sources"] = sorted(set(cluster["sources"] + [d["source"]]))
                        placed = True
                        break
                if not placed:
                    clusters.append({"label": cls, "prompt": d.get("prompt"), "box": d["box"],
                                     "score": d["score"], "sources": [d["source"]]})
            merged += clusters
        family = row.get("family", "")
        ftype = {"controlled_fire": "controlled", "spark_screen": "spark",
                 "uncontrolled_fire": "uncontrolled", "both": "uncontrolled"}.get(family, "unknown")
        if row["id"] in blind_ids:
            merged = []
        images[row["id"]] = {
            "blind": row["id"] in blind_ids,
            "proposals": merged,
            "suggested_has_fire": int(any(p["label"] == "fire" and p["score"] >= 0.05 for p in merged)),
            "suggested_has_smoke": int(any(p["label"] == "smoke" and p["score"] >= 0.05 for p in merged)),
            "suggested_flame_type": ftype if any(p["label"] == "fire" for p in merged) else None,
        }
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-annotate bench_R with independent open-vocabulary proposers; no evaluated detector.")
    parser.add_argument("--model", choices=["owlv2", "dino"], required=True)
    parser.add_argument("--scope", choices=["positives", "all"], default="positives")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--per-family", type=int, default=0)
    parser.add_argument("--missing-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = load_index()
    if args.scope == "positives":
        rows = [r for r in rows if r.get("proposed_has_fire") or r.get("proposed_has_smoke")]
    if args.missing_only:
        prop_path = PROJECT_ROOT / "bench_R/proposals.json"
        have = set()
        if prop_path.exists():
            have = set(json.loads(prop_path.read_text(encoding="utf-8")).get("images", {}).keys())
        rows = [r for r in rows if r["id"] not in have]
    if args.per_family:
        by = defaultdict(list)
        for r in rows:
            by[r.get("family", "other")].append(r)
        picked = []
        for fam, members in by.items():
            random.Random(7).shuffle(members)
            picked += members[:args.per_family]
        rows = picked
    out = PROJECT_ROOT / (args.out or f"bench_R/proposals_raw/{args.model}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "owlv2":
        run_owlv2(rows, out, limit=args.limit, offset=args.offset)
    else:
        run_dino(rows, out, limit=args.limit, offset=args.offset)
    print(json.dumps({"model": args.model, "raw": str(out), "rows_available": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
