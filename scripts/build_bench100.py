from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest

SEED = 20261022
LICENSE = {
    "dfire": "D-Fire (Kaggle mirror CC0; upstream per-source terms)",
    "fasdd": "CC BY 4.0 (FASDD_CV; Wang et al. 2025)",
    "local_seed": "Proprietary internal (SmartSight)",
    "local_negative": "Proprietary internal (SmartSight)",
    "cctv_nc": "CC BY-NC 4.0 (Simuletic; evaluation only)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brightness(path: Path) -> float:
    img = cv2.imread(str(path))
    if img is None:
        return 0.0
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def max_box_ratio(rec) -> float:
    if not rec.labels:
        return 0.0
    return max((g.box.area / (rec.width * rec.height)) for g in rec.labels)


def pick_unique(pool, n, key, reverse=True):
    seen = set()
    out = []
    for rec, score in sorted(pool, key=lambda x: x[1], reverse=reverse):
        if rec.id in seen:
            continue
        seen.add(rec.id)
        out.append(rec)
        if len(out) >= n:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the 100-image candidate bench; labels remain PENDING human review.")
    parser.add_argument("--out", default="testbench100")
    parser.add_argument("--inventory", default="hard_negatives/clip_inventory.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    import random
    rng = random.Random(args.seed)
    out = PROJECT_ROOT / args.out
    if out.exists():
        print("refusing to overwrite existing testbench100; move it away first")
        return 1
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    test = load_manifest(PROJECT_ROOT / "data/splits/test.jsonl", root=PROJECT_ROOT)
    train_scenes = {r.scene for r in load_manifest(PROJECT_ROOT / "data/splits/train.jsonl", root=PROJECT_ROOT)}
    val_scenes = {r.scene for r in load_manifest(PROJECT_ROOT / "data/splits/val.jsonl", root=PROJECT_ROOT)}
    test = [r for r in test if r.scene not in train_scenes and r.scene not in val_scenes]

    def bright(r):
        img = cv2.imread(str(PROJECT_ROOT / r.path))
        return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2].mean()) if img is not None else 0.0

    fire_pool = [r for r in test if r.source == "dfire" and any(g.label == "fire" for g in r.labels) and not any(g.label == "smoke" for g in r.labels)]
    smoke_pool = [r for r in test if any(g.label == "smoke" for g in r.labels) and not any(g.label == "fire" for g in r.labels)]
    both_pool = [r for r in test if any(g.label == "fire" for g in r.labels) and any(g.label == "smoke" for g in r.labels)]
    # Reject the tiny/dark boxes that are the dominant label noise in D-Fire test.
    fire_pool = [r for r in fire_pool if max_box_ratio(r) >= 0.002 and bright(r) >= 25]
    smoke_pool = [r for r in smoke_pool if max_box_ratio(r) >= 0.0005 and bright(r) >= 25]
    both_pool = [r for r in both_pool if max_box_ratio(r) >= 0.002 and bright(r) >= 25]
    for pool in (fire_pool, smoke_pool, both_pool):
        rng.shuffle(pool)

    def stratum(pool, large_n, small_n, dark_n):
        scored = [(r, max_box_ratio(r)) for r in pool]
        bright = [(r, brightness(PROJECT_ROOT / r.path)) for r in pool]
        dark_ids = {r.id for r, b in sorted(bright, key=lambda x: x[1])[:max(dark_n * 2, 1)]}
        picks = []
        picks += pick_unique([(r, s) for r, s in scored], large_n, None, True)
        small = [x for x in scored if x[0].id not in {p.id for p in picks}]
        picks += pick_unique(small, small_n, None, False)
        dark = [x for x in scored if x[0].id in dark_ids and x[0].id not in {p.id for p in picks}]
        picks += pick_unique(dark, dark_n, None, False)
        return picks

    positives = []
    positives += stratum(fire_pool, 13, 7, 5)[:25]
    positives += stratum(smoke_pool, 8, 4, 3)[:15]
    positives += stratum(both_pool, 5, 3, 2)[:10]
    # fill if any stratum fell short
    for pool in (fire_pool, smoke_pool, both_pool):
        for r in pool:
            if len(positives) >= 50:
                break
            if all(r.id != p.id for p in positives):
                positives.append(r)
    positives = positives[:50]

    inv = json.loads((PROJECT_ROOT / args.inventory).read_text(encoding="utf-8"))
    inv_rows = inv["rows"]
    by_cat = defaultdict(list)
    for row in inv_rows:
        by_cat[(row["family"], row["category"])].append(row)
    for key in by_cat:
        by_cat[key].sort(key=lambda r: r["clip_score"], reverse=True)

    neg_plan = {
        ("light_colour", "sunset_sunrise"): 3,
        ("light_colour", "street_head_lights"): 3,
        ("light_colour", "lamps_neon_brake"): 2,
        ("light_colour", "reflection_glare"): 2,
        ("light_colour", "orange_clothing_ppe"): 2,
        ("light_colour", "chandelier_festival"): 1,
        ("smoke_like", "clouds"): 6,
        ("smoke_like", "fog_mist_haze"): 5,
        ("smoke_like", "dust_snow"): 2,
        ("camera_artifact", "blur_noise_compression"): 4,
        ("camera_artifact", "infrared_night"): 4,
    }
    rec_by_id = {r.id: r for r in test}
    negatives = []
    for key, n in neg_plan.items():
        got = 0
        for row in by_cat.get(key, []):
            rec = rec_by_id.get(row["id"])
            if rec is None or rec.labels or any(rec.id == x.id for x in negatives):
                continue
            rec.meta["bench_category"] = key[1]
            rec.meta["bench_family"] = key[0]
            negatives.append(rec)
            got += 1
            if got >= n or len(negatives) >= 50:
                break
        if len(negatives) >= 50:
            break
    hard = [r for r in test if not r.labels and all(r.id != x.id for x in negatives)]
    rng.shuffle(hard)
    synth_bases = (by_cat.get(("light_colour", "street_head_lights"), [])
                   + by_cat.get(("smoke_like", "fog_mist_haze"), [])
                   + by_cat.get(("smoke_like", "clouds"), []))
    synth_ids = [row["id"] for row in synth_bases if all(row["id"] != x.id for x in negatives)][:4]
    synth_sources = [rec_by_id[i] for i in synth_ids if i in rec_by_id][:4]
    for idx, rec in enumerate(synth_sources):
        src = PROJECT_ROOT / rec.path
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        overlay = img.copy()
        for _ in range(120):
            x, y = rng.randrange(w), rng.randrange(h)
            cv2.circle(overlay, (x, y), rng.randint(1, 4), (200, 200, 210), -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        out_path = out / "images" / f"synth_rain_{idx}_{Path(rec.path).stem}.jpg"
        cv2.imwrite(str(out_path), img)
        rec.meta["synthetic_path"] = str(out_path)
        rec.meta["synthetic_how"] = "synthetic_rain"
        rec.meta["bench_category"] = "rain_on_lens"
        rec.meta["bench_family"] = "camera_artifact"
        negatives.append(rec)

    wm_meta = PROJECT_ROOT / "hard_negatives/sources/wikimedia/attribution.jsonl"
    wm_rows = [json.loads(line) for line in wm_meta.read_text(encoding="utf-8").splitlines() if line.strip()] if wm_meta.exists() else []
    take = {"candle": 3, "match_lighter": 2, "gas_stove": 1, "campfire": 2, "welding": 2, "fireworks": 1, "fireplace": 1}
    wm_selected = []
    used = set()
    for cat, n in take.items():
        got = 0
        for row in wm_rows:
            if row["category"] != cat or row["title"] in used:
                continue
            if any(w in row["title"].lower() for w in ("prentenboek", "aardig", "ormonde", "burning match", "dpla")):
                continue
            wm_selected.append(row)
            used.add(row["title"])
            got += 1
            if got >= n:
                break
    negatives = negatives[:38]

    index = []
    for rec in positives:
        clone = rec.meta.get("synthetic_path")
        src = Path(clone) if clone else PROJECT_ROOT / rec.path
        name = f"{rec.id.replace(':','_')}.jpg" if not clone else Path(clone).name
        dst = out / "images" / name
        if not clone:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), dst) if not dst.exists() else None
        fire_boxes = [g.box.to_list() for g in rec.labels if g.label == "fire"]
        smoke_boxes = [g.box.to_list() for g in rec.labels if g.label == "smoke"]
        ratio = max_box_ratio(rec)
        bright = brightness(src)
        difficulty = "easy" if ratio >= 0.03 and bright >= 70 else ("medium" if ratio >= 0.002 else "hard")
        index.append({
            "id": rec.id, "file": name, "source": rec.source,
            "license": LICENSE.get(rec.source, "unknown"), "origin": rec.origin,
            "kind": "positive", "category": rec.category, "difficulty": difficulty,
            "has_fire": 1 if fire_boxes else 0, "has_smoke": 1 if smoke_boxes else 0,
            "boxes_fire": fire_boxes, "boxes_smoke": smoke_boxes,
            "width": rec.width, "height": rec.height, "sha256": sha256(dst),
            "flags": {"small_distant": ratio < 0.002, "low_light": bright < 60,
                      "max_box_ratio": round(ratio, 6), "brightness": round(bright, 1)},
            "review_status": "pending", "reviewer": "", "reviewed_at": "", "ambiguous": 0, "notes": "",
        })
    for rec in negatives:
        clone = rec.meta.get("synthetic_path")
        name = Path(clone).name if clone else f"{rec.id.replace(':','_')}.jpg"
        src = Path(clone) if clone else (PROJECT_ROOT / rec.path)
        dst = out / "images" / name
        if clone:
            if not dst.exists():
                shutil.move(str(clone), dst)
        else:
            shutil.copy2(src, dst)
        index.append({
            "id": rec.id, "file": name, "source": rec.source,
            "license": LICENSE.get(rec.source, "unknown"), "origin": rec.origin,
            "kind": "negative", "category": rec.meta.get("bench_category", "hard_negative"),
            "family": rec.meta.get("bench_family", "other"),
            "difficulty": "hard", "has_fire": 0, "has_smoke": 0,
            "boxes_fire": [], "boxes_smoke": [], "width": rec.width, "height": rec.height,
            "sha256": sha256(dst),
            "flags": {"synthetic": bool(clone), "synthetic_how": rec.meta.get("synthetic_how")},
            "review_status": "pending", "reviewer": "", "reviewed_at": "", "ambiguous": 0, "notes": "",
        })

    for row in wm_selected:
        src = PROJECT_ROOT / row["file"]
        if not src.exists():
            continue
        name = f"wm_{row['category']}_{Path(src).name}"
        dst = out / "images" / name
        shutil.copy2(src, dst)
        index.append({
            "id": f"wikimedia:{Path(src).stem[:60]}", "file": name, "source": "wikimedia_commons",
            "license": row["license"], "origin": row.get("source_url", ""),
            "attribution": {"artist": row.get("artist", ""), "license_url": row.get("license_url", ""),
                            "title": row.get("title", "")},
            "kind": "negative", "category": row["category"], "family": "fire_adjacent",
            "difficulty": "hard", "has_fire": 0, "has_smoke": 0,
            "boxes_fire": [], "boxes_smoke": [], "width": None, "height": None,
            "sha256": sha256(dst),
            "flags": {"policy": "controlled/non-emergency flame, not an alarm event"},
            "review_status": "pending", "reviewer": "", "reviewed_at": "", "ambiguous": 0, "notes": "",
        })

    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "file", "has_fire", "has_smoke", "category", "difficulty", "source", "license",
                         "boxes_fire", "boxes_smoke", "ambiguous", "reviewer", "reviewed_at", "review_status", "notes"])
        for row in index:
            writer.writerow([row["id"], row["file"], row["has_fire"], row["has_smoke"], row["category"], row["difficulty"],
                             row["source"], row["license"], json.dumps(row["boxes_fire"]), json.dumps(row["boxes_smoke"]),
                             row["ambiguous"], row["reviewer"], row["reviewed_at"], row["review_status"], row["notes"]])
    for row in index:
        lines = []
        for box in row["boxes_fire"]:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / row["width"], (y1 + y2) / 2 / row["height"]
            bw, bh = (x2 - x1) / row["width"], (y2 - y1) / row["height"]
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        for box in row["boxes_smoke"]:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / row["width"], (y1 + y2) / 2 / row["height"]
            bw, bh = (x2 - x1) / row["width"], (y2 - y1) / row["height"]
            lines.append(f"1 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / "labels" / (Path(row["file"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    summary = {
        "seed": args.seed,
        "images": len(index),
        "positives": sum(1 for r in index if r["kind"] == "positive"),
        "negatives": sum(1 for r in index if r["kind"] == "negative"),
        "positive_categories": {k: sum(1 for r in index if r["kind"] == "positive" and r["category"] == k) for k in ("fire", "smoke", "both")},
        "negative_families": {k: sum(1 for r in index if r["kind"] == "negative" and r.get("family") == k) for k in sorted({r.get("family") for r in index if r["kind"] == "negative"})},
        "negative_subcategories": {k: sum(1 for r in index if r["kind"] == "negative" and r["category"] == k) for k in sorted({r["category"] for r in index if r["kind"] == "negative"})},
        "review_status": "PENDING HUMAN REVIEW - not locked, not for benchmarking",
    }
    (out / "README.md").write_text(
        "# testbench100\n\n"
        "**Status: PENDING HUMAN REVIEW. Do not run a model on this bench and do not report metrics until it is locked.**\n\n"
        "Composition: 50 positives (25 fire, 15 smoke-only, 10 fire+smoke) and 50 hard negatives.\n"
        "Provisional labels come from the source datasets' human annotations; every image must be\n"
        "accepted/edited in the dashboard *Bench review* page. The reviewer name and date are stored\n"
        "per image, and `POST /api/bench100/lock` writes `labels.csv`, YOLO labels and a SHA-256\n"
        "manifest. Benchmarking is blocked while the bench is unlocked.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
