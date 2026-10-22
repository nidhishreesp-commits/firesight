from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT
from collect_bench_R import NEGATIVE_GROUPS, POSITIVE_GROUPS, fetch_pool, sha256


def phash(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB"), hash_size=8))


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def safe_title(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", title.replace("File:", ""))[:90]


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize already-downloaded bench_R candidates; no new model run.")
    parser.add_argument("--out", default="bench_R")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-per-scene", type=int, default=3)
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    session = requests.Session()
    session.headers["User-Agent"] = "FireSightBenchR/0.1 (local research)"
    # Rebuild metadata by re-querying the same Commons searches (no downloads).
    meta_by_stem = {}
    for group, spec in list(POSITIVE_GROUPS.items()) + [(k, (v[0], v[1])) for k, v in NEGATIVE_GROUPS.items()]:
        queries = spec["queries"] if "queries" in spec else spec[1]
        for query in queries:
            for row in fetch_pool(session, query, limit=50):
                key = safe_title(row["title"])
                meta_by_stem[key] = {**row, "group": group}
                meta_by_stem[Path(key).stem] = {**row, "group": group}
    rows = []
    for group in list(POSITIVE_GROUPS) + list(NEGATIVE_GROUPS):
        for path in sorted((out / "candidates" / group).glob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            meta = meta_by_stem.get(path.stem) or meta_by_stem.get(path.stem[:90])
            if not meta:
                continue
            rows.append({**meta, "file": str(path.relative_to(PROJECT_ROOT)), "group": group,
                         "sha256": sha256(path), "phash": phash(path)})
    # Banded pHash lookup against train/val + testbench100.
    cache_path = PROJECT_ROOT / "data/processed/manifests/phash_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    refs = list(cache.values())
    idx = PROJECT_ROOT / "testbench100/index.json"
    if idx.exists():
        for r in json.loads(idx.read_text(encoding="utf-8")):
            p = PROJECT_ROOT / "testbench100/images" / r["file"]
            if p.exists():
                refs.append(phash(p))
    bands = [defaultdict(list) for _ in range(4)]
    for h in refs:
        v = int(h, 16)
        for b in range(4):
            bands[b][(v >> (b * 16)) & 0xFFFF].append(h)
    removed = {"train_val_bench100_overlap": 0, "duplicate": 0, "scene_cap": 0}
    final = []
    for r in rows:
        v = int(r["phash"], 16)
        cand = set()
        for b in range(4):
            cand.update(bands[b][(v >> (b * 16)) & 0xFFFF])
        if any(ham(r["phash"], h) <= 6 for h in cand):
            removed["train_val_bench100_overlap"] += 1
            continue
        if any(ham(r["phash"], x["phash"]) <= 6 for x in final):
            removed["duplicate"] += 1
            continue
        if sum(1 for x in final if ham(r["phash"], x["phash"]) <= 6) >= args.max_per_scene:
            removed["scene_cap"] += 1
            continue
        final.append(r)
    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    for r in final:
        by_group[r["group"]].append(r)
    selected = []
    for group, spec in POSITIVE_GROUPS.items():
        rows_g = by_group.get(group, [])
        rng.shuffle(rows_g)
        selected += rows_g[:spec["n"]]
        if len(rows_g) < spec["n"]:
            print("SHORT", group, len(rows_g), spec["n"])
    # negatives: take quotas; if one is short, take the remainder from the other ordinary groups.
    for group, spec in NEGATIVE_GROUPS.items():
        rows_g = by_group.get(group, [])
        rng.shuffle(rows_g)
        selected += rows_g[:spec[0]]
        if len(rows_g) < spec[0]:
            print("SHORT", group, len(rows_g), spec[0])
    negative_count = sum(1 for r in selected if r["group"] in NEGATIVE_GROUPS)
    if negative_count < 500:
        extras = []
        for group in ("ordinary_streets_warehouses_offices", "ordinary_indoor", "ordinary_other"):
            chosen = {r["title"] for r in selected}
            extras += [r for r in by_group.get(group, []) if r["title"] not in chosen]
        rng.shuffle(extras)
        selected += extras[:500 - negative_count]
    if len(selected) != 1000:
        print("WARNING selected", len(selected))
    index = []
    for r in selected:
        if r["group"] in POSITIVE_GROUPS:
            spec = POSITIVE_GROUPS[r["group"]]
            kind, hf, hs, ft = "positive", spec["has_fire"], spec["has_smoke"], spec["flame_type"]
        else:
            kind, hf, hs, ft = "negative", 0, 0, None
        index.append({
            "id": f"benchR:{Path(r['file']).stem[:70]}", "file": r["file"], "kind": kind,
            "family": r["group"], "has_fire": hf, "has_smoke": hs, "flame_type": ft,
            "proposed_has_fire": hf, "proposed_has_smoke": hs, "proposed_flame_type": ft,
            "boxes_fire": [], "boxes_smoke": [],
            "source": "wikimedia_commons", "license": r["license"], "origin": r["source_url"],
            "attribution": {"title": r["title"], "artist": r["artist"], "license_url": r["license_url"]},
            "sha256": r["sha256"], "phash": r["phash"], "width": r["width"], "height": r["height"],
            "difficulty": "medium", "ambiguous": 0, "review_status": "pending",
            "reviewer": "", "reviewed_at": "", "notes": "",
        })
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "file", "has_fire", "has_smoke", "flame_type", "category", "difficulty",
                         "source", "license", "boxes_fire", "boxes_smoke", "ambiguous", "reviewer",
                         "reviewed_at", "review_status", "notes"])
        for r in index:
            writer.writerow([r["id"], r["file"], r["has_fire"], r["has_smoke"], r["flame_type"] or "",
                             r["family"], r["difficulty"], r["source"], r["license"], "[]", "[]", 0, "", "",
                             "pending", ""])
    counts = defaultdict(int)
    for r in index:
        counts[(r["kind"], r["family"])] += 1
    summary = {"selected": len(index), "positives": sum(1 for r in index if r["kind"] == "positive"),
               "negatives": sum(1 for r in index if r["kind"] == "negative"),
               "counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())},
               "removed": removed, "pool": len(rows)}
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# bench_R candidates\n\nPENDING HUMAN REVIEW — no model has been run on these images.\n"
        "Sources: Wikimedia Commons (CC0/PD/CC BY/CC BY-SA) only. Boxes are intentionally empty; "
        "the reviewer must draw them. `labels.csv` is provisional and not a label file until locked.\n\n"
        "```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
