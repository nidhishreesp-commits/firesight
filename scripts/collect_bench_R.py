from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

POSITIVE_GROUPS = {
    "uncontrolled_fire": {
        "n": 200, "has_fire": 1, "has_smoke": 0, "flame_type": "uncontrolled",
        "queries": ["wildfire", "forest fire", "house fire", "building fire", "car fire", "vehicle fire",
                    "grass fire", "night fire", "firefighters fire", "small fire", "industrial fire", "brush fire"],
    },
    "controlled_fire": {
        "n": 80, "has_fire": 1, "has_smoke": 0, "flame_type": "controlled",
        "queries": ["candle flame", "lit candle", "lighter flame", "burning match", "gas stove flame",
                    "tandoor", "fireplace fire", "campfire", "barbecue grill flame", "diya lamp", "incense burning"],
    },
    "spark_screen": {
        "n": 40, "has_fire": 1, "has_smoke": 0, "flame_type": "spark",
        "queries": ["welding sparks", "grinding sparks", "fireworks", "fire on television screen", "movie fire scene"],
    },
    "smoke_only": {
        "n": 100, "has_fire": 0, "has_smoke": 1, "flame_type": None,
        "queries": ["smoke plume", "wildfire smoke", "chimney smoke", "indoor smoke", "thick smoke"],
    },
    "both": {
        "n": 80, "has_fire": 1, "has_smoke": 1, "flame_type": "uncontrolled",
        "queries": ["wildfire flames smoke", "burning building smoke", "car fire smoke", "forest fire smoke", "house fire smoke"],
    },
}

NEGATIVE_GROUPS = {
    "sunset_orange_sky": (30, ["sunset", "sunrise", "orange sky", "dusk sky"]),
    "red_orange_neon_lights": (30, ["neon sign", "red lantern", "orange lamp", "neon lights"]),
    "vehicle_lights_streetlights": (30, ["car headlights night", "street lights night", "car tail lights", "brake lights"]),
    "chandelier_stage_lights": (25, ["chandelier", "stage lighting", "concert lights"]),
    "reflections_lens_flare_glare": (25, ["sun glare water", "lens flare", "reflection glass building"]),
    "orange_ppe_vests_cones": (30, ["orange safety vest", "traffic cones", "orange helmet worker", "orange work clothes"]),
    "flowers_autumn_leaves": (25, ["orange flowers", "autumn leaves", "red flowers"]),
    "clouds": (30, ["clouds sky", "cumulus clouds", "storm clouds"]),
    "fog_haze_mist": (30, ["fog forest", "mist landscape", "haze city", "foggy road"]),
    "steam": (25, ["steam cooking", "steam kettle", "steam shower", "manhole steam"]),
    "exhaust": (25, ["vehicle exhaust", "factory chimney water vapour", "cooling tower steam"]),
    "dust_construction_plaster": (25, ["construction dust", "plaster dust", "demolition dust", "sand dust site"]),
    "snow": (20, ["snow landscape", "blowing snow", "snowy street"]),
    "overexposed_white": (20, ["overexposed sky", "white wall bright", "bright white background"]),
    "camera_artifacts": (40, ["blurry photo", "noisy image", "compression artifacts", "infrared camera",
                              "night vision camera", "rain on camera lens"]),
    "ordinary_indoor": (30, ["living room interior", "kitchen interior", "office interior", "bedroom interior"]),
    "ordinary_streets_warehouses_offices": (30, ["warehouse interior", "street scene", "office building interior",
                                                 "factory floor"]),
    "ordinary_other": (20, ["landscape no fire", "park scene", "city square", "market scene"]),
}

BLOCK = ("prentenboek", "aardig", "book", "poster", "logo", "diagram", "map", "painting", "drawing",
         "cartoon", "clipart", "icon", "stamp", "cover")


def clean(v: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(v or "")).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def allowed_license(lic: str) -> bool:
    low = lic.lower()
    return any(a in low for a in ("cc0", "public domain", "cc by", "cc-by")) and "nc" not in low and "nd" not in low


def fetch_pool(session, query, limit=50):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": str(limit),
              "prop": "imageinfo", "iiprop": "url|size|extmetadata|mime", "iiurlwidth": "1280"}
    try:
        data = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=45).json()
    except Exception as exc:
        print("query failed", query, exc)
        return []
    out = []
    for page in list((data.get("query") or {}).get("pages", {}).values()):
        if "imageinfo" not in page:
            continue
        title = page.get("title", "")
        low = title.lower()
        if any(b in low for b in BLOCK):
            continue
        info = page["imageinfo"][0]
        md = info.get("extmetadata", {})
        lic = clean(md.get("LicenseShortName", {}).get("value", ""))
        if not allowed_license(lic):
            continue
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        if int(info.get("width", 0)) < 500 or int(info.get("height", 0)) < 400:
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        out.append({
            "title": title, "url": url, "source_url": info.get("descriptionurl", ""),
            "license": lic, "license_url": clean(md.get("LicenseUrl", {}).get("value", "")),
            "artist": clean(md.get("Artist", {}).get("value", "")), "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
        })
    return out


def download(session, row, target: Path) -> bool:
    try:
        r = session.get(row["url"], timeout=60)
        r.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(r.content)
        with Image.open(target) as im:
            im.convert("RGB")
        return True
    except Exception as exc:
        print("download failed", row["title"], exc)
        return False


def phash(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB"), hash_size=8))


def ham(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect fresh Wikimedia candidates for bench_R; no model is run.")
    parser.add_argument("--out", default="bench_R")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--pool-factor", type=float, default=1.35)
    parser.add_argument("--max-per-scene", type=int, default=3)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    out = PROJECT_ROOT / args.out
    if out.exists():
        raise SystemExit(f"{out} exists; move it away first")
    (out / "candidates").mkdir(parents=True)
    session = requests.Session()
    session.headers["User-Agent"] = "FireSightBenchR/0.1 (local research)"
    bench100_titles = set()
    idx = PROJECT_ROOT / "testbench100/index.json"
    if idx.exists():
        for r in json.loads(idx.read_text(encoding="utf-8")):
            if r.get("attribution"):
                bench100_titles.add(r["attribution"].get("title", ""))
    pool = defaultdict(list)
    seen_titles = set()

    def collect(group, spec, is_positive):
        need = int(round(spec["n"] * args.pool_factor)) if is_positive else int(round(spec[0] * args.pool_factor))
        queries = spec["queries"] if is_positive else spec[1]
        for q in queries:
            for row in fetch_pool(session, q):
                if len(pool[group]) >= need:
                    break
                if row["title"] in seen_titles or row["title"] in bench100_titles:
                    continue
                ext = ".jpg" if "jpeg" in row["url"].lower() or "jpg" in row["url"].lower() else ".png"
                name = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["title"].replace("File:", ""))[:90]
                target = out / "candidates" / group / (name if name.lower().endswith(ext) else name + ext)
                if not download(session, row, target):
                    continue
                seen_titles.add(row["title"])
                row = {**row, "group": group, "file": str(target.relative_to(PROJECT_ROOT)),
                       "sha256": sha256(target), "phash": phash(target)}
                pool[group].append(row)
            print(group, q, len(pool[group]), "/", need, flush=True)
            if len(pool[group]) >= need:
                break

    for group, spec in POSITIVE_GROUPS.items():
        collect(group, spec, True)
    for group, spec in NEGATIVE_GROUPS.items():
        collect(group, spec, False)

    # Global pHash dedupe + max 3 per scene, plus remove testbench100 / train / val near-duplicates.
    cached = {}
    cache_path = PROJECT_ROOT / "data/processed/manifests/phash_cache.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    ref_hashes = list(cached.values())
    all_rows = [r for rows in pool.values() for r in rows]
    kept = []
    removed = {"duplicate": 0, "scene_cap": 0, "train_val_overlap": 0}
    clusters = defaultdict(list)
    for r in all_rows:
        if any(ham(r["phash"], h) <= 6 for h in ref_hashes):
            removed["train_val_overlap"] += 1
            continue
        clusters[r["phash"][:8]].append(r)
    # refine with exact hamming within band buckets
    final = []
    for r in all_rows:
        if r not in final and all(r["title"] != x["title"] for x in final):
            if any(ham(r["phash"], x["phash"]) <= 6 for x in final):
                removed["duplicate"] += 1
                continue
        if any(r["title"] == x["title"] for x in final):
            continue
        final.append(r)
    # cap per near-duplicate cluster
    cap = []
    for r in final:
        if sum(1 for x in cap if ham(r["phash"], x["phash"]) <= 6) >= args.max_per_scene:
            removed["scene_cap"] += 1
            continue
        cap.append(r)
    by_group = defaultdict(list)
    for r in cap:
        by_group[r["group"]].append(r)
    selected = []
    for group, spec in POSITIVE_GROUPS.items():
        rows = by_group.get(group, [])
        rng.shuffle(rows)
        selected += rows[:spec["n"]]
        if len(rows) < spec["n"]:
            print("SHORT", group, len(rows), "of", spec["n"])
    for group, spec in NEGATIVE_GROUPS.items():
        rows = by_group.get(group, [])
        rng.shuffle(rows)
        selected += rows[:spec[0]]
        if len(rows) < spec[0]:
            print("SHORT", group, len(rows), "of", spec[0])

    index = []
    for r in selected:
        if r["group"] in POSITIVE_GROUPS:
            spec = POSITIVE_GROUPS[r["group"]]
            has_fire, has_smoke, ftype = spec["has_fire"], spec["has_smoke"], spec["flame_type"]
            kind = "positive"
        else:
            has_fire = has_smoke = 0
            ftype = None
            kind = "negative"
        index.append({
            "id": f"benchR:{Path(r['file']).stem[:70]}", "file": r["file"], "kind": kind,
            "family": r["group"], "proposed_has_fire": has_fire, "proposed_has_smoke": has_smoke,
            "proposed_flame_type": ftype, "has_fire": has_fire, "has_smoke": has_smoke,
            "flame_type": ftype, "boxes_fire": [], "boxes_smoke": [],
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
               "removed": removed, "pool": len(all_rows)}
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
