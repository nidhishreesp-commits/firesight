"""Fast Wikimedia Commons category collector (licence-clean, append-only).

Uses category members instead of free-text search, which gives much higher yield per API call
for lookalike families such as sunsets, fog, clouds, traffic lights and street lamps.  Reuses
the same licence filter and provenance fields as collect_stage_a_images.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from collect_stage_a_images import (API, BLOCK_TITLE, USER_AGENT, allowed_license, clean,
                                    fetch, save, sha256_bytes)

CATEGORY_MAP: dict[str, list[str]] = {
    "sun_glare_sky": ["Category:Sunlight", "Category:Backlighting", "Category:Silhouettes",
                      "Category:Lens flare", "Category:Contre-jour"],
    "sunset_sunrise": ["Category:Sunsets", "Category:Sunrises", "Category:Orange sky"],
    "warm_artificial_light": ["Category:Street lamps", "Category:Lanterns", "Category:Light bulbs",
                              "Category:City lights"],
    "traffic_light": ["Category:Traffic lights", "Category:Railway signals"],
    "vehicle_lights": ["Category:Automobile headlights", "Category:Automotive lamps",
                       "Category:Tail lights"],
    "haze_cloud_fog_mist": ["Category:Clouds", "Category:Fog", "Category:Mist", "Category:Haze",
                            "Category:Overcast"],
    "dust_plaster": ["Category:Dust", "Category:Dust storms", "Category:Construction dust",
                     "Category:Sawdust", "Category:Plaster"],
    "screen_fire": ["Category:Television screens", "Category:Computer monitors",
                    "Category:Video game screenshots"],
    "reflection_lens_flare": ["Category:Reflections", "Category:Lens flare",
                              "Category:Water reflections"],
    "orange_ppe": ["Category:High-visibility clothing", "Category:Traffic cones",
                   "Category:Hard hats"],
    "skin_person": ["Category:Portrait photographs", "Category:Selfies", "Category:Human faces"],
    "cave_warm_rock": ["Category:Caves", "Category:Red rocks", "Category:Rock formations"],
    "smoke_steam_nonfire": ["Category:Steam", "Category:Water vapor", "Category:Cooling towers",
                            "Category:Geysers"],
    "orange_vegetation": ["Category:Orange flowers", "Category:Autumn leaves"],
    "neon_lights": ["Category:Neon signs"],
    "stage_chandelier": ["Category:Chandeliers", "Category:Stage lighting",
                         "Category:Concert lighting"],
    "snow_white_overexposed": ["Category:Snowfall", "Category:Blizzards", "Category:Snow"],
    "exhaust": ["Category:Vehicle exhaust", "Category:Exhaust pipes"],
    "camera_artifacts": ["Category:Motion blur", "Category:Lens flare"],
    "ordinary_indoor": ["Category:Office interiors", "Category:Warehouse interiors",
                        "Category:Hallways"],
    "ordinary_outdoor": ["Category:Streets", "Category:Office buildings", "Category:Parks",
                         "Category:Car parks"],
}


def pages(session: requests.Session, category: str, max_pages: int):
    params = {
        "action": "query", "format": "json", "generator": "categorymembers",
        "gcmtitle": category, "gcmtype": "file", "gcmlimit": "50",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime", "iiurlwidth": "1280",
        "maxlag": "5",
    }
    for _ in range(max_pages):
        data = None
        for attempt in range(4):
            try:
                r = session.get(API, params=params, timeout=45)
                if r.status_code in (429, 503):
                    time.sleep(8 + attempt * 5)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                print(f"  api retry {category}: {exc}", flush=True)
                time.sleep(3 + attempt * 3)
        if data is None:
            return
        for page in (data.get("query") or {}).get("pages", {}).values():
            yield page
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)
        time.sleep(0.2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--min-width", type=int, default=500)
    ap.add_argument("--out", default="data/mined_pools/wikimedia_stage_a")
    ap.add_argument("--sleep", type=float, default=0.10)
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in cats if c not in CATEGORY_MAP]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")
    out = PROJECT_ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    seen_titles: set[str] = set()
    seen_sha: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seen_titles.add(r.get("title", ""))
                seen_sha.add(r.get("sha256", ""))

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    counts = Counter()
    rejected = Counter()
    for cause in cats:
        got = 0
        for wiki_cat in CATEGORY_MAP[cause]:
            if got >= args.per_category:
                break
            for page in pages(session, wiki_cat, args.max_pages):
                if got >= args.per_category:
                    break
                title = page.get("title", "")
                if title in seen_titles or BLOCK_TITLE.search(title):
                    rejected["title"] += 1
                    continue
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png"):
                    rejected["mime"] += 1
                    continue
                md = info.get("extmetadata", {})
                licence = clean(md.get("LicenseShortName", {}).get("value", ""))
                if not allowed_license(licence):
                    rejected["licence"] += 1
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    rejected["url"] += 1
                    continue
                raw = fetch(session, url)
                if raw is None:
                    rejected["download"] += 1
                    continue
                digest = sha256_bytes(raw)
                if digest in seen_sha:
                    rejected["sha"] += 1
                    continue
                dest = out / "images" / cause / f"{digest[:16]}.jpg"
                size = save(raw, dest, args.min_width)
                if size is None:
                    rejected["save"] += 1
                    continue
                row = {
                    "id": f"commons:{digest[:16]}", "path": str(dest.relative_to(PROJECT_ROOT)),
                    "cause": cause, "source": "wikimedia_commons", "title": title,
                    "page_url": info.get("descriptionurl", ""), "image_url": url,
                    "licence": licence,
                    "licence_url": clean(md.get("LicenseUrl", {}).get("value", "")),
                    "artist": clean(md.get("Artist", {}).get("value", "")),
                    "credit": clean(md.get("Credit", {}).get("value", "")),
                    "sha256": digest, "width": size[0], "height": size[1],
                    "query": wiki_cat,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                with FileLock(str(manifest) + ".lock", timeout=120):
                    with manifest.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    with (out / "attribution.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"id": row["id"], "title": title,
                                             "page_url": row["page_url"], "artist": row["artist"],
                                             "credit": row["credit"], "licence": licence,
                                             "licence_url": row["licence_url"]}) + "\n")
                seen_titles.add(title)
                seen_sha.add(digest)
                counts[cause] += 1
                got += 1
                if got % 25 == 0:
                    print(f"  {cause}: {got}", flush=True)
                time.sleep(args.sleep)
        print(f"{cause}: {counts[cause]} new", flush=True)
    print(json.dumps({"new": dict(counts), "rejected": dict(rejected),
                      "manifest_rows": sum(1 for _ in manifest.open(encoding="utf-8"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
