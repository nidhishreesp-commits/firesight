"""Stage A external image collection from Wikimedia Commons (licence-clean).

Only CC0 / public domain / CC BY / CC BY-SA are accepted.  NC and ND are rejected.  Every
downloaded image records its own licence, licence URL, artist/credit, Commons page URL and
thumbnail URL.  Images are saved as <=1280 px JPEG thumbnails (a Commons-permitted derivative
under the same licence), so the whole pool stays well below the 2 GB approval threshold.

This feeds scripts/mine_stage_a.py; it does not train or tune anything.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image
from filelock import FileLock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from hardneg_causes import CATEGORIES, QUERIES

API = "https://commons.wikimedia.org/w/api.php"
BLOCK_TITLE = re.compile(
    r"\b(fire|flame|burning|wildfire|bonfire|candle|welding|firework|smoke|diagram|map|"
    r"drawing|painting|engraving|illustration|logo|coat of arms|chart|svg)\b",
    re.IGNORECASE,
)
USER_AGENT = "FireSightStageA/1.0 (hard-negative research; contact: local)"


def clean(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def allowed_license(name: str) -> bool:
    low = clean(name).lower()
    if not any(t in low for t in ("cc0", "public domain", "cc by", "cc-by", "pd-", "pd ")):
        return False
    return not any(t in low for t in (" nc", "-nc", " nd", "-nd", "non-commercial", "noderiv"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wiki_pages(session: requests.Session, query: str, max_pages: int):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "50",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime", "iiurlwidth": "1280",
        "maxlag": "5",
    }
    for _ in range(max_pages):
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
                print(f"  api retry {query!r}: {exc}", flush=True)
                time.sleep(3 + attempt * 3)
        else:
            return
        for page in (data.get("query") or {}).get("pages", {}).values():
            yield page
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)
        time.sleep(0.3)


def fetch(session: requests.Session, url: str) -> bytes | None:
    for attempt in range(3):
        try:
            r = session.get(url, timeout=60)
            if r.status_code in (429, 503):
                time.sleep(8 + attempt * 5)
                continue
            r.raise_for_status()
            if len(r.content) < 4096:
                return None
            return r.content
        except Exception:
            time.sleep(2 + attempt * 2)
    return None


def save(raw: bytes, dest: Path, min_width: int) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            if im.width < min_width:
                return None
            if max(im.size) > 1280:
                s = 1280 / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, "JPEG", quality=88)
            return im.size
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", default=None, help="comma list; default all")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--max-pages", type=int, default=5)
    ap.add_argument("--min-width", type=int, default=600)
    ap.add_argument("--out", default="data/mined_pools/wikimedia_stage_a")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else list(CATEGORIES)
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")

    out = PROJECT_ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    seen_titles: set[str] = set()
    seen_sha: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            seen_titles.add(r.get("title", ""))
            seen_sha.add(r.get("sha256", ""))

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    new_rows: list[dict] = []
    counts: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    rejected: Counter[str] = Counter()

    for category in cats:
        existing = sum(1 for r in new_rows if r["cause"] == category)
        if manifest_path.exists():
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("cause") == category:
                    existing += 1
        got = existing
        if got >= args.per_category:
            print(f"{category}: already {got}", flush=True)
            continue
        for query in QUERIES[category]:
            if got >= args.per_category:
                break
            for page in wiki_pages(session, query, args.max_pages):
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
                dest = out / "images" / category / f"{digest[:16]}.jpg"
                size = save(raw, dest, args.min_width)
                if size is None:
                    rejected["save"] += 1
                    continue
                row = {
                    "id": f"commons:{digest[:16]}",
                    "path": str(dest.relative_to(PROJECT_ROOT)),
                    "cause": category,
                    "source": "wikimedia_commons",
                    "title": title,
                    "page_url": info.get("descriptionurl", ""),
                    "image_url": url,
                    "licence": licence,
                    "licence_url": clean(md.get("LicenseUrl", {}).get("value", "")),
                    "artist": clean(md.get("Artist", {}).get("value", "")),
                    "credit": clean(md.get("Credit", {}).get("value", "")),
                    "sha256": digest,
                    "width": size[0],
                    "height": size[1],
                    "query": query,
                    "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                new_rows.append(row)
                seen_titles.add(title)
                seen_sha.add(digest)
                counts[category] += 1
                licences[licence] += 1
                got += 1
                # append immediately so a restart/kill never loses provenance; lock across workers
                with FileLock(str(manifest_path) + ".lock", timeout=120):
                    with manifest_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                    with (out / "attribution.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "id": row["id"], "title": row["title"], "page_url": row["page_url"],
                            "artist": row["artist"], "credit": row["credit"],
                            "licence": row["licence"], "licence_url": row["licence_url"],
                        }) + "\n")
                if got % 25 == 0:
                    print(f"  {category}: {got}", flush=True)
                time.sleep(args.sleep)
        print(f"wikimedia {category}: {counts[category]} new ({got} total)", flush=True)

    summary = {
        "new": len(new_rows), "total_in_manifest": len(seen_titles),
        "per_category_new": dict(counts), "licences_new": dict(licences),
        "rejected": dict(rejected), "measured": "actual downloads in this run",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
