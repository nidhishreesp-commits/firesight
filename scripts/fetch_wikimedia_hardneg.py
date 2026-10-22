from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

QUERIES = {
    "candle": ["candle flame", "candlelight", "lit candle"],
    "match_lighter": ["burning match", "lighter flame"],
    "gas_stove": ["gas stove flame", "cooking flame"],
    "campfire": ["campfire", "bonfire"],
    "welding": ["welding sparks", "grinding sparks"],
    "fireworks": ["fireworks"],
    "fireplace": ["fireplace fire"],
}
ALLOWED = ("cc0", "public domain", "cc by", "cc-by")


def clean(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CC-licensed fire-adjacent hard-negative images from Wikimedia Commons.")
    parser.add_argument("--out", default="hard_negatives/sources/wikimedia")
    parser.add_argument("--per-query", type=int, default=2)
    parser.add_argument("--max", type=int, default=24)
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "attribution.jsonl"
    session = requests.Session()
    session.headers["User-Agent"] = "FireSightBench/0.1 (local research; contact: none)"
    rows = []
    seen = set()
    for category, queries in QUERIES.items():
        got = 0
        for query in queries:
            if got >= args.per_query:
                break
            params = {
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "30",
                "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1024",
            }
            try:
                data = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=40).json()
            except Exception as exc:
                print("query failed", query, exc)
                continue
            pages = list((data.get("query") or {}).get("pages", {}).values())
            for page in pages:
                if got >= args.per_query or len(rows) >= args.max:
                    break
                title = page.get("title", "")
                if title in seen or "imageinfo" not in page:
                    continue
                info = page["imageinfo"][0]
                md = info.get("extmetadata", {})
                license_name = clean(md.get("LicenseShortName", {}).get("value", ""))
                low = license_name.lower()
                if not any(a in low for a in ALLOWED) or "nc" in low or "nd" in low:
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue
                fname = re.sub(r"[^A-Za-z0-9_.-]+", "_", title.replace("File:", ""))[:100]
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fname += ".jpg"
                target = out / category / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    r = session.get(url, timeout=60)
                    r.raise_for_status()
                    target.write_bytes(r.content)
                except Exception as exc:
                    print("download failed", title, exc)
                    continue
                seen.add(title)
                row = {
                    "category": category, "query": query, "title": title,
                    "file": str(target.relative_to(PROJECT_ROOT)),
                    "source_url": info.get("descriptionurl", ""),
                    "license": license_name,
                    "license_url": clean(md.get("LicenseUrl", {}).get("value", "")),
                    "artist": clean(md.get("Artist", {}).get("value", "")),
                    "credit": clean(md.get("Credit", {}).get("value", "")),
                }
                rows.append(row)
                with meta_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                got += 1
                print("saved", category, title, license_name)
    print(json.dumps({"images": len(rows), "attribution": str(meta_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
