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
    "candle": ["candle flame", "lit candle", "candles burning"],
    "campfire": ["campfire", "bonfire", "fire pit"],
    "welding": ["welding sparks", "grinding sparks", "angle grinder sparks"],
    "fireworks": ["fireworks", "firecracker"],
    "fireplace": ["burning fireplace", "wood stove fire"],
    "gas_flame": ["gas stove flame", "bunsen burner flame", "lighter flame"],
}
ALLOWED = ("cc0", "public domain", "cc by", "cc-by")


def clean(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect CC-licensed controlled-flame validation images (NOT testbench100).")
    parser.add_argument("--out", default="hard_negatives/sources/val_flames")
    parser.add_argument("--per-query", type=int, default=3)
    parser.add_argument("--max", type=int, default=36)
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    bench_titles = set()
    idx = PROJECT_ROOT / "testbench100/index.json"
    if idx.exists():
        for row in json.loads(idx.read_text(encoding="utf-8")):
            if row.get("attribution"):
                bench_titles.add(row["attribution"].get("title", ""))
    meta_path = out / "attribution.jsonl"
    session = requests.Session()
    session.headers["User-Agent"] = "FireSightBench/0.1 (local research)"
    rows, seen = [], set()
    for category, queries in QUERIES.items():
        got = 0
        for query in queries:
            if got >= args.per_query or len(rows) >= args.max:
                break
            params = {"action": "query", "format": "json", "generator": "search",
                      "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "30",
                      "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": "1024"}
            try:
                data = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=40).json()
            except Exception as exc:
                print("query failed", query, exc)
                continue
            for page in list((data.get("query") or {}).get("pages", {}).values()):
                if got >= args.per_query or len(rows) >= args.max or "imageinfo" not in page:
                    continue
                title = page.get("title", "")
                if title in seen or title in bench_titles:
                    continue
                info = page["imageinfo"][0]
                md = info.get("extmetadata", {})
                lic = clean(md.get("LicenseShortName", {}).get("value", ""))
                low = lic.lower()
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
                    target.write_bytes(session.get(url, timeout=60).content)
                except Exception as exc:
                    print("download failed", title, exc)
                    continue
                seen.add(title)
                rows.append({"category": category, "title": title, "file": str(target.relative_to(PROJECT_ROOT)),
                             "source_url": info.get("descriptionurl", ""), "license": lic,
                             "license_url": clean(md.get("LicenseUrl", {}).get("value", "")),
                             "artist": clean(md.get("Artist", {}).get("value", ""))})
                got += 1
                print("saved", category, title, lic)
    meta_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(rows), "out": str(meta_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
