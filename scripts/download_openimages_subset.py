"""Download a random Open Images V7 validation subset (image listing: CC BY 2.0).

Official Open Images statement: annotations are CC BY 4.0; images are listed as CC BY 2.0,
with a warning to verify each image.  This script records the dataset-level image licence and
the source URL per image.  Default 3,000 additional images (~600 MB), under the 2 GB cap.

Resumable and append-only: existing manifest rows are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

IDS_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{id}.jpg"
LICENCE = "CC BY 2.0 (Open Images V7 image listing; annotations CC BY 4.0)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--out", default="data/mined_pools/openimages")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"

    existing: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["id"])
                except Exception:
                    pass
    print("existing manifest rows:", len(existing), flush=True)

    r = requests.get(IDS_URL, timeout=120)
    r.raise_for_status()
    ids = [row["ImageID"] for row in csv.DictReader(io.StringIO(r.text))]
    rng = random.Random(args.seed)
    pool = [i for i in ids if i not in existing]
    pick = rng.sample(pool, min(args.n, len(pool)))
    print("validation ids:", len(ids), "available:", len(pool), "picked:", len(pick), flush=True)

    s = requests.Session()
    s.headers["User-Agent"] = "FireSightResearch/0.1"
    ok = fail = 0
    for i, iid in enumerate(pick):
        dest = out / "images" / f"{iid}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        try:
            resp = s.get(IMG_URL.format(id=iid), timeout=30)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content)
                ok += 1
                row = {"id": iid, "path": str(dest.relative_to(PROJECT_ROOT)),
                       "licence": LICENCE, "source": "Open Images V7 validation",
                       "source_url": IMG_URL.format(id=iid)}
                with manifest.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            else:
                fail += 1
        except Exception:
            fail += 1
        if (i + 1) % 100 == 0:
            print("downloaded", i + 1, "/", len(pick), "ok", ok, "fail", fail, flush=True)
    print(json.dumps({"picked": len(pick), "ok": ok, "fail": fail,
                      "manifest_rows": sum(1 for _ in manifest.open(encoding="utf-8"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
