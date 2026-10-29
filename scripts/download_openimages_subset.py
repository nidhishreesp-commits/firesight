"""Download a random Open Images V7 validation subset (CC BY 4.0).

Records licence + source URL per image. Default 2,000 images (~400 MB), under the 2 GB cap.
"""
from __future__ import annotations

import argparse, csv, io, json, random, sys, time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

IDS_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{id}.jpg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--out", default="data/mined_pools/openimages")
    args = ap.parse_args()
    out = PROJECT_ROOT/args.out
    (out/"images").mkdir(parents=True, exist_ok=True)
    r = requests.get(IDS_URL, timeout=120)
    r.raise_for_status()
    ids = [row["ImageID"] for row in csv.DictReader(io.StringIO(r.text))]
    rng = random.Random(args.seed)
    pick = rng.sample(ids, min(args.n, len(ids)))
    print("validation ids:", len(ids), "picked:", len(pick), flush=True)
    s = requests.Session(); s.headers["User-Agent"] = "FireSightResearch/0.1"
    rows, ok, fail = [], 0, 0
    for i, iid in enumerate(pick):
        dest = out/"images"/f"{iid}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1; continue
        try:
            resp = s.get(IMG_URL.format(id=iid), timeout=30)
            if resp.status_code == 200 and resp.content:
                dest.write_bytes(resp.content); ok += 1
                rows.append({"id": iid, "path": str(dest.relative_to(PROJECT_ROOT)),
                             "licence": "CC BY 4.0", "source": "Open Images V7 validation",
                             "source_url": IMG_URL.format(id=iid)})
            else:
                fail += 1
        except Exception:
            fail += 1
        if (i+1) % 100 == 0:
            print("downloaded", i+1, "/", len(pick), "ok", ok, "fail", fail, flush=True)
    with (out/"manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row)+"\n")
    print(json.dumps({"picked": len(pick), "ok": ok, "fail": fail}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
