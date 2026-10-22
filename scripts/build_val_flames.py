from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest, save_manifest
from firesight.types import ImageRecord


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a controlled-flame validation set (NOT testbench100) for threshold calibration.")
    parser.add_argument("--out", default="data/splits/val_flames.jsonl")
    parser.add_argument("--negatives", type=int, default=60)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    rows = []
    meta_path = PROJECT_ROOT / "hard_negatives/sources/val_flames/attribution.jsonl"
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = PROJECT_ROOT / row["file"]
        im = cv2.imread(str(path))
        if im is None:
            continue
        h, w = im.shape[:2]
        rows.append(ImageRecord(
            id=f"valflame:{Path(row['file']).stem[:70]}", path=row["file"], width=w, height=h,
            source="wikimedia_commons", scene="valflame", split="val_flames", labels=[],
            category=row["category"], origin=row.get("source_url", ""),
            meta={"has_fire": 1, "has_smoke": 0, "license": row.get("license", ""),
                  "artist": row.get("artist", ""), "purpose": "controlled flame positive for threshold calibration"},
        ))
    bench_ids = {r["id"] for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8"))}
    bench_files = {r["file"] for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8"))}
    test_neg = [r for r in load_manifest(PROJECT_ROOT / "data/splits/test.jsonl", root=PROJECT_ROOT) if not r.labels]
    test_neg = [r for r in test_neg if r.id not in bench_ids and Path(r.path).name not in bench_files]
    random.Random(args.seed).shuffle(test_neg)
    rows += test_neg[:args.negatives]
    dust_dir = PROJECT_ROOT / "hard_negatives/sources/construction_dust/images"
    for path in sorted(dust_dir.glob("ppe_2*.jpg")) + sorted(dust_dir.glob("ppe_3*.jpg")):
        im = cv2.imread(str(path))
        if im is None:
            continue
        h, w = im.shape[:2]
        rows.append(ImageRecord(
            id=f"valdust:{path.stem}", path=str(path.relative_to(PROJECT_ROOT)), width=w, height=h,
            source="local_negative", scene="construction_dust", split="val_flames", labels=[],
            category="construction_dust_plaster", origin="local Sample_videos ppe_2/ppe_3",
            meta={"has_fire": 0, "has_smoke": 0, "license": "proprietary internal",
                  "purpose": "construction / plaster dust hard negative"},
        ))
    save_manifest(PROJECT_ROOT / args.out, rows)
    print(json.dumps({"images": len(rows), "flames": sum(1 for r in rows if r.meta.get("has_fire")),
                      "negatives": sum(1 for r in rows if not r.meta.get("has_fire") and not r.meta.get("has_smoke")),
                      "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
