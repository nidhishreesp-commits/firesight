from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest, manifest_stats, save_manifest


def md5_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create immutable TEST lock and deterministic benchmark subsets.")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--bench-size", type=int, default=3000)
    parser.add_argument("--mini-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_dir = PROJECT_ROOT / args.split_dir
    test_path = split_dir / "test.jsonl"
    records = load_manifest(test_path, root=PROJECT_ROOT)
    train_scenes = {r.scene for r in load_manifest(split_dir / "train.jsonl", root=PROJECT_ROOT)}
    val_scenes = {r.scene for r in load_manifest(split_dir / "val.jsonl", root=PROJECT_ROOT)}
    overlap = {r.scene for r in records} & (train_scenes | val_scenes)
    if overlap:
        raise SystemExit(f"REFUSING TO FREEZE: {len(overlap)} test scenes also appear in train/val")

    stats = manifest_stats(records)
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file": str(test_path.relative_to(PROJECT_ROOT)),
        "sha256": md5_file(test_path),
        "images": stats["images"],
        "positive_images": stats["positive_images"],
        "negative_images": stats["negative_images"],
        "boxes": stats["boxes"],
        "fire_boxes": stats["fire_boxes"],
        "smoke_boxes": stats["smoke_boxes"],
        "scene_overlap_with_train_val": len(overlap),
        "rule": "Never train, tune thresholds, select models, or cherry-pick on this file. Report once.",
    }
    (split_dir / "TEST.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    rng = random.Random(args.seed)
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for rec in records:
        groups[(rec.source, rec.category)].append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: r.id)

    def stratified(size: int, out_name: str) -> None:
        total = sum(len(v) for v in groups.values())
        picks = []
        for key, members in groups.items():
            n = max(1, round(size * len(members) / total))
            n = min(n, len(members))
            picks.extend(rng.sample(members, n))
        rng.shuffle(picks)
        picks = picks[:size]
        save_manifest(split_dir / out_name, picks)
        data = {
            "source_file": "test.jsonl",
            "parent_sha256": lock["sha256"],
            "seed": args.seed,
            "images": len(picks),
            "positive_images": sum(1 for r in picks if r.labels),
            "negative_images": sum(1 for r in picks if not r.labels),
            "sha256": hashlib.sha256((split_dir / out_name).read_bytes()).hexdigest(),
        }
        (split_dir / (out_name.replace(".jsonl", ".lock.json"))).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(out_name, data)

    stratified(args.bench_size, "test_bench.jsonl")
    stratified(args.mini_size, "test_mini.jsonl")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
