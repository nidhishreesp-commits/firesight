from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest

LABELS = {0: "fire", 1: "smoke"}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(records, out: Path, split: str) -> int:
    images = out / split / "images"
    labels = out / split / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    count = 0
    for rec in records:
        src_img = PROJECT_ROOT / rec.path
        if not src_img.exists():
            continue
        name = rec.id.replace(":", "_") + src_img.suffix
        link_or_copy(src_img, images / name)
        src_lbl = PROJECT_ROOT / "data/processed/labels" / rec.source.replace("local_seed", "local").replace("local_negative", "local") / rec.split / (src_img.stem + ".txt")
        if not src_lbl.exists():
            # fall back to any matching label file
            candidates = list((PROJECT_ROOT / "data/processed/labels").rglob(src_img.stem + ".txt"))
            src_lbl = candidates[0] if candidates else None
        dst_lbl = labels / (Path(name).stem + ".txt")
        if src_lbl and src_lbl.exists():
            link_or_copy(src_lbl, dst_lbl)
        else:
            dst_lbl.write_text("", encoding="utf-8")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a YOLO-format dataset with hardlinks and a balanced training subset.")
    parser.add_argument("--out", default="data/yolo")
    parser.add_argument("--train-positive", type=int, default=3200)
    parser.add_argument("--train-negative", type=int, default=3200)
    parser.add_argument("--val-size", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = PROJECT_ROOT / args.out
    rng = random.Random(args.seed)
    train = load_manifest(PROJECT_ROOT / "data/splits/train.jsonl", root=PROJECT_ROOT)
    val = load_manifest(PROJECT_ROOT / "data/splits/val_bench.jsonl", root=PROJECT_ROOT)
    positives = [r for r in train if r.labels]
    negatives = [r for r in train if not r.labels]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    train_pick = positives[:args.train_positive] + negatives[:args.train_negative]
    rng.shuffle(train_pick)
    val_pick = val[:args.val_size]
    n_train = write_split(train_pick, out, "train")
    n_val = write_split(val_pick, out, "val")
    data_yaml = {
        "path": str(out),
        "train": "train/images",
        "val": "val/images",
        "names": LABELS,
        "nc": 2,
    }
    import yaml
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    info = {"train": n_train, "val": n_val, "train_positives": sum(1 for r in train_pick if r.labels),
            "train_negatives": sum(1 for r in train_pick if not r.labels), "seed": args.seed}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
