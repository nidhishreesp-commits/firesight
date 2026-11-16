from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from ..types import ImageRecord


def sha256_file(path: str | Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(path: str | Path, root: Path | None = None) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(ImageRecord.from_dict(json.loads(line), root=root))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_manifest(path: str | Path, records: Sequence[ImageRecord]) -> None:
    write_jsonl(path, (r.to_dict() for r in records))


def manifest_stats(records: Sequence[ImageRecord]) -> dict:
    by_split = Counter(r.split for r in records)
    by_source = defaultdict(lambda: {"images": 0, "fire_boxes": 0, "smoke_boxes": 0, "negative_images": 0,
                                     "fire_images": 0, "smoke_images": 0, "both_images": 0})
    for r in records:
        s = by_source[r.source]
        s["images"] += 1
        fire = any(g.label == "fire" for g in r.labels)
        smoke = any(g.label == "smoke" for g in r.labels)
        s["fire_boxes"] += sum(1 for g in r.labels if g.label == "fire")
        s["smoke_boxes"] += sum(1 for g in r.labels if g.label == "smoke")
        if not r.labels:
            s["negative_images"] += 1
        if fire:
            s["fire_images"] += 1
        if smoke:
            s["smoke_images"] += 1
        if fire and smoke:
            s["both_images"] += 1
    total_boxes = sum(len(r.labels) for r in records)
    negatives = sum(1 for r in records if not r.labels)
    positives = len(records) - negatives
    return {
        "images": len(records),
        "positive_images": positives,
        "negative_images": negatives,
        "negative_ratio": round(negatives / len(records), 4) if records else 0.0,
        "boxes": total_boxes,
        "fire_boxes": sum(1 for r in records for g in r.labels if g.label == "fire"),
        "smoke_boxes": sum(1 for r in records for g in r.labels if g.label == "smoke"),
        "by_split": dict(sorted(by_split.items())),
        "by_split_negative": dict(sorted(Counter(r.split for r in records if not r.labels).items())),
        "by_source": {k: dict(v) for k, v in sorted(by_source.items())},
        "scenes": len({r.scene for r in records}),
    }


def write_stats_markdown(path: str | Path, stats: dict, title: str = "Dataset") -> None:
    lines = [f"# {title}", "", f"Generated from the frozen manifest. Images: {stats['images']} "
             f"({stats['positive_images']} positive / {stats['negative_images']} negative, "
             f"negative ratio {stats['negative_ratio']:.2%}), boxes: {stats['boxes']} "
             f"(fire {stats['fire_boxes']}, smoke {stats['smoke_boxes']}), scenes: {stats['scenes']}.", ""]
    lines += ["## By split", "", "| Split | Images | Negative |", "|---|---:|---:|"]
    for split, count in stats["by_split"].items():
        lines.append(f"| {split} | {count} | {stats['by_split_negative'].get(split, 0)} |")
    lines += ["", "## By source", "",
              "| Source | Images | Fire boxes | Smoke boxes | Negative images | Fire images | Smoke images | Both |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, s in stats["by_source"].items():
        lines.append(f"| {source} | {s['images']} | {s['fire_boxes']} | {s['smoke_boxes']} | "
                     f"{s['negative_images']} | {s['fire_images']} | {s['smoke_images']} | {s['both_images']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
