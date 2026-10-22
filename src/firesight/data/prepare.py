from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image

from ..types import Box, GroundTruth, ImageRecord
from .dedupe import dedupe_records
from .manifest import manifest_stats, save_manifest, write_jsonl, write_stats_markdown

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_CLASS_NAMES = {
    "dfire": {0: "smoke", 1: "fire"},
    "fasdd": {0: "fire", 1: "smoke"},
    "msfsdb": {0: "fire", 1: "smoke"},
    "local": {0: "fire", 1: "smoke"},
}


@dataclass
class PreparePaths:
    root: Path
    processed: Path = field(init=False)
    images: Path = field(init=False)
    labels: Path = field(init=False)
    splits: Path = field(init=False)
    manifests: Path = field(init=False)

    def __post_init__(self):
        self.processed = self.root / "data" / "processed"
        self.images = self.processed / "images"
        self.labels = self.processed / "labels"
        self.splits = self.root / "data" / "splits"
        self.manifests = self.processed / "manifests"
        for p in (self.images, self.labels, self.splits, self.manifests):
            p.mkdir(parents=True, exist_ok=True)


def _category(labels: Sequence[GroundTruth], filename: str = "") -> str:
    fire = any(g.label == "fire" for g in labels)
    smoke = any(g.label == "smoke" for g in labels)
    if fire and smoke:
        return "both"
    if fire:
        return "fire"
    if smoke:
        return "smoke"
    low = filename.lower()
    if "neitherfirenorsmoke" in low:
        return "none"
    return "none"


def _parse_yolo(text: str, width: int, height: int, class_names: dict[int, str], source_name: str) -> list[GroundTruth]:
    labels: list[GroundTruth] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        cls = class_names.get(cls_id)
        if cls is None:
            log.warning("%s: unknown class id %s", source_name, cls_id)
            continue
        if bw <= 0.001 or bh <= 0.001:
            continue
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        box = Box(x1, y1, x2, y2)
        if box.area < 16.0:
            continue
        labels.append(GroundTruth(label=cls, box=box))
    return labels


def _write_image(path: Path, data: bytes | None = None, src: Path | None = None) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if data is not None:
        path.write_bytes(data)
    elif src is not None:
        try:
            os.link(src, path)
        except OSError:
            shutil.copy2(src, path)
    else:
        raise ValueError("data or src required")
    with Image.open(path) as im:
        return im.size


def _write_label(path: Path, labels: Sequence[GroundTruth], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {"fire": 0, "smoke": 1}
    lines = []
    for gt in labels:
        if gt.box.area <= 0:
            continue
        cx = ((gt.box.x1 + gt.box.x2) / 2) / width
        cy = ((gt.box.y1 + gt.box.y2) / 2) / height
        bw = (gt.box.x2 - gt.box.x1) / width
        bh = (gt.box.y2 - gt.box.y1) / height
        lines.append(f"{order[gt.label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _zip_image_entries(zf: zipfile.ZipFile) -> Iterator[tuple[str, str, str]]:
    for name in zf.namelist():
        if not name.lower().endswith(tuple(IMAGE_EXTS)):
            continue
        parts = name.split("/")
        split = next((p for p in parts if p in {"train", "valid", "val", "test"}), "unspecified")
        if split == "valid":
            split = "val"
        yield name, split, parts[-1]


def prepare_dfire(root: Path, zip_path: Path, limit: int | None = None, split_map: dict[str, str] | None = None) -> list[ImageRecord]:
    paths = PreparePaths(root)
    records: list[ImageRecord] = []
    class_names = SOURCE_CLASS_NAMES["dfire"]
    with zipfile.ZipFile(zip_path) as zf:
        entries = list(_zip_image_entries(zf))
        if limit:
            by_split: dict[str, list] = defaultdict(list)
            for e in entries:
                by_split[e[1]].append(e)
            entries = [e for split_entries in by_split.values() for e in split_entries[:limit]]
        for idx, (name, split, basename) in enumerate(entries, 1):
            split = (split_map or {}).get(split, split)
            if "/images/" not in name:
                continue
            label_name = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            text = zf.read(label_name).decode("utf-8", errors="replace") if label_name in zf.namelist() else ""
            data = zf.read(name)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
            except Exception as exc:
                log.warning("skip unreadable %s: %s", name, exc)
                continue
            labels = _parse_yolo(text, width, height, class_names, name)
            rel = f"data/processed/images/dfire/{split}/{basename}"
            out_img = root / rel
            if not out_img.exists():
                _write_image(out_img, data=data)
            _write_label(paths.labels / "dfire" / split / (Path(basename).stem + ".txt"), labels, width, height)
            rec = ImageRecord(
                id=f"dfire:{split}:{Path(basename).stem}",
                path=rel, width=width, height=height,
                source="dfire", scene="unknown", split=split,
                labels=labels, category=_category(labels, basename),
                origin=f"kaggle:sayedgamal99/smoke-fire-detection-yolo/{split}",
                meta={"class_remap": "source 0=smoke,1=fire -> unified 0=fire,1=smoke"},
            )
            records.append(rec)
            if idx % 2000 == 0:
                log.info("dfire prepared %d/%d", idx, len(entries))
    return records


def prepare_fasdd(root: Path, zip_path: Path, limit: int | None = None, split_map: dict[str, str] | None = None) -> list[ImageRecord]:
    paths = PreparePaths(root)
    records: list[ImageRecord] = []
    class_names = SOURCE_CLASS_NAMES["fasdd"]
    with zipfile.ZipFile(zip_path) as zf:
        entries = list(_zip_image_entries(zf))
        if limit:
            by_split: dict[str, list] = defaultdict(list)
            for e in entries:
                by_split[e[1]].append(e)
            entries = [e for split_entries in by_split.values() for e in split_entries[:limit]]
        names = set(zf.namelist())
        for idx, (name, split, basename) in enumerate(entries, 1):
            split = (split_map or {}).get(split, split)
            if "/images/" not in name:
                continue
            label_name = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            text = zf.read(label_name).decode("utf-8", errors="replace") if label_name in names else ""
            data = zf.read(name)
            try:
                with Image.open(io.BytesIO(data)) as im:
                    width, height = im.size
            except Exception as exc:
                log.warning("skip unreadable %s: %s", name, exc)
                continue
            labels = _parse_yolo(text, width, height, class_names, name)
            rel = f"data/processed/images/fasdd/{split}/{basename}"
            out_img = root / rel
            if not out_img.exists():
                _write_image(out_img, data=data)
            _write_label(paths.labels / "fasdd" / split / (Path(basename).stem + ".txt"), labels, width, height)
            records.append(ImageRecord(
                id=f"fasdd:{split}:{Path(basename).stem}",
                path=rel, width=width, height=height,
                source="fasdd", scene="unknown", split=split,
                labels=labels, category=_category(labels, basename),
                origin="huggingface:weslleyskah/fasdd_cv",
                meta={"class_remap": "source 0=fire,1=smoke -> unified unchanged"},
            ))
            if idx % 5000 == 0:
                log.info("fasdd prepared %d/%d", idx, len(entries))
    return records


def prepare_local(root: Path, local_root: Path, limit: int | None = None) -> list[ImageRecord]:
    paths = PreparePaths(root)
    records: list[ImageRecord] = []
    class_names = SOURCE_CLASS_NAMES["local"]

    positives = local_root / "fire_dataset" / "positives"
    negatives_dirs = [
        ("web_negative", local_root / "fire_dataset" / "web_negatives" / "images"),
        ("hard_negative_video", local_root / "fire_hardneg" / "images"),
    ]
    for img in sorted((positives / "images").glob("*.jpg")):
        stem = img.stem
        if stem.startswith("pos_fire_2_"):
            scene, split = "local:positive_video_2", "val"
        else:
            scene, split = "local:positive_video_1", "train"
        label_file = positives / "labels" / (stem + ".txt")
        with Image.open(img) as im:
            width, height = im.size
        labels = _parse_yolo(label_file.read_text() if label_file.exists() else "", width, height, class_names, stem)
        rel = f"data/processed/images/local/{split}/{img.name}"
        out = root / rel
        if not out.exists():
            _write_image(out, src=img)
        _write_label(paths.labels / "local" / split / (stem + ".txt"), labels, width, height)
        records.append(ImageRecord(
            id=f"local:{split}:{stem}:pos", path=rel, width=width, height=height,
            source="local_seed", scene=scene, split=split, labels=labels,
            category=_category(labels, img.name), origin="local:fire_dataset/positives",
        ))
        if limit and len(records) >= limit:
            break

    for neg_name, neg_dir in negatives_dirs:
        imgs = sorted(neg_dir.glob("*.jpg"))
        for img in imgs:
            stem = img.stem
            split = "val" if neg_name == "hard_negative_video" else "train"
            scene = f"local:{neg_name}:{stem.rsplit('_', 1)[0]}"
            with Image.open(img) as im:
                width, height = im.size
            rel = f"data/processed/images/local/{split}/{img.name}"
            out = root / rel
            if not out.exists():
                _write_image(out, src=img)
            _write_label(paths.labels / "local" / split / (stem + ".txt"), [], width, height)
            records.append(ImageRecord(
                id=f"local:{split}:{stem}:{neg_name}", path=rel, width=width, height=height,
                source="local_negative", scene=scene, split=split, labels=[],
                category="none", origin=f"local:{neg_name}",
            ))
    return records


def build_splits(records: Sequence[ImageRecord], paths: PreparePaths) -> dict:
    for split in ("train", "val", "test"):
        subset = [r for r in records if r.split == split]
        save_manifest(paths.splits / f"{split}.jsonl", subset)
        log.info("%s: %d images", split, len(subset))
    all_stats = manifest_stats(records)
    write_stats_markdown(paths.splits / "README.md", all_stats, "Frozen dataset splits")
    preview = {
        "splits": all_stats["by_split"],
        "sources": all_stats["by_source"],
        "counts": {k: all_stats[k] for k in ("images", "positive_images", "negative_images", "boxes", "fire_boxes", "smoke_boxes", "scenes")},
    }
    (paths.processed / "stats.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
    save_manifest(paths.manifests / "all.jsonl", records)
    return all_stats


def prepare_all(
    root: Path,
    sources: Sequence[str],
    local_root: Path | None = None,
    limit_per_split: int | None = None,
    workers: int = 4,
    phash_threshold: int = 6,
    split_map: dict[str, str] | None = None,
) -> dict:
    records: list[ImageRecord] = []
    if "dfire" in sources:
        zip_path = root / "data" / "raw" / "dfire.zip"
        if zip_path.exists():
            records += prepare_dfire(root, zip_path, limit=limit_per_split, split_map=split_map)
        else:
            log.warning("D-Fire zip not found: %s", zip_path)
    if "fasdd" in sources:
        zip_path = root / "data" / "raw" / "fasdd" / "FASDD_CV.zip"
        if zip_path.exists():
            records += prepare_fasdd(root, zip_path, limit=limit_per_split, split_map=split_map)
        else:
            log.warning("FASDD zip not found: %s", zip_path)
    if "local" in sources:
        if local_root is None:
            raise ValueError("local_root required for local source")
        records += prepare_local(root, local_root, limit=limit_per_split)
    if not records:
        raise RuntimeError("No records prepared; check downloads")

    before = len(records)
    records, dedupe_stats = dedupe_records(
        records, workers=workers, threshold=phash_threshold,
        cache_path=root / "data" / "processed" / "manifests" / "phash_cache.json",
    )
    log.info("dedupe: %d -> %d images", before, len(records))
    paths = PreparePaths(root)
    stats = build_splits(records, paths)
    stats["dedupe"] = dedupe_stats
    stats["sources_requested"] = list(sources)
    (root / "data" / "processed" / "stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats
