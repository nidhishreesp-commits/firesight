from __future__ import annotations

import logging
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

from ..types import ImageRecord

log = logging.getLogger(__name__)


def _phash_one(path: str) -> tuple[str, str | None, str | None]:
    try:
        import imagehash
        from PIL import Image
        with Image.open(path) as im:
            h = str(imagehash.phash(im.convert("RGB"), hash_size=8))
        return path, h, None
    except Exception as exc:
        return path, None, str(exc)


def compute_phashes(paths: Sequence[str], workers: int = 4, progress_every: int = 1000) -> dict[str, str]:
    result: dict[str, str] = {}
    if not paths:
        return result
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_phash_one, p) for p in paths]
        for idx, fut in enumerate(as_completed(futures), 1):
            path, h, err = fut.result()
            if h:
                result[path] = h
            elif err:
                log.warning("phash failed for %s: %s", path, err)
            if progress_every and idx % progress_every == 0:
                log.info("phashed %d/%d", idx, len(paths))
    return result


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def cluster_phashes(records: Sequence[ImageRecord], phashes: dict[str, str], threshold: int = 6) -> dict[str, int]:
    hashed = [r for r in records if r.path in phashes]
    uf = _UnionFind(len(hashed))
    bands: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    for idx, rec in enumerate(hashed):
        value = int(phashes[rec.path], 16)
        for band in range(4):
            bands[band][(value >> (band * 16)) & 0xFFFF].append(idx)
            for other_idx in bands[band][(value >> (band * 16)) & 0xFFFF]:
                if other_idx == idx:
                    continue
                if _hamming(phashes[hashed[other_idx].path], phashes[rec.path]) <= threshold:
                    uf.union(idx, other_idx)
    groups: dict[int, int] = {}
    cluster_counts: Counter[int] = Counter()
    for idx in range(len(hashed)):
        cluster_counts[uf.find(idx)] += 1
    for idx, rec in enumerate(hashed):
        root = uf.find(idx)
        groups[rec.id] = root
        if cluster_counts[root] > 1:
            rec.meta["duplicate_cluster"] = f"phash:{root}"
    return groups


SPLIT_PRIORITY = {"test": 3, "val": 2, "train": 1, "unspecified": 0}


def enforce_split_isolation(records: Sequence[ImageRecord]) -> tuple[list[ImageRecord], dict]:
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        key = rec.scene or rec.id
        cluster_splits[key].add(rec.split)
    dropped_test = 0
    val_to_train = 0
    kept: list[ImageRecord] = []
    for rec in records:
        key = rec.scene or rec.id
        splits = cluster_splits[key]
        if rec.split == "test" and len(splits - {"test"}) > 0:
            dropped_test += 1
            continue
        if rec.split == "val" and "train" in splits:
            rec.split = "train"
            val_to_train += 1
        kept.append(rec)
    stats = {"test_dropped_duplicate_or_scene_overlap": dropped_test, "val_moved_to_train": val_to_train}
    return kept, stats


def dedupe_records(
    records: Sequence[ImageRecord],
    workers: int = 4,
    threshold: int = 6,
    exact_threshold: int = 2,
    cache_path: Path | None = None,
) -> tuple[list[ImageRecord], dict]:
    import json

    cached: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [r.path for r in records if r.path not in cached]
    if missing:
        log.info("computing perceptual hashes for %d images", len(missing))
        cached.update(compute_phashes(missing, workers=workers))
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cached), encoding="utf-8")
    for rec in records:
        rec.phash = cached.get(rec.path)

    hashed = [r for r in records if r.path in cached]
    exact_roots = cluster_phashes(hashed, cached, threshold=exact_threshold)
    exact_groups: dict[int, list[ImageRecord]] = defaultdict(list)
    for rec in hashed:
        exact_groups[exact_roots[rec.id]].append(rec)

    kept: list[ImageRecord] = []
    exact_drops = 0
    for members in exact_groups.values():
        members_sorted = sorted(
            members,
            key=lambda r: (SPLIT_PRIORITY.get(r.split, 0), len(r.labels), r.path),
            reverse=True,
        )
        keep = members_sorted[0]
        kept.append(keep)
        for other in members_sorted[1:]:
            exact_drops += 1
            other.meta["exact_duplicate_of"] = keep.id
    unhashed = [r for r in records if r.path not in cached]
    kept.extend(unhashed)

    scene_roots = cluster_phashes(kept, cached, threshold=threshold)
    for rec in kept:
        rec.meta.pop("duplicate_cluster", None)
        root = scene_roots.get(rec.id)
        scene_key = f"phash:{root}" if root is not None else f"unique:{rec.id}"
        if not rec.scene or rec.scene == "unknown":
            rec.scene = scene_key
        elif rec.scene != scene_key:
            rec.meta["phash_scene"] = scene_key

    kept, isolation = enforce_split_isolation(kept)
    stats = {
        "input_images": len(records),
        "kept_images": len(kept),
        "exact_duplicate_drops": exact_drops,
        **isolation,
        "scene_threshold": threshold,
        "exact_threshold": exact_threshold,
    }
    log.info("dedupe stats: %s", stats)
    return kept, stats
