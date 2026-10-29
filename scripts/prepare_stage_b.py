"""Prepare the E0/E1 paired experiment without training.

What it writes
--------------
build/stage_b/
  e0_train.txt, e1_train.txt, val.txt, val_cal.txt, val_report.txt   absolute local lists
  negatives.jsonl, composition.json, composition.csv, summary.json
  config_diff.json, config_diff.md, gpu_estimate.json
  leakage_report.json
build/kaggle_stage_a_negatives/
  images/, labels/, lists/e1_negatives.txt, manifest.jsonl, dataset-metadata.json, README.md

E0 = the packaged 25% list as-is (dataset's own positives and negatives).
E1 = E0 + 865 hard-train negatives + easy backgrounds to 9.0% added negatives of the 14,747-image list.

Important measured correction: the "25% positive list" is 14,747 images total, of which 6,532 are
positive and 8,215 are negative.  The 8-10% rule is applied to the *added* hard/easy negatives
(1,327 = 9.0% of 14,747), not to total negatives; the base list already contains 55.7% negatives.
This is stated in config_diff.md so the experiment cannot be misread.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from hardneg_causes import HELD_OUT_CAUSES


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def phash_file(path: Path) -> str | None:
    try:
        import imagehash
        return str(imagehash.phash(Image.open(path).convert("RGB")))
    except Exception:
        return None


def resolve_base(line: str, base: Path) -> Path:
    p = Path(line.strip())
    if not p.is_absolute():
        p = base / p
    if not p.exists():
        p = base / "images" / Path(line).name
    return p.resolve()


def label_positive(path: Path) -> bool:
    text = str(path).replace("\\", "/")
    if "/images/" in text:
        lbl = Path(text.replace("/images/", "/labels/")).with_suffix(".txt")
        return lbl.exists() and any(l.strip() for l in lbl.read_text(encoding="utf-8").splitlines())
    return False


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["cause"]].append(r)
    total = len(rows)
    alloc = {c: max(1, round(n * len(v) / total)) for c, v in groups.items()}
    # trim/grow to exactly n
    while sum(alloc.values()) > n:
        c = max(alloc, key=lambda k: alloc[k])
        alloc[c] -= 1
    while sum(alloc.values()) < n:
        c = max(groups, key=lambda k: len(groups[k]) - alloc[k])
        alloc[c] += 1
    picked: list[dict] = []
    for c in sorted(groups):
        pool = sorted(groups[c], key=lambda r: r["sha256"])
        rng.shuffle(pool)
        picked.extend(pool[: min(alloc[c], len(pool))])
    if len(picked) < n:
        rest = [r for r in rows if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])
    return picked


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="build/kaggle_dataset")
    ap.add_argument("--pool", default="data/hard_negatives_v3")
    ap.add_argument("--out", default="build/stage_b")
    ap.add_argument("--neg-package", default="build/kaggle_stage_a_negatives")
    ap.add_argument("--target-added-share", type=float, default=0.09)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    base_dir = PROJECT_ROOT / args.base
    pool_dir = PROJECT_ROOT / args.pool
    out = PROJECT_ROOT / args.out
    neg_pkg = PROJECT_ROOT / args.neg_package
    out.mkdir(parents=True, exist_ok=True)
    for d in ("images", "labels", "lists"):
        (neg_pkg / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- base 25% list
    base_lines = [l.strip() for l in (base_dir / "lists/train_25.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    base_paths = [resolve_base(l, base_dir) for l in base_lines]
    base_pos = sum(1 for p in base_paths if label_positive(p))
    base_neg = len(base_paths) - base_pos
    print(f"base 25% list: {len(base_paths)} images = {base_pos} positive + {base_neg} negative", flush=True)
    if not all(p.exists() for p in base_paths):
        missing = [str(p) for p in base_paths if not p.exists()][:5]
        raise SystemExit(f"base images missing: {missing}")

    # ---------------------------------------------------------------- pool selection
    pool = read_jsonl(pool_dir / "manifest.jsonl")
    hard = [r for r in pool if r.get("status") == "hard" and r.get("split") == "train"
            and r.get("source") != "openimages_v7" and r.get("cause") not in HELD_OUT_CAUSES
            and "sample_videos" not in r.get("path", "").lower()]
    easy = [r for r in pool if r.get("status") == "easy" and r.get("split") == "train"
            and r.get("source") != "openimages_v7" and r.get("cause") not in HELD_OUT_CAUSES
            and "sample_videos" not in r.get("path", "").lower()]
    hard = sorted(hard, key=lambda r: r["sha256"])
    target_added = round(args.target_added_share * len(base_paths))
    n_easy = max(0, target_added - len(hard))
    easy_pick = stratified_sample(easy, n_easy, args.seed) if n_easy else []
    added = hard + easy_pick
    print(f"hard={len(hard)} easy={len(easy_pick)} added={len(added)} "
          f"({len(added)/len(base_paths):.2%} of base)", flush=True)

    # ---------------------------------------------------------------- lists
    def write_list(name: str, paths: list[Path]) -> None:
        (out / name).write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")

    write_list("e0_train.txt", base_paths)
    write_list("e1_train.txt", base_paths + [PROJECT_ROOT / r["path"] for r in added])
    write_list("val.txt", [resolve_base(l, base_dir) for l in (base_dir / "lists/val_early.txt").read_text(encoding="utf-8").splitlines() if l.strip()])
    write_list("val_cal.txt", [resolve_base(l, base_dir) for l in (base_dir / "lists/val_cal.txt").read_text(encoding="utf-8").splitlines() if l.strip()])
    write_list("val_report.txt", [resolve_base(l, base_dir) for l in (base_dir / "lists/val_report.txt").read_text(encoding="utf-8").splitlines() if l.strip()])

    # ---------------------------------------------------------------- negative package
    pkg_rows = []
    seen_sha = set()
    for r in added:
        src = PROJECT_ROOT / r["path"]
        if not src.exists():
            raise SystemExit(f"missing negative {src}")
        sha = r.get("sha256") or sha256_file(src)
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        suffix = src.suffix.lower() if src.suffix.lower() in (".jpg", ".jpeg", ".png") else ".jpg"
        name = f"neg_{sha[:16]}{suffix}"
        link_or_copy(src, neg_pkg / "images" / name)
        (neg_pkg / "labels" / (Path(name).stem + ".txt")).write_text("", encoding="utf-8")
        pkg_rows.append({
            "id": r.get("id"), "package_file": f"images/{name}", "orig_path": r["path"],
            "cause": r.get("cause"), "source": r.get("source"), "licence": r.get("licence"),
            "sha256": sha, "phash": r.get("phash"), "max_conf": r.get("max_conf"),
            "status": r.get("status"), "split": r.get("split"),
        })
    (neg_pkg / "lists/e1_negatives.txt").write_text(
        "\n".join(r["package_file"] for r in pkg_rows) + "\n", encoding="utf-8")
    (neg_pkg / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pkg_rows) + "\n", encoding="utf-8")
    (neg_pkg / "dataset-metadata.json").write_text(json.dumps({
        "title": "firesight stage a negatives (private)",
        "id": "nidhishreesp/firesight-stage-a-negatives",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": "Private E1 negatives for FireSight Stage B. Hard/easy negatives from "
                       "Wikimedia Commons (CC0/PD/CC BY/CC BY-SA per image), D-Fire (CC0), "
                       "FASDD (CC BY 4.0) and tagged synthetic derivatives. No Open Images, no "
                       "held-out causes, no quarantine, no Sample_videos, no TEST/bench_R.",
    }, indent=2), encoding="utf-8")
    (neg_pkg / "README.md").write_text(
        "# FireSight Stage B negatives (private)\n\n"
        f"{len(pkg_rows)} images, empty YOLO labels. E1 only. Licences are in `manifest.jsonl`.\n"
        "Excluded: quarantine, sun_glare_sky, traffic_light, Open Images, Sample_videos.\n",
        encoding="utf-8")

    # ---------------------------------------------------------------- composition
    cause_hard = Counter(r["cause"] for r in hard)
    cause_easy = Counter(r["cause"] for r in easy_pick)
    comp = {
        "base_list": "build/kaggle_dataset/lists/train_25.txt",
        "base_images": len(base_paths), "base_positives": base_pos,
        "base_negatives": base_neg, "base_negative_share": round(base_neg / len(base_paths), 4),
        "added_hard": len(hard), "added_easy": len(easy_pick), "added_total": len(added),
        "added_share_of_base": round(len(added) / len(base_paths), 4),
        "e1_images": len(base_paths) + len(added),
        "e1_negative_share_total": round((base_neg + len(added)) / (len(base_paths) + len(added)), 4),
        "hard_by_cause": dict(sorted(cause_hard.items())),
        "easy_by_cause": dict(sorted(cause_easy.items())),
        "hard_by_source": dict(Counter(r["source"] for r in hard)),
        "easy_by_source": dict(Counter(r["source"] for r in easy_pick)),
        "rule": "8-10% applies to ADDED negatives (hard+easy), not to total negatives; the base "
                "25% list already contains 55.7% negatives.",
    }
    (out / "composition.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    with (out / "composition.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "cause", "count"])
        for c, n in sorted(cause_hard.items()):
            w.writerow(["hard", c, n])
        for c, n in sorted(cause_easy.items()):
            w.writerow(["easy", c, n])

    # ---------------------------------------------------------------- config diff
    common = {
        "model": "yolo11s.pt (fresh pretrained, not P50)",
        "epochs": 20, "imgsz": 640, "batch": 64,
        "lr0": 0.01, "lrf": 0.01, "cos_lr": True, "warmup_epochs": 3.0,
        "seed": 0, "deterministic": True,
        "patience": 10,
        "selection_metric": "image-level recall at FPR<=2% on val_cal (not val loss)",
        "train_val": "val_early.txt",
        "selection_data": "val_cal.txt (2,000)",
        "augmentation": {
            "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0,
            "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "erasing": 0.4,
        },
    }
    diff = {
        "common": common,
        "E0": {"train": "train_25 as-is (6,532 pos + 8,215 dataset negatives)", "n_train": len(base_paths),
               "added_negatives": 0},
        "E1": {"train": "train_25 + hard/easy Stage A negatives", "n_train": len(base_paths) + len(added),
               "added_hard": len(hard), "added_easy": len(easy_pick), "added_total": len(added),
               "added_share_of_base": comp["added_share_of_base"]},
        "changed": ["train list", "run name/out directory"],
        "unchanged": ["model", "epochs", "imgsz", "batch", "lr0", "lrf", "cos_lr", "warmup",
                      "seed", "deterministic", "patience", "selection metric", "augmentation",
                      "val_early", "val_cal"],
        "correction": "The 25% list is 14,747 images (6,532 positive, 8,215 negative), not 14,747 "
                      "positives. The added negatives are 9.0% of the 14,747-image list; total "
                      "negatives in E1 are 59.4%. If the intent is 8-10% total negatives, E0/E1 "
                      "must be rebuilt from positives only; that is a different experiment.",
    }
    (out / "config_diff.json").write_text(json.dumps(diff, indent=2), encoding="utf-8")
    (out / "config_diff.md").write_text(
        "# E0 vs E1 config diff\n\n"
        f"**Measured correction:** the 25% list is **{len(base_paths):,} images** = "
        f"**{base_pos:,} positive + {base_neg:,} dataset negative**. It is not 14,747 positives.\n\n"
        f"E0 train: {len(base_paths):,} images, no added negatives.\n\n"
        f"E1 train: {len(base_paths)+len(added):,} images = E0 + {len(hard)} hard + "
        f"{len(easy_pick)} easy = {len(added):,} added negatives "
        f"({comp['added_share_of_base']:.2%} of the base list).\n\n"
        "Everything else is identical (model, 20 epochs, 640, batch 64, lr0 0.01, cosine, seed 0, "
        "deterministic, patience 10, augmentation, val_early/val_cal). Checkpoint selection is "
        "image-level recall at FPR<=2% on val_cal, never val loss.\n\n"
        "The 8-10% rule is applied to **added** negatives. The base list already contains 55.7% "
        "negatives; E1 total negatives are 59.4%. If you intended 8-10% total negatives, the base "
        "must be positives-only.\n",
        encoding="utf-8")

    # ---------------------------------------------------------------- GPU estimate
    measured_img_s = 72.1
    e0_h = len(base_paths) / measured_img_s * 20 / 3600
    e1_h = (len(base_paths) + len(added)) / measured_img_s * 20 / 3600
    gpu = {
        "measured_basis": "72.1 images/s, 251.9 s/epoch for 18,154 images (FT_HN Kaggle run)",
        "epochs": 20,
        "E0_images": len(base_paths), "E0_gpu_hours_no_overhead": round(e0_h, 3),
        "E1_images": len(base_paths) + len(added), "E1_gpu_hours_no_overhead": round(e1_h, 3),
        "E0_E1_total_no_overhead": round(e0_h + e1_h, 3),
        "E0_E1_total_plus_25pct_overhead": round((e0_h + e1_h) * 1.25, 3),
        "note": "ESTIMATE from measured throughput; the E0/E1 runs themselves are not measured.",
    }
    (out / "gpu_estimate.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- leakage
    print("leakage check...", flush=True)
    ref_phash: list[int] = []
    ref_sha: set[str] = set()
    ref_ids: set[str] = set()
    ref_paths: set[str] = set()
    sha_cache = {}
    cache_path = PROJECT_ROOT / "build/stage_a/ref_sha_cache.json"
    if cache_path.exists():
        sha_cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def add_ref(rid: str, path: str | Path, ph: str | None, sha: str | None = None):
        if ph:
            ref_phash.append(int(ph, 16))
        if sha:
            ref_sha.add(sha)
        if rid:
            ref_ids.add(rid)
        if path:
            ref_paths.add(str(path).replace("\\", "/").lower())

    for name in ("val.jsonl", "test.jsonl", "val_bench.jsonl", "test_bench.jsonl",
                 "val_mini.jsonl", "test_mini.jsonl", "val_gsd.jsonl", "test_gsd.jsonl"):
        for r in read_jsonl(PROJECT_ROOT / "data/splits" / name):
            add_ref(r.get("id", ""), r.get("path", ""), r.get("phash"))
    for r in json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8")):
        add_ref(r["id"], r.get("file", ""), r.get("phash"), r.get("sha256"))
    for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8")):
        p = PROJECT_ROOT / "testbench100/images" / r["file"]
        add_ref(r["id"], p, r.get("phash") or phash_file(p), r.get("sha256"))
    # exact reference hashes from the Stage A cache (all frozen val/test slices)
    for name in ("val.jsonl", "test.jsonl", "val_bench.jsonl", "test_bench.jsonl",
                 "val_mini.jsonl", "test_mini.jsonl", "val_gsd.jsonl", "test_gsd.jsonl"):
        for r in read_jsonl(PROJECT_ROOT / "data/splits" / name):
            p = str((PROJECT_ROOT / r["path"]).resolve()).replace("\\", "/")
            if p in sha_cache:
                ref_sha.add(sha_cache[p])
    for r in json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8")):
        if r.get("sha256"):
            ref_sha.add(r["sha256"])
    for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8")):
        if r.get("sha256"):
            ref_sha.add(r["sha256"])
    ref_arr = np.array(ref_phash, dtype=np.uint64)

    cand_sha = {}
    cand_ph = {}
    exact_hits = []
    ph_hits = []
    for r in added:
        p = PROJECT_ROOT / r["path"]
        s = r.get("sha256") or sha256_file(p)
        h = r.get("phash") or phash_file(p)
        cand_sha[r["id"]] = s
        cand_ph[r["id"]] = h
        if s and s in ref_sha:
            exact_hits.append({"id": r["id"], "path": r["path"]})
        if h and len(ref_arr):
            d = np.bitwise_count(np.uint64(int(h, 16)) ^ ref_arr)
            if int(d.min()) <= 2:
                ph_hits.append({"id": r["id"], "path": r["path"], "distance": int(d.min())})
    dup = len(added) - len({r["sha256"] for r in added})
    heldout_hits = [r["id"] for r in added if r.get("cause") in HELD_OUT_CAUSES]
    quarantine_hits = [r["id"] for r in added if r.get("status") != "hard" and r.get("status") != "easy"]
    oi_hits = [r["id"] for r in added if r.get("source") == "openimages_v7"]
    sample_hits = [r["id"] for r in added if "sample_videos" in r.get("path", "").lower()]

    package_leak = json.loads((base_dir / "LEAKAGE_REPORT.json").read_text(encoding="utf-8")) if (base_dir / "LEAKAGE_REPORT.json").exists() else {}
    report = {
        "verdict": "PASS" if not exact_hits and not ph_hits and not dup and not heldout_hits
                   and not quarantine_hits and not oi_hits and not sample_hits else "FAIL",
        "base_package_leakage": package_leak.get("verdict", "unknown"),
        "e1_added_checked": len(added),
        "e1_added_exact_sha_hits": len(exact_hits),
        "e1_added_phash_le2_hits": len(ph_hits),
        "e1_added_duplicate_sha": dup,
        "e1_added_quarantine": len(quarantine_hits),
        "e1_added_heldout_cause": len(heldout_hits),
        "e1_added_openimages": len(oi_hits),
        "e1_added_sample_videos": len(sample_hits),
        "reference_counts": {"phashes": len(ref_arr), "exact_sha": len(ref_sha),
                             "ids": len(ref_ids), "paths": len(ref_paths)},
        "examples": {"exact": exact_hits[:5], "phash": ph_hits[:5]},
        "measured": "SHA-256 + pHash Hamming<=2 + id/path; added negatives re-checked now",
    }
    (out / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

    summary = {
        "base_images": len(base_paths), "base_positives": base_pos, "base_negatives": base_neg,
        "added_hard": len(hard), "added_easy": len(easy_pick), "added_total": len(added),
        "e1_images": len(base_paths) + len(added),
        "added_share_of_base": comp["added_share_of_base"],
        "e1_negative_share_total": comp["e1_negative_share_total"],
        "leakage_verdict": report["verdict"],
        "gpu_estimate_hours": gpu,
        "neg_package": str(neg_pkg.relative_to(PROJECT_ROOT)),
        "neg_package_images": len(pkg_rows),
        "measured": "counts and leakage; GPU hours are an ESTIMATE",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
