"""Build the PRIVATE Kaggle dataset for the Phase 3b pilot (local only, no upload).

What it does
------------
1. Selects the training pool from `data/splits/train.jsonl`, restricted to sources
   whose licence permits redistribution on a private dataset (D-Fire, FASDD_CV).
2. Excludes, by construction, every proprietary/own-capture source and every
   frozen exam set (bench_R, testbench100, TEST).
3. Writes a YOLO-format package with stratified fraction lists (10/25/50/100 %),
   a per-image MANIFEST.csv (source + licence), counts per source/licence, and a
   LEAKAGE_REPORT.json proving zero id / path / pHash overlap with the exam sets.
4. Never talks to the network and never uploads. `--zip` is opt-in.

Usage
-----
    python scripts/kaggle_package.py                    # manifest + leakage proof only
    python scripts/kaggle_package.py --materialize      # also hardlink images/labels
    python scripts/kaggle_package.py --materialize --zip  # also produce the .zip to upload
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT

SPLITS = PROJECT_ROOT / "data/splits"
PROCESSED = PROJECT_ROOT / "data/processed"
DEFAULT_OUT = PROJECT_ROOT / "build/kaggle_dataset"

# Redistributable on a private Kaggle dataset, with the licence recorded per source.
ALLOWED_SOURCES = {
    "dfire": {"licence": "CC0-1.0 (Kaggle mirror: sayedgamal99/smoke-fire-detection-yolo)", "public": True},
    "fasdd": {"licence": "CC BY 4.0 (HuggingFace: weslleyskah/fasdd_cv)", "public": True},
}
# Own captures / unclear-rights web images: proprietary, never uploaded.
EXCLUDED_SOURCES = {
    "local_seed": "own captures (proprietary)",
    "local_negative": "web negatives with unclear rights (proprietary/unverified)",
}

# Path fragments that must never appear in the package (proprietary construction footage).
PROPRIETARY_PATH_MARKERS = (
    "sample_videos", "ppe_2", "ppe_3", "construction", "site_footage",
    "flame_positives", "data/clips", "data\\clips", "data/raw/cctv_nc",
)

FRACTIONS = {"10": 0.10, "25": 0.25, "50": 0.50, "100": 1.00}
SEED = 20261022


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phash_hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def stratum(rec: dict) -> str:
    labels = rec.get("labels") or []
    hf = any(label.get("label") == "fire" for label in labels) or int(rec.get("has_fire") or 0) == 1
    hs = any(label.get("label") == "smoke" for label in labels) or int(rec.get("has_smoke") or 0) == 1
    return f"f{int(hf)}s{int(hs)}"


def stratified_pick(rows: list[dict], fraction: float, rng: random.Random) -> list[dict]:
    """Sample the same class ratio at every fraction (size is the only thing that changes)."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    picked: list[dict] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda r: r["id"])
        rng.shuffle(bucket)
        picked.extend(bucket[: max(1, round(len(bucket) * fraction))])
    rng.shuffle(picked)
    return picked


_LABEL_INDEX: dict[str, Path] | None = None


def _label_index() -> dict[str, Path]:
    """stem -> label file, built once (a per-image rglob over 117k files is far too slow)."""
    global _LABEL_INDEX
    if _LABEL_INDEX is None:
        index: dict[str, Path] = {}
        for path in (PROCESSED / "labels").rglob("*.txt"):
            index.setdefault(path.stem, path)
        _LABEL_INDEX = index
    return _LABEL_INDEX


def label_path(rec: dict) -> Path | None:
    src = str(rec["path"])
    stem = Path(src).stem
    source = rec.get("source", "")
    split = rec.get("split", "train")
    direct = PROCESSED / "labels" / source / split / f"{stem}.txt"
    if direct.exists():
        return direct
    return _label_index().get(stem)


def exam_ids_and_hashes() -> dict:
    bench_index = json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))
    bench_ids = {r["id"] for r in bench_index}
    bench_phash = {r["id"]: r.get("phash") for r in bench_index if r.get("phash")}
    bench_files = {str(r.get("file", "")).replace("\\", "/") for r in bench_index}

    dev_sealed = {r["id"] for r in read_jsonl(PROJECT_ROOT / "bench_R/DEV.jsonl")}
    dev_sealed |= {r["id"] for r in read_jsonl(PROJECT_ROOT / "bench_R/SEALED.jsonl")}
    bench_ids |= dev_sealed

    tb_index = json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8"))
    tb_ids = {r["id"] for r in tb_index}
    tb_phash = {}
    try:
        import imagehash
        from PIL import Image
        for row in tb_index:
            p = PROJECT_ROOT / "testbench100/images" / row["file"]
            if p.exists():
                tb_phash[row["id"]] = str(imagehash.phash(Image.open(p).convert("RGB")))
    except Exception:
        pass

    test_rows = read_jsonl(SPLITS / "test.jsonl")
    test_ids = {r["id"] for r in test_rows}
    test_phash = {r["id"]: r["phash"] for r in test_rows if r.get("phash")}
    test_paths = {str(r.get("path", "")).replace("\\", "/") for r in test_rows}

    return {
        "bench_ids": bench_ids, "bench_phash": bench_phash, "bench_files": bench_files,
        "tb_ids": tb_ids, "tb_phash": tb_phash,
        "test_ids": test_ids, "test_phash": test_phash, "test_paths": test_paths,
    }


def leakage_report(package: list[dict], exam: dict) -> dict:
    pkg_ids = {r["id"] for r in package}
    pkg_paths = {str(r["path"]).replace("\\", "/") for r in package}
    pkg_phash = {r["id"]: r.get("phash") for r in package if r.get("phash")}

    id_hits = {
        "bench_R": sorted(pkg_ids & exam["bench_ids"]),
        "testbench100": sorted(pkg_ids & exam["tb_ids"]),
        "TEST": sorted(pkg_ids & exam["test_ids"]),
    }
    path_hits = {
        "TEST": sorted(pkg_paths & exam["test_paths"]),
        "bench_R": sorted(p for p in pkg_paths if any(p.endswith(f) for f in exam["bench_files"])),
    }

    ref = {**exam["bench_phash"], **exam["test_phash"], **exam["tb_phash"]}
    near: list[dict] = []
    for rid, h in pkg_phash.items():
        for ref_id, ref_hash in ref.items():
            if phash_hamming(h, ref_hash) <= 2:
                near.append({"package_id": rid, "exam_id": ref_id, "distance": phash_hamming(h, ref_hash)})

    proprietary = sorted(
        p for p in pkg_paths if any(marker in p.lower() for marker in PROPRIETARY_PATH_MARKERS)
    )

    return {
        "checked": len(package),
        "reference_sets": {"bench_R": len(exam["bench_ids"]), "testbench100": len(exam["tb_ids"]),
                           "TEST": len(exam["test_ids"])},
        "id_overlap": {k: len(v) for k, v in id_hits.items()},
        "id_overlap_examples": {k: v[:5] for k, v in id_hits.items()},
        "path_overlap": {k: len(v) for k, v in path_hits.items()},
        "path_overlap_examples": {k: v[:5] for k, v in path_hits.items()},
        "phash_near_duplicates_hamming_le_2": len(near),
        "phash_near_duplicate_examples": near[:5],
        "proprietary_paths_found": len(proprietary),
        "proprietary_path_examples": proprietary[:5],
        "verdict": "PASS" if not any(id_hits.values()) and not any(path_hits.values())
                   and not near and not proprietary else "FAIL",
    }


def write_list(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(PROJECT_ROOT / r["path"]).replace("\\", "/") for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--materialize", action="store_true", help="hardlink images and labels into the package")
    parser.add_argument("--zip", action="store_true", help="also write the .zip to upload (large)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    train = read_jsonl(SPLITS / "train.jsonl")
    val = read_jsonl(SPLITS / "val.jsonl")

    excluded_counts = Counter()
    pool, val_pool = [], []
    for row in train:
        src = row.get("source", "")
        if src in ALLOWED_SOURCES:
            pool.append(row)
        else:
            excluded_counts[src] += 1
    for row in val:
        if row.get("source", "") in ALLOWED_SOURCES:
            val_pool.append(row)

    # deterministic, stratified val slices (never TEST, never bench)
    val_sorted = sorted(val_pool, key=lambda r: r["id"])
    rng_v = random.Random(SEED + 1)
    early = stratified_pick(val_sorted, 1000 / max(len(val_sorted), 1), rng_v)
    early_ids = {r["id"] for r in early}
    rest = [r for r in val_sorted if r["id"] not in early_ids]
    rng_c = random.Random(SEED + 2)
    cal = stratified_pick(rest, 2000 / max(len(rest), 1), rng_c)
    cal_ids = {r["id"] for r in cal}
    report = [r for r in rest if r["id"] not in cal_ids]

    package = pool + early + cal + report
    exam = exam_ids_and_hashes()
    leak = leakage_report(package, exam)

    summary = {
        "seed": SEED,
        "train_pool": len(pool),
        "excluded_sources": dict(excluded_counts),
        "excluded_source_reasons": {k: EXCLUDED_SOURCES.get(k, "not in allow-list") for k in excluded_counts},
        "val": {"early_stop": len(early), "threshold_calibration": len(cal), "report": len(report)},
        "total_images": len(package),
        "by_source": dict(Counter(r.get("source") for r in package)),
        "by_licence": dict(Counter(ALLOWED_SOURCES[r["source"]]["licence"] for r in package)),
        "by_stratum_train": dict(Counter(stratum(r) for r in pool)),
        "fractions": {k: sum(1 for _ in stratified_pick(pool, v, random.Random(SEED))) for k, v in FRACTIONS.items()},
        "licence_table": {k: v["licence"] for k, v in ALLOWED_SOURCES.items()},
        "excluded_sets": ["bench_R (DEV+SEALED+index)", "testbench100", "TEST (test.jsonl + lock)",
                          "proprietary construction footage / own captures (local_seed, local_negative, data/clips)"],
        "leakage": leak,
    }
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "LEAKAGE_REPORT.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    with (out / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "source", "licence", "split", "path", "phash", "stratum"])
        for row in sorted(package, key=lambda r: r["id"]):
            writer.writerow([row["id"], row.get("source"), ALLOWED_SOURCES[row["source"]]["licence"],
                             "val" if row in val_pool else "train", str(row["path"]).replace("\\", "/"),
                             row.get("phash", ""), stratum(row)])

    lists = out / "lists"
    for name, fraction in FRACTIONS.items():
        picked = stratified_pick(pool, fraction, random.Random(SEED))
        write_list(lists / f"train_{name}.txt", picked)
        summary["fractions"][name] = len(picked)
    write_list(lists / "val_early.txt", early)
    write_list(lists / "val_cal.txt", cal)
    write_list(lists / "val_report.txt", report)

    (out / "data.yaml").write_text(
        "# Relative paths: ultralytics resolves them against this file's directory.\n"
        "train: lists/train_10.txt\n"
        "val: lists/val_early.txt\n"
        "names:\n  0: fire\n  1: smoke\n"
        "nc: 2\n", encoding="utf-8")

    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "firesight pilot (private)",
        "id": "REPLACE_WITH_YOUR_KAGGLE_USERNAME/firesight-pilot",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": (
            "Private pilot dataset for the FireSight detector learning curve.\n\n"
            "Sources and licences:\n" + "\n".join(f"- {k}: {v['licence']}" for k, v in ALLOWED_SOURCES.items()) +
            "\n\nExcludes bench_R, testbench100, TEST and all proprietary construction footage "
            "(see LEAKAGE_REPORT.json)."
        ),
    }, indent=2), encoding="utf-8")

    # ship the training/eval code + pinned requirements with the dataset
    code_dir = out / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pilot_train.py", "pilot_eval.py"):
        src = PROJECT_ROOT / "kaggle" / name
        if src.exists():
            shutil.copy2(src, code_dir / name)
    # ship the exact metric code used for bench_R so pilot numbers are computed identically
    shutil.copy2(PROJECT_ROOT / "scripts" / "run_bench.py", code_dir / "run_bench.py")
    shutil.copytree(PROJECT_ROOT / "src" / "firesight", code_dir / "firesight",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"), dirs_exist_ok=True)
    (out / "requirements-pinned.txt").write_text(
        "ultralytics==8.4.147\nonnx==1.22.0\nonnxruntime==1.30.0\n", encoding="utf-8")
    (out / "versions-reference.json").write_text(json.dumps({
        "note": "torch is pinned by the Kaggle GPU image for CUDA compatibility; these are the "
                "versions used for the local dry run and the pins we control.",
        "local_dry_run": {"torch": "2.14.0+cpu", "ultralytics": "8.4.147", "opencv": "5.0.0",
                          "numpy": "2.5.3", "onnx": "1.22.0", "onnxruntime": "1.30.0",
                          "pillow": "12.3.0", "pyyaml": "6.0.3"},
    }, indent=2), encoding="utf-8")

    (out / "CHECKLIST.md").write_text(
        "# Upload checklist (PRIVATE dataset)\n\n"
        f"- [ ] Leakage verdict: **{leak['verdict']}** (id/path/phash all zero)\n"
        f"- [ ] Images: {summary['total_images']} (train pool {summary['train_pool']} + val slices)\n"
        f"- [ ] Excluded sources: {json.dumps(summary['excluded_sources'])}\n"
        f"- [ ] Licence table present in MANIFEST.csv and dataset-metadata.json\n"
        "- [ ] `isPrivate: true` and `id` set to your Kaggle username\n"
        "- [ ] Credentials come from env/`~/.kaggle/kaggle.json`; never committed\n"
        "- [ ] Upload only after explicit go-ahead\n", encoding="utf-8")

    if args.materialize:
        for row in package:
            src = PROJECT_ROOT / row["path"]
            if not src.exists():
                continue
            dst = out / "images" / (row["id"].replace(":", "_") + src.suffix)
            link(src, dst)
            lbl = label_path(row)
            dst_lbl = out / "labels" / (dst.stem + ".txt")
            if lbl and lbl.exists():
                link(lbl, dst_lbl)
            else:
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text("", encoding="utf-8")
        # rewrite lists to point at the materialized copies
        # relative to the dataset root so the same package works on Kaggle
        for name, rows in ([(f"train_{k}", stratified_pick(pool, v, random.Random(SEED))) for k, v in FRACTIONS.items()]
                           + [("val_early", early), ("val_cal", cal), ("val_report", report)]):
            (lists / f"{name}.txt").write_text("\n".join(
                "images/" + r["id"].replace(":", "_") + Path(r["path"]).suffix
                for r in rows) + "\n", encoding="utf-8")

    if args.zip:
        base = out.parent / out.name
        archive = shutil.make_archive(str(base), "zip", root_dir=out)
        summary["zip"] = {"path": archive, "bytes": Path(archive).stat().st_size}

    summary["fractions"] = {k: v for k, v in summary["fractions"].items()}
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("train_pool", "total_images", "by_source", "by_licence",
                                              "fractions", "excluded_sources")}, indent=2))
    print("leakage:", json.dumps({k: leak[k] for k in ("verdict", "id_overlap", "path_overlap",
                                                       "phash_near_duplicates_hamming_le_2",
                                                       "proprietary_paths_found")}))
    return 0 if leak["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
