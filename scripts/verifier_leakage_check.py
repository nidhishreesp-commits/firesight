"""Leakage check: verifier training crops vs every evaluation set.

Compares source images, not crops: exact SHA-256, pHash Hamming<=2, id/path, scene id, and
source video ids.  Evaluation sets: val_cal, val_report, bench_R DEV, independent negatives,
and a 1-in-5s sample of the 11 Sample_videos frames.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha_file(p: Path):
    try:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except Exception:
        return None


def phash_file(p: Path):
    try:
        import imagehash
        from PIL import Image
        return str(imagehash.phash(Image.open(p).convert("RGB")))
    except Exception:
        return None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="build/verifier/sample_train_v2_clean.json")
    args = ap.parse_args()
    train_rows = {r["id"]: r for r in read_jsonl(PROJECT_ROOT / "data/splits/train.jsonl")}
    train_sample = json.loads((PROJECT_ROOT / args.sample).read_text(encoding="utf-8"))
    train_ids = {r["id"] for r in train_sample}
    train_paths = [PROJECT_ROOT / r["path"] for r in train_sample]
    train_ph = {r["id"]: train_rows[r["id"]].get("phash") for r in train_sample if r["id"] in train_rows}
    train_scenes = {train_rows[r["id"]].get("scene") for r in train_sample if r["id"] in train_rows and train_rows[r["id"]].get("scene")}
    train_sha = {sha_file(p) for p in train_paths if p.exists()}
    train_sha.discard(None)
    train_arr = np.array([int(v, 16) for v in train_ph.values() if v], dtype=np.uint64)
    print(f"training source images: {len(train_sample)} sha={len(train_sha)} phash={len(train_arr)} scenes={len(train_scenes)}", flush=True)

    sets = {}
    # val_cal + val_report
    val = read_jsonl(PROJECT_ROOT / "data/splits/val.jsonl")
    sets["val"] = [{"id": r["id"], "path": r["path"], "phash": r.get("phash"), "scene": r.get("scene")} for r in val]
    # TEST
    test = read_jsonl(PROJECT_ROOT / "data/splits/test.jsonl")
    sets["TEST"] = [{"id": r["id"], "path": r["path"], "phash": r.get("phash"), "scene": r.get("scene")} for r in test]
    # bench_R DEV
    bench = [r for r in json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))
             if r.get("kind") in ("positive", "negative")]
    dev_ids = {r["id"] for r in read_jsonl(PROJECT_ROOT / "bench_R/DEV.jsonl")}
    sets["bench_R_DEV"] = [{"id": r["id"], "path": r["file"], "phash": r.get("phash"), "scene": None,
                            "sha256": r.get("sha256")} for r in bench if r["id"] in dev_ids]
    # independent negatives
    pool = read_jsonl(PROJECT_ROOT / "data/hard_negatives_v3/manifest.jsonl")
    sets["independent_neg"] = [{"id": r["id"], "path": r["path"], "phash": r.get("phash"), "scene": None,
                                "sha256": r.get("sha256")} for r in pool
                               if r.get("status") in ("hard", "easy", "quarantine") and r.get("split") == "holdout"]
    # video frames: 1 per 5s
    import cv2
    video_rows = []
    for p in sorted(Path(r"C:\Users\Nidhi\My projects\Sample_videos").glob("*")):
        if p.suffix.lower() not in (".mp4", ".mov", ".avi", ".mkv"):
            continue
        cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps * 5)); idx = 0
        while cap.isOpened():
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                tmp = PROJECT_ROOT / "build/verifier/_video_frame.jpg"
                cv2.imwrite(str(tmp), frame)
                video_rows.append({"id": f"{p.stem}:{idx}", "path": str(tmp), "phash": phash_file(tmp),
                                   "scene": f"video:{p.name}", "sha256": sha_file(tmp)})
            idx += 1
        cap.release()
    sets["sample_videos"] = video_rows

    report = {"training_images": len(train_sample), "training_scenes": len(train_scenes), "sets": {}, "verdict": "PASS"}
    for name, rows in sets.items():
        ph = [int(r["phash"], 16) for r in rows if r.get("phash")]
        arr = np.array(ph, dtype=np.uint64)
        exact = []
        near = []
        for r in rows:
            if r.get("sha256") and r["sha256"] in train_sha:
                exact.append({"eval_id": r["id"]})
            if r.get("phash") and len(train_arr):
                d = np.bitwise_count(np.uint64(int(r["phash"], 16)) ^ train_arr)
                if int(d.min()) <= 2:
                    near.append({"eval_id": r["id"], "distance": int(d.min())})
        id_hits = {r["id"] for r in rows} & train_ids
        scene_hits = {r.get("scene") for r in rows if r.get("scene")} & train_scenes
        path_hits = {p for p in (r.get("path") for r in rows) if p and "sample_videos" in str(p).lower()}
        report["sets"][name] = {"n": len(rows), "exact_sha": len(exact), "phash_le2": len(near),
                                "id_overlap": len(id_hits), "scene_overlap": len(scene_hits),
                                "sample_videos_paths_in_training": len(path_hits),
                                "examples": {"exact": exact[:3], "phash": near[:3],
                                             "ids": sorted(id_hits)[:3], "scenes": sorted(scene_hits)[:3]}}
        if exact or near or id_hits or scene_hits:
            report["verdict"] = "FAIL"
        print(name, json.dumps(report["sets"][name]), flush=True)
    (PROJECT_ROOT / "build/verifier/leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VERDICT", report["verdict"])
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
