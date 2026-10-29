"""Pilot loop: monthly mining of pilot footage into the same gated retrain loop.

Reads pilot frames/videos (never C:\\Users\\Nidhi\\My projects\\Sample_videos), runs frozen
P50 at a low confidence, keeps frames where P50 fires, dedupes by pHash, and writes a manifest
that scripts/mine_stage_a.py can ingest (source=pilot_footage, proprietary internal licence).

The full leakage gate is applied by mine_stage_a.py, not here.  This script only produces
candidates; nothing is trained automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

SAMPLE_VIDEOS = (PROJECT_ROOT.parent / "Sample_videos").resolve()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def guard(path: Path) -> None:
    try:
        path.resolve().relative_to(SAMPLE_VIDEOS)
    except ValueError:
        return
    raise SystemExit(f"refusing to mine final held-out Sample_videos: {path}")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path) -> str | None:
    try:
        import imagehash
        return str(imagehash.phash(Image.open(path).convert("RGB")))
    except Exception:
        return None


def sample_video(path: Path, out_dir: Path, max_frames: int, sample_fps: float):
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native / max(sample_fps, 0.1))))
    idx = saved = 0
    while cap.isOpened() and saved < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            dest = out_dir / f"{path.stem}_{idx:07d}.jpg"
            cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            saved += 1
        idx += 1
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="directory of pilot frames or videos")
    ap.add_argument("--out", default="data/pilot_mined")
    ap.add_argument("--model", default="build/kaggle_weights/runs/P50/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--max-frames-per-video", type=int, default=20)
    ap.add_argument("--hamming", type=int, default=4)
    ap.add_argument("--camera", default="pilot")
    args = ap.parse_args()

    src = PROJECT_ROOT / args.input
    guard(src)
    out = PROJECT_ROOT / args.out
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT | VID_EXT)
    print(f"input files: {len(files)}", flush=True)
    frames: list[Path] = []
    for p in files:
        if p.suffix.lower() in IMG_EXT:
            frames.append(p)
        else:
            sample_video(p, img_dir, args.max_frames_per_video, args.sample_fps)
    frames.extend(sorted(img_dir.glob("*.jpg")))
    # dedupe input frames by pHash before inference
    seen: list[int] = []
    unique: list[Path] = []
    for p in frames:
        h = phash(p)
        if not h:
            continue
        v = int(h, 16)
        if any((v ^ o).bit_count() <= args.hamming for o in seen):
            continue
        seen.append(v)
        unique.append(p)
    print(f"unique frames: {len(unique)}", flush=True)

    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / args.model), task="detect")
    lst = out / "paths.txt"
    lst.write_text("\n".join(str(p) for p in unique) + "\n", encoding="utf-8")
    p2p = {str(p.resolve()): p for p in unique}
    rows = []
    for res in model.predict(str(lst), imgsz=640, conf=0.01, batch=8, device="cpu",
                             verbose=False, stream=True):
        p = p2p.get(str(Path(res.path).resolve()))
        if p is None:
            continue
        dets = [{"label": "fire" if int(b.cls.item()) == 0 else "smoke",
                 "conf": round(float(b.conf.item()), 6),
                 "box": [round(float(v), 2) for v in b.xyxy[0].tolist()]} for b in res.boxes]
        mc = max((d["conf"] for d in dets), default=0.0)
        if mc < args.conf:
            continue
        try:
            im = Image.open(p)
            w, h = im.width, im.height
        except Exception:
            continue
        rows.append({
            "id": f"pilot:{sha_file(p)[:16]}", "path": str(p.relative_to(PROJECT_ROOT)),
            "cause": "", "source": "pilot_footage",
            "licence": "proprietary internal (pilot footage)",
            "licence_url": "", "page_url": "", "sha256": sha_file(p), "phash": phash(p),
            "width": w, "height": h, "max_conf": round(mc, 6), "dets": dets,
            "camera_id": args.camera,
        })
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summary = {
        "input_files": len(files), "unique_frames": len(unique), "false_alarms_kept": len(rows),
        "by_camera": dict(Counter(r["camera_id"] for r in rows)),
        "conf_min": args.conf,
        "next": "python scripts/mine_stage_a.py --manifests " + str((out / "manifest.jsonl").relative_to(PROJECT_ROOT)),
        "measured": "P50 ONNX CPU inference on the supplied pilot frames",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
