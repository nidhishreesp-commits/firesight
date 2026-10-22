from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.data.manifest import load_manifest
from firesight.detectors.base import load_detector

MODEL = "../smartsight-models/best-fire-v3/model.onnx"


def crop_box(frame, box, margin=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin)); y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin)); y2 = min(h, int(y2 + bh * margin))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine hard negatives from TRAIN/val non-fire data only (never TEST).")
    parser.add_argument("--manifest", default="data/splits/train_negatives_4000.jsonl")
    parser.add_argument("--clips", default="data/clips/clips.jsonl")
    parser.add_argument("--out", default="data/negatives/mined")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--video-fps", type=float, default=1.0)
    args = parser.parse_args()
    cfg = load_config()
    out = PROJECT_ROOT / args.out
    crop_dir = out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "mined.jsonl"
    meta = []
    det = load_detector("onnx", PROJECT_ROOT / MODEL, imgsz=cfg.detector.imgsz,
                        device="cpu", name="v3_miner", license="proprietary", family="YOLOv12n")
    records = load_manifest(PROJECT_ROOT / args.manifest, root=PROJECT_ROOT)[:args.limit]
    try:
        for idx, rec in enumerate(records, 1):
            frame = cv2.imread(str(PROJECT_ROOT / rec.path))
            if frame is None:
                continue
            dets = [d for d in det.predict(frame) if d.confidence >= args.conf][:args.max_per_image]
            for j, d in enumerate(dets):
                crop = crop_box(frame, d.box.to_list())
                if crop is None:
                    continue
                name = f"{rec.id.replace(':','_')}_{j}.jpg"
                cv2.imwrite(str(crop_dir / name), crop)
                meta.append({"crop": str((crop_dir / name).relative_to(PROJECT_ROOT)), "label": "lookalike",
                             "source": rec.source, "image_id": rec.id, "pred_label": d.label,
                             "confidence": round(d.confidence, 4), "origin": "train_negative"})
            if idx % 500 == 0:
                print(f"images {idx}/{len(records)} crops={len(meta)}", flush=True)

        clips = [json.loads(line) for line in (PROJECT_ROOT / args.clips).read_text(encoding="utf-8").splitlines() if line.strip()]
        for clip in clips:
            if clip.get("label") != "nonfire":
                continue
            path = PROJECT_ROOT / clip["path"]
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            stride = max(1, int(round(fps / args.video_fps)))
            frame_idx = 0
            saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride:
                    frame_idx += 1
                    continue
                dets = [d for d in det.predict(frame) if d.confidence >= args.conf][:args.max_per_image]
                for j, d in enumerate(dets):
                    crop = crop_box(frame, d.box.to_list())
                    if crop is None:
                        continue
                    name = f"video_{clip['id']}_{frame_idx:07d}_{j}.jpg"
                    cv2.imwrite(str(crop_dir / name), crop)
                    meta.append({"crop": str((crop_dir / name).relative_to(PROJECT_ROOT)), "label": "lookalike",
                                 "source": "nonfire_video", "image_id": f"{clip['id']}:{frame_idx}",
                                 "pred_label": d.label, "confidence": round(d.confidence, 4), "origin": "nonfire_video"})
                    saved += 1
                frame_idx += 1
            cap.release()
            print(f"video {clip['id']} saved={saved}", flush=True)
    finally:
        det.close()
    meta_path.write_text("\n".join(json.dumps(m) for m in meta) + "\n", encoding="utf-8")
    print(json.dumps({"crops": len(meta), "out": str(meta_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
