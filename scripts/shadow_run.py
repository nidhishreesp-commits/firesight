from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.detectors.base import load_detector
from firesight.logging_utils import setup_logging
from firesight.video import run_video


def main() -> int:
    parser = argparse.ArgumentParser(description="Shadow-mode run: log detections/alarms alongside existing alarms; never actuates.")
    parser.add_argument("--source", required=True, help="video file, webcam index, or RTSP URI")
    parser.add_argument("--model", default="models/best_fire_v3.onnx")
    parser.add_argument("--backend", default="onnx")
    parser.add_argument("--out", default="artifacts/shadow")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--verifier", default="models/verifier.onnx")
    parser.add_argument("--verifier-threshold", type=float, default=0.6)
    args = parser.parse_args()
    cfg = load_config()
    setup_logging(cfg.app.log_level)
    detector = load_detector(args.backend, PROJECT_ROOT / args.model, imgsz=cfg.detector.imgsz, device="cpu",
                             name=Path(args.model).stem, license="proprietary (SmartSight)", family="YOLOv12n")
    verifier = None
    if args.verifier and (PROJECT_ROOT / args.verifier).exists():
        from firesight.verifier import CropVerifier
        verifier = CropVerifier(PROJECT_ROOT / args.verifier, classes=cfg.verifier.classes, threshold=args.verifier_threshold)
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"shadow_{int(time.time())}.jsonl"
    try:
        result = run_video(detector, args.source, cfg.detector, cfg.temporal, cfg.alarm,
                           sample_fps=args.sample_fps, max_seconds=args.max_seconds, camera_id="shadow",
                           collect_frames=True, verifier=verifier)
    finally:
        detector.close()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "summary", **result.to_dict()}) + "\n")
        for frame in result.frames:
            if frame["detections"] or frame["alarm"].get("active"):
                fh.write(json.dumps({"type": "frame", **frame}) + "\n")
    print(json.dumps({"log": str(log_path), "frames_processed": result.frames_processed,
                      "alarm_events": len(result.alarm_events), "first_alarm_s": result.first_alarm_s}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
