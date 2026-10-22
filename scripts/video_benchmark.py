from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT, load_config
from firesight.detectors.base import load_detector
from firesight.logging_utils import setup_logging
from firesight.video import load_clips, run_video

log = logging.getLogger("video_benchmark")


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / p
    return candidate if candidate.exists() else p.resolve()


def _registry(path: Path) -> dict[str, dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {e["id"]: e for e in data.get("models", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure time-to-detection and false alarms per camera-day on video clips.")
    parser.add_argument("--registry", default="models/registry.yaml")
    parser.add_argument("--models", default="ssprod_best_fire_v2")
    parser.add_argument("--clips", default="data/clips/clips.jsonl")
    parser.add_argument("--out", default="artifacts/video_benchmark")
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", default=None)
    parser.add_argument("--verifier", default=None, help="Path to verifier ONNX; disables if omitted")
    parser.add_argument("--verifier-threshold", type=float, default=0.6)
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.no_temporal:
        cfg.temporal.enabled = False
    setup_logging(cfg.app.log_level)
    registry = _registry(_resolve(args.registry))
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    clips = load_clips(_resolve(args.clips))
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for model_id in model_ids:
        entry = registry[model_id]
        backend = entry.get("backend", "yolo")
        model_ref = entry["path"] if backend in {"grounding_dino", "grounding-dino", "gdino", "rtdetr_v2", "rtdetrv2", "rtdetr2"} else _resolve(entry["path"])
        detector = load_detector(
            backend, model_ref,
            imgsz=cfg.detector.imgsz, iou=cfg.detector.iou, max_det=cfg.detector.max_det,
            device=args.device, name=model_id, license=entry.get("license", "unknown"),
            family=entry.get("family", "yolo"),
            **({"preprocess_mode": entry["preprocess_mode"]} if entry.get("preprocess_mode") else {}),
            **({"threshold": entry["threshold"]} if entry.get("threshold") is not None else {}),
        )
        verifier = None
        if args.verifier:
            from firesight.verifier import CropVerifier
            verifier = CropVerifier(_resolve(args.verifier), classes=cfg.verifier.classes, threshold=args.verifier_threshold)
        rows = []
        try:
            for clip in clips:
                path = _resolve(clip["path"])
                if not path.exists():
                    log.warning("clip missing: %s", path)
                    continue
                log.info("%s: %s", model_id, path.name)
                result = run_video(
                    detector, path, cfg.detector, cfg.temporal, cfg.alarm,
                    sample_fps=args.sample_fps, max_seconds=args.max_seconds,
                    camera_id=clip["id"], collect_frames=False, verifier=verifier,
                )
                row = {**clip, **result.to_dict()}
                if clip["label"] == "fire":
                    fire_ttd = None
                    smoke_ttd = None
                    first = result.first_confirmed
                    if "fire" in first and clip.get("fire_onset_s") is not None:
                        fire_ttd = first["fire"] - clip["fire_onset_s"]
                    if "smoke" in first and clip.get("smoke_onset_s") is not None:
                        smoke_ttd = first["smoke"] - clip["smoke_onset_s"]
                    earliest_onset = min(v for v in (clip.get("fire_onset_s"), clip.get("smoke_onset_s")) if v is not None)
                    any_ttd = result.first_alarm_s - earliest_onset if result.first_alarm_s is not None else None
                    row["metrics"] = {"fire_ttd_s": fire_ttd, "smoke_ttd_s": smoke_ttd, "any_ttd_s": any_ttd}
                else:
                    duration_h = result.duration_s / 3600.0
                    events = len(result.alarm_events)
                    fa_per_24h = events / duration_h * 24.0 if duration_h > 0 else None
                    detection_frames = sum(1 for e in result.frames if e["detections"])
                    row["metrics"] = {
                        "alarm_events": events,
                        "fa_per_camera_24h": fa_per_24h,
                        "video_hours": duration_h,
                    }
                rows.append(row)
        finally:
            detector.close()
        fire_rows = [r for r in rows if r["label"] == "fire"]
        nonfire_rows = [r for r in rows if r["label"] == "nonfire"]
        fire_ttds = [r["metrics"]["any_ttd_s"] for r in fire_rows if r["metrics"].get("any_ttd_s") is not None]
        detected = sum(1 for r in fire_rows if r["metrics"].get("any_ttd_s") is not None)
        total_hours = sum(r["metrics"]["video_hours"] for r in nonfire_rows)
        total_events = sum(r["metrics"]["alarm_events"] for r in nonfire_rows)
        summary = {
            "model": model_id,
            "fire_clips": len(fire_rows),
            "fire_clips_detected": detected,
            "fire_clip_recall": round(detected / len(fire_rows), 4) if fire_rows else None,
            "ttd_mean_s": round(sum(fire_ttds) / len(fire_ttds), 3) if fire_ttds else None,
            "ttd_max_s": round(max(fire_ttds), 3) if fire_ttds else None,
            "nonfire_hours": round(total_hours, 4),
            "nonfire_alarm_events": total_events,
            "fa_per_camera_24h": round(total_events / total_hours * 24.0, 3) if total_hours else None,
            "sample_fps": args.sample_fps,
            "temporal_enabled": cfg.temporal.enabled,
            "verifier": args.verifier,
            "verifier_threshold": args.verifier_threshold if args.verifier else None,
            "warning": "Non-fire video coverage is short; 24h rates are extrapolations with wide uncertainty.",
        }
        all_results[model_id] = {"summary": summary, "clips": rows}
        model_dir = out_dir / (args.tag or model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "video_metrics.json").write_text(json.dumps(all_results[model_id], indent=2, default=str), encoding="utf-8")
        log.info("video summary %s: %s", model_id, summary)
    (out_dir / "summary.json").write_text(json.dumps({k: v["summary"] for k, v in all_results.items()}, indent=2), encoding="utf-8")
    if args.tag:
        for k, v in all_results.items():
            (out_dir / f"{args.tag}_video_metrics.json").write_text(json.dumps(v, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v["summary"] for k, v in all_results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
