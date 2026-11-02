"""Step 1+4: frozen P50 on the 11 sample videos with temporal confirmation and the video layer.

Production alarm thresholds 0.29 (fire) / 0.33 (smoke); temporal N-of-M fire 3/5, smoke 4/6
(IoU tracker for determinism; production uses ByteTrack - noted). Sampling 5 fps.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter

NONFIRE = ["Attendance.MOV", "Car.mp4", "drone.mp4", "number_plate.mp4",
           "ppe_1.mp4", "ppe_2.mp4", "ppe_3.mp4", "ppe_4.mp4", "theft.mp4"]
FIRE = ["fire.mp4", "fire_2.mp4"]
THR = {"fire": 0.29, "smoke": 0.33}
FPS = 5.0


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d
    h = z*math.sqrt((p*(1-p) + z*z/(4*n))/n)/d
    return [round(max(0.0, c-h), 4), round(min(1.0, c+h), 4)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/day_video")
    args = ap.parse_args()
    out = PROJECT_ROOT/args.out; (out/"frames").mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT/"build/kaggle_weights/runs/P50/weights/best.onnx"), task="detect")
    cfg = load_config()
    cfg.temporal.tracker = "iou"

    videos = NONFIRE + FIRE
    results = {}
    for name in videos:
        path = Path(r"C:\Users\Nidhi\My projects\Sample_videos")/name
        cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps/FPS)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n/max(fps, 1)
        confirmer = TemporalConfirmer(cfg.temporal)
        vfilter = VideoFalseAlarmFilter(FilterConfig(), out/f"filter_{name}.jsonl")
        idx = sampled = 0
        raw_frames = 0; raw_dets = 0; max_conf = {"fire": 0.0, "smoke": 0.0}
        curve = []
        conf_frames = []; layer_frames = []
        first_raw = {"fire": None, "smoke": None}; first_conf = None; first_layer = None
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % step:
                idx += 1; continue
            t = idx/fps
            r = model.predict(frame, imgsz=640, conf=0.05, device="cpu", verbose=False)[0]
            dets = []
            for b in r.boxes:
                lab = "fire" if int(b.cls.item()) == 0 else "smoke"
                c = float(b.conf.item())
                dets.append(Detection(lab, c, Box(*[float(v) for v in b.xyxy[0].tolist()])))
            if dets:
                raw_frames += 1; raw_dets += len(dets)
                for d in dets:
                    max_conf[d.label] = max(max_conf[d.label], d.confidence)
                    if d.confidence >= 0.10 and first_raw[d.label] is None:
                        first_raw[d.label] = round(t, 2)
                if saved < 3 and any(d.confidence >= 0.10 for d in dets):
                    ann = r.plot()
                    cv2.imwrite(str(out/"frames"/f"{Path(name).stem}_{idx:05d}.jpg"), ann); saved += 1
            curve.append(round(max((d.confidence for d in dets), default=0.0), 4))
            temp = confirmer.update(sampled, t, dets, size=(frame.shape[0], frame.shape[1]))
            if temp.confirmed:
                conf_frames.append(round(t, 2))
                if first_conf is None: first_conf = round(t, 2)
                kept, review, supp = vfilter.update(name, t, frame, temp.confirmed)
                if kept:
                    layer_frames.append(round(t, 2))
                    if first_layer is None: first_layer = round(t, 2)
            sampled += 1; idx += 1
        cap.release()

        def episodes(times, gap=1.5):
            eps = []
            for t in times:
                if not eps or t - eps[-1][-1] > gap: eps.append([t])
                else: eps[-1].append(t)
            return eps
        ce = episodes(conf_frames); le = episodes(layer_frames)
        results[name] = {"group": "nonfire" if name in NONFIRE else "fire", "duration_s": round(dur, 1),
                         "sampled_frames": sampled, "raw_frames_with_det": raw_frames, "raw_detections": raw_dets,
                         "max_conf": {k: round(v, 4) for k, v in max_conf.items()},
                         "first_raw_ge_010": first_raw,
                         "confirmed_frames": len(conf_frames), "confirmed_episodes": len(ce),
                         "first_confirmed_s": first_conf, "confirmed_times": conf_frames[:20],
                         "layer_frames": len(layer_frames), "layer_episodes": len(le), "first_layer_s": first_layer,
                         "curve": curve, "annotated_saved": saved}
        print(name, json.dumps({k: results[name][k] for k in ("duration_s","raw_frames_with_det","max_conf",
              "confirmed_episodes","first_confirmed_s","layer_episodes","first_layer_s")}), flush=True)

    nf = [results[n] for n in NONFIRE]
    nf_min = sum(r["duration_s"] for r in nf)/60.0
    conf_eps = sum(r["confirmed_episodes"] for r in nf)
    layer_eps = sum(r["layer_episodes"] for r in nf)
    summary = {"nonfire_minutes": round(nf_min, 2),
               "confirmed_alarms": conf_eps, "confirmed_fa_per_cam_min": round(conf_eps/nf_min, 4),
               "confirmed_fa_per_cam_min_ci": wilson(conf_eps, max(1, int(round(nf_min)))),
               "layer_alarms": layer_eps, "layer_fa_per_cam_min": round(layer_eps/nf_min, 4),
               "per_video": results}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps({k: summary[k] for k in ("nonfire_minutes","confirmed_alarms",
          "confirmed_fa_per_cam_min","layer_alarms","layer_fa_per_cam_min")}))
    for n in FIRE:
        r = results[n]
        print(n, "first raw>=0.10:", r["first_raw_ge_010"], "first confirmed:", r["first_confirmed_s"],
              "first layer alarm:", r["first_layer_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
