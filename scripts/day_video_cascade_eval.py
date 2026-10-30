"""One authorized pass over the 11 sample videos.

Compares, on the same frames:
  A) P50 calibrated (0.29/0.33) + temporal + video layer
  B) P50@0.10 + verifier v3 + temporal + video layer

The verifier uses the same box/context/full-image CLIP features as training. Sample_videos is
evaluated once and never used for fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter

NONFIRE = ["Attendance.MOV", "Car.mp4", "drone.mp4", "number_plate.mp4",
           "ppe_1.mp4", "ppe_2.mp4", "ppe_3.mp4", "ppe_4.mp4", "theft.mp4"]
FIRE = ["fire.mp4", "fire_2.mp4"]
ONSET = {"fire.mp4": 40.5, "fire_2.mp4": 6.0}
THR = {"fire": 0.29, "smoke": 0.33}
CASCADE_CONF = 0.10
VERIFIER_V = 0.011936064809560776
FPS = 5.0
SAMPLE_DIR = Path(r"C:\Users\Nidhi\My projects\Sample_videos")


def clip_crop(im: Image.Image, box, scale=1.0):
    W, H = im.size
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = (box[2] - box[0]) * scale, (box[3] - box[1]) * scale
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (64, 64))
    return im.crop((x1, y1, x2, y2))


def episodes(times, gap=1.5):
    eps = []
    for t in times:
        if not eps or t - eps[-1][-1] > gap:
            eps.append([t])
        else:
            eps[-1].append(t)
    return eps


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class Verifier:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.clip = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").eval()
        self.proc = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

    def score(self, frame_bgr, dets):
        if not dets:
            return []
        im = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        W, H = im.size
        full = im.copy()
        full.thumbnail((448, 448))
        crops = []
        for d in dets:
            b = d.box.to_list()
            crops.append(clip_crop(im, b, 1.15))
            crops.append(clip_crop(im, b, 2.5))
        embs = []
        for s in range(0, len(crops), 32):
            inp = self.proc(images=crops[s:s + 32], return_tensors="pt")
            with torch.no_grad():
                vo = self.clip.vision_model(pixel_values=inp["pixel_values"])
                f = self.clip.visual_projection(vo.pooler_output)
                f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f)
        E = torch.cat(embs)
        inp = self.proc(images=[full], return_tensors="pt")
        with torch.no_grad():
            vo = self.clip.vision_model(pixel_values=inp["pixel_values"])
            ff = self.clip.visual_projection(vo.pooler_output)
            ff = ff / ff.norm(dim=-1, keepdim=True)
        rows = []
        for j, d in enumerate(dets):
            b = d.box.to_list()
            bw, bh = (b[2] - b[0]) / max(W, 1), (b[3] - b[1]) / max(H, 1)
            nums = np.array([d.confidence, 1.0 if d.label == "fire" else 0.0, bw, bh,
                             bw * bh, bw / max(bh, 1e-6), W / 640.0, H / 640.0], dtype=np.float32)
            rows.append(np.concatenate([E[2 * j].numpy(), E[2 * j + 1].numpy(), ff[0].numpy(), nums]))
        X = np.stack(rows)
        return self.model.predict_proba(X)[:, 1].tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="build/day_video_cascade")
    ap.add_argument("--verifier", default="build/verifier/model_v4.joblib")
    ap.add_argument("--verifier-v", type=float, default=VERIFIER_V)
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg.temporal.tracker = "iou"
    from ultralytics import YOLO
    detector = YOLO(str(PROJECT_ROOT / "build/kaggle_weights/runs/P50/weights/best.onnx"), task="detect")
    verifier = Verifier(PROJECT_ROOT / args.verifier)

    results = {}
    for name in NONFIRE + FIRE:
        path = SAMPLE_DIR / name
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / max(fps, 1)
        conf = {
            "p50": (TemporalConfirmer(cfg.temporal), VideoFalseAlarmFilter(FilterConfig(), out / f"p50_{name}.jsonl")),
            "cascade": (TemporalConfirmer(cfg.temporal), VideoFalseAlarmFilter(FilterConfig(), out / f"cascade_{name}.jsonl")),
        }
        conf_times = {"p50": [], "cascade": []}
        layer_times = {"p50": [], "cascade": []}
        first_conf = {"p50": None, "cascade": None}
        first_layer = {"p50": None, "cascade": None}
        idx = sampled = 0
        raw = {"frames": 0, "dets": 0, "max_conf": {"fire": 0.0, "smoke": 0.0}}
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            t = idx / fps
            r = detector.predict(frame, imgsz=640, conf=0.05, device="cpu", verbose=False)[0]
            dets = [Detection("fire" if int(b.cls.item()) == 0 else "smoke", float(b.conf.item()),
                              Box(*[float(v) for v in b.xyxy[0].tolist()])) for b in r.boxes]
            if dets:
                raw["frames"] += 1
                raw["dets"] += len(dets)
                for d in dets:
                    raw["max_conf"][d.label] = max(raw["max_conf"][d.label], d.confidence)
            cand = [d for d in dets if d.confidence >= CASCADE_CONF]
            scores = verifier.score(frame, cand) if cand else []
            sets = {
                "p50": [d for d in dets if d.confidence >= THR.get(d.label, 1.0)],
                "cascade": [d for d, s in zip(cand, scores) if s >= args.verifier_v],
            }
            for arm, ds in sets.items():
                tc, vf = conf[arm]
                temp = tc.update(sampled, t, ds, size=(frame.shape[0], frame.shape[1]))
                if temp.confirmed:
                    conf_times[arm].append(round(t, 2))
                    if first_conf[arm] is None:
                        first_conf[arm] = round(t, 2)
                    kept, review, supp = vf.update(name, t, frame, temp.confirmed)
                    if kept:
                        layer_times[arm].append(round(t, 2))
                        if first_layer[arm] is None:
                            first_layer[arm] = round(t, 2)
            sampled += 1
            idx += 1
        cap.release()
        results[name] = {
            "group": "nonfire" if name in NONFIRE else "fire",
            "duration_s": round(dur, 1),
            "sampled_frames": sampled,
            "raw_frames_with_det": raw["frames"],
            "raw_detections": raw["dets"],
            "max_conf": {k: round(v, 4) for k, v in raw["max_conf"].items()},
            "p50": {"confirmed_episodes": len(episodes(conf_times["p50"])),
                    "first_confirmed_s": first_conf["p50"],
                    "layer_episodes": len(episodes(layer_times["p50"])),
                    "first_layer_s": first_layer["p50"]},
            "cascade": {"confirmed_episodes": len(episodes(conf_times["cascade"])),
                        "first_confirmed_s": first_conf["cascade"],
                        "layer_episodes": len(episodes(layer_times["cascade"])),
                        "first_layer_s": first_layer["cascade"]},
        }
        print(name, json.dumps(results[name]), flush=True)

    nf = [results[n] for n in NONFIRE]
    nf_min = sum(r["duration_s"] for r in nf) / 60.0
    summary = {"nonfire_minutes": round(nf_min, 2), "per_video": results}
    for arm in ("p50", "cascade"):
        eps = sum(r[arm]["layer_episodes"] for r in nf)
        summary[arm] = {"layer_alarms": eps, "layer_fa_per_cam_min": round(eps / nf_min, 4),
                        "layer_fa_per_cam_min_ci": wilson(eps, max(1, int(round(nf_min))))}
    fire = [results[n] for n in FIRE]
    summary["fire"] = {}
    for arm in ("p50", "cascade"):
        detected = sum(1 for r in fire if r[arm]["first_layer_s"] is not None)
        onset_detected = sum(1 for n, r in zip(FIRE, fire)
                             if r[arm]["first_layer_s"] is not None and r[arm]["first_layer_s"] >= ONSET[n])
        summary["fire"][arm] = {"clips": len(fire), "clips_detected": detected,
                                "clip_recall": round(detected / len(fire), 4),
                                "clip_recall_ci": wilson(detected, len(fire)),
                                "detected_after_onset": onset_detected}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY", json.dumps({k: summary[k] for k in ("nonfire_minutes", "p50", "cascade", "fire")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
