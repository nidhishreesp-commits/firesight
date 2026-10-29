"""Synthetic hard-negative supplement (clearly tagged, capped downstream at 20%).

Applies realistic sun glare, lens flare, sunset colour grading, bright-sky overexposure and
haze to ordinary non-fire scenes, then keeps only variants on which frozen P50 fires at
max_conf >= 0.05.  Synthetic rows are training-only: scripts/mine_stage_a.py marks source
'synthetic' as train and never puts it in an evaluation list.

Base images must be Wikimedia Commons ordinary scenes (not Open Images, which is held out as a
whole source) and must not belong to a held-out cause.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from hardneg_causes import EASY_CAUSES, HELD_OUT_CAUSES


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _to_np(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def _to_im(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "RGB")


def sun_glare(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = _to_np(im)
    h, w = a.shape[:2]
    cx, cy = int(w * rng.uniform(0.05, 0.35)), int(h * rng.uniform(0.05, 0.35))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(w, h)
    glow = np.exp(-(d / rng.uniform(0.18, 0.35)) ** 2)
    tint = np.array([1.0, 0.82, 0.55], dtype=np.float32)
    a = a + glow[..., None] * tint * rng.uniform(0.55, 0.95)
    # lens flare: a few translucent discs along a line from the sun
    im2 = _to_im(a)
    from PIL import ImageDraw
    d2 = ImageDraw.Draw(im2, "RGBA")
    for k in range(rng.integers(2, 5)):
        fx = int(cx + (w * 0.5 - cx) * rng.uniform(0.2, 1.4) * (k + 1) / 4)
        fy = int(cy + (h * 0.5 - cy) * rng.uniform(0.2, 1.4) * (k + 1) / 4)
        r = int(rng.uniform(0.03, 0.10) * min(w, h))
        col = tuple(int(v) for v in rng.uniform(0.55, 0.95, 3) * 255) + (int(rng.uniform(30, 80)),)
        d2.ellipse([fx - r, fy - r, fx + r, fy + r], fill=col)
    return im2.filter(ImageFilter.GaussianBlur(1.2))


def sunset_grade(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = _to_np(im)
    # warm colour matrix + gamma
    m = np.array([[1.12, 0.04, 0.0], [0.03, 0.94, 0.0], [0.0, 0.03, 0.78]], dtype=np.float32)
    a = np.clip(a @ m.T, 0, 1.4)
    a = np.clip(a ** rng.uniform(0.85, 1.15), 0, 1)
    # warm horizon gradient
    h, w = a.shape[:2]
    g = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    a = a + (1 - g) * np.array([0.12, 0.05, -0.02], dtype=np.float32)
    return _to_im(np.clip(a, 0, 1))


def bright_sky(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = _to_np(im)
    h, w = a.shape[:2]
    g = np.linspace(0.2, 0.98, h, dtype=np.float32)[:, None, None]
    sky = np.array([1.0, 0.96, 0.82], dtype=np.float32)
    a = a * (1 - g) + sky * g * rng.uniform(0.75, 1.0)
    a = np.clip(a * rng.uniform(1.05, 1.25), 0, 1)
    return _to_im(a)


def haze(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = _to_np(im)
    veil = np.array([0.92, 0.90, 0.86], dtype=np.float32)
    a = a * rng.uniform(0.35, 0.65) + veil * rng.uniform(0.35, 0.65)
    a = np.clip(a, 0, 1)
    im2 = _to_im(a)
    return ImageEnhance.Contrast(im2).enhance(rng.uniform(0.6, 0.9))


EFFECTS = {"sun_glare": sun_glare, "sunset_grade": sunset_grade,
           "bright_sky": bright_sky, "haze": haze}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/hard_negatives_v3/manifest.jsonl")
    ap.add_argument("--out", default="data/hard_negatives_v3_synthetic")
    ap.add_argument("--model", default="build/kaggle_weights/runs/P50/weights/best.onnx")
    ap.add_argument("--max-base", type=int, default=300)
    ap.add_argument("--effects", default="sun_glare,sunset_grade,bright_sky,haze")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--hard-min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    pool = PROJECT_ROOT / args.pool
    out = PROJECT_ROOT / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in pool.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Any easy, non-fire Wikimedia scene is a valid base; ordinary scenes are preferred.
    bases = [r for r in rows if r.get("status") == "easy"
             and r.get("source") == "wikimedia_commons"
             and r.get("cause") not in HELD_OUT_CAUSES
             and r.get("split") == "train"]
    bases.sort(key=lambda r: 0 if r.get("cause") in EASY_CAUSES else 1)
    bases = bases[: args.max_base]
    effects = [e.strip() for e in args.effects.split(",") if e.strip()]
    rng = np.random.default_rng(args.seed)
    print(f"base images: {len(bases)} effects: {effects}", flush=True)

    generated: list[dict] = []
    for i, r in enumerate(bases):
        try:
            im = Image.open(PROJECT_ROOT / r["path"]).convert("RGB")
        except Exception:
            continue
        for eff in effects:
            try:
                variant = EFFECTS[eff](im, rng)
            except Exception:
                continue
            raw = variant.tobytes()
            digest = hashlib.sha256(f"{r['id']}:{eff}:{args.seed}".encode()).hexdigest()
            dest = out / "images" / f"{digest[:16]}.jpg"
            variant.save(dest, "JPEG", quality=90)
            generated.append({
                "id": f"synthetic:{digest[:16]}", "path": str(dest.relative_to(PROJECT_ROOT)),
                "cause": r["cause"], "source": "synthetic", "synthetic": True,
                "synthetic_effect": eff, "base_id": r["id"], "base_path": r["path"],
                "licence": f"{r.get('licence', 'unrecorded')} (synthetic derivative; internal research)",
                "licence_url": r.get("licence_url", ""), "page_url": r.get("page_url", ""),
                "sha256": digest, "width": variant.width, "height": variant.height,
            })
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{len(bases)} -> {len(generated)}", flush=True)

    # keep only variants that are hard for P50
    if generated:
        from ultralytics import YOLO
        model = YOLO(str(PROJECT_ROOT / args.model), task="detect")
        lst = out / "generated.txt"
        lst.write_text("\n".join(str(PROJECT_ROOT / g["path"]) for g in generated) + "\n", encoding="utf-8")
        p2g = {str((PROJECT_ROOT / g["path"]).resolve()): g for g in generated}
        for res in model.predict(str(lst), imgsz=640, conf=args.conf, batch=8, device="cpu",
                                 verbose=False, stream=True):
            g = p2g.get(str(Path(res.path).resolve()))
            if g is None:
                continue
            g["max_conf"] = round(max((float(b.conf.item()) for b in res.boxes), default=0.0), 6)
    kept = [g for g in generated if g.get("max_conf", 0.0) >= args.hard_min]
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for g in kept:
            fh.write(json.dumps(g) + "\n")
    summary = {
        "base_images": len(bases), "generated": len(generated), "kept_hard": len(kept),
        "effects": effects, "by_effect": dict(Counter(g["synthetic_effect"] for g in kept)),
        "by_cause": dict(Counter(g["cause"] for g in kept)),
        "hard_min": args.hard_min,
        "measured": "P50 ONNX CPU inference on generated variants",
        "rule": "training-only; mine_stage_a.py caps synthetic at 20% of hard negatives and excludes it from evaluation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
