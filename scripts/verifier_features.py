"""Build verifier features from P50 detections.

Per detection: CLIP ViT-B/32 embedding of (a) box crop, (b) 2.5x context crop, (c) full image
(computed once per image) + numeric features (P50 conf, class, box area/aspect). Cause label from
the context embedding vs cause prompts (triage for the per-cause scorecard).

Labels: TP if IoU>=0.5 with a same-class GT box; FP if IoU<0.1 with every GT box; others skipped.
"""
from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

CAUSES = {
    "warm_light": "a street lamp, lantern or chandelier glowing warm at night",
    "sunset": "a sunset or sunrise with an orange sky",
    "haze_cloud": "haze, fog, mist or clouds in the sky",
    "screen": "a television or monitor screen showing fire or orange graphics",
    "reflection": "a reflection on water, wet ground, glass or metal",
    "orange_ppe": "a person wearing an orange safety vest or a traffic cone",
    "cave_rock": "a cave interior or red/orange rock formation",
    "skin": "a close-up of a person's face or skin",
    "smoke_steam": "steam or smoke from a chimney, cooling tower or cooking",
    "construction_dust": "construction dust, plaster dust or sawdust",
    "sun_bright": "a bright sun or overexposed white sky",
    "other": "an ordinary scene with no fire and no smoke",
}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2-ix1)*max(0.0, iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def clip_crop(im, box, scale=1.0):
    W, H = im.size
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    w, h = (box[2]-box[0])*scale, (box[3]-box[1])*scale
    x1, y1 = max(0, int(cx-w/2)), max(0, int(cy-h/2))
    x2, y2 = min(W, int(cx+w/2)), min(H, int(cy+h/2))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (64, 64))
    return im.crop((x1, y1, x2, y2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-fp-ratio", type=float, default=2.0)
    ap.add_argument("--keep-all", action="store_true", help="eval mode: keep ambiguous detections (y=-1)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    rows = json.loads((PROJECT_ROOT/args.sample).read_text(encoding="utf-8"))
    gt = {r["id"]: r["labels"] for r in rows}
    path_of = {r["id"]: r["path"] for r in rows}
    dets = [json.loads(l) for l in (PROJECT_ROOT/args.detections).read_text(encoding="utf-8").splitlines() if l.strip()]
    print("images with detections:", len(dets), flush=True)

    import torch
    from transformers import CLIPModel, CLIPProcessor
    mid = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    model = CLIPModel.from_pretrained(mid).eval(); proc = CLIPProcessor.from_pretrained(mid)
    names = list(CAUSES)
    with torch.no_grad():
        t = proc(text=[CAUSES[n] for n in names], return_tensors="pt", padding=True)
        tfeat = model.get_text_features(**t); tfeat = tfeat/tfeat.norm(dim=-1, keepdim=True)

    items = []
    for row in dets:
        rid = row["id"]
        labels = gt.get(rid, [])
        W, H = row.get("w") or 0, row.get("h") or 0
        for d in row["detections"]:
            box = d["box"]
            same = [g for g in labels if g["label"] == d["label"]]
            ious = [iou(box, g["box"]) for g in same]
            best = max(ious, default=0.0)
            any_gt = [iou(box, g["box"]) for g in labels]
            if best >= 0.5: y = 1
            elif max(any_gt, default=0.0) < 0.1: y = 0
            elif args.keep_all: y = -1
            else: continue
            items.append({"id": rid, "label": d["label"], "conf": d["conf"], "box": box,
                          "w": W, "h": H, "y": y})
    print("kept detections:", len(items), "TP", sum(i["y"] for i in items), "FP", sum(1-i["y"] for i in items), flush=True)

    if args.keep_all:
        tp = [i for i in items if i["y"] == 1]; fp = [i for i in items if i["y"] == 0]
        unk = [i for i in items if i["y"] == -1]
        items = tp + fp + unk
    # balance (training only): cap FP at max_fp_ratio x TP
    tp = [i for i in items if i["y"] == 1]; fp = [i for i in items if i["y"] == 0]
    unk = [i for i in items if i["y"] == -1]
    import random
    rng = random.Random(20261028)
    rng.shuffle(fp)
    if not args.keep_all:
        fp = fp[: int(len(tp)*args.max_fp_ratio)]
        if fp and len(tp) > 2*len(fp):
            rng.shuffle(tp); tp = tp[: 2*len(fp)]
        items = tp + fp
    print("after balance:", len(items), "TP", len(tp), "FP", len(fp), "unknown", len(unk), flush=True)

    # group by image for full-image embeddings
    by_img = defaultdict(list)
    for it in items:
        by_img[it["id"]].append(it)
    feats, nums, ys, groups, metas = [], [], [], [], []
    for rid, group in by_img.items():
        try:
            im = Image.open(PROJECT_ROOT/path_of[rid]).convert("RGB")
        except Exception:
            continue
        crops, kinds = [], []
        full = im.copy(); full.thumbnail((448, 448))
        for it in group:
            crops.append(clip_crop(im, it["box"], 1.15)); kinds.append("box")
            crops.append(clip_crop(im, it["box"], 2.5)); kinds.append("context")
        # batch encode crops
        embs = []
        for s in range(0, len(crops), args.batch):
            batch = crops[s:s+args.batch]
            inp = proc(images=batch, return_tensors="pt")
            with torch.no_grad():
                f = model.get_image_features(**inp); f = f/f.norm(dim=-1, keepdim=True)
            embs.append(f)
        E = torch.cat(embs) if embs else torch.zeros((0, 512))
        inp = proc(images=[full], return_tensors="pt")
        with torch.no_grad():
            ff = model.get_image_features(**inp); ff = ff/ff.norm(dim=-1, keepdim=True)
        # causes from context embeddings
        ctx = E[1::2]
        sims = ctx @ tfeat.T
        causes = [names[int(s.argmax())] for s in sims]
        for j, it in enumerate(group):
            W, H = max(it["w"], 1), max(it["h"], 1)
            bw, bh = (it["box"][2]-it["box"][0])/W, (it["box"][3]-it["box"][1])/H
            feats.append(torch.cat([E[2*j], E[2*j+1], ff[0]]).numpy())
            nums.append([it["conf"], 1.0 if it["label"] == "fire" else 0.0, bw, bh, bw*bh,
                         bw/max(bh, 1e-6), W/640.0, H/640.0])
            ys.append(it["y"]); groups.append(rid)
            metas.append({"id": rid, "label": it["label"], "conf": it["conf"],
                          "box": it["box"], "w": W, "h": H, "cause": causes[j], "y": it["y"]})
    X = np.hstack([np.asarray(feats, dtype=np.float32), np.asarray(nums, dtype=np.float32)])
    np.savez_compressed(PROJECT_ROOT/args.out, X=X, y=np.asarray(ys, dtype=np.int8),
                        groups=np.asarray(groups), meta=np.asarray([json.dumps(m) for m in metas]))
    print("saved", args.out, X.shape, "pos", int(sum(ys)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
