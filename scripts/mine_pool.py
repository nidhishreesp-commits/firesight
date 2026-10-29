"""P50-driven hard-negative mining on a licensed background pool.

Steps: P50 inference at conf 0.01 -> keep max_conf >= --min-conf -> exclude anything whose
pHash is within --hamming of P50's training set (train_50), bench_R, testbench100 or TEST ->
CLIP zero-shot cause clustering -> per-cause review grids with detector boxes.
"""
from __future__ import annotations

import argparse, csv, json, sys, textwrap, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    "other": "an ordinary scene with no fire and no smoke",
}


def phash(path):
    import imagehash
    try:
        return imagehash.phash(Image.open(path).convert("RGB"))
    except Exception:
        return None


def font(size=11):
    try: return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try: return ImageFont.load_default(size=size)
        except Exception: return ImageFont.load_default()


def draw(rec, dets, cause, out_size=(340, 255)):
    im = Image.open(rec["path"]).convert("RGB"); im.thumbnail(out_size)
    t = Image.new("RGB", (out_size[0], out_size[1]+44), (18,18,20))
    ox, oy = (out_size[0]-im.width)//2, (out_size[1]-im.height)//2
    t.paste(im, (ox, oy)); d = ImageDraw.Draw(t)
    sx, sy = im.width/rec["w"], im.height/rec["h"]
    for lab, conf, box in dets[:3]:
        d.rectangle([ox+box[0]*sx, oy+box[1]*sy, ox+box[2]*sx, oy+box[3]*sy], outline=(255,60,60), width=2)
        d.text((ox+box[0]*sx+2, oy+box[1]*sy+1), f"{lab} {conf:.2f}", fill=(255,230,0), font=font(11))
    d.text((5, out_size[1]+2), f"{rec['id'][:36]} | {cause}", fill=(170,170,170), font=font(10))
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="dir with manifest.jsonl (id,path,licence,source)")
    ap.add_argument("--out", default="build/mined")
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--hamming", type=int, default=2)
    ap.add_argument("--grid-per-cause", type=int, default=30)
    args = ap.parse_args()
    pool = PROJECT_ROOT/args.pool
    out = PROJECT_ROOT/args.out/pool.name; (out/"grids").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (pool/"manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    print("pool rows", len(rows), flush=True)

    # exclusion hashes
    pkg = PROJECT_ROOT/"build/kaggle_dataset"
    t50 = {Path(l.strip()).name for l in (pkg/"lists"/"train_50.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    train_ph = set()
    for line in (PROJECT_ROOT/"data/splits/train.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r["id"].replace(":", "_") + Path(r["path"]).suffix in t50 and r.get("phash"):
            train_ph.add(int(r["phash"], 16))
    bench_ph = {int(r["phash"], 16) for r in json.loads((PROJECT_ROOT/"bench_R/index.json").read_text(encoding="utf-8")) if r.get("phash")}
    test_ph = {int(r["phash"], 16) for r in (json.loads(l) for l in (PROJECT_ROOT/"data/splits/test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()) if r.get("phash")}
    tb_ph = set()
    for r in json.loads((PROJECT_ROOT/"testbench100/index.json").read_text(encoding="utf-8")):
        p = PROJECT_ROOT/"testbench100/images"/r["file"]
        if p.exists():
            h = phash(p)
            if h: tb_ph.add(int(str(h), 16))
    ref = train_ph | bench_ph | test_ph | tb_ph
    print("reference hashes: train_50", len(train_ph), "bench_R", len(bench_ph), "TEST", len(test_ph), "testbench100", len(tb_ph), flush=True)

    # P50 inference
    from ultralytics import YOLO
    onnx = PROJECT_ROOT/"build/kaggle_weights/runs/P50/weights/best.onnx"
    model = YOLO(str(onnx), task="detect")
    lst = out/"paths.txt"; lst.write_text("\n".join(str(PROJECT_ROOT/r["path"]) for r in rows)+"\n", encoding="utf-8")
    p2i = {str((PROJECT_ROOT/r["path"]).resolve()): r["id"] for r in rows}
    dets = defaultdict(list)
    t0 = time.time()
    for res in model.predict(str(lst), imgsz=640, conf=0.01, batch=8, device="cpu", verbose=False, stream=True):
        key = p2i.get(str(Path(res.path).resolve()), Path(res.path).stem)
        dets[key] = [("fire" if int(b.cls.item())==0 else "smoke", float(b.conf.item()),
                      [float(v) for v in b.xyxy[0].tolist()]) for b in res.boxes]
    print("inference", round(time.time()-t0,1), "s", flush=True)

    kept, excluded = [], Counter()
    for r in rows:
        d = dets.get(r["id"], [])
        mc = max((c for _, c, _ in d), default=0.0)
        if mc < args.min_conf:
            excluded["below_conf"] += 1; continue
        h = phash(PROJECT_ROOT/r["path"])
        if h is not None and any(bin(int(str(h),16) ^ ref_h).count("1") <= args.hamming for ref_h in ref):
            excluded["phash_overlap"] += 1; continue
        kept.append({**r, "max_conf": round(mc,4), "dets": d})
    print("kept", len(kept), "excluded", dict(excluded), flush=True)

    # CLIP clustering
    from transformers import CLIPModel, CLIPProcessor
    mid = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    cm = CLIPModel.from_pretrained(mid).eval(); cp = CLIPProcessor.from_pretrained(mid)
    import torch
    names = list(CAUSES)
    txt = cp(text=[CAUSES[n] for n in names], return_tensors="pt", padding=True)
    with torch.no_grad():
        tfeat = cm.get_text_features(**txt); tfeat = tfeat/tfeat.norm(dim=-1, keepdim=True)
    for start in range(0, len(kept), 32):
        batch = kept[start:start+32]
        imgs = []
        for r in batch:
            try: imgs.append(Image.open(PROJECT_ROOT/r["path"]).convert("RGB"))
            except Exception: imgs.append(Image.new("RGB",(224,224)))
        inp = cp(images=imgs, return_tensors="pt")
        with torch.no_grad():
            f = cm.get_image_features(**inp); f = f/f.norm(dim=-1, keepdim=True)
        sims = (f @ tfeat.T)
        for r, s in zip(batch, sims):
            r["cause"] = names[int(s.argmax())]
    print("causes", dict(Counter(r["cause"] for r in kept)), flush=True)

    with (out/"mined.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps({k: r[k] for k in ("id","path","licence","source","max_conf","cause")})+"\n")

    # grids per cause (top by confidence)
    for cause in names:
        sub = sorted([r for r in kept if r["cause"]==cause], key=lambda r: r["max_conf"], reverse=True)[:args.grid_per_cause]
        if not sub: continue
        cols = 6; tw, th = 340, 299
        sheet = Image.new("RGB", (cols*tw, ((len(sub)+cols-1)//cols)*th), (10,10,12))
        for i, r in enumerate(sub):
            im = Image.open(PROJECT_ROOT/r["path"]).convert("RGB")
            rec = {"id": r["id"], "path": PROJECT_ROOT/r["path"], "w": im.width, "h": im.height}
            sheet.paste(draw(rec, r["dets"], cause), ((i%cols)*tw, (i//cols)*th))
        p = out/"grids"/f"{cause}.jpg"; sheet.save(p, quality=87)
        print("grid", p, len(sub), flush=True)
    (out/"summary.json").write_text(json.dumps({"pool": str(pool.relative_to(PROJECT_ROOT)), "rows": len(rows),
        "kept": len(kept), "excluded": dict(excluded), "causes": dict(Counter(r["cause"] for r in kept))}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
