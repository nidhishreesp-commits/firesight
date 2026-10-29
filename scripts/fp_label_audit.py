"""Label audit: contact sheets of false positives with detector box + caption.

Sets:
  * ALL 88 FPs used in the taxonomy (83 val_report + 5 bench_R DEV)
  * top-N highest-confidence detections on val_report negatives (default 200)

Outputs (build/fp_audit/):
  audit_index.csv      one row per image (set,id,path,conf,label,caption,suspect_label_noise)
  sheets/sheet_XX.jpg  visual grids for human review
No benchmark labels are modified.
"""
from __future__ import annotations

import argparse, csv, json, sys, textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from firesight.config import PROJECT_ROOT
from run_bench import calibrate_image_level, decision, load_val
from run_benchR import load_benchR

P50_ONNX = PROJECT_ROOT/"build/kaggle_weights/runs/P50/weights/best.onnx"
VAL_PREDS = PROJECT_ROOT/"build/pilot_eval/preds_P50.jsonl"
OUT = PROJECT_ROOT/"build/fp_audit"

FIRE_WORDS = ("fire", "flame", "burning", "bonfire", "campfire", "blaze", "wildfire",
              "smoke", "smouldering", "smoldering", "ember", "candle", "torch", "lantern fire")


def _font(size=12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()


def draw_tile(path: Path, det, caption: str, out_size=(360, 270)):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", out_size, (30, 30, 30))
    im.thumbnail(out_size)
    tile = Image.new("RGB", (out_size[0], out_size[1] + 64), (18, 18, 20))
    ox = (out_size[0] - im.width)//2
    oy = (out_size[1] - im.height)//2
    tile.paste(im, (ox, oy))
    if det:
        sx = im.width / max(det["w"], 1); sy = im.height / max(det["h"], 1)
        x1, y1, x2, y2 = det["box"]
        draw = ImageDraw.Draw(tile)
        draw.rectangle([ox + x1*sx, oy + y1*sy, ox + x2*sx, oy + y2*sy], outline=(255, 60, 60), width=3)
        draw.text((ox + x1*sx + 4, oy + y1*sy + 2), f'{det["label"]} {det["conf"]:.2f}', fill=(255, 255, 0), font=_font(13))
    draw = ImageDraw.Draw(tile)
    draw.text((6, out_size[1] + 3), f'{det["set"] if det else ""} {det["id"][:44] if det else ""}', fill=(180, 180, 180), font=_font(11))
    wrapped = textwrap.wrap(caption, 58)[:3]
    for i, line in enumerate(wrapped):
        draw.text((6, out_size[1] + 18 + i*14), line, fill=(220, 220, 220), font=_font(11))
    return tile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--caption-missing", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"sheets").mkdir(exist_ok=True)

    from pilot_eval import load_list, load_records, read_preds
    from firesight.types import Box, Detection
    from ultralytics import YOLO

    ds = PROJECT_ROOT/"build/kaggle_dataset"
    report = load_records(load_list(ds/"lists"/"val_report.txt", ds), ds)
    preds = read_preds(VAL_PREDS)
    thr = json.loads((PROJECT_ROOT/"build/pilot_eval/metrics_P50.json").read_text())["thresholds"]
    stem2path = {}
    with (ds/"MANIFEST.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem2path[row["image_id"].replace(":", "_")] = row["path"]

    def best_det(rec):
        ds_ = preds.get(rec.id, [])
        if not ds_:
            return None
        d = max(ds_, key=lambda x: x.confidence)
        return {"label": d.label, "conf": round(d.confidence, 4), "box": d.box.to_list(),
                "w": rec.width, "h": rec.height, "id": rec.id, "set": "val_report"}

    neg = [r for r in report if not (r.meta.get("has_fire") or r.meta.get("has_smoke"))]
    ranked = sorted(neg, key=lambda r: max((d.confidence for d in preds.get(r.id, [])), default=0), reverse=True)
    top = ranked[: args.top]
    rows = []
    for r in top:
        det = best_det(r)
        rows.append({"set": "val_report", "id": r.id, "path": str(PROJECT_ROOT/stem2path.get(r.id, r.path)),
                     "det": det})

    # bench_R DEV FPs (need boxes: run P50 on the FP images)
    records, member = load_benchR()
    dev = [r for r in records if member[r.id] == "dev"]
    cal = load_val()
    def predict(recs, tag):
        lst = OUT/f"paths_{tag}.txt"
        lst.write_text("\n".join(str(PROJECT_ROOT/r.path) for r in recs)+"\n", encoding="utf-8")
        m = YOLO(str(P50_ONNX), task="detect"); out = {}
        p2i = {str((PROJECT_ROOT/r.path).resolve()): r.id for r in recs}
        for res in m.predict(str(lst), imgsz=640, conf=0.01, batch=8, device="cpu", verbose=False, stream=True):
            out[p2i.get(str(Path(res.path).resolve()), Path(res.path).stem)] = [
                Detection("fire" if int(b.cls.item())==0 else "smoke", float(b.conf.item()),
                          Box(*[float(v) for v in b.xyxy[0].tolist()])) for b in res.boxes]
        return out
    cal_thr = calibrate_image_level(cal, predict(cal, "cal"), target_fpr=0.02)
    dev_preds = predict(dev, "dev")
    for r in dev:
        if r.meta.get("has_fire") or r.meta.get("has_smoke"):
            continue
        if decision(r, dev_preds.get(r.id, []), cal_thr):
            ds_ = dev_preds.get(r.id, [])
            d = max(ds_, key=lambda x: x.confidence)
            rows.append({"set": "bench_R_dev", "id": r.id, "path": str(PROJECT_ROOT/r.path),
                         "det": {"label": d.label, "conf": round(d.confidence,4), "box": d.box.to_list(),
                                 "w": r.width, "h": r.height, "id": r.id, "set": "bench_R_dev"}})

    # captions: reuse taxonomy file, caption the rest
    cap = {}
    tax = PROJECT_ROOT/"build/fp_taxonomy/fps_categorized.csv"
    if tax.exists():
        for row in csv.DictReader(tax.open(encoding="utf-8")):
            cap[row["id"]] = row.get("caption", "")
    need = [r for r in rows if not cap.get(r["id"])]
    if need and args.caption_missing:
        from transformers import AutoModelForCausalLM, AutoProcessor
        mid = "microsoft/Florence-2-base"
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).eval()
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        task = "<MORE_DETAILED_CAPTION>"
        for i, r in enumerate(need):
            try:
                img = Image.open(r["path"]).convert("RGB")
                inputs = proc(text=task, images=img, return_tensors="pt")
                with torch.no_grad():
                    ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                         max_new_tokens=96, num_beams=2)
                txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
                cap[r["id"]] = proc.post_process_generation(txt, task=task, image_size=img.size)[task].strip()
            except Exception as exc:
                cap[r["id"]] = f"CAPTION_ERROR {type(exc).__name__}"
            if (i+1) % 20 == 0:
                print("captioned", i+1, "/", len(need), flush=True)

    with (OUT/"audit_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["set","id","path","det_label","det_conf","caption","suspect_label_noise"])
        for r in rows:
            c = cap.get(r["id"], "")
            low = c.lower()
            suspect = int(any(word in low for word in FIRE_WORDS))
            w.writerow([r["set"], r["id"], r["path"], r["det"]["label"] if r["det"] else "",
                        r["det"]["conf"] if r["det"] else "", c, suspect])

    # sheets, 30 tiles each (6x5)
    per, cols = 30, 6
    for si in range(0, len(rows), per):
        chunk = rows[si:si+per]
        tile_w, tile_h = 360, 334
        rows_n = (len(chunk)+cols-1)//cols
        sheet = Image.new("RGB", (cols*tile_w, rows_n*tile_h), (10,10,12))
        for i, r in enumerate(chunk):
            t = draw_tile(Path(r["path"]), r["det"], cap.get(r["id"], ""))
            sheet.paste(t, ((i % cols)*tile_w, (i//cols)*tile_h))
        out = OUT/"sheets"/f"sheet_{si//per + 1:02d}.jpg"
        sheet.save(out, quality=88)
        print("sheet", out, len(chunk), flush=True)

    suspect = sum(1 for r in rows if any(w in cap.get(r["id"], "").lower() for w in FIRE_WORDS))
    print(json.dumps({"images": len(rows), "val": sum(1 for r in rows if r["set"]=="val_report"),
                      "dev": sum(1 for r in rows if r["set"]=="bench_R_dev"),
                      "provisional_label_noise": suspect,
                      "provisional_share": round(suspect/max(len(rows),1), 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
