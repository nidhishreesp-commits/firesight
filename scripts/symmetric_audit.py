"""Symmetric label audit: random unflagged negatives + missed positives.

Builds two contact-sheet sets with Florence-2 captions:
  A. 200 random val_report negatives that P50 did NOT flag (no alarm)
  B. every val_report positive that P50 missed (false negatives; smoke-only and small flames included)

Green boxes = ground truth (positives only). Red boxes = P50 detections.
Outputs build/audit_symmetric/{sheets_negatives,sheets_missed}/ + index.csv.
No labels are modified.
"""
from __future__ import annotations

import argparse, csv, json, random, sys, textwrap
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle"))

from firesight.config import PROJECT_ROOT
from pilot_eval import load_list, load_records, read_preds

OUT = PROJECT_ROOT/"build/audit_symmetric"
FIRE_WORDS = ("fire", "flame", "burning", "bonfire", "campfire", "blaze", "wildfire",
              "smoke", "smouldering", "smoldering", "ember", "candle", "torch")


def font(size=11):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()


def tile(rec, dets, caption, out_size=(360, 270)):
    try:
        im = Image.open(rec.path).convert("RGB")
    except Exception:
        return Image.new("RGB", (out_size[0], out_size[1]+64), (30, 30, 30))
    im.thumbnail(out_size)
    t = Image.new("RGB", (out_size[0], out_size[1]+64), (18, 18, 20))
    ox, oy = (out_size[0]-im.width)//2, (out_size[1]-im.height)//2
    t.paste(im, (ox, oy))
    sx, sy = im.width/max(rec.width, 1), im.height/max(rec.height, 1)
    d = ImageDraw.Draw(t)
    for g in rec.labels:
        x1, y1, x2, y2 = g.box.to_list()
        d.rectangle([ox+x1*sx, oy+y1*sy, ox+x2*sx, oy+y2*sy], outline=(0, 220, 80), width=3)
        d.text((ox+x1*sx+3, oy+y1*sy+1), f"GT {g.label}", fill=(0, 255, 100), font=font(12))
    for det in dets[:4]:
        x1, y1, x2, y2 = det.box.to_list()
        d.rectangle([ox+x1*sx, oy+y1*sy, ox+x2*sx, oy+y2*sy], outline=(255, 60, 60), width=2)
        d.text((ox+x1*sx+3, oy+y2*sy-14), f'{det.label} {det.confidence:.2f}', fill=(255, 220, 0), font=font(11))
    d.text((6, out_size[1]+3), f"{rec.id[:52]}", fill=(170, 170, 170), font=font(11))
    for i, line in enumerate(textwrap.wrap(caption, 58)[:3]):
        d.text((6, out_size[1]+18+i*14), line, fill=(220, 220, 220), font=font(11))
    return t


def write_sheets(rows, folder: Path, per=30, cols=6):
    folder.mkdir(parents=True, exist_ok=True)
    for si in range(0, len(rows), per):
        chunk = rows[si:si+per]
        tw, th = 360, 334
        sheet = Image.new("RGB", (cols*tw, ((len(chunk)+cols-1)//cols)*th), (10, 10, 12))
        for i, r in enumerate(chunk):
            sheet.paste(tile(r["rec"], r["dets"], r["caption"]), ((i % cols)*tw, (i//cols)*th))
        p = folder/f"sheet_{si//per+1:02d}.jpg"
        sheet.save(p, quality=88)
        print("sheet", p, len(chunk), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--caption-missing", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ds = PROJECT_ROOT/"build/kaggle_dataset"
    records = load_records(load_list(ds/"lists"/"val_report.txt", ds), ds)
    preds = read_preds(PROJECT_ROOT/"build/pilot_eval/preds_P50.jsonl")
    thr = json.loads((PROJECT_ROOT/"build/pilot_eval/metrics_P50.json").read_text())["thresholds"]

    def alarm(rec):
        return any(d.confidence >= thr.get(d.label, 1.0) for d in preds.get(rec.id, []))

    negs = [r for r in records if not (r.meta.get("has_fire") or r.meta.get("has_smoke")) and not alarm(r)]
    missed = [r for r in records if (r.meta.get("has_fire") or r.meta.get("has_smoke")) and not alarm(r)]
    rng = random.Random(args.seed)
    neg_pick = rng.sample(negs, min(args.negatives, len(negs)))
    print("unflagged negatives available:", len(negs), "picked:", len(neg_pick))
    print("missed positives available:", len(missed), "(all included)")
    comp = Counter(("fire+smoke" if r.meta.get("has_fire") and r.meta.get("has_smoke")
                    else "fire" if r.meta.get("has_fire") else "smoke") for r in missed)
    print("missed positives composition:", dict(comp))

    rows = []
    for r in neg_pick:
        rows.append({"group": "random_unflagged_negative", "rec": r, "dets": preds.get(r.id, []), "caption": ""})
    for r in missed:
        rows.append({"group": "missed_positive", "rec": r, "dets": preds.get(r.id, []), "caption": ""})

    if args.caption_missing:
        from transformers import AutoModelForCausalLM, AutoProcessor
        mid = "microsoft/Florence-2-base"
        model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True).eval()
        proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        task = "<MORE_DETAILED_CAPTION>"
        for i, r in enumerate(rows):
            try:
                img = Image.open(r["rec"].path).convert("RGB")
                inputs = proc(text=task, images=img, return_tensors="pt")
                with torch.no_grad():
                    ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                                         max_new_tokens=96, num_beams=2)
                txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
                r["caption"] = txt.strip()
            except Exception as exc:
                r["caption"] = f"CAPTION_ERROR {type(exc).__name__}"
            if (i+1) % 25 == 0:
                print("captioned", i+1, "/", len(rows), flush=True)

    with (OUT/"index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group","id","path","gt_labels","max_conf","caption","keyword_flag"])
        for r in rows:
            gt = "+".join(sorted({g.label for g in r["rec"].labels})) or "negative"
            mc = max((d.confidence for d in r["dets"]), default=0.0)
            flag = int(any(wd in r["caption"].lower() for wd in FIRE_WORDS))
            w.writerow([r["group"], r["rec"].id, r["rec"].path, gt, round(mc,4), r["caption"], flag])

    write_sheets([r for r in rows if r["group"]=="random_unflagged_negative"], OUT/"sheets_negatives")
    write_sheets([r for r in rows if r["group"]=="missed_positive"], OUT/"sheets_missed")

    for g in ("random_unflagged_negative", "missed_positive"):
        sub = [r for r in rows if r["group"] == g]
        flag = sum(1 for r in sub if any(w in r["caption"].lower() for w in FIRE_WORDS))
        print(json.dumps({"group": g, "n": len(sub), "keyword_flag": flag,
                          "keyword_share": round(flag/max(len(sub),1), 4)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
