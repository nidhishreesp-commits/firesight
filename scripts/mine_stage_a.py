"""Stage A hard-negative miner and pool builder (frozen P50, CPU).

Inputs are licence-clean candidate manifests (Wikimedia Commons, Open Images, the free
out-of-sample train-negative pool, optionally synthetic).  For every candidate this script:

  * computes SHA-256 + pHash and rejects exact/pHash/id overlap with P50's train_50 set,
    val, TEST, bench_R, testbench100 and the other frozen benches;
  * runs frozen P50 at conf=0.01 and records hardness (max fire/smoke confidence);
  * tags cause with CLIP, keeping the requested cause for targeted sources and flagging
    mismatches;
  * quarantines high-confidence/high-CLIP-fire candidates for human review;
  * selects hard negatives (P50 >= 0.05) and easy background negatives;
  * holds out whole sources (Open Images V7) and whole cause types (traffic_light,
    sun_glare_sky) so Stage D can measure generalisation;
  * writes composition tables, leakage report, review grids and train/holdout lists.

Nothing here trains or tunes a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from firesight.config import PROJECT_ROOT
from hardneg_causes import (CATEGORIES, EASY_CAUSES, HELD_OUT_CAUSES,
                            SENSITIVE_CAUSES, TIER1)


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def phash_of(path: Path) -> str | None:
    try:
        import imagehash
        return str(imagehash.phash(Image.open(path).convert("RGB")))
    except Exception:
        return None


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def font(size=11):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except Exception:
            return ImageFont.load_default()


def draw_grid(rows: list[dict], path: Path, title: str, cols: int = 6,
              tw: int = 340, th: int = 265) -> None:
    if not rows:
        return
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, sheet_rows * th + 24), (10, 10, 12))
    d0 = ImageDraw.Draw(sheet)
    d0.text((6, 5), title, fill=(235, 235, 235), font=font(12))
    for i, r in enumerate(rows):
        try:
            im = Image.open(PROJECT_ROOT / r["path"]).convert("RGB")
        except Exception:
            continue
        im.thumbnail((tw - 8, th - 48))
        ox = (i % cols) * tw + (tw - im.width) // 2
        oy = (i // cols) * th + 26
        sheet.paste(im, (ox, oy))
        d = ImageDraw.Draw(sheet)
        sx, sy = im.width / max(r.get("width") or im.width, 1), im.height / max(r.get("height") or im.height, 1)
        for det in (r.get("dets") or [])[:3]:
            b = det["box"]
            d.rectangle([ox + b[0] * sx, oy + b[1] * sy, ox + b[2] * sx, oy + b[3] * sy],
                        outline=(255, 60, 60), width=2)
            d.text((ox + b[0] * sx + 2, oy + b[1] * sy + 1),
                   f"{det['label']} {det['conf']:.2f}", fill=(255, 230, 0), font=font(10))
        label = f"{r['cause']} | {r['status']} | {r['source']} | {r['max_conf']:.2f}"
        d.text(((i % cols) * tw + 4, (i // cols) * th + th - 19), label[:52], fill=(175, 175, 175), font=font(9))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=87)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        r["_manifest"] = str(path)
        rows.append(r)
    if bad:
        print(f"  WARN {path.name}: skipped {bad} malformed line(s)", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", nargs="+", default=[
        "data/mined_pools/wikimedia_stage_a/manifest.jsonl",
        "data/mined_pools/openimages/manifest.jsonl",
        "build/stage_a/free_pool_false.jsonl",
    ])
    ap.add_argument("--out", default="data/hard_negatives_v3")
    ap.add_argument("--model", default="build/kaggle_weights/runs/P50/weights/best.onnx")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--hard-min", type=float, default=0.05)
    ap.add_argument("--easy-max", type=float, default=0.02)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--clip-batch", type=int, default=32)
    ap.add_argument("--grid-per-cause", type=int, default=30)
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--max-candidates", type=int, default=0)
    args = ap.parse_args()

    out = PROJECT_ROOT / args.out
    (out / "lists").mkdir(parents=True, exist_ok=True)
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- candidates
    rows: list[dict] = []
    for m in args.manifests:
        p = PROJECT_ROOT / m
        if not p.exists():
            print(f"WARN missing manifest {m}", flush=True)
            continue
        got = load_manifest(p)
        print(f"loaded {len(got)} from {m}", flush=True)
        rows.extend(got)
    if args.max_candidates:
        rows = rows[: args.max_candidates]
    print(f"candidates: {len(rows)}", flush=True)

    # normalise provenance fields
    for r in rows:
        r.setdefault("cause", "")
        is_free = "free_pool" in str(r.get("_manifest", ""))
        if is_free:
            r["origin_source"] = r.get("source", "unknown")
            r["source"] = f"train_negative_free_{r['origin_source']}"
        # openimages manifest uses id/source; normalise source label
        if r.get("source") in ("Open Images V7 validation", "openimages_v7"):
            r["source"] = "openimages_v7"
        if not r.get("source"):
            r["source"] = "unknown"
        if not r.get("licence"):
            if is_free:
                r["licence"] = "CC0-1.0" if r.get("origin_source") == "dfire" else "CC BY 4.0"
            elif r["source"] == "openimages_v7":
                r["licence"] = "CC BY 4.0"
            else:
                r["licence"] = "unrecorded"
        if is_free:
            r.setdefault("licence_url", "")
            r.setdefault("page_url", r.get("origin", ""))
        if r["source"] == "openimages_v7":
            r.setdefault("page_url", r.get("source_url", ""))
            # Official statement: annotations CC BY 4.0, images listed as CC BY 2.0.
            if not r.get("licence") or r["licence"].startswith("CC BY 4.0"):
                r["licence"] = "CC BY 2.0 (Open Images V7 image listing; annotations CC BY 4.0)"
        if not r.get("width") or not r.get("height"):
            try:
                with Image.open(PROJECT_ROOT / r["path"]) as im:
                    r["width"], r["height"] = im.width, im.height
            except Exception:
                r["width"], r["height"] = None, None

    # ---------------------------------------------------------------- reference exclusion
    print("building reference sets...", flush=True)
    ref_rows: list[dict] = []

    def add_ref(rid: str, path: str | Path, ph: str | None, sha: str | None = None):
        if ph or sha or rid:
            ref_rows.append({"id": rid, "path": str(path), "phash": ph, "sha256": sha})

    train = [json.loads(l) for l in (PROJECT_ROOT / "data/splits/train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pkg = PROJECT_ROOT / "build/kaggle_dataset/lists"
    t50 = {Path(l.strip()).name for l in (pkg / "train_50.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    for r in train:
        key = r["id"].replace(":", "_") + Path(r["path"]).suffix
        if key in t50:
            add_ref(r["id"], r["path"], r.get("phash"))
    # every non-train frozen split (val/cal/report/mini/flames/gsd/bench, TEST, test benches)
    for name in ("val.jsonl", "test.jsonl", "val_bench.jsonl", "test_bench.jsonl",
                 "val_mini.jsonl", "test_mini.jsonl", "val_flames.jsonl",
                 "val_gsd.jsonl", "test_gsd.jsonl"):
        p = PROJECT_ROOT / "data/splits" / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            add_ref(r["id"], r["path"], r.get("phash"))
    # external CCTV NC reference is never trainable, but exclude it too
    p = PROJECT_ROOT / "data/splits/external_cctv_nc.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                add_ref(r.get("id", ""), r.get("path", ""), r.get("phash"))
    for r in json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8")):
        add_ref(r["id"], r["file"], r.get("phash"), r.get("sha256"))
    for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8")):
        add_ref(r["id"], PROJECT_ROOT / "testbench100/images" / r["file"], r.get("phash"), r.get("sha256"))

    ref_ph = [int(r["phash"], 16) for r in ref_rows if r.get("phash")]
    ref_ph_arr = np.array(ref_ph, dtype=np.uint64)
    ref_ids = {r["id"] for r in ref_rows if r.get("id")}
    ref_paths = {str(r["path"]).replace("\\", "/").lower() for r in ref_rows}
    print(f"reference rows={len(ref_rows)} phash={len(ref_ph)} ids={len(ref_ids)}", flush=True)

    # exact hashes: only reference images whose SHA is actually looked up
    cand_sha: set[str] = set()
    for r in rows:
        s = r.get("sha256")
        if not s:
            s = sha256_file(PROJECT_ROOT / r["path"])
            r["sha256"] = s
        if s:
            cand_sha.add(s)
    sha_cache_path = PROJECT_ROOT / "build/stage_a/ref_sha_cache.json"
    sha_cache: dict[str, str] = {}
    if sha_cache_path.exists():
        sha_cache = json.loads(sha_cache_path.read_text(encoding="utf-8"))
    exact_hits: list[dict] = []
    missing_ref = 0
    t0 = time.time()
    for i, rr in enumerate(ref_rows):
        pth = Path(rr["path"])
        if not pth.is_absolute():
            pth = PROJECT_ROOT / pth
        key = str(pth).replace("\\", "/")
        s = rr.get("sha256") or sha_cache.get(key)
        if not s:
            if not pth.exists():
                missing_ref += 1
                continue
            s = sha256_file(pth)
            if s:
                sha_cache[key] = s
        if s and s in cand_sha:
            exact_hits.append({"ref_id": rr["id"], "ref_path": key, "sha256": s})
        if (i + 1) % 10000 == 0:
            print(f"  hashed refs {i+1}/{len(ref_rows)} ({time.time()-t0:.0f}s)", flush=True)
    sha_cache_path.parent.mkdir(parents=True, exist_ok=True)
    sha_cache_path.write_text(json.dumps(sha_cache), encoding="utf-8")
    exact_sha = {h["sha256"] for h in exact_hits}
    print(f"exact-sha matches={len(exact_hits)} missing_ref={missing_ref}", flush=True)

    # ---------------------------------------------------------------- dedupe + leakage
    kept: list[dict] = []
    rejected: Counter[str] = Counter()
    leakage_examples: list[dict] = []
    seen_sha: dict[str, str] = {}
    seen_ph: list[tuple[int, str]] = []
    for idx, r in enumerate(rows):
        r["_row"] = idx
        p = PROJECT_ROOT / r["path"]
        if not p.exists():
            rejected["missing_file"] += 1
            continue
        if r["sha256"] in exact_sha:
            rejected["exact_sha_leak"] += 1
            leakage_examples.append({"id": r["id"], "reason": "exact_sha", "path": r["path"]})
            continue
        if r["id"] in ref_ids or str(r["path"]).replace("\\", "/").lower() in ref_paths:
            rejected["id_or_path_leak"] += 1
            leakage_examples.append({"id": r["id"], "reason": "id_or_path", "path": r["path"]})
            continue
        if r["sha256"] in seen_sha:
            rejected["duplicate_sha"] += 1
            continue
        ph = phash_of(p)
        if ph is None:
            rejected["unreadable"] += 1
            continue
        r["phash"] = ph
        # near-duplicate against any frozen reference
        if len(ref_ph_arr):
            x = np.array([int(ph, 16)], dtype=np.uint64) ^ ref_ph_arr
            d = np.bitwise_count(x)
            if int(d.min()) <= 2:
                rejected["phash_leak_le2"] += 1
                leakage_examples.append({"id": r["id"], "reason": "phash_le2", "path": r["path"],
                                         "distance": int(d.min())})
                continue
        # near-duplicate inside the candidate pool
        dup = False
        if seen_ph:
            arr = np.array([v for v, _ in seen_ph], dtype=np.uint64)
            x = np.array([int(ph, 16)], dtype=np.uint64) ^ arr
            if int(np.bitwise_count(x).min()) <= 4:
                dup = True
        if dup:
            rejected["duplicate_phash"] += 1
            continue
        seen_sha[r["sha256"]] = r["id"]
        seen_ph.append((int(ph, 16), r["id"]))
        kept.append(r)
        if (idx + 1) % 2000 == 0:
            print(f"  dedup/leakage {idx+1}/{len(rows)} kept={len(kept)}", flush=True)
    print(f"after leakage+dedupe: {len(kept)} rejected={dict(rejected)}", flush=True)

    # ---------------------------------------------------------------- P50 hardness
    det_path = out / "p50_dets.jsonl"
    det_by_id: dict[str, dict] = {}
    if det_path.exists():
        for line in det_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                det_by_id[d["id"]] = d
    # Free-pool rows already carry P50 detections from mine_free_pool.py; do not re-run them.
    reused = 0
    for r in kept:
        if r["id"] not in det_by_id and str(r.get("source", "")).startswith("train_negative_free") \
                and r.get("max_conf") is not None and isinstance(r.get("dets"), list):
            det_by_id[r["id"]] = {"id": r["id"], "max_conf": float(r["max_conf"]), "dets": r["dets"]}
            reused += 1
    todo = [r for r in kept if r["id"] not in det_by_id]
    print(f"P50 cache={len(det_by_id)} (reused free-pool={reused}) todo={len(todo)}", flush=True)
    if todo:
        from ultralytics import YOLO
        model = YOLO(str(PROJECT_ROOT / args.model), task="detect")
        lst = out / "p50_todo.txt"
        lst.write_text("\n".join(str(PROJECT_ROOT / r["path"]) for r in todo) + "\n", encoding="utf-8")
        p2r = {str((PROJECT_ROOT / r["path"]).resolve()): r for r in todo}
        t0 = time.time()
        n = 0
        with det_path.open("a", encoding="utf-8") as fh:
            for res in model.predict(str(lst), imgsz=args.imgsz, conf=args.conf, batch=args.batch,
                                     device="cpu", verbose=False, stream=True):
                r = p2r.get(str(Path(res.path).resolve()))
                if r is None:
                    continue
                dets = [{"label": "fire" if int(b.cls.item()) == 0 else "smoke",
                         "conf": round(float(b.conf.item()), 6),
                         "box": [round(float(v), 2) for v in b.xyxy[0].tolist()]}
                        for b in res.boxes]
                rec = {"id": r["id"], "max_conf": round(max((d["conf"] for d in dets), default=0.0), 6),
                       "dets": dets}
                det_by_id[r["id"]] = rec
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n += 1
                if n % 200 == 0:
                    el = time.time() - t0
                    print(f"  P50 {n}/{len(todo)} {n/el:.2f} img/s eta {(len(todo)-n)/max(n/el,1e-6)/60:.1f} min", flush=True)
        print(f"P50 inference done {(time.time()-t0)/60:.1f} min", flush=True)

    for r in kept:
        d = det_by_id.get(r["id"], {"max_conf": 0.0, "dets": []})
        r["max_conf"] = d["max_conf"]
        r["dets"] = d["dets"]
        r["is_hard"] = d["max_conf"] >= args.hard_min

    # ---------------------------------------------------------------- CLIP cause + fire score
    clip_path = out / "clip_tags.jsonl"
    clip_by_id: dict[str, dict] = {}
    if clip_path.exists():
        for line in clip_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                c = json.loads(line)
                clip_by_id[c["id"]] = c
    to_tag = [r for r in kept if r["id"] not in clip_by_id]
    print(f"CLIP cache={len(clip_by_id)} todo={len(to_tag)}", flush=True)
    if to_tag:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        mid = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        cm = CLIPModel.from_pretrained(mid).eval()
        cp = CLIPProcessor.from_pretrained(mid)
        causes = list(CATEGORIES)
        prompts = [CATEGORIES[c] for c in causes] + ["a photo of fire", "a photo of smoke"]
        txt = cp(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            tout = cm.text_model(input_ids=txt["input_ids"], attention_mask=txt["attention_mask"])
            tf = cm.text_projection(tout.pooler_output)
            tf = tf / tf.norm(dim=-1, keepdim=True)
        t0 = time.time()
        with clip_path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(to_tag), args.clip_batch):
                batch = to_tag[start:start + args.clip_batch]
                imgs = []
                for r in batch:
                    try:
                        imgs.append(Image.open(PROJECT_ROOT / r["path"]).convert("RGB"))
                    except Exception:
                        imgs.append(Image.new("RGB", (224, 224)))
                inp = cp(images=imgs, return_tensors="pt")
                with torch.no_grad():
                    vout = cm.vision_model(pixel_values=inp["pixel_values"])
                    f = cm.visual_projection(vout.pooler_output)
                    f = f / f.norm(dim=-1, keepdim=True)
                sims = f @ tf.T
                probs = sims.softmax(dim=-1)
                labels = causes + ["fire", "smoke"]
                for r, sim, prob in zip(batch, sims, probs):
                    cidx = int(sim[:len(causes)].argmax())
                    fire_idx, smoke_idx = len(causes), len(causes) + 1
                    order = torch.argsort(sim, descending=True).tolist()
                    fire_rank = order.index(fire_idx) + 1
                    smoke_rank = order.index(smoke_idx) + 1
                    top_idx = order[0]
                    fire_p = float(prob[fire_idx].item())
                    smoke_p = float(prob[smoke_idx].item())
                    rec = {"id": r["id"], "cause_clip": causes[cidx],
                           "cause_clip_sim": round(float(sim[cidx].item()), 4),
                           "top_label": labels[top_idx],
                           "fire_rank": fire_rank, "smoke_rank": smoke_rank,
                           "fire_sim": round(float(sim[fire_idx].item()), 4),
                           "smoke_sim": round(float(sim[smoke_idx].item()), 4),
                           "max_sim": round(float(sim[top_idx].item()), 4),
                           "fire_prob": round(fire_p, 4), "smoke_prob": round(smoke_p, 4)}
                    clip_by_id[r["id"]] = rec
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                if (start + len(batch)) % 320 == 0:
                    print(f"  CLIP {start+len(batch)}/{len(to_tag)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"CLIP done {(time.time()-t0)/60:.1f} min", flush=True)

    for r in kept:
        c = clip_by_id.get(r["id"], {})
        r["cause_clip"] = c.get("cause_clip", "")
        r["top_label"] = c.get("top_label", "")
        r["fire_rank"] = c.get("fire_rank", 999)
        r["smoke_rank"] = c.get("smoke_rank", 999)
        r["fire_prob"] = c.get("fire_prob", 0.0)
        r["smoke_prob"] = c.get("smoke_prob", 0.0)
        r["clips_fire"] = max(r["fire_prob"], r["smoke_prob"])
        requested = r.get("cause") or ""
        if not requested:
            r["cause"] = r["cause_clip"] or "other"
            r["cause_mismatch"] = False
        else:
            r["cause_mismatch"] = bool(r["cause_clip"] and r["cause_clip"] != requested)

    # ---------------------------------------------------------------- status + splits
    rng = np.random.default_rng(args.seed)
    for r in kept:
        reasons = []
        fire_rank, smoke_rank = r.get("fire_rank", 999), r.get("smoke_rank", 999)
        best_rank = min(fire_rank, smoke_rank)
        # CLIP softmax is diluted across 23 prompts, so rank is the reliable signal.
        if r["max_conf"] >= 0.50:
            reasons.append("conf>=0.50 review")
        elif r["max_conf"] >= 0.30 and best_rank <= 3:
            reasons.append("conf>=0.30+clip_top3_fire_smoke")
        elif r["max_conf"] >= 0.10 and best_rank <= 2:
            reasons.append("conf>=0.10+clip_top2_fire_smoke")
        if r["cause"] in SENSITIVE_CAUSES and r["max_conf"] >= 0.05 and best_rank <= 2:
            reasons.append("sensitive_cause+clip_top2_fire_smoke")
        if reasons:
            r["status"] = "quarantine"
            r["reason"] = "; ".join(reasons)
        elif r["is_hard"]:
            r["status"] = "hard"
            r["reason"] = ""
        elif r["max_conf"] < args.easy_max:
            r["status"] = "easy"
            r["reason"] = ""
        else:
            r["status"] = "rejected_weak"
            r["reason"] = "between easy and hard thresholds"

        # Synthetic images are a training-only supplement: never in any evaluation set.
        held = (r["source"] == "openimages_v7" or r["cause"] in HELD_OUT_CAUSES)
        if not held and r["source"] != "synthetic" and rng.random() < args.holdout_frac:
            held = True
        r["split"] = "holdout" if held else "train"

    # Enforce the synthetic cap: S <= 20% of (N + S)  <=>  S <= 0.25 * N.
    non_syn_hard = [r for r in kept if r["status"] == "hard" and r["source"] != "synthetic"]
    syn_hard = sorted([r for r in kept if r["status"] == "hard" and r["source"] == "synthetic"],
                      key=lambda r: r["max_conf"], reverse=True)
    syn_cap = int(0.25 * len(non_syn_hard))
    for r in syn_hard[syn_cap:]:
        r["status"] = "rejected_synthetic_cap"
        r["reason"] = f"synthetic cap: kept top {syn_cap} of {len(syn_hard)}"

    # ---------------------------------------------------------------- composition
    comp = defaultdict(int)
    for r in kept:
        comp[(r["cause"], r["source"], r["status"], r["split"])] += 1
    comp_rows = [{"cause": k[0], "source": k[1], "status": k[2], "split": k[3], "count": v}
                 for k, v in sorted(comp.items())]
    (out / "composition.csv").write_text("", encoding="utf-8")
    with (out / "composition.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cause", "source", "status", "split", "count"])
        w.writeheader()
        w.writerows(comp_rows)

    lic = Counter((r["source"], r["licence"]) for r in kept if r["status"] in ("hard", "easy"))
    with (out / "source_licence.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "licence", "count"])
        for (src, lc), n in sorted(lic.items(), key=lambda x: -x[1]):
            w.writerow([src, lc, n])

    # ---------------------------------------------------------------- write manifests/lists
    keep_fields = ("id", "path", "cause", "cause_clip", "cause_mismatch", "source", "licence",
                   "licence_url", "page_url", "title", "artist", "credit", "sha256", "phash",
                   "width", "height", "max_conf", "is_hard", "top_label", "fire_rank",
                   "smoke_rank", "fire_prob", "smoke_prob", "status", "reason", "split")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            row = {k: r.get(k) for k in keep_fields}
            row["dets"] = r.get("dets", [])[:5]
            fh.write(json.dumps(row) + "\n")

    def paths(statuses, split=None):
        return [str(PROJECT_ROOT / r["path"]) for r in kept
                if r["status"] in statuses and (split is None or r["split"] == split)]

    (out / "lists/train.txt").write_text("\n".join(paths({"hard", "easy"}, "train")) + "\n", encoding="utf-8")
    (out / "lists/holdout.txt").write_text("\n".join(paths({"hard", "easy"}, "holdout")) + "\n", encoding="utf-8")
    (out / "lists/quarantine.txt").write_text("\n".join(paths({"quarantine"})) + "\n", encoding="utf-8")
    # Independent negative evaluation set: all held-out hard/easy rows plus low-confidence
    # quarantine rows (P50 < 0.10).  High-confidence quarantine rows are never used for eval.
    eval_neg = paths({"hard", "easy"}, "holdout")
    eval_neg += [str(PROJECT_ROOT / r["path"]) for r in kept
                 if r["status"] == "quarantine" and r["split"] == "holdout" and r["max_conf"] < 0.10]
    (out / "lists/eval_neg.txt").write_text("\n".join(eval_neg) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- leakage report
    # The gate rejects a candidate when it overlaps a frozen reference.  A high number of
    # rejected leaks is a good sign (the gate worked), not a failure.  PASS means the FINAL
    # pool contains no leak; FAIL means a leak survived into the pool.
    final_exact = sum(1 for r in kept if r.get("sha256") in exact_sha)
    final_id = sum(1 for r in kept if r["id"] in ref_ids
                   or str(r["path"]).replace("\\", "/").lower() in ref_paths)
    leak = {
        "candidates": len(rows), "kept": len(kept),
        "rejected": dict(rejected),
        "removed_leaks": {
            "exact_sha": rejected.get("exact_sha_leak", 0),
            "phash_le_2": rejected.get("phash_leak_le2", 0),
            "id_or_path": rejected.get("id_or_path_leak", 0),
        },
        "reference_sets": {
            "all_reference_rows": len(ref_rows), "reference_phashes": len(ref_ph_arr),
        },
        "exact_sha_matches_removed": len(exact_hits), "exact_sha_examples": exact_hits[:10],
        "leakage_examples_removed": leakage_examples[:20],
        "final_pool_leaks": {"exact_sha": final_exact, "id_or_path": final_id, "phash_le_2": 0},
        "verdict": "PASS" if final_exact == 0 and final_id == 0 else "FAIL",
        "measured": "SHA-256 exact + pHash Hamming<=2 + id/path against train_50, all val/test frozen splits, bench_R and testbench100; leaks are removed before selection",
    }
    (out / "leakage_report.json").write_text(json.dumps(leak, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- review grids
    by_cause = defaultdict(list)
    for r in kept:
        if r["status"] in ("hard", "quarantine"):
            by_cause[r["cause"]].append(r)
    for cause, rs in by_cause.items():
        rs = sorted(rs, key=lambda r: r["max_conf"], reverse=True)[: args.grid_per_cause]
        draw_grid(rs, out / "contact_sheets" / f"{cause}.jpg",
                  f"{cause} - P50 conf >= {args.hard_min} (top {len(rs)})")
    easy = [r for r in kept if r["status"] == "easy"]
    easy = sorted(easy, key=lambda r: r["cause"])[: args.grid_per_cause]
    draw_grid(easy, out / "contact_sheets" / "easy_backgrounds.jpg", "easy backgrounds")

    # ---------------------------------------------------------------- summary
    def count(pred):
        return sum(1 for r in kept if pred(r))
    summary = {
        "candidates": len(rows), "kept_after_leakage_and_dedupe": len(kept),
        "hard_all": count(lambda r: r["status"] == "hard"),
        "easy_all": count(lambda r: r["status"] == "easy"),
        "quarantine": count(lambda r: r["status"] == "quarantine"),
        "rejected_weak": count(lambda r: r["status"] == "rejected_weak"),
        "rejected_synthetic_cap": count(lambda r: r["status"] == "rejected_synthetic_cap"),
        "synthetic_hard_kept": count(lambda r: r["status"] == "hard" and r["source"] == "synthetic"),
        "hard_train": count(lambda r: r["status"] == "hard" and r["split"] == "train"),
        "easy_train": count(lambda r: r["status"] == "easy" and r["split"] == "train"),
        "hard_holdout": count(lambda r: r["status"] == "hard" and r["split"] == "holdout"),
        "easy_holdout": count(lambda r: r["status"] == "easy" and r["split"] == "holdout"),
        "independent_eval_negatives": len(eval_neg),
        "held_out_causes": sorted(HELD_OUT_CAUSES),
        "held_out_sources": ["openimages_v7"],
        "synthetic_rule": "training-only supplement; excluded from every evaluation list",
        "by_cause_hard": dict(Counter(r["cause"] for r in kept if r["status"] == "hard")),
        "by_cause_hard_train": dict(Counter(r["cause"] for r in kept if r["status"] == "hard" and r["split"] == "train")),
        "by_cause_hard_holdout": dict(Counter(r["cause"] for r in kept if r["status"] == "hard" and r["split"] == "holdout")),
        "by_source_hard": dict(Counter(r["source"] for r in kept if r["status"] == "hard")),
        "licence_hard_easy": {f"{k[0]} | {k[1]}": v for k, v in lic.items()},
        "target_hard": 3000, "target_met": count(lambda r: r["status"] == "hard") >= 3000,
        "measured": "P50 ONNX CPU inference and CLIP tagging on the listed manifests",
        "estimated": "none in this summary",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
