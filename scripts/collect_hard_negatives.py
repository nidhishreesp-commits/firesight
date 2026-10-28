from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT

CATEGORIES: dict[str, str] = {
    "sunset_orange_sky": "a sunset or sunrise with an orange sky",
    "red_orange_neon_lights": "red or orange neon signs or lamps",
    "vehicle_lights_streetlights": "bright car headlights, brake lights or street lights at night",
    "chandelier_stage_lights": "a chandelier or stage and festival lighting",
    "reflections_lens_flare_glare": "a reflection or sun glare on water, glass or metal",
    "orange_ppe_vests_cones": "people wearing orange safety vests, helmets or traffic cones",
    "flowers_autumn_leaves": "orange flowers or autumn leaves",
    "clouds": "clouds in the sky",
    "fog_haze_mist": "fog, mist or haze in a landscape",
    "steam": "steam from a kettle, cooking, a shower or a manhole",
    "exhaust": "vehicle exhaust, a chimney or a cooling tower emitting vapour",
    "dust_construction_plaster": "construction dust, plaster dust, sawdust or sandblasting",
    "snow": "snowfall, a blizzard or blowing snow",
    "overexposed_white": "a bright overexposed white wall or sky",
    "camera_artifacts": "motion blur, lens dirt, rain on the lens or sensor noise",
    "ordinary_indoor": "a normal indoor office, warehouse or hallway with no fire",
    "ordinary_streets_warehouses_offices": "a normal street, warehouse or office exterior with no fire",
    "ordinary_other": "a normal landscape or park with no fire",
    "screen_fire": "a television, monitor or phone screen showing fire or orange graphics",
    "cave_warm_rock": "a cave interior or red/orange rock formation lit warmly",
    "person_portrait": "a close-up portrait of a person's face or skin",
}

QUERIES: dict[str, list[str]] = {
    "sunset_orange_sky": ["sunset", "sunrise", "orange sky", "red sky"],
    "red_orange_neon_lights": ["neon sign", "red neon sign", "orange neon light", "red lamp"],
    "vehicle_lights_streetlights": ["car headlights", "street lights at night", "brake lights", "tail lights"],
    "chandelier_stage_lights": ["chandelier", "stage lighting", "concert stage lights", "festival lights"],
    "reflections_lens_flare_glare": ["lens flare", "sun glare", "water reflection", "glass reflection"],
    "orange_ppe_vests_cones": ["orange safety vest", "traffic cone", "high visibility vest", "construction helmet"],
    "flowers_autumn_leaves": ["orange flowers", "red flowers", "autumn leaves", "autumn foliage"],
    "clouds": ["clouds", "cumulus clouds", "cloudscape", "overcast sky"],
    "fog_haze_mist": ["fog", "mist", "haze", "foggy landscape"],
    "steam": ["steam", "water vapour", "kettle steam", "steam cooking"],
    "exhaust": ["exhaust smoke", "chimney smoke", "cooling tower", "factory chimney"],
    "dust_construction_plaster": ["construction dust", "plaster dust", "sawdust", "sandblasting", "dust cloud"],
    "snow": ["snowfall", "blizzard", "snow landscape", "blowing snow"],
    "overexposed_white": ["overexposed photo", "white sky", "bright sky", "white wall"],
    "camera_artifacts": ["motion blur", "lens dirt", "rain on lens", "sensor noise"],
    "ordinary_indoor": ["office interior", "warehouse interior", "indoor hallway"],
    "ordinary_streets_warehouses_offices": ["street scene", "warehouse exterior", "office building"],
    "ordinary_other": ["landscape", "park"],
    "screen_fire": ["television screen", "computer monitor", "video game screenshot", "movie poster fire"],
    "cave_warm_rock": ["cave interior", "red rock canyon", "slot canyon", "sandstone formation"],
    "person_portrait": ["portrait photograph", "selfie", "close-up face"],
}

TIERS = {
    1: ["clouds", "fog_haze_mist", "dust_construction_plaster", "reflections_lens_flare_glare", "sunset_orange_sky"],
    2: ["vehicle_lights_streetlights", "red_orange_neon_lights", "orange_ppe_vests_cones", "exhaust",
        "snow", "overexposed_white", "steam", "chandelier_stage_lights", "flowers_autumn_leaves",
        "screen_fire", "cave_warm_rock", "person_portrait"],
    3: ["camera_artifacts", "ordinary_indoor", "ordinary_streets_warehouses_offices", "ordinary_other"],
}

VIDEO_MAP = {
    "ppe": ["dust_construction_plaster", "orange_ppe_vests_cones", "ordinary_other"],
    "car": ["vehicle_lights_streetlights", "ordinary_streets_warehouses_offices"],
    "number_plate": ["vehicle_lights_streetlights", "ordinary_streets_warehouses_offices"],
    "attendance": ["ordinary_indoor"],
    "theft": ["ordinary_indoor"],
    "drone": ["ordinary_other", "clouds"],
}

BLOCK_TITLE = re.compile(
    r"\b(fire|flame|burning|wildfire|firework|candle|welding|diagram|map|drawing|"
    r"painting|engraving|illustration|logo|coat of arms|poster|chart|svg)\b",
    re.IGNORECASE,
)


def clean(value: str) -> str:
    return re.sub(r"<[^>]+>", "", html.unescape(value or "")).strip()


def allowed_license(name: str) -> bool:
    low = clean(name).lower()
    if not any(token in low for token in ("cc0", "public domain", "cc by", "cc-by")):
        return False
    return not any(token in low for token in ("nc", "nd"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def image_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: Path) -> int:
    import imagehash
    with Image.open(path) as im:
        return int(str(imagehash.dhash(im.convert("RGB"))), 16)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class Deduper:
    def __init__(self, existing: list[int] | None = None, threshold: int = 4) -> None:
        self.hashes = list(existing or [])
        self.threshold = threshold

    def add(self, value: int) -> None:
        self.hashes.append(value)

    def duplicate(self, value: int) -> bool:
        return any(hamming(value, other) <= self.threshold for other in self.hashes)


def wiki_pages(session: requests.Session, query: str, max_pages: int):
    params = {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6", "gsrlimit": "50",
        "prop": "imageinfo", "iiprop": "url|extmetadata|size|mime", "iiurlwidth": "1280",
        "maxlag": "5",
    }
    for _ in range(max_pages):
        try:
            response = session.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=40)
            if response.status_code == 429:
                time.sleep(10)
                continue
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            print(f"  query failed {query!r}: {exc}", flush=True)
            return
        for page in (data.get("query") or {}).get("pages", {}).values():
            yield page
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)
        time.sleep(0.3)


def fetch_image(session: requests.Session, url: str) -> bytes | None:
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60)
            if response.status_code == 429:
                time.sleep(8 + attempt * 5)
                continue
            response.raise_for_status()
            if len(response.content) < 4096:
                return None
            return response.content
        except Exception:
            time.sleep(2 + attempt * 2)
    return None


def save_candidate(raw: bytes, dest: Path, min_width: int) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = im.convert("RGB")
            if im.width < min_width:
                return None
            if max(im.size) > 1600:
                scale = 1600 / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
            dest.parent.mkdir(parents=True, exist_ok=True)
            im.save(dest, "JPEG", quality=90)
            return im.size
    except Exception:
        return None


def collect_wikimedia(session, categories, args, cand_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for category in categories:
        got = 0
        for query in QUERIES[category]:
            if got >= args.candidates_per_category:
                break
            for page in wiki_pages(session, query, args.max_pages):
                if got >= args.candidates_per_category:
                    break
                title = page.get("title", "")
                if BLOCK_TITLE.search(title):
                    continue
                info = (page.get("imageinfo") or [{}])[0]
                if info.get("mime") not in ("image/jpeg", "image/png"):
                    continue
                md = info.get("extmetadata", {})
                license_name = clean(md.get("LicenseShortName", {}).get("value", ""))
                if not allowed_license(license_name):
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue
                raw = fetch_image(session, url)
                if raw is None:
                    continue
                digest = sha256_bytes(raw)
                dest = cand_dir / category / f"{digest[:16]}.jpg"
                size = save_candidate(raw, dest, args.min_width)
                if size is None:
                    continue
                rows.append({
                    "candidate_path": str(dest.relative_to(PROJECT_ROOT)),
                    "requested": category,
                    "allowed": [category],
                    "source": "wikimedia",
                    "query": query,
                    "title": title,
                    "source_url": info.get("descriptionurl", ""),
                    "license": license_name,
                    "license_url": clean(md.get("LicenseUrl", {}).get("value", "")),
                    "artist": clean(md.get("Artist", {}).get("value", "")),
                    "credit": clean(md.get("Credit", {}).get("value", "")),
                    "sha256": digest,
                    "reuse": False,
                })
                got += 1
                time.sleep(args.sleep)
        print(f"  wikimedia {category}: {got} candidates", flush=True)
    return rows


def collect_video(args, cand_dir: Path) -> list[dict]:
    import cv2
    clips_path = PROJECT_ROOT / "data/clips/clips.jsonl"
    if not clips_path.exists():
        return []
    rows: list[dict] = []
    for line in clips_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clip = json.loads(line)
        if clip.get("label") != "nonfire":
            continue
        clip_id = clip["id"]
        allowed = None
        for key, cats in VIDEO_MAP.items():
            if key in clip_id:
                allowed = cats
                break
        if allowed is None:
            continue
        path = (PROJECT_ROOT / clip["path"]).resolve()
        if not path.exists():
            print(f"  video missing {path}", flush=True)
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / args.video_fps))
        idx = 0
        got = 0
        while got < args.video_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step:
                idx += 1
                continue
            idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 35:
                continue
            scale = min(1.0, 1280 / max(frame.shape[:2]))
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                continue
            raw = buf.tobytes()
            digest = sha256_bytes(raw)
            dest = cand_dir / clip_id / f"{digest[:16]}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            rows.append({
                "candidate_path": str(dest.relative_to(PROJECT_ROOT)),
                "requested": None,
                "allowed": allowed,
                "source": "own_video",
                "query": clip_id,
                "title": f"{clip_id} frame {idx}",
                "source_url": clip.get("source", ""),
                "license": "proprietary internal (own capture)",
                "license_url": "",
                "artist": "FireSight internal",
                "credit": clip.get("note", ""),
                "sha256": digest,
                "reuse": False,
            })
            got += 1
        cap.release()
        print(f"  video {clip_id}: {got} candidates", flush=True)
    return rows


def collect_train_mine(args) -> list[dict]:
    manifest = PROJECT_ROOT / "data/splits/train_negatives_4000.jsonl"
    if not manifest.exists():
        print("  train_mine: data/splits/train_negatives_4000.jsonl missing", flush=True)
        return []
    rows: list[dict] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        path = PROJECT_ROOT / rec["path"]
        if not path.exists():
            continue
        license_name = "CC0-1.0" if rec.get("source") == "dfire" else "CC BY 4.0"
        rows.append({
            "candidate_path": rec["path"],
            "requested": None,
            "allowed": list(CATEGORIES),
            "source": "train_negative",
            "query": rec.get("scene", ""),
            "title": rec.get("id", path.stem),
            "source_url": "",
            "license": license_name,
            "license_url": "",
            "artist": rec.get("source", ""),
            "credit": "existing TRAIN negative",
            "sha256": "",
            "reuse": True,
            "id": rec.get("id", path.stem),
        })
        if len(rows) >= args.mine_limit:
            break
    print(f"  train_mine: {len(rows)} candidates", flush=True)
    return rows


class ClipScorer:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.categories = list(CATEGORIES)
        texts = [f"a photo of {CATEGORIES[c]}" for c in self.categories] + ["a photo of fire", "a photo of smoke"]
        with torch.no_grad():
            feats = self.model.get_text_features(**self.processor(text=texts, return_tensors="pt", padding=True))
            feats = getattr(feats, "pooler_output", feats)
            self.text = feats / feats.norm(dim=-1, keepdim=True)

    def score(self, paths: list[Path], batch: int = 32) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            images, valid = [], []
            for path in chunk:
                try:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                        valid.append(path)
                except Exception:
                    continue
            if not images:
                continue
            with self.torch.no_grad():
                inputs = self.processor(images=images, return_tensors="pt")
                feats = self.model.get_image_features(**inputs)
                feats = getattr(feats, "pooler_output", feats)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                probs = (feats @ self.text.T).softmax(dim=-1)
            for path, prob in zip(valid, probs):
                values = [float(v) for v in prob]
                out[str(path)] = {
                    "probs": {c: round(values[i], 4) for i, c in enumerate(self.categories)},
                    "fire": round(values[-2], 4),
                    "smoke": round(values[-1], 4),
                }
        return out


def detector_hardness(paths: list[Path], model: Path, out_dir: Path, batch: int) -> dict[str, dict]:
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_VERBOSE"] = "False"
    if not model.exists():
        return {}
    try:
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        net = YOLO(str(model))
        list_file = out_dir / "_hardness_paths.txt"
        list_file.parent.mkdir(parents=True, exist_ok=True)
        list_file.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
        scores: dict[str, dict] = {}
        for res in net.predict(str(list_file), imgsz=640, conf=0.01, batch=batch, device="cpu",
                               verbose=False, stream=True):
            fire = smoke = 0.0
            for box in res.boxes:
                value = float(box.conf.item())
                if int(box.cls.item()) == 0:
                    fire = max(fire, value)
                else:
                    smoke = max(smoke, value)
            scores[str(Path(res.path).resolve())] = {"fire": round(fire, 4), "smoke": round(smoke, 4)}
        return scores
    except Exception as exc:
        print(f"  hardness skipped: {exc}", flush=True)
        return {}


def contact_sheet(rows: list[dict], out_path: Path, cols: int = 6, thumb=(220, 160)) -> None:
    if not rows:
        return
    rows_per = (len(rows) + cols - 1) // cols
    width, height = cols * thumb[0], rows_per * (thumb[1] + 18)
    sheet = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        try:
            with Image.open(PROJECT_ROOT / row["file"]) as im:
                im = im.convert("RGB")
                im.thumbnail(thumb)
                x = (i % cols) * thumb[0] + (thumb[0] - im.width) // 2
                y = (i // cols) * (thumb[1] + 18) + (thumb[1] - im.height) // 2
                sheet.paste(im, (x, y))
        except Exception:
            continue
        label = f"{'H' if row.get('is_hard') else '-'} {row.get('hardness', {}).get('fire', 0):.2f}/{row.get('hardness', {}).get('smoke', 0):.2f}"
        draw.text(((i % cols) * thumb[0] + 3, (i // cols) * (thumb[1] + 18) + thumb[1] + 2),
                  f"{row['id'][:26]} {label}", fill=(230, 230, 230))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect licence-filtered hard negatives for detector fine-tuning.")
    parser.add_argument("--sources", default="wikimedia,video")
    parser.add_argument("--categories", default=None)
    parser.add_argument("--tier", type=int, default=None)
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--candidates-per-category", type=int, default=300)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--min-width", type=int, default=800)
    parser.add_argument("--min-clip", type=float, default=0.0)
    parser.add_argument("--quarantine", type=float, default=0.80)
    parser.add_argument("--hard-threshold", type=float, default=0.10)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-frames", type=int, default=400)
    parser.add_argument("--mine-limit", type=int, default=4000)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--model", default="models/best_fire_v3.onnx")
    parser.add_argument("--out", default="data/hard_negatives")
    parser.add_argument("--contact-sheets", action="store_true")
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.tier:
        categories = TIERS[args.tier]
    else:
        categories = list(CATEGORIES)
    unknown = [c for c in categories if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"unknown categories: {unknown}")

    out = PROJECT_ROOT / args.out
    cand_dir = out / "_candidates"
    for sub in ("images", "labels", "contact_sheets", "lists"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "FireSightHardNegativeResearch/0.1 (local; non-commercial dataset build)"

    print(json.dumps({"sources": sources, "categories": categories, "per_category": args.per_category}), flush=True)
    rows: list[dict] = []
    if "wikimedia" in sources:
        rows += collect_wikimedia(session, categories, args, cand_dir)
    if "video" in sources and any(c in {"dust_construction_plaster", "orange_ppe_vests_cones", "ordinary_indoor",
                                        "ordinary_streets_warehouses_offices", "ordinary_other", "clouds",
                                        "vehicle_lights_streetlights"} for c in categories):
        rows += collect_video(args, cand_dir)
    if "train_mine" in sources:
        rows += collect_train_mine(args)
    if not rows:
        print("no candidates collected", flush=True)
        return 1

    existing = []
    existing_ids = set()
    manifest_path = out / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            existing_ids.add(row.get("id"))
            if row.get("dhash"):
                existing.append(int(row["dhash"], 16))
    deduper = Deduper(existing)

    paths = [PROJECT_ROOT / r["candidate_path"] for r in rows]
    print(f"scoring {len(paths)} candidates with CLIP", flush=True)
    clip = ClipScorer("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").score(paths)
    print(f"scoring {len(paths)} candidates for detector hardness", flush=True)
    hardness = detector_hardness(paths, PROJECT_ROOT / args.model, out, batch=1)

    scored: list[dict] = []
    for row in rows:
        path = PROJECT_ROOT / row["candidate_path"]
        key = str(path.resolve())
        signals = clip.get(str(path)) or clip.get(key)
        if signals is None:
            continue
        pool = [c for c in (row.get("allowed") or categories) if c in CATEGORIES]
        if row.get("requested"):
            pool = list(CATEGORIES)
        top = max(pool, key=lambda c: signals["probs"].get(c, 0.0))
        score = signals["probs"].get(top, 0.0)
        h = hardness.get(key, {})
        hard = max(h.get("fire", 0.0), h.get("smoke", 0.0))
        row_id = f"{top}:{row.get('sha256') or row.get('id') or Path(row['candidate_path']).stem}"[:120]
        row.update({
            "id": row_id,
            "file": row["candidate_path"],
            "split": "train",
            "category": top,
            "clip_top": top,
            "clip_score": score,
            "clip_fire": signals["fire"],
            "clip_smoke": signals["smoke"],
            "hardness": h,
            "max_conf": hard,
            "is_hard": hard >= args.hard_threshold,
            "dhash": None,
        })
        if score < args.min_clip:
            row["status"] = "rejected_low_clip"
        elif row.get("requested") and top != row["requested"]:
            row["status"] = "rejected_other_category"
        elif top not in categories:
            row["status"] = "rejected_other_category"
        elif hard >= args.quarantine and (signals["fire"] >= 0.25 or signals["smoke"] >= 0.25):
            row["status"] = "suspect_positive"
        else:
            row["status"] = "selected"
        if not row.get("reuse"):
            try:
                row["dhash"] = f"{dhash(path):016x}"
            except Exception:
                row["status"] = "rejected_unreadable"
        scored.append(row)

    selected: dict[str, list[dict]] = {c: [] for c in categories}
    for row in scored:
        if row["status"] == "selected" and row["category"] in selected:
            selected[row["category"]].append(row)
    for category in selected:
        selected[category].sort(
            key=lambda r: (r.get("is_hard", False), r.get("max_conf", 0.0), r.get("clip_score", 0.0)),
            reverse=True,
        )

    final_rows: list[dict] = list(scored)
    kept: dict[str, int] = Counter()
    final_hashes: list[int] = list(existing)
    for category in categories:
        for row in selected[category]:
            if kept[category] >= args.per_category:
                row["status"] = "rejected_quota"
                continue
            if row.get("dhash"):
                value = int(row["dhash"], 16)
                if any(hamming(value, other) <= 4 for other in final_hashes):
                    row["status"] = "rejected_duplicate"
                    continue
                final_hashes.append(value)
            row["split"] = "train" if row.get("reuse") else (
                "holdout" if (int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 100) < 10 else "train"
            )
            if row.get("reuse"):
                row["file"] = row["candidate_path"]
            else:
                src = PROJECT_ROOT / row["candidate_path"]
                dest = out / "images" / category / (src.stem + ".jpg")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(src.read_bytes())
                row["file"] = str(dest.relative_to(PROJECT_ROOT))
                label = out / "labels" / category / (src.stem + ".txt")
                label.parent.mkdir(parents=True, exist_ok=True)
                label.write_text("", encoding="utf-8")
            kept[category] += 1
        print(f"kept {category}: {kept[category]}", flush=True)

    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out / "attribution.jsonl").open("w", encoding="utf-8") as fh:
        for row in final_rows:
            if row["status"] != "selected" or row["source"] not in ("wikimedia", "own_video"):
                continue
            fh.write(json.dumps({
                "id": row["id"], "category": row["category"], "source": row["source"],
                "title": row.get("title", ""), "source_url": row.get("source_url", ""),
                "license": row.get("license", ""), "license_url": row.get("license_url", ""),
                "artist": row.get("artist", ""), "credit": row.get("credit", ""),
            }, ensure_ascii=False) + "\n")

    train_list = [r["file"] for r in final_rows if r["status"] == "selected" and r.get("split") == "train"]
    holdout = [r["file"] for r in final_rows if r["status"] == "selected" and r.get("split") == "holdout"]
    (out / "lists/train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8")
    (out / "lists/holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")

    if args.contact_sheets:
        for category in categories:
            rows_cat = [r for r in final_rows if r["category"] == category and r["status"] in ("selected", "suspect_positive")]
            contact_sheet(rows_cat, out / "contact_sheets" / f"{category}.jpg")

    counts = Counter(r["category"] for r in final_rows if r["status"] == "selected")
    hard_counts = Counter(r["category"] for r in final_rows if r["status"] == "selected" and r.get("is_hard"))
    sources_mix = Counter(r["source"] for r in final_rows if r["status"] == "selected")
    licences = Counter(r["license"] for r in final_rows if r["status"] == "selected")
    suspect = Counter(r["category"] for r in final_rows if r["status"] == "suspect_positive")
    summary = {
        "categories": {c: counts.get(c, 0) for c in categories},
        "hard_categories": {c: hard_counts.get(c, 0) for c in categories},
        "selected": sum(counts.values()),
        "hard_selected": sum(hard_counts.values()),
        "suspect_positive": dict(suspect),
        "sources": dict(sources_mix),
        "licences": dict(licences),
        "train": len(train_list),
        "holdout": len(holdout),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Hard-negative collection summary", "", f"Selected: **{summary['selected']}** "
             f"(hard: {summary['hard_selected']})", "",
             "| Category | Selected | Hard | Suspect |", "|---|---:|---:|---:|"]
    for c in categories:
        lines.append(f"| {c} | {counts.get(c, 0)} | {hard_counts.get(c, 0)} | {suspect.get(c, 0)} |")
    lines += ["", "## Sources", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(sources_mix.items())]
    lines += ["", "## Licences", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(licences.items())]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

