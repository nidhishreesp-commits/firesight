from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest

# family|category|prompt
CATEGORIES = [
    ("light_colour", "sunset_sunrise", "a sunset or sunrise with an orange sky"),
    ("light_colour", "street_head_lights", "bright street lights or car headlights at night"),
    ("light_colour", "lamps_neon_brake", "red or orange lamps, neon signs, or car brake lights"),
    ("light_colour", "chandelier_festival", "a chandelier, fairy lights, or festival lighting"),
    ("light_colour", "reflection_glare", "a reflection or sun glare on water, glass, or metal"),
    ("light_colour", "orange_clothing_ppe", "people wearing orange or red clothing, safety vests, or helmets"),
    ("light_colour", "cones_machinery", "traffic cones, orange construction machinery, or equipment"),
    ("light_colour", "flowers_leaves", "orange flowers or autumn leaves"),
    ("smoke_like", "fog_mist_haze", "fog, mist, or haze in a landscape"),
    ("smoke_like", "clouds", "clouds in the sky"),
    ("smoke_like", "steam_cooking", "steam from cooking, a kettle, a shower, or a manhole"),
    ("smoke_like", "exhaust_chimney", "vehicle exhaust or a chimney emitting water vapour"),
    ("smoke_like", "dust_snow", "dust, a sandstorm, or blowing snow"),
    ("smoke_like", "overexposed_white", "a bright overexposed white wall or sky"),
    ("fire_adjacent", "candle_match_stove", "a candle, match, lighter, incense, or gas stove flame"),
    ("fire_adjacent", "campfire_bbq", "a controlled bonfire, campfire, or barbecue"),
    ("fire_adjacent", "welding_grinding", "welding or grinding sparks"),
    ("fire_adjacent", "fireworks", "fireworks in the sky"),
    ("fire_adjacent", "screen_fire", "a fire on a television or movie screen"),
    ("camera_artifact", "blur_noise_compression", "a blurry, noisy, or low-quality camera image"),
    ("camera_artifact", "rain_fogged_lens", "rain drops or fog on a camera lens"),
    ("camera_artifact", "infrared_night", "an infrared or night-vision camera image"),
    ("easy_negative", "plain_scene", "a plain landscape or building with no fire or smoke"),
    ("easy_negative", "people_vehicles", "people or vehicles in a normal scene"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="CLIP-assisted inventory of existing hard negatives (for selection only).")
    parser.add_argument("--manifest", default="data/splits/test.jsonl")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="hard_negatives/clip_inventory.json")
    args = parser.parse_args()
    records = [r for r in load_manifest(PROJECT_ROOT / args.manifest, root=PROJECT_ROOT) if not r.labels][:args.limit]
    model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    texts = [f"a photo of {c[2]}" for c in CATEGORIES]
    with torch.no_grad():
        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rows = []
    counts = Counter()
    for start in range(0, len(records), args.batch):
        batch = records[start:start + args.batch]
        images = []
        valid = []
        for rec in batch:
            try:
                images.append(Image.open(PROJECT_ROOT / rec.path).convert("RGB"))
                valid.append(rec)
            except Exception:
                continue
        if not images:
            continue
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (feats @ text_features.T).softmax(dim=-1)
        for rec, prob in zip(valid, probs):
            idx = int(prob.argmax())
            family, category, _ = CATEGORIES[idx]
            rows.append({
                "id": rec.id, "path": rec.path, "source": rec.source, "family": family,
                "category": category, "clip_score": round(float(prob[idx]), 4),
                "all_scores": {CATEGORIES[i][1]: round(float(prob[i]), 4) for i in range(len(CATEGORIES))},
            })
            counts[(family, category)] += 1
        if start % 256 == 0:
            print(f"categorized {len(rows)}/{len(records)}", flush=True)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model_id, "images": len(rows), "counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())}, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
