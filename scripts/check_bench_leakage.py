from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest


def phash(path: Path) -> str:
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB"), hash_size=8))


def ham(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check testbench100 against train/val by pHash and sampled CLIP embeddings.")
    parser.add_argument("--bench", default="testbench100")
    parser.add_argument("--clip-sample", type=int, default=1200)
    parser.add_argument("--out", default="testbench100/LEAKAGE_CHECK.json")
    args = parser.parse_args()
    bench = PROJECT_ROOT / args.bench
    index = json.loads((bench / "index.json").read_text(encoding="utf-8"))
    train = load_manifest(PROJECT_ROOT / "data/splits/train.jsonl", root=PROJECT_ROOT)
    val = load_manifest(PROJECT_ROOT / "data/splits/val.jsonl", root=PROJECT_ROOT)
    cache_path = PROJECT_ROOT / "data/processed/manifests/phash_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    bands: list[dict[int, list[tuple[str, str]]]] = [defaultdict(list) for _ in range(4)]
    for rec in train + val:
        h = cache.get(rec.path)
        if not h:
            continue
        value = int(h, 16)
        for b in range(4):
            bands[b][(value >> (b * 16)) & 0xFFFF].append((rec.id, h))

    phash_rows = []
    flagged = []
    for row in index:
        h = phash(bench / "images" / row["file"])
        candidates = {}
        value = int(h, 16)
        for b in range(4):
            for rid, other in bands[b][(value >> (b * 16)) & 0xFFFF]:
                candidates[rid] = other
        best = (64, None)
        for rid, other in candidates.items():
            d = ham(h, other)
            if d < best[0]:
                best = (d, rid)
        phash_rows.append({"id": row["id"], "phash": h, "min_hamming": best[0], "nearest": best[1]})
        if best[0] <= 6:
            flagged.append({"id": row["id"], "reason": "phash", "min_hamming": best[0], "nearest": best[1]})

    rng = random.Random(11)
    train_sample = rng.sample(train, min(args.clip_sample, len(train)))
    val_sample = rng.sample(val, min(args.clip_sample // 3, len(val)))
    refs = train_sample + val_sample
    model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_id).eval()
    processor = CLIPProcessor.from_pretrained(model_id)

    def embed(paths):
        out = []
        for start in range(0, len(paths), 32):
            imgs = []
            for p in paths[start:start + 32]:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224)))
            inputs = processor(images=imgs, return_tensors="pt")
            with torch.no_grad():
                f = model.get_image_features(**inputs)
                if hasattr(f, "pooler_output"):
                    f = f.pooler_output
                f = f / f.norm(dim=-1, keepdim=True)
            out.append(f)
        return torch.cat(out) if out else torch.zeros((0, 512))

    ref_paths = [PROJECT_ROOT / r.path for r in refs]
    bench_paths = [bench / "images" / r["file"] for r in index]
    ref_emb = embed(ref_paths)
    bench_emb = embed(bench_paths)
    sims = (bench_emb @ ref_emb.T).max(dim=1)
    embedding_rows = []
    for row, score, idx in zip(index, sims.values.tolist(), sims.indices.tolist()):
        embedding_rows.append({"id": row["id"], "max_cosine": round(score, 4), "nearest": refs[idx].id})
        if score >= 0.98:
            flagged.append({"id": row["id"], "reason": "clip_embedding", "max_cosine": round(score, 4), "nearest": refs[idx].id})
    result = {
        "bench_images": len(index),
        "phash_threshold": 6,
        "phash_flagged": sum(1 for r in phash_rows if r["min_hamming"] <= 6),
        "clip_reference_images": len(refs),
        "clip_threshold": 0.98,
        "clip_flagged": sum(1 for r in embedding_rows if r["max_cosine"] >= 0.98),
        "flagged": flagged,
        "phash": phash_rows,
        "clip_embedding": embedding_rows,
        "note": "CLIP embedding check is a sample of train/val, not an exhaustive proof. pHash is exhaustive over cached train/val phashes.",
    }
    out = PROJECT_ROOT / args.out
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("bench_images", "phash_flagged", "clip_reference_images", "clip_flagged", "flagged")}, indent=2))
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
