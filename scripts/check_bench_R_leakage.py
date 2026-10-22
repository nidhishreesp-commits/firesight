from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest


def main() -> int:
    index = json.loads((PROJECT_ROOT / "bench_R/index.json").read_text(encoding="utf-8"))
    train = load_manifest(PROJECT_ROOT / "data/splits/train.jsonl", root=PROJECT_ROOT)
    val = load_manifest(PROJECT_ROOT / "data/splits/val.jsonl", root=PROJECT_ROOT)
    rng = random.Random(7)
    refs = rng.sample(train, min(1400, len(train))) + rng.sample(val, min(400, len(val)))
    from transformers import CLIPModel, CLIPProcessor
    mid = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    model = CLIPModel.from_pretrained(mid).eval()
    processor = CLIPProcessor.from_pretrained(mid)

    def embed(paths):
        chunks = []
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
            chunks.append(f)
        return torch.cat(chunks)

    ref_paths = [PROJECT_ROOT / r.path for r in refs]
    bench_paths = [PROJECT_ROOT / r["file"] for r in index]
    ref_emb = embed(ref_paths)
    bench_emb = embed(bench_paths)
    sims = (bench_emb @ ref_emb.T).max(dim=1)
    rows = []
    flagged = []
    for row, score, idx in zip(index, sims.values.tolist(), sims.indices.tolist()):
        rows.append({"id": row["id"], "max_cosine": round(score, 4), "nearest": refs[idx].id})
        if score >= 0.98:
            flagged.append({"id": row["id"], "max_cosine": round(score, 4), "nearest": refs[idx].id})
    result = {"bench_images": len(index), "clip_reference_images": len(refs), "clip_threshold": 0.98,
              "clip_flagged": len(flagged), "flagged": flagged,
              "pHash_note": "59 candidate images were removed during collection by pHash Hamming<=6 against train/val/testbench100.",
              "clip_rows": rows}
    (PROJECT_ROOT / "bench_R/LEAKAGE_CHECK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("bench_images", "clip_reference_images", "clip_flagged", "flagged")}, indent=2))
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
