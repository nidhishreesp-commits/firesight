"""Build the YOLO training lists for a hard-negative fine-tune.

Positives come from the packaged pilot dataset (25% or 50% list); hard negatives come
from a collect_hard_negatives.py manifest (status=selected, split=train). No copying:
paths are absolute and ultralytics resolves labels by replacing /images/ with /labels/.

Usage:
    python scripts/build_finetune_dataset.py --hard-neg data/hard_negatives_v2 \
        --positives build/kaggle_dataset/lists/train_25.txt \
        --val build/kaggle_dataset/lists/val_early.txt --out build/ft_dataset \
        --max-neg 3000 --oversample-tier1 2
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

TIER1 = {"clouds", "fog_haze_mist", "dust_construction_plaster",
         "reflections_lens_flare_glare", "sunset_orange_sky"}


def resolve(line: str, base: Path) -> str:
    p = Path(line.strip())
    if not p.is_absolute():
        p = base / p
    return str(p.resolve()).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-neg", default="data/hard_negatives_v2")
    ap.add_argument("--positives", default="build/kaggle_dataset/lists/train_25.txt")
    ap.add_argument("--val", default="build/kaggle_dataset/lists/val_early.txt")
    ap.add_argument("--out", default="build/ft_dataset")
    ap.add_argument("--max-neg", type=int, default=3000)
    ap.add_argument("--oversample-tier1", type=int, default=2)
    args = ap.parse_args()

    hn = PROJECT_ROOT / args.hard_neg
    manifest = hn / "manifest.jsonl"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}")
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    selected = [r for r in rows if r.get("status") == "selected" and r.get("split") == "train"]
    by_cat = Counter(r["category"] for r in selected)

    # tier-1 first, then the rest, capped
    ordered = sorted(selected, key=lambda r: (r["category"] not in TIER1, -float(r.get("max_conf") or 0)))
    picked = ordered[: args.max_neg]
    neg_lines = []
    for r in picked:
        path = resolve(r["file"], PROJECT_ROOT)
        neg_lines.extend([path] * (args.oversample_tier1 if r["category"] in TIER1 else 1))

    pkg = PROJECT_ROOT / "build/kaggle_dataset"
    pos = [resolve(l, pkg) for l in (PROJECT_ROOT / args.positives).read_text(encoding="utf-8").splitlines() if l.strip()]
    val = [resolve(l, pkg) for l in (PROJECT_ROOT / args.val).read_text(encoding="utf-8").splitlines() if l.strip()]

    out = PROJECT_ROOT / args.out
    (out / "lists").mkdir(parents=True, exist_ok=True)
    (out / "lists/train.txt").write_text("\n".join(pos + neg_lines) + "\n", encoding="utf-8")
    (out / "lists/val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out).replace("\\", "/"),
        "train": "lists/train.txt",
        "val": "lists/val.txt",
        "names": {0: "fire", 1: "smoke"},
        "nc": 2,
    }, sort_keys=False), encoding="utf-8")

    info = {"positives": len(pos), "negatives_unique": len(picked),
            "negative_lines": len(neg_lines), "neg_share_of_train": round(len(neg_lines) / max(len(pos) + len(neg_lines), 1), 4),
            "val": len(val), "by_category": dict(by_cat), "max_neg": args.max_neg,
            "oversample_tier1": args.oversample_tier1}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
