from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest

KEYWORDS = {
    "candle": ["candle", "candles", "candlestick"],
    "match": ["match", "matches", "lighter"],
    "sunset": ["sunset", "sunrise", "dusk"],
    "steam": ["steam", "kettle", "boiler"],
    "fog": ["fog", "mist", "haze"],
    "headlight": ["headlight", "taillight", "headlamp"],
    "lamp": ["lamp", "lantern", "chandelier"],
    "cloud": ["cloud", "clouds"],
    "kitchen": ["kitchen", "cooking", "bbq", "barbecue", "grill"],
    "welding": ["weld", "welding", "spark"],
    "orange_object": ["orange", "traffic_cone", "cone"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Flag/optionally quarantine likely label noise by filename keywords.")
    parser.add_argument("--manifest", default="data/splits/train.jsonl")
    parser.add_argument("--out", default="data/processed/label_noise_review.jsonl")
    parser.add_argument("--quarantine", action="store_true", help="Empty labels for flagged images (moves originals to labels_quarantine).")
    args = parser.parse_args()
    records = load_manifest(PROJECT_ROOT / args.manifest, root=PROJECT_ROOT)
    flagged = []
    for rec in records:
        low = Path(rec.path).name.lower()
        reasons = [cat for cat, words in KEYWORDS.items() if any(w in low for w in words)]
        if not reasons:
            continue
        flagged.append({"id": rec.id, "path": rec.path, "source": rec.source, "labels": len(rec.labels),
                        "categories": reasons, "action": "review"})
        if args.quarantine and rec.labels:
            label = PROJECT_ROOT / "data/processed/labels" / rec.source.replace("local_seed", "local").replace("local_negative", "local") / rec.split / (Path(rec.path).stem + ".txt")
            if label.exists():
                q = label.with_suffix(".quarantine.txt")
                shutil.move(label, q)
                label.write_text("", encoding="utf-8")
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(f) for f in flagged) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": args.manifest, "flagged": len(flagged), "out": str(out),
                      "categories": {c: sum(1 for f in flagged if c in f["categories"]) for c in KEYWORDS}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
