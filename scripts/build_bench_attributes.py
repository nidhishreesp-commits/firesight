from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT

CONTROLLED = {
    "candle": "candle", "match_lighter": "lighter", "gas_stove": "gas_stove",
    "campfire": "campfire", "fireplace": "fireplace", "screen_fire": "screen_fire",
}
SPARK = {"welding": "welding_grinding", "fireworks": "fireworks"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Side-car flame_type attributes for testbench100; does NOT modify locked bench files.")
    parser.add_argument("--out", default="bench_attributes.csv")
    args = parser.parse_args()
    index = {r["id"]: r for r in json.loads((PROJECT_ROOT / "testbench100/index.json").read_text(encoding="utf-8"))}
    rows = []
    with (PROJECT_ROOT / "testbench100/labels.csv").open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            meta = index.get(row["image_id"], {})
            boxes = json.loads(row["boxes_fire"] or "[]")
            if not int(row["has_fire"]) or not boxes:
                continue
            category = row["category"]
            if category in SPARK:
                ftype, subtype = "spark", SPARK[category]
            elif category in CONTROLLED:
                ftype, subtype = "controlled", CONTROLLED[category]
            elif category == "fireworks":
                ftype, subtype = "spark", "fireworks"
            elif category == "screen_fire":
                ftype, subtype = "screen", "tv_movie_fire"
            else:
                ftype, subtype = "uncontrolled", category
            for i, box in enumerate(boxes):
                rows.append({
                    "image_id": row["image_id"], "file": row["file"], "box_index": i,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                    "flame_type": ftype, "flame_subtype": subtype,
                    "has_fire": 1, "has_smoke": int(row["has_smoke"]),
                    "category": category, "difficulty": row["difficulty"],
                    "source": row["source"], "license": row["license"],
                    "reviewer": row["reviewer"], "reviewed_at": row["reviewed_at"],
                    "notes": row.get("notes", "") + ("; controlled flame is positive by reviewer policy" if ftype != "uncontrolled" else ""),
                })
    out = PROJECT_ROOT / args.out
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "out": str(out),
                      "types": {t: sum(1 for r in rows if r["flame_type"] == t) for t in ("uncontrolled", "controlled", "spark", "screen", "unknown")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
