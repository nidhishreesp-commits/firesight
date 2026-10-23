from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench_R"


def new_id(row):
    return f"benchR:{row['sha256'][:8]}:{Path(row['file']).stem[:60]}"


def main():
    index = json.loads((BENCH / "index.json").read_text(encoding="utf-8"))
    old_to_new = {}
    for row in index:
        old_to_new.setdefault(row["id"], new_id(row))
    for row in index:
        row["id"] = new_id(row)
    (BENCH / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    reviews_path = BENCH / "reviews.json"
    if reviews_path.exists():
        reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
        fixed = {}
        for old, val in reviews.items():
            if old in old_to_new:
                fixed[old_to_new[old]] = val
        reviews_path.write_text(json.dumps(fixed, indent=2), encoding="utf-8")
        print("reviews remapped", len(fixed))

    prop_path = BENCH / "proposals.json"
    if prop_path.exists():
        props = json.loads(prop_path.read_text(encoding="utf-8"))
        images = {}
        for old, val in props.get("images", {}).items():
            if old in old_to_new:
                images[old_to_new[old]] = val
        props["images"] = images
        props["meta"]["blind_ids"] = [old_to_new.get(x, x) for x in props["meta"].get("blind_ids", [])]
        prop_path.write_text(json.dumps(props, indent=2), encoding="utf-8")
        print("proposals remapped", len(images))

    labels_path = BENCH / "labels.csv"
    if labels_path.exists():
        rows = list(csv.DictReader(labels_path.open(encoding="utf-8")))
        for row in rows:
            row["image_id"] = old_to_new.get(row["image_id"], row["image_id"])
        with labels_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for raw_name in ("owlv2", "dino"):
        raw_path = BENCH / "proposals_raw" / f"{raw_name}.jsonl"
        if not raw_path.exists():
            continue
        out = []
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["id"] = old_to_new.get(row["id"], row["id"])
            out.append(json.dumps(row))
        raw_path.write_text("\n".join(out) + "\n", encoding="utf-8")

    print("index", len(index), "unique", len({r["id"] for r in index}))


if __name__ == "__main__":
    main()
