from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="testbench100")
    parser.add_argument("--kind", choices=["positive", "negative"], required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()
    bench = PROJECT_ROOT / args.bench
    index = json.loads((bench / "index.json").read_text(encoding="utf-8"))
    rows = [r for r in index if r["kind"] == args.kind]
    tiles = []
    for row in rows:
        img = cv2.imread(str(bench / "images" / row["file"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        for box in row.get("boxes_fire", []):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for box in row.get("boxes_smoke", []):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.resize(img, (320, 200))
        label = f"{row['category']} | {row['difficulty']}"
        cv2.putText(img, label[:44], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(img, row["id"][-38:], (4, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        tiles.append(img)
    while len(tiles) % args.cols:
        tiles.append(np.zeros((200, 320, 3), np.uint8))
    grid = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(grid)
    out = PROJECT_ROOT / (args.out or f"artifacts/bench100_{args.kind}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
