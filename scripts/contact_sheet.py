from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.manifest import load_manifest
from firesight.eval.runner import load_predictions


def draw(rec, dets, tile=(360, 240)):
    img = cv2.imread(str(PROJECT_ROOT / rec.path))
    if img is None:
        return np.zeros((tile[1], tile[0], 3), np.uint8)
    h, w = img.shape[:2]
    for gt in rec.labels:
        x1, y1, x2, y2 = map(int, gt.box.to_list())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), max(2, int(min(w, h) / 300)))
        cv2.putText(img, f"GT {gt.label}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    for det in dets:
        x1, y1, x2, y2 = map(int, det.box.to_list())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), max(2, int(min(w, h) / 300)))
        cv2.putText(img, f"{det.label} {det.confidence:.2f}", (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    img = cv2.resize(img, tile)
    cv2.putText(img, rec.id[-42:], (4, tile[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Contact sheet of GT vs predictions for visual audit.")
    parser.add_argument("--manifest", default="data/splits/test_bench.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="artifacts/contact_sheet.jpg")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--filter", default="all", choices=["all", "fire", "smoke", "negative", "negative_fp", "fp", "fn", "correct"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    records = load_manifest(PROJECT_ROOT / args.manifest, root=PROJECT_ROOT)
    preds, _ = load_predictions(PROJECT_ROOT / args.predictions)
    from firesight.eval.metrics import _greedy_match
    selected = []
    for rec in records:
        dets = [d for d in preds.get(rec.id, []) if d.confidence >= args.threshold]
        mp, mg = _greedy_match(dets, rec.labels, 0.5)
        fp = [d for d, m in zip(dets, mp) if not m]
        fn = [g for g, m in zip(rec.labels, mg) if not m]
        if args.filter == "negative" and rec.labels:
            continue
        if args.filter == "fire" and not any(g.label == "fire" for g in rec.labels):
            continue
        if args.filter == "smoke" and not any(g.label == "smoke" for g in rec.labels):
            continue
        if args.filter == "fp" and not fp:
            continue
        if args.filter == "negative_fp" and (rec.labels or not dets):
            continue
        if args.filter == "fn" and not fn:
            continue
        if args.filter == "correct" and (fp or fn or not rec.labels):
            continue
        selected.append((rec, dets))
        if len(selected) >= args.n:
            break
    tiles = [draw(rec, dets) for rec, dets in selected]
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(rows)
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    print(out, sheet.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
