from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT
from firesight.data.prepare import prepare_all
from firesight.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge public fire/smoke datasets into a unified, deduplicated, scene-disjoint manifest.")
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    parser.add_argument("--sources", default="dfire,fasdd,local")
    parser.add_argument("--local-root", default=str(PROJECT_ROOT.parent))
    parser.add_argument("--limit-per-split", type=int, default=None, help="Debug/subsample cap; leave unset for full data.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--split-map", default=None, help="JSON mapping e.g. '{\"valid\":\"val\"}'")
    args = parser.parse_args()

    setup_logging("INFO")
    split_map = json.loads(args.split_map) if args.split_map else None
    stats = prepare_all(
        Path(args.root),
        [s.strip() for s in args.sources.split(",") if s.strip()],
        local_root=Path(args.local_root),
        limit_per_split=args.limit_per_split,
        workers=args.workers,
        phash_threshold=args.phash_threshold,
        split_map=split_map,
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
