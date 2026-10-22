from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FireSight FastAPI service.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    import os
    if args.config:
        os.environ["FIRESIGHT_CONFIG"] = args.config
    import uvicorn
    uvicorn.run("firesight.api.main:app", host=args.host or cfg.server.host, port=args.port or cfg.server.port,
                reload=args.reload, log_level=cfg.app.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
