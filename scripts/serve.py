from __future__ import annotations

import argparse
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import load_config


def _probe_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((_probe_host(host), port)) == 0


def _listening_pid(port: int) -> int | None:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            check=True,
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING":
            try:
                return int(parts[-1])
            except ValueError:
                return None
    return None


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
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    if _port_in_use(host, port):
        pid = _listening_pid(port)
        owner = f" (PID {pid})" if pid else ""
        print(
            f"error: {host}:{port} is already in use{owner}. "
            "Another FireSight server is probably still running.",
            file=sys.stderr,
        )
        if pid:
            print(f"  stop it:  Stop-Process -Id {pid}", file=sys.stderr)
        print(f"  or use:   python scripts/serve.py --port {port + 1}", file=sys.stderr)
        print("           (then update the proxy target in frontend/vite.config.ts)", file=sys.stderr)
        return 1
    import uvicorn
    uvicorn.run("firesight.api.main:app", host=host, port=port,
                reload=args.reload, log_level=cfg.app.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
