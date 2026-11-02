from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("camera_id", "model", "event", "latency_ms", "fps", "alarm", "request_id"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_logs: bool | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    if json_logs is None:
        json_logs = sys.stdout.isatty() is False
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers = [handler]
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    _CONFIGURED = True


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
