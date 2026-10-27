from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass
class DetectorConfig:
    model_path: str = "models/best_fire_v2.onnx"
    backend: str = "onnx"
    imgsz: int = 640
    iou: float = 0.45
    max_det: int = 100
    device: str = "auto"
    class_map: dict[int, str] = field(default_factory=lambda: {0: "fire", 1: "smoke"})
    conf: dict[str, float] = field(default_factory=lambda: {"fire": 0.35, "smoke": 0.25})


@dataclass
class TemporalConfig:
    enabled: bool = True
    n_of_m: dict[str, list[int]] = field(default_factory=lambda: {"fire": [3, 5], "smoke": [4, 6]})
    max_gap_s: float = 2.0
    min_area_px: int = 400
    smoke_growth_window: int = 8
    smoke_growth_min_ratio: float = 0.15
    tracker: str = "bytetrack"
    track_iou: float = 0.3


@dataclass
class VerifierConfig:
    enabled: bool = False
    model_path: str = "models/verifier.onnx"
    threshold: float = 0.60
    classes: list[str] = field(default_factory=lambda: ["fire", "lookalike", "smoke"])


@dataclass
class AlarmConfig:
    min_duration_s: float = 1.0
    cooldown_s: float = 30.0
    escalation: list[dict[str, Any]] = field(default_factory=list)
    fire_type_include: list[str] = field(default_factory=lambda: ["uncontrolled", "unknown"])
    fire_type_exclude: list[str] = field(default_factory=lambda: ["controlled", "spark_benign", "screen"])
    fire_type_warning: list[str] = field(default_factory=lambda: ["electrical_arc"])


@dataclass
class StreamConfig:
    sample_fps: float = 5.0
    reconnect_s: float = 3.0
    jpeg_quality: int = 80
    max_cameras: int = 9


@dataclass
class BenchmarkConfig:
    test_manifest: str = "data/splits/test.jsonl"
    val_manifest: str = "data/splits/val.jsonl"
    target_fpr: float = 0.001
    iou_threshold: float = 0.5
    frames_per_day: int = 86400
    clip_labels: str = "data/clips/clips.jsonl"
    bootstrap_samples: int = 500


@dataclass
class AppConfig:
    name: str = "firesight"
    data_root: str = "data"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    max_upload_mb: int = 512


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    monitoring: dict[str, Any] = field(default_factory=dict)

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def data_root(self) -> Path:
        return self.path(self.app.data_root)

    @property
    def models_dir(self) -> Path:
        return self.path(self.app.models_dir)

    @property
    def artifacts_dir(self) -> Path:
        return self.path(self.app.artifacts_dir)


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_overrides() -> dict[str, Any]:
    override: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith("FIRESIGHT__"):
            continue
        parts = key[len("FIRESIGHT__"):].lower().split("__")
        cursor = override
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw)
    return override


def _from_dict(data: dict[str, Any]) -> Config:
    def section(name: str, cls):
        return cls(**data.get(name, {}))

    return Config(
        app=section("app", AppConfig),
        server=section("server", ServerConfig),
        detector=section("detector", DetectorConfig),
        temporal=section("temporal", TemporalConfig),
        verifier=section("verifier", VerifierConfig),
        alarm=section("alarm", AlarmConfig),
        stream=section("stream", StreamConfig),
        benchmark=section("benchmark", BenchmarkConfig),
        monitoring=data.get("monitoring", {}),
    )


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else Path(os.getenv("FIRESIGHT_CONFIG", DEFAULT_CONFIG))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = _deep_merge(raw, _env_overrides())
    cfg = _from_dict(raw)
    cfg.detector.class_map = {int(k): v for k, v in cfg.detector.class_map.items()}
    return cfg
