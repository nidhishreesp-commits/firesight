from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def clip(self, w: int, h: int) -> "Box":
        return Box(max(0.0, min(self.x1, w)), max(0.0, min(self.y1, h)),
                   max(0.0, min(self.x2, w)), max(0.0, min(self.y2, h)))

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    label: str
    confidence: float
    box: Box
    track_id: int | None = None
    source: str = "model"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": [round(v, 2) for v in self.box.to_list()],
            "track_id": self.track_id,
            "source": self.source,
        }
        if self.meta:
            data["meta"] = self.meta
        return data


@dataclass
class GroundTruth:
    label: str
    box: Box

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "box": self.box.to_list()}


@dataclass
class ImageRecord:
    id: str
    path: str
    width: int
    height: int
    source: str
    scene: str
    split: str
    labels: list[GroundTruth] = field(default_factory=list)
    category: str = "none"
    phash: str | None = None
    origin: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        return len(self.labels) == 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["labels"] = [gt.to_dict() for gt in self.labels]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], root: Path | None = None) -> "ImageRecord":
        path = data["path"]
        if root is not None and not Path(path).is_absolute():
            path = str(root / path)
        return cls(
            id=data["id"],
            path=path,
            width=int(data["width"]),
            height=int(data["height"]),
            source=data.get("source", "unknown"),
            scene=data.get("scene", "unknown"),
            split=data.get("split", "unspecified"),
            labels=[GroundTruth(label=l["label"], box=Box(*l["box"])) for l in data.get("labels", [])],
            category=data.get("category", "none"),
            phash=data.get("phash"),
            origin=data.get("origin"),
            meta=data.get("meta", {}),
        )
