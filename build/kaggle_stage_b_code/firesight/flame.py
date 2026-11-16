from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .types import Detection

log = logging.getLogger(__name__)


def flame_fraction(frame: np.ndarray, box, min_saturation: int = 100, min_value: int = 120) -> float:
    """Fraction of pixels in the box that match a broad visible-flame colour mask."""
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box.to_list()]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    mask = (((hh < 30) | (hh > 160)) & (ss > min_saturation) & (vv > min_value))
    return float(mask.mean())


def relabel_flame_evidence(
    frame: np.ndarray,
    detections: Sequence[Detection],
    min_fraction: float = 0.08,
    min_pixels: int = 120,
) -> list[Detection]:
    """Relabel smoke boxes that clearly contain flame-coloured pixels.

    This does not create new detections, so it cannot create a new false alarm on a
    clean negative image. It only fixes the fire/smoke class of an existing box and
    records the evidence in ``meta``.
    """
    for det in detections:
        if det.label != "smoke" or det.confidence < 0.25:
            continue
        frac = flame_fraction(frame, det.box)
        pixels = frac * det.box.area
        if frac >= min_fraction and pixels >= min_pixels:
            det.meta["original_label"] = "smoke"
            det.meta["flame_fraction"] = round(frac, 4)
            det.meta["relabel_reason"] = "flame_color_evidence"
            det.label = "fire"
    return list(detections)
