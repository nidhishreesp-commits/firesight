import numpy as np

from firesight.flame import flame_fraction, relabel_flame_evidence
from firesight.types import Box, Detection


def test_flame_fraction_on_orange_patch():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:80, 20:80] = (0, 100, 240)
    assert flame_fraction(frame, Box(10, 10, 90, 90)) > 0.3


def test_flame_fraction_on_gray_smoke():
    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    assert flame_fraction(frame, Box(10, 10, 90, 90)) < 0.01


def test_smoke_box_with_flame_is_relabelled():
    frame = np.full((100, 100, 3), 60, dtype=np.uint8)
    frame[30:70, 30:70] = (0, 120, 230)
    det = Detection("smoke", 0.8, Box(10, 10, 90, 90))
    out = relabel_flame_evidence(frame, [det])
    assert out[0].label == "fire"
    assert out[0].meta["original_label"] == "smoke"
    assert out[0].meta["flame_fraction"] > 0.08


def test_fire_detection_is_not_changed():
    frame = np.full((100, 100, 3), 60, dtype=np.uint8)
    det = Detection("fire", 0.8, Box(10, 10, 90, 90))
    out = relabel_flame_evidence(frame, [det])
    assert out[0].label == "fire"
    assert "relabel_reason" not in out[0].meta
