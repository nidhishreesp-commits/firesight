import numpy as np
import pytest

from firesight.eval.metrics import box_iou, compute_ap, image_level_curve, recall_at_fpr, select_operating_point
from firesight.types import Box, Detection, GroundTruth, ImageRecord


def make_record(i, labels, negative=False):
    return ImageRecord(id=f"img{i}", path=f"img{i}.jpg", width=100, height=100, source="test", scene=f"s{i}",
                       split="test", labels=[] if negative else [GroundTruth(l, Box(10, 10, 50, 50)) for l in labels])


def test_box_iou_identity_and_disjoint():
    a = Box(0, 0, 10, 10)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, Box(20, 20, 30, 30)) == 0.0
    assert box_iou(a, Box(0, 0, 5, 10)) == pytest.approx(0.5)


def test_compute_ap_perfect_and_none():
    records = [make_record(0, ["fire"]), make_record(1, ["fire"])]
    perfect = {"img0": [Detection("fire", 0.9, Box(10, 10, 50, 50))], "img1": [Detection("fire", 0.8, Box(10, 10, 50, 50))]}
    assert compute_ap(records, perfect, "fire", 0.5) == pytest.approx(1.0)
    empty = {"img0": [], "img1": []}
    assert compute_ap(records, empty, "fire", 0.5) == pytest.approx(0.0)


def test_select_operating_point_respects_fpr():
    val = [make_record(0, ["fire"]), make_record(1, [], negative=True), make_record(2, [], negative=True)]
    test = [make_record(0, ["fire"]), make_record(1, [], negative=True)]
    val_preds = {
        "img0": [Detection("fire", 0.9, Box(10, 10, 50, 50))],
        "img1": [Detection("fire", 0.05, Box(1, 1, 5, 5))],
        "img2": [],
    }
    test_preds = {
        "img0": [Detection("fire", 0.9, Box(10, 10, 50, 50))],
        "img1": [Detection("fire", 0.05, Box(1, 1, 5, 5))],
    }
    op = select_operating_point(val, val_preds, test, test_preds, target_fpr=0.0)
    assert op.fpr == 0.0
    assert op.thresholds["fire"] >= 0.05


def test_recall_at_fpr_interpolates():
    curve = [{"threshold": 0.9, "recall": 0.2, "fpr": 0.0}, {"threshold": 0.5, "recall": 0.6, "fpr": 0.02},
             {"threshold": 0.1, "recall": 0.9, "fpr": 0.5}]
    assert recall_at_fpr(curve, 0.01)["recall"] == pytest.approx(0.6)
    assert recall_at_fpr(curve, 0.0)["recall"] == pytest.approx(0.2)


def test_image_level_curve_monotonic_recall():
    records = [make_record(0, ["fire"]), make_record(1, [], negative=True)]
    preds = {"img0": [Detection("fire", 0.9, Box(10, 10, 50, 50))], "img1": [Detection("fire", 0.4, Box(1, 1, 5, 5))]}
    curve = image_level_curve(records, preds, 0.5)
    recalls = [row["recall"] for row in sorted(curve, key=lambda r: r["threshold"])]
    assert recalls == sorted(recalls)
