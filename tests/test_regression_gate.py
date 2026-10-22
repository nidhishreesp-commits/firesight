"""Regression gate.

Fails the build if the frozen-TEST operating metrics of the shipped detector
worsen beyond tolerance relative to the committed baseline. Update the baseline
ONLY with an explicit, reviewed change (and never by tuning on TEST).
"""
import json
from pathlib import Path

import pytest

from firesight.config import PROJECT_ROOT

BASELINE = PROJECT_ROOT / "tests/regression_baseline.json"
CURRENT = PROJECT_ROOT / "artifacts/benchmark_target0001/ssprod_best_fire_v3/metrics.json"
TOLERANCE = {"recall_drop": 0.02, "fpr_increase": 0.005}


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline not committed")
@pytest.mark.skipif(not CURRENT.exists(), reason="run scripts/benchmark.py first to produce current metrics")
def test_recall_does_not_regress():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    b_recall = baseline["operating_point"]["image_recall"]
    c_recall = current["operating_point"]["image_recall"]
    assert c_recall >= b_recall - TOLERANCE["recall_drop"], (
        f"recall regressed: baseline={b_recall:.4f} current={c_recall:.4f}")


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline not committed")
@pytest.mark.skipif(not CURRENT.exists(), reason="run scripts/benchmark.py first to produce current metrics")
def test_false_alarm_rate_does_not_regress():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    current = json.loads(CURRENT.read_text(encoding="utf-8"))
    b_fpr = baseline["operating_point"]["fpr"]
    c_fpr = current["operating_point"]["fpr"]
    assert c_fpr <= b_fpr + TOLERANCE["fpr_increase"], (
        f"false-alarm rate regressed: baseline={b_fpr:.4f} current={c_fpr:.4f}")
