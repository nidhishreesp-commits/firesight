from pathlib import Path

import numpy as np
import pytest

from firesight.config import PROJECT_ROOT
from firesight.types import Box, Detection

VERIFIER = PROJECT_ROOT / "models/verifier.onnx"


@pytest.mark.skipif(not VERIFIER.exists(), reason="verifier.onnx not built")
def test_verifier_scores_crops():
    from firesight.verifier import CropVerifier
    v = CropVerifier(VERIFIER, classes=("fire", "lookalike", "smoke"), threshold=0.5)
    frame = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype(np.uint8)
    dets = [Detection("fire", 0.8, Box(20, 20, 120, 120))]
    kept, rows = v.verify(frame, dets)
    assert len(rows) == 1
    assert set(rows[0]).issuperset({"fire", "lookalike", "smoke"})
