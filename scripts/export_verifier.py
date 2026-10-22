from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from firesight.config import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="data/verifier/best.pt")
    parser.add_argument("--out", default="models/verifier.onnx")
    parser.add_argument("--classes", type=int, default=3)
    args = parser.parse_args()
    from torchvision.models import mobilenet_v3_small
    model = mobilenet_v3_small(weights=None)
    import torch.nn as nn
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, args.classes)
    state = torch.load(PROJECT_ROOT / args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 128, 128), str(out),
                      input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    print(out, out.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
