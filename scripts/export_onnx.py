from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firesight.config import PROJECT_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a YOLO/RT-DETR .pt to ONNX or TensorRT and write a versioned manifest.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine"])
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, 0 (TensorRT needs CUDA)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    args = parser.parse_args()
    from ultralytics import YOLO
    weights = PROJECT_ROOT / args.weights if not Path(args.weights).is_absolute() else Path(args.weights)
    out_dir = PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    exported = model.export(format=args.format, imgsz=args.imgsz, opset=args.opset, dynamic=args.dynamic,
                            half=args.half, int8=args.int8, simplify=True, device=args.device)
    exported_path = Path(exported)
    target = out_dir / (Path(args.out).name if args.out else exported_path.name)
    if exported_path.resolve() != target.resolve():
        target.write_bytes(exported_path.read_bytes())
    manifest = {
        "source_pt": str(weights), "artifact": str(target.relative_to(PROJECT_ROOT)),
        "format": args.format, "sha256": sha256(target), "imgsz": args.imgsz, "opset": args.opset,
        "device": args.device, "dynamic": args.dynamic, "half": args.half, "int8": args.int8,
        "classes": model.names,
    }
    (out_dir / (target.stem + ".manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
