# Models

Weights are git-ignored. Provision them locally:

| File | How |
|---|---|
| `models/best_fire_v3.onnx` | Copied from `../smartsight-models/best-fire-v3/model.onnx` (ss-prod reference). |
| `models/verifier.onnx` + `models/verifier.onnx.data` | Built by `python scripts/train_verifier.py` then `python scripts/export_verifier.py`. |
| `runs/train/<name>/weights/best.pt` | Built by `python scripts/train_yolo.py`. |
| FireViewer RT-DETRv2 | `python scripts/download_data.py --fireviewer --accept-restricted` (non-redistributable). |
| Grounding DINO Tiny | Downloaded automatically by `transformers` on first use (Apache-2.0). |

`registry.yaml` is the benchmark registry. `default.yaml` points the API at
`models/best_fire_v3.onnx`.

Version/rollback: `scripts/retrain_pipeline.py` writes `models/versions/<ts>/`
and `models/active.json`; keep both for rollback.
