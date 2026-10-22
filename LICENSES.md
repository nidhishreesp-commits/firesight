# Licenses and attribution

This file records the license of every dataset and model used. **Do not commit
restricted weights or data to a public repository.** Files under `data/raw/`,
`models/*.onnx`, `models/*.pt`, and `models/*.safetensors` are git-ignored.

## Datasets

| Dataset | Source | License | Use here | Redistribution |
|---|---|---|---|---|
| D-Fire | `gaia-solutions-on-demand/DFireDataset`; Kaggle mirror `sayedgamal99/smoke-fire-detection-yolo` | Kaggle mirror CC0; upstream repo "Other" with per-source terms | Training + test | Images may carry per-source terms (HPWREN, FESB, FireNet). Cite Venâncio et al. 2022 and upstream providers. |
| FASDD (FASDD_CV) | Science Data Bank `10.57760/sciencedb.j00104.00103`; HF mirror `weslleyskah/fasdd_cv` | CC BY 4.0 (HF mirror) | Training + test | Attribution required: Wang, M., Yue, P., Jiang, L., Yu, D., Tuo, T., Li, J. (2025), Geo-spatial Information Science 28(2). |
| MS-FSDB (arXiv 2410.16631) | `XiaoyiHan6/MS-FSDB` | Code MIT; dataset download link (Google Drive) was **not retrievable** without a Google account at build time | Not used | See `REPORT.md` limitations. |
| Local seed (`fire_dataset`, `fire_hardneg`, `fire-assets*`) | Internal / SmartSight | Proprietary | Training + hard negatives + video | Internal only. |
| Simuletic CCTV Fire/Smoke Emergency Dataset | HF `Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset` | **CC BY-NC 4.0 (non-commercial)** | External evaluation only | Not redistributed. Used only to measure generalization. |
| Wikimedia Commons hard negatives (14+2 images) | Commons API, `flame_positives/sources/wikimedia/` | CC0 / Public Domain / CC BY / CC BY-SA per image | testbench100 fire-adjacent negatives | Per-image author, licence and source URL in `attribution.jsonl` and `testbench100/index.json`. |
| FireViewer fire-smoke-detection-corpus-v1 | HF `fireviewer/fire-smoke-detection-corpus-v1` | "other" (rights matrix in repo) | Not downloaded | Reference only. |

## Models

| Model | Source | License | Notes |
|---|---|---|---|
| `best_fire_v2.pt` | ss-prod (`iSight-python-licenseclean`) | Proprietary (SmartSight) | YOLOv12n, 2.57M params, trained on an internal `/data/merged` dataset. |
| `best-fire-v3/model.onnx` | `smartsight-models` | Proprietary (SmartSight) | YOLOv12n hard-negative retrain; **deployed as default** in this repo (`models/best_fire_v3.onnx`). |
| `best-fire-v4/model.onnx` | `smartsight-models` | Proprietary (SmartSight) | YOLOv12n class-balanced retrain. |
| `fire.pt` | ss-prod | Proprietary (SmartSight) | YOLO11s, inactive in ss-prod. |
| FireViewer RT-DETRv2-R50 | HF `fireviewer/rtdetr-v2-r50-fire-smoke` | FireViewer Model Use License v1.0: commercial use **with attribution**, internal conversion/fine-tune allowed, **redistribution and shipping weights are NOT permitted** | Evaluation only. Weights are not committed. |
| Grounding DINO Tiny | HF `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | Zero-shot baseline only. |
| `models/verifier.onnx` | Trained here | Internal | MobileNetV3-Small crop classifier trained on TRAIN-split crops + mined train hard negatives. |

## Third-party software

Ultralytics (AGPL-3.0 by default — **check your Ultralytics license for commercial
deployment**), PyTorch (BSD-3), ONNX Runtime (MIT), OpenCV (Apache-2.0),
FastAPI (MIT), React (MIT), Tailwind CSS (MIT), ByteTrack implementation via
Ultralytics.

| Pyronear pyro-sdis | HF pyronear/pyro-sdis | Apache-2.0 | Phase B smoke positives / validation | 12,378 images, 1.32 GB; **not used in bench_R** to avoid Pyronear-model training overlap. |
| bench_R candidates | Wikimedia Commons API | CC0 / PD / CC BY / CC BY-SA per image | bench_R reference images | Per-image title/author/URL in ench_R/index.json; images are local and git-ignored. |
