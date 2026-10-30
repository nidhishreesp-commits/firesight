# External fire/smoke models: download and DEV check

Status: **two of three repos are not usable as supplied.** No SEALED read. `bench_R` is
1,000 images = 300 DEV + 700 SEALED; I used **DEV only** so the SEALED exam is not spent on
four-model selection.

## Models

| repo | licence | status | class order |
|---|---|---|---|
| `TommyNgx/YOLOv10-Fire-and-Smoke-Detection` | Apache-2.0 (card) | **gated (HTTP 401 without HF token)** | unknown |
| `aryantyagiase/yolov8-fire-smoke-detection` | **no licence** | **corrupt/truncated**: `best.pt` size mismatch (expected 9,437,184, got 10,485,760); `best.onnx` invalid protobuf | unknown |
| `rabahdev/fire-smoke-yolov8n` | AGPL-3.0 | loads; 3.01 M params | `0=smoke, 1=fire` |
| **P50 (ours)** | internal | reference | `0=fire, 1=smoke` |

The aryantyagiase repo should be re-uploaded or a valid file supplied. The TommyNgx repo
requires accepting the gate and an `HF_TOKEN`; do not paste a token in chat.

## DEV result (measured)

Fair operating point: one image-level max-confidence threshold calibrated on **val_cal** at
FPR <= 2%, then applied unchanged to DEV.

| model | val_cal recall @2% FPR | DEV recall | DEV FPR | DEV n |
|---|---:|---:|---:|---:|
| **P50 (ours)** | **92.35%** | **60.00%** [49.4, 69.8] | 2.79% [1.29, 5.95] | 85 pos / 215 neg |
| rabahdev yolov8n | 71.35% | 42.35% [32.4, 53.0] | 1.40% [0.48, 4.02] | 85 / 215 |

DEV ROC (descriptive; thresholds fitted on DEV, not used for selection):

| model | AUC | AP | recall @1.5% FPR | @2.0% | @3.0% |
|---|---:|---:|---:|---:|---:|
| **P50** | 0.8526 | **0.7896** | **48.24%** | **50.59%** | **62.35%** |
| rabahdev yolov8n | **0.8663** | 0.7861 | 44.71% | 45.88% | 55.29% |

**Conclusion:** rabahdev is **not better** than P50. It is slightly higher AUC but lower AP and
lower recall at every matched DEV FPR; its val_cal recall at 2% FPR is 20.9 pp lower. The two
remaining models cannot be judged until the gated access and corrupt files are fixed.

## What would make the 1,000-image comparison valid

Spending the 700-image SEALED read on all four models would destroy the one-SEALED-read rule.
Options:

1. Use DEV 300 for all models (done) and give **one** SEALED read to the winner.
2. Explicitly authorize a one-time exploratory SEALED read for all four, accepting that the
   SEALED set then cannot be used as a clean final exam.

No SEALED image has been read in this work.
