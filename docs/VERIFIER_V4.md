# Verifier v3 (CPU fallback): retrain + calibration fix

Status: **fallback experiment, not shipped into the detector path.** It runs after the detector and
only re-scores detections; it does not change P50. The model is `build/verifier/model_v3.joblib`.

## 1. Why v1 was rejected

The v1 verifier used only 327 TP crops (class-strict, IoU>=0.5), so at low FPR it rejected true
positives and lost recall (`docs/VERIFIER_RESULT.md`). The calibration search also collapsed 1.5%
and 2.0% onto the same operating point (both reported 0.91%).

## 2. v3 training data (measured)

| item | value |
|---|---:|
| out-of-sample positive images | 4,000 |
| free-pool false-detection negative images | 1,091 |
| P50 inference conf | 0.05 |
| detections after top-2/positive filter | 9,163 |
| class-agnostic matching | IoU >= 0.3 |
| **TP crops** | **5,736** |
| **FP crops** | **3,028** |
| groups (source images) | 4,969 |
| grouped CV AUC (MLP) | **0.9771** |
| grouped CV AUC (LR) | 0.9631 |

The 0.10-0.30 confidence band is included because P50 inference starts at 0.05 and every
positive's top-2 detections at >=0.10 are kept. Ambiguous IoU 0.1-0.3 detections are skipped, not
used as positives or negatives.

## 3. Calibration fix (measured)

`scripts/verifier_eval.py` now sweeps **every P50 confidence** and **every verifier score**,
separately for each target FPR. The old 101-quantile grid was too coarse and returned the same
point for 1.5% and 2.0%.

| target | P50 threshold | verifier threshold | val_cal FPR | val_cal recall | val_report recall | val_report FPR |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5% | 0.1003 | 0.00211 | 1.06% | 97.41% | **97.66%** | **1.74%** |
| 2.0% | 0.1003 | 0.00063 | 1.74% | 97.52% | **97.70%** | **2.19%** |

P50 calibrated baseline on the same val_report: recall **92.97%**, FPR **3.14%**.

McNemar on identical val_report images, v3 @1.5% vs P50 calibrated:

- positives: cascade-only 119 vs P50-only 3, p = **1.1e-31**
- negatives: cascade-only 18 vs P50-only 55, p = **1.7e-5**

At 2.0%: positives p = 5.6e-33, negatives p = 0.0035.

## 4. Caveats

- The val_report features were built before the CLIP API fix. The val_cal features were rebuilt
  with the fixed API; the v3 evaluation on rebuilt val_cal + reused val_report features matches
  the all-old-feature evaluation to within 1 FP, which is the compatibility check. A full
  val_report rebuild was attempted but exceeded the CPU shell window; it is not claimed as done.
- DEV is one small run: recall 0.685, FPR 3.79% (89 positives / 211 negatives); CIs are wide.
- This is not a replacement for the detector retrain and is not part of the E0/E1 gate.

## 5. Reproduction

```powershell
# 4,000 out-of-sample positives + 1,091 free-pool false negatives
python scripts\verifier_infer.py --sample build\verifier\sample_train_v2.json `
  --out build\verifier\detections_train_v2.jsonl --conf 0.05 --batch 16
# keep top-2 detections per positive, all negative detections
python scripts\verifier_features.py --detections build\verifier\detections_train_v2_filtered.jsonl `
  --sample build\verifier\sample_train_v2.json --out build\verifier\features_train_v3.npz `
  --class-agnostic --iou-tp 0.3 --max-fp-ratio 2
python scripts\verifier_train.py --features build\verifier\features_train_v3.npz `
  --out build\verifier\model_v3.joblib
python scripts\verifier_eval.py --model build\verifier\model_v3.joblib `
  --targets 0.015,0.02 --out build\verifier\eval_v3.json
```
