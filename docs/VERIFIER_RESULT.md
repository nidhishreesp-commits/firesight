# Verifier cascade result (2026-10-28)

**Decision: NOT SHIPPED.** The cascade reduces false alarms but loses more recall than P50 at
the same operating point; the McNemar test on positives is strongly against it (p ~ 1e-25).
P50 remains frozen and unchanged.

## Headline table (val_report: 2,475 pos / 2,644 neg; Wilson 95% CIs)

| system | recall | 95% CI | FPR/neg | 95% CI | accuracy | McNemar vs P50-cal (pos / neg) |
|---|---:|---|---:|---|---:|---|
| P50 calibrated (0.29/0.33) | **0.9297** | [0.919, 0.939] | 3.14% | [2.54, 3.87] | 0.9498 | reference |
| P50 @ 0.10 (no verifier) | **0.9786** | [0.972, 0.984] | 6.66% | [5.77, 7.67] | - | 121 vs 0 (p=8e-37) |
| **Cascade @1.5% target** (P50 0.10 + MLP v=0.0018) | 0.8481 | [0.833, 0.862] | **0.91%** | [0.61, 1.35] | 0.9219 | 92 vs 294 (p=1.0e-25) / 15 vs 74 (p=1.5e-10) |
| **Cascade @2.0% target** | 0.8481 | [0.833, 0.862] | 0.91% | [0.61, 1.35] | 0.9219 | same point |
| Cascade, LR variant @2.0% | 0.8533 | [0.839, 0.867] | 1.55% | [1.15, 2.10] | 0.9211 | 79 vs 268 (p=3.5e-25) |
| bench_R DEV (cascade, sanity) | 0.5393 | [0.436, 0.639] | 4.27% | [2.26, 7.91] | - | one DEV run only |

Verifier latency: **0.004-0.010 ms per detection** (MLP), negligible; feature extraction (CLIP)
dominates and runs once per detection.

## Per-cause false alarms removed (val_report negatives, cascade @1.5%)

Counts of negative images whose top detection fell in each cause (P50 FAs -> cascade FAs):

| cause | P50 FAs | cascade FAs | removed | images in bucket | verdict |
|---|---:|---:|---:|---:|---|
| construction dust | 19 | 0 | 100% | 19 | too few to measure |
| haze / cloud | 28 | 9 | 68% | 28 | too few to measure |
| sun / bright sky | 29 | 4 | 86% | 29 | too few to measure |
| smoke / steam | 13 | 4 | 69% | 13 | too few to measure |
| reflection | 15 | 2 | 87% | 15 | too few to measure |
| cave / rock | 11 | 0 | 100% | 11 | too few to measure |
| skin | 10 | 0 | 100% | 10 | too few to measure |
| screen | 9 | 0 | 100% | 9 | too few to measure |
| orange PPE | 8 | 0 | 100% | 8 | too few to measure |
| warm light | 7 | 1 | 86% | 7 | too few to measure |
| sunset | 6 | 1 | 83% | 6 | too few to measure |
| other | 21 | 3 | 86% | 21 | too few to measure |

**Honest caveat on this table:** the denominators are the number of *negative images that
produced a detection* in that cause, so "FPR before" is 1.0 by construction. A true per-cause
FPR needs every negative image classified by cause (not only the ones that alarmed); with 6-29
images per cause, **every cause is below the 30-image bar and is reported as "too few to measure"**.
The removal counts are valid; the per-cause FPR is not.

## What was measured vs estimated

**Measured:** the val_report/val_cal numbers, the McNemar tests, DEV (one run), latency,
training CV AUC (MLP 0.9845, LR 0.9806, 5-fold grouped by source image), per-cause removal counts.
**Not measured:** per-cause FPR with proper denominators; dust on held-out construction videos;
video-layer false alarms per camera-hour; the >=5,000-image independent negative set.

## Why it failed

1. **Too few positive crops.** After IoU>=0.5 matching, the out-of-sample run produced only
   **327 TP** (vs 654 FP). The verifier is trained on a thin, imbalanced slice.
2. **The low-FPR operating point is brutal.** At 1.5% FPR the MLP threshold rejects ~15% of true
   positives; the aggregate AUC (0.98) hides this.
3. **TP/FP matching is class-strict and localization-strict**, so many correct alarms became
   "unknown" and were dropped from training.
4. The cascade therefore trades recall for FPR: 93.0% -> 84.8% recall for 3.14% -> 0.91% FPR.

## What would fix it (not done, no retraining of P50)

- More TP crops: relax matching to any-GT IoU>=0.5 (class-agnostic) and include near-threshold
  positives from P10/P25 checkpoints; target >=3,000 TP.
- Per-cause balancing of the training set (currently dominated by dust/reflection/skin).
- A small fine-tuned CNN on crops (the user's item 4 option) instead of frozen CLIP + MLP.
- Re-evaluate at 1.5% and 2.0% with paired McNemar; ship only if recall >= P50-calibrated.

## Video layer (implemented, not yet measured)

`src/firesight/video_filter.py` implements:
- **flicker/growth requirement** (static tracks without area growth or confidence flicker are
  down-weighted and routed to review),
- **static-scene memory**,
- **per-camera suppression learned from reviewer dismissals** (region IoU + colour-histogram
  appearance, configurable TTL and thresholds), logged to JSONL.
Unit tests: `tests/test_video_filter.py` (4/4 pass). **No camera-hour measurement yet** — the
non-fire video run was not completed in this session; this is reported as unmeasured, not zero.

## Reproduction

```bash
python scripts/make_verifier_samples.py
python scripts/verifier_infer.py --sample build/verifier/sample_train.json --out build/verifier/detections_train.jsonl
python scripts/verifier_features.py --detections build/verifier/detections_train.jsonl --sample build/verifier/sample_train.json --out build/verifier/features_train.npz
python scripts/verifier_train.py --features build/verifier/features_train.npz --out build/verifier/model.joblib
python scripts/verifier_infer.py --sample build/verifier/sample_val_cal.json --out build/verifier/det_val_cal.jsonl
python scripts/verifier_infer.py --sample build/verifier/sample_dev.json --out build/verifier/det_dev.jsonl
python scripts/verifier_features.py --detections build/verifier/det_val_cal.jsonl --sample build/verifier/sample_val_cal.json --out build/verifier/features_val_cal.npz --keep-all
python scripts/verifier_features.py --detections build/verifier/det_val_report.jsonl --sample build/verifier/sample_val_report.json --out build/verifier/features_val_report.npz --keep-all
python scripts/verifier_features.py --detections build/verifier/det_dev.jsonl --sample build/verifier/sample_dev.json --out build/verifier/features_dev.npz --keep-all
python scripts/verifier_eval.py --model build/verifier/model.joblib --targets 0.015,0.02
```

## Data provenance / leakage

- Training detections come from **29,492 out-of-sample images** (train_100 minus train_50; verified
  25 subset of 50 subset of 100, overlap 0). 3,552 positives + 4,000 negatives were processed.
- Exclusions checked: pHash vs train_50 (29,493), bench_R (1,000), testbench100 (100), TEST (8,113).
- val_report was **not** used for training or thresholds; thresholds were calibrated on val_cal.
- bench_R DEV: one run only; SEALED/TEST untouched.
