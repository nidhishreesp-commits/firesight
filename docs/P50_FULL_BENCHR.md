# P50 on the full 1,000-image bench_R — production review

Run: **2026-11-01T12:22Z**, P50 (`build/kaggle_weights/runs/P50/weights/best.onnx`),
all **1,000** locked bench_R images (DEV 300 + SEALED 700), thresholds calibrated only on
`val_mini + val_flames` at target FPR 2%: **fire 0.13 / smoke 0.39**. SEALED was read **once**;
no tuning after the read. Logged in `bench_results/BENCH_USAGE.md`.

Raw: `bench_results/benchR/20261101_122236_P50.json`,
`docs/benchR/P50_full_review.json`.

## 1. Headline (all 1,000; 284 positives / 716 negatives)

| metric | value | 95% CI |
|---|---:|---|
| accuracy | **85.50%** | [83.2, 87.5] |
| precision | **88.40%** | [82.9, 92.3] |
| recall / sensitivity | **56.34%** | [50.5, 62.0] |
| specificity | **97.07%** | [95.6, 98.1] |
| FPR per negative | **2.10%** | - |
| F1 | **68.82%** | - |
| balanced accuracy | **76.70%** | - |
| MCC | **0.6255** | - |
| image-level AUC | **0.8458** | - |
| image-level AP | **0.7511** | - |
| Brier score | **0.1431** | - |

Confusion matrix: **TP 160 / FP 21 / TN 695 / FN 124**.

DEV: accuracy 86.33%, precision 89.29%, recall 58.82%, FPR 2.00%, MCC 0.648, AUC 0.8526.
SEALED: accuracy 85.14%, precision 88.00%, recall 55.28%, FPR 2.14%, MCC 0.6158, AUC 0.8429.
DEV and SEALED are consistent; the full-set number is not hiding a SEALED collapse.

## 2. Per class

| class | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| fire | 107 | 18 | 71 | 85.60% | **60.11%** | 70.63% |
| smoke | 90 | 14 | 98 | 86.54% | **47.87%** | 61.64% |

Smoke is the weak class: more than half of smoke-bearing images are missed at the 2% FPR point.

## 3. Policy view (deployment-relevant)

| view | n | recall | specificity | accuracy | suppressed |
|---|---:|---:|---:|---:|---|
| uncontrolled fire + smoke | 878 | **67.28%** | 97.07% | 91.57% | 122 controlled/spark images |
| strict uncontrolled-only | 872 | **69.87%** | 97.07% | 92.20% | 128 non-uncontrolled |

Even under the strict product policy, **~30% of qualifying events are missed**. The high accuracy
is inflated by the 716 negatives (71.6% of the set); balanced accuracy is 76.7%, MCC 0.626.

## 4. Per flame type

| type | n | detected | recall | 95% CI |
|---|---:|---:|---:|---|
| uncontrolled | 156 | 109 | 69.87% | [62.3, 76.5] |
| controlled | 85 | 49 | 57.65% | [47.0, 67.6] |
| smoke-only | 106 | 48 | 45.28% | [36.1, 54.8] |
| spark-benign | 37 | 2 | 5.41% | [1.5, 17.7] |
| unknown | 6 | 0 | 0% | - |

## 5. Lookalike / family behaviour

| family | n | positives | precision | recall | FPR |
|---|---:|---:|---:|---:|---:|
| uncontrolled_fire | 200 | 101 | 98.53% | 66.34% | 1.01% |
| controlled_fire | 80 | 49 | 100% | 71.43% | 0% |
| both (fire+smoke) | 80 | 54 | 90.70% | 72.22% | 15.38% |
| smoke_only | 100 | 28 | 83.33% | 35.71% | 2.78% |
| spark_screen | 40 | 37 | 100% | 5.41% | 0% |
| clouds | 30 | 0 | - | - | **10.00%** |
| fog_haze_mist | 30 | 0 | - | - | **10.00%** |
| vehicle_lights_streetlights | 30 | 0 | - | - | **6.67%** |
| snow | 20 | 0 | - | - | 5.00% |
| chandelier_stage_lights | 25 | 0 | - | - | 4.00% |
| dust_construction_plaster | 25 | 0 | - | - | 4.00% |
| flowers_autumn_leaves | 25 | 0 | - | - | 4.00% |
| orange_ppe_vests_cones | 30 | 0 | - | - | 3.33% |
| ordinary_indoor | 38 | 1 | 0% | 0% | 2.70% |
| camera_artifacts | 40 | 0 | - | - | 0% |
| red_orange_neon_lights | 30 | 0 | - | - | 0% |
| reflections_lens_flare_glare | 25 | 0 | - | - | 0% |
| sunset_orange_sky | 30 | 0 | - | - | 0% |
| steam | 25 | 2 | - | 0% | 0% |

The dominant residual false alarms are **clouds, fog/haze, vehicle/street lights**, not dust.

## 6. Localization is weak

| class | AP50 | AP75 | AP50-95 | recall at IoU50 | mean IoU (matched) |
|---|---:|---:|---:|---:|---:|
| fire | 0.1812 | 0.1011 | 0.0950 | 15.37% | 0.751 |
| smoke | 0.3719 | 0.1777 | 0.2021 | 24.83% | 0.770 |

Image-level alarms are far better than box localization. Do not use P50 boxes as precise evidence
for area/clipping without human confirmation.

## 7. Threshold sweep (global confidence; class-specific point is above)

| threshold | accuracy | recall | FPR |
|---:|---:|---:|---:|
| 0.10 | 86.2% | 63.7% | 3.5% |
| 0.20 | 86.0% | 59.5% | 2.5% |
| 0.25 | 85.7% | 57.4% | 2.2% |
| 0.30 | 85.8% | 56.7% | 1.9% |
| 0.50 | 83.8% | 47.2% | 1.2% |

## 8. Verdict: is P50 production-ready?

**No, not as an autonomous fire/smoke alarm at this operating point.** At a 2.1% false-alarm rate
it misses **43.7% of positive images** (124/284), including **52.1% of smoke-bearing images**.
Even the strict uncontrolled-fire policy misses ~30%. That is not acceptable for life-safety
without a human in the loop.

It is usable as a **shadow/assistive candidate** only: precision 88.4% and FPR 2.1% are good, but
the recall and box localization are not. A production decision needs an explicit recall target.
If the requirement is e.g. recall >=90%, P50 fails clearly. If the requirement is "reviewer
triages alerts", it can be piloted in shadow mode.

The upgrade path remains E1 (Gate 1 passed, Gate 2 not passed) plus the verifier cascade;
neither is shipped.
