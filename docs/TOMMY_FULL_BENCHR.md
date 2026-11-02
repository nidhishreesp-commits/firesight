# TommyNgx vs P50 on the full 1,000-image bench_R

Same protocol for both: locked bench_R (DEV 300 + SEALED 700), thresholds calibrated only on
`val_mini + val_flames` at target FPR 2%, no tuning on bench_R. SEALED was read once per model;
both reads are logged in `bench_results/BENCH_USAGE.md`.

P50 thresholds: fire 0.13 / smoke 0.39.
TommyNgx thresholds: fire 0.67 / smoke 0.61.

Raw: `bench_results/benchR/20261101_122236_P50.json`,
`bench_results/benchR/20261102_070231_TommyNgx.json`.

## Overall (1,000; 284 positives / 716 negatives)

| metric | **P50** | TommyNgx | winner |
|---|---:|---:|---|
| accuracy | **85.50%** [83.2, 87.5] | 78.00% [75.3, 80.5] | P50 |
| precision | **88.40%** [82.9, 92.3] | 76.67% [68.3, 83.3] | P50 |
| recall | **56.34%** [50.5, 62.0] | 32.39% [27.2, 38.0] | P50 |
| specificity | **97.07%** [95.6, 98.1] | 96.09% [94.4, 97.3] | P50 |
| FPR per negative | **2.10%** | 2.80% | P50 |
| F1 | **68.82%** | 45.54% | P50 |
| MCC | **0.6255** | 0.3953 | P50 |
| image AUC | **0.8458** | 0.8351 | P50 |
| image AP | **0.7511** | 0.6673 | P50 |
| Brier | **0.1431** | 0.1414 | Tommy (marginally) |

Confusion matrix: P50 **TP 160 / FP 21 / TN 695 / FN 124**; TommyNgx **TP 92 / FP 28 / TN 688 / FN 192**.

## DEV and SEALED

| split | P50 recall | P50 FPR | Tommy recall | Tommy FPR |
|---|---:|---:|---:|---:|
| DEV 300 | **58.82%** | **2.00%** | 34.12% | 2.00% |
| SEALED 700 | **55.28%** | **2.14%** | 31.66% | 3.14% |
| all 1,000 | **56.34%** | **2.10%** | 32.39% | 2.80% |

## Per class (all 1,000)

| class | P50 precision | P50 recall | Tommy precision | Tommy recall |
|---|---:|---:|---:|---:|
| fire | **85.60%** | **60.11%** | 77.55% | 21.35% |
| smoke | **86.54%** | **47.87%** | 78.21% | 32.45% |

## Deployment policy view

| view | P50 recall | Tommy recall | P50 FPR | Tommy FPR |
|---|---:|---:|---:|---:|
| uncontrolled fire + smoke | **67.28%** | 43.21% | **2.10%** | 2.80% |
| strict uncontrolled-only | **69.87%** | 43.59% | **2.10%** | 2.80% |

## Localization

| class | P50 AP50 | Tommy AP50 | P50 AP50-95 | Tommy AP50-95 |
|---|---:|---:|---:|---:|
| fire | **0.1812** | 0.1298 | **0.0950** | 0.0581 |
| smoke | **0.3719** | 0.3334 | **0.2021** | 0.1766 |

## Verdict

**TommyNgx is clearly worse than P50 on the complete 1,000-image bench_R.** It recovers only
**92 of 284** positives (P50: 160), misses **192** positives (P50: 124), has lower precision and a
higher false-alarm rate, and its strict-policy recall is only **43.6%** vs P50 **69.9%**. Its
calibration also required much higher thresholds (0.67/0.61), and its box AP is lower.

P50 remains the better model. Neither is production-ready as an autonomous alarm on these
numbers; P50 is the only candidate for shadow/assistive evaluation.
