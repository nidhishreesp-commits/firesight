# Cascade + resolution study, and the mining source/licence table

Date: 2026-10-28. Model frozen: P50. No retraining.

## 1. Split provenance (verifier training data)

`train_25 ⊂ train_50 ⊂ train_100` verified on the packaged lists:

| list | images |
|---|---:|
| train_25 (P25 saw) | 14,747 |
| train_50 (P50 saw) | 29,493 |
| train_100 (pool) | 58,985 |
| **out-of-sample for P50** (100 - 50) | **29,492** (overlap with train_50 = 0) |

The verifier will be trained on P50 detections from these **29,492 unseen images** plus the
mined pools. Never val_report, bench_R, testbench100, SEALED or TEST.

## 2. Detector cascade (val_report; 2,475 pos / 2,644 neg)

| operating point | recall | 95% CI | FPR/neg | 95% CI | McNemar vs calibrated |
|---|---:|---|---:|---|---|
| P50 calibrated (0.29/0.33) | 0.9297 | [0.919, 0.939] | 3.14% | [2.54, 3.87] | - |
| P50 @ 0.15 | **0.9693** | [0.962, 0.975] | 5.18% | [4.40, 6.09] | 98 vs 0, p=6e-30 |
| P50 @ 0.10 | **0.9786** | [0.972, 0.984] | 6.66% | [5.77, 7.67] | 121 vs 0, p=8e-37 |

Lowering to 0.10 recovers exactly **121 of the 174** missed positives (53 remain), at the cost of
**+3.5 pp FPR**. That is the cascade slot the verifier must fill: keep the 121, reject the added
false alarms.

## 3. Resolution / tiling on the remaining 53 misses

Controls: 200 positives that P50 already detects, 300 random negatives. Same thresholds (0.29/0.33).

| mode | throughput (img/s) | misses recovered (of 53) | control positives | control negatives flagged (FPR) |
|---|---:|---:|---:|---:|
| 640 (baseline) | 6.67 | 0 | 200/200 (100%) | 8/300 (2.67%) |
| 960 | 2.71 | **5 (9.4%)** | 197/200 (98.5%) | 16/300 (5.33%) |
| 1280 | 1.47 | **14 (26.4%)** | 188/200 (94.0%) | 29/300 (9.67%) |
| tiled 2x2 @640 | 2.42 | 8 (15.1%) | 169/200 (84.5%) | 25/300 (8.33%) |

Vantage breakdown of the 53 misses (caption-based: aerial/high vs ground/CCTV):

| mode | aerial/high (n=13) | ground/CCTV (n=40) |
|---|---:|---:|
| 640 | 0% | 0% |
| 960 | 0% | 12.5% |
| 1280 | **46.2%** | 20.0% |
| tiled | 23.1% | 12.5% |

**Reading:** 1280 is the only mode that meaningfully recovers distant/aerial smoke, but it costs
3.4x the false-alarm rate and is 4.5x slower; tiled inference loses control positives (large
flames cut by tiles). Recommendation: keep **640 + lowered threshold + verifier** as the main
cascade, and consider **1280 as a second pass only on aerial/high cameras** (where it recovers
46% of the misses) if the deployment has such cameras.

## 4. Mining source / licence table (approved set)

| source | licence | planned volume | download | status |
|---|---|---|---|---|
| Wikimedia Commons API | CC0 / PD / CC BY / CC BY-SA (per image, attribution stored) | ~1,400 candidates | per-image, no single archive | approved, running |
| Own non-fire video (`Sample_videos`, nonfire clips only) | proprietary internal, local only | ~1,000 frames | none | approved, running |
| Open Images V7 (validation subset) | **CC BY 4.0** (dataset-wide) | 2,000 images (~400 MB) | < 2 GB | approved, running |
| COCO val2017 | **no per-image licence field** in the annotations | - | - | **excluded**: cannot filter on a licence field as required |

## 5. Wikimedia caveat (bench_R contamination)

bench_R's negatives were themselves sourced from Wikimedia, and the earlier leakage gate removed
**47** exact/near duplicates. For the new pool we additionally exclude, where the Commons metadata
allows, files from the **same uploader and same Commons categories** as the bench_R items; the
remaining risk is reported rather than hidden. Verifier results will be reported separately on:

1. **bench_R DEV** (sanity check only, 300 images),
2. the **independent >=5,000-image negative set** (the verdict),
3. **own non-fire video** (false alarms per camera-day).
