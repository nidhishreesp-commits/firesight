# DEV miss analysis (no SEALED)

Input: bench_R **DEV only** (300 images; 85 positives, 215 negatives). Baseline P50 thresholds
from `val_mini+val_flames` at target FPR 2%: fire 0.13 / smoke 0.39. No SEALED image or SEALED
error was used.

Raw: `build/miss_analysis/DEV/missed_positives.csv`, `summary.json`.
Contact sheets: `build/miss_analysis/DEV/contact_sheets/`.

## 1. Misses at the accepted baseline

| item | count |
|---|---:|
| DEV positives | 85 |
| detected at baseline | 50 (58.8%) |
| **missed** | **35** |
| missed fire-only | 14 |
| missed smoke-only | 20 |
| missed both | 1 |

By flame type: **uncontrolled 15**, controlled 11, spark-benign 8, unknown 1.
By family: smoke-only 10, spark/screen 8, uncontrolled-fire 7, controlled-fire 4, steam 2,
exhaust 2, both 1, ordinary-indoor 1.

Primary cause tags (heuristic, then checked on the sheets):

| cause | missed | what it is |
|---|---:|---|
| **thin_smoke** | **21** | thin/white/distant smoke plume, often low contrast |
| low_light | 7 | night, welding sparks, dark scenes |
| small_distant | 4 | tiny flame/box, distant subject, indoor small box |
| unusual_scene | 3 | controlled/spark/historical/nonstandard scene |

The dominant failure is **thin smoke**, not boxes on large flames. Several "smoke" misses are
steam/food/historical images (`Steamed_Rice`, `Lamb_chops_cooking`, ship exhaust) and are
**label-noise suspects**; review them before using them as training positives.

## 2. Recovery at lower global thresholds (DEV)

Global image-level max confidence (same threshold for fire and smoke):

| threshold | recall | 95% CI | FPR | 95% CI | TP | FP |
|---:|---:|---|---:|---|---:|---:|
| 0.20 | 62.35% | [51.7, 71.9] | 2.79% | [1.29, 5.95] | 53 | 6 |
| 0.15 | 65.88% | [55.3, 75.1] | 3.26% | [1.59, 6.57] | 56 | 7 |
| 0.10 | 67.06% | [56.5, 76.1] | 4.65% | [2.55, 8.35] | 57 | 10 |

Lowering the threshold recovers only ~7-8 of the 35 misses and roughly doubles FPR. It is not a
free fix.

## 3. Verifier v4 cascade on DEV

Cascade = P50@0.10 + verifier v4, threshold chosen on val_cal (1.5% and 2.0% points). This is
**not** matched to the 2% DEV FPR; it is reported as a recall-recovery option.

| val_cal point | DEV recall | 95% CI | DEV FPR | 95% CI | baseline misses recovered |
|---|---:|---|---:|---|---:|
| 1.5% | 70.59% | [60.2, 79.2] | 5.12% | [2.88, 8.93] | 11/35 |
| 2.0% | 71.76% | [61.4, 80.2] | 5.12% | [2.88, 8.93] | 11/35 |

The verifier recovers ~11 missed positives but at ~5.1% DEV FPR, i.e. it buys recall with false
alarms. It is not a clean production gate without re-calibration on a larger validation set.

## 4. Contact sheets

- `missed_uncontrolled_fire.jpg` — all missed positives whose family/flame type is uncontrolled
- `missed_smoke.jpg` — all missed positives with smoke
- `missed_all.jpg` — all 35
- `cause_thin_smoke.jpg`, `cause_low_light.jpg`, `cause_small_distant.jpg`,
  `cause_unusual_scene.jpg`

Green boxes are ground truth; red boxes are P50. The sheets show the thin-smoke/distant pattern
clearly. The "steam/food" tiles are label-noise suspects, not detector failures.

## 5. Aerial/high-mounted subset

Keyword proxy (aerial/drone/wildfire/landscape/mountain/forest-fire/view) on DEV found **31
images: 4 positives / 27 negatives**. Only 4 positives is too few to measure a resolution effect;
resolution results are in `docs/RESOLUTION_DEV.md` and should be treated as inconclusive.
