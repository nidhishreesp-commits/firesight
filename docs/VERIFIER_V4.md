# Verifier v4 (CPU fallback): retrain, leakage fix, calibration fix

Status: **fallback experiment, not shipped.** It re-scores P50 detections; it does not change the
detector. Model: `build/verifier/model_v4.joblib`. The earlier v3 model is **invalid**: the
leakage check found 149 of its training negatives were also in the independent evaluation set.

## 1. Leakage catch and fix (measured)

`scripts/verifier_leakage_check.py` compares verifier source images against:

- val (val_cal + val_report): 8,252
- TEST: 8,113
- bench_R DEV: 300
- independent negatives: 5,535 pool rows
- Sample_videos: 104 sampled frames (1 every 5 s, 11 clips)

v3 training (5,091 source images) overlapped **149 independent negatives** (exact SHA and
pHash 0). v3 was discarded. v4 removes those 149 source images from the training sample
(4,000 positives + 942 negatives = **4,942 source images**) and re-runs the feature builder.

**v4 leakage verdict: PASS.**

| set | n | exact SHA | pHash<=2 | id overlap | scene overlap |
|---|---:|---:|---:|---:|---:|
| val | 8,252 | 0 | 0 | 0 | 0 |
| TEST | 8,113 | 0 | 0 | 0 | 0 |
| bench_R DEV | 300 | 0 | 0 | 0 | 0 |
| independent negatives | 5,535 | 0 | 0 | 0 | 0 |
| Sample_videos frames | 104 | 0 | 0 | 0 | 0 |

## 2. v4 training data (measured)

| item | value |
|---|---:|
| source images | 4,942 |
| P50 detections at conf 0.05 | 8,839 |
| TP (class-agnostic, IoU>=0.3) | **5,736** |
| FP (IoU<0.1) | **2,703** |
| grouped CV AUC (MLP) | **0.9738** |
| grouped CV AUC (LR) | 0.9612 |

The 0.10-0.30 confidence band is included (top-2 positive detections at >=0.10). Ambiguous
IoU 0.1-0.3 detections are skipped.

## 3. Calibration (val_cal only)

The old 101-quantile search was replaced by a full sweep of every P50 confidence and every
verifier score. v4 thresholds:

| target | P50 threshold | verifier threshold | val_cal recall | val_cal FPR |
|---:|---:|---:|---:|---:|
| 1.5% | 0.10267 | 0.011936 | 97.31% | 1.16% |
| 2.0% | 0.10034 | 0.004953 | 97.52% | 1.94% |

## 4. Results (measured)

### val_report (2,475 pos / 2,644 neg)

| system | recall | 95% CI | FPR | 95% CI | McNemar vs P50 calibrated |
|---|---:|---|---:|---|---|
| P50 calibrated (0.29/0.33) | 92.97% | [91.9, 93.9] | 3.14% | [2.54, 3.87] | reference |
| cascade v4 @1.5% | **97.58%** | [96.9, 98.1] | **1.85%** | [1.40, 2.44] | pos p=3.4e-30, neg p=4.5e-5 |
| cascade v4 @2.0% | **97.70%** | [97.0, 98.2] | **2.34%** | [1.83, 2.99] | pos p=5.6e-33, neg p=0.014 |

### bench_R DEV (89 pos / 211 neg, one run)

| system | recall | 95% CI | FPR | 95% CI |
|---|---:|---|---:|---|
| P50 calibrated | 60.67% | [50.3, 70.2] | 2.84% | [1.31, 6.06] |
| cascade v4 @1.5% | **69.66%** | [59.5, 78.2] | 4.27% | [2.26, 7.91] |
| P50 at matched FPR (2.37%) | 50.56% | [40.4, 60.7] | 2.37% | [1.02, 5.43] |

The cascade improves DEV recall but **raises DEV FPR**; it is not a clean DEV win. The DEV set is
small (89 positives), so CIs are wide.

### independent negatives (5,301 images; all negative)

| system | FPR | 95% CI |
|---|---:|---|
| P50 calibrated | 0.585% | [0.41, 0.83] |
| P50@0.10 | 3.622% | [3.15, 4.16] |
| cascade v4 | **0.528%** | [0.37, 0.76] |

The cascade vs P50 calibrated on independent negatives: 22 cascade-only vs 25 P50-only,
McNemar p=0.771 — **not significant**. On val_report the same comparison is significant in both
directions; the independent set remains the harder test.

### held-out causes (independent negatives)

| cause | n | P50 FPR | cascade FPR | hard holdout images | flag |
|---|---:|---:|---:|---:|---|
| sun_glare_sky | 323 | 2.79% | 1.86% | 67 | measurable |
| skin_person | 303 | 0.33% | 0.00% | 26 | measurable |
| traffic_light | 92 | 1.09% | 0.00% | 7 | **too few hard images for a gate** |

Per-cause table (all causes) is in `docs/stage_b/verifier_v4_cascade.json`. Causes that got worse
under the cascade: vehicle_lights (0.00% -> 1.37%), orange_ppe (0.00% -> 1.67%), neon_lights
(0.00% -> 1.96%), smoke_steam_nonfire (1.49% -> 2.99%), cave_warm_rock (0.00% -> 0.81%),
ordinary_outdoor (0.00% -> 1.02%). These are small denominators and need review before claiming a
per-cause improvement.

## 5. One authorized Sample_videos pass (measured)

P50 calibrated + temporal + video layer vs P50@0.10 + v4 + temporal + video layer, same frames:

| arm | nonfire layer alarms | rate (per camera-min) | 95% CI | fire clips detected | first layer alarm |
|---|---:|---:|---|---:|---|
| P50 calibrated | 1 | 0.345/min | [0.06, 0.79] | 2/2 | 45.67 s, 10.2 s |
| cascade v4 | **0** | **0.000/min** | [0.00, 0.56] | 2/2 | 45.67 s, 10.2 s |

Nonfire coverage is only 2.9 minutes; these rates are **not** a camera-hour estimate. Fire onsets
were 40.5 s and 6.0 s; both clips alarmed after onset. Clips: 2 fire, 9 nonfire, 11 total.

## 6. Latency and memory (measured, CPU i7-1165G7)

30 frames with detections from `fire_2.mp4`:

| component | mean | median |
|---|---:|---:|
| P50 ONNX | 82.9 ms/frame | 75.9 ms |
| verifier v4 (CLIP box+context+full) | 174.8 ms/frame | 206.2 ms |
| cascade total | **257.7 ms/frame** | - |

RSS: 495 MB after P50, **1.14 GB** after CLIP+verifier, **1.39 GB** peak. Model load 2.8 s for
CLIP/verifier. This is slower than the real-time requirement on CPU; it is a fallback, not a
production latency claim.

## 7. Pilot-candidate configuration (report only, not shipped)

A candidate configuration exists:

**P50 detector @ conf 0.10 → verifier v4 (v=0.011936, calibrated on val_cal) → temporal N-of-M →
video false-alarm layer.**

Measured on the 11 sample videos: **2/2 fire clips detected after onset; 0 alarms on 2.9 minutes
of nonfire video** (vs 1 P50 alarm). On val_report: 97.58% recall / 1.85% FPR at the 1.5% val_cal
point. On bench_R DEV: 69.66% / 4.27% (mixed vs P50 60.67% / 2.84%). On independent negatives:
0.528% vs 0.585% (not significant).

**Still unmeasured:** camera-hours, night/IR, rain, cross-site generalisation, real reviewer
workload, video-layer event-level rates over days, and the full-run detector retrain. Nothing is
shipped.

## 8. Reproduction

```powershell
python scripts\verifier_leakage_check.py --sample build\verifier\sample_train_v2_clean.json
python scripts\verifier_features.py --detections build\verifier\detections_train_v2_filtered_clean.jsonl `
  --sample build\verifier\sample_train_v2_clean.json --out build\verifier\features_train_v4.npz `
  --class-agnostic --iou-tp 0.3 --max-fp-ratio 2
python scripts\verifier_train.py --features build\verifier\features_train_v4.npz --out build\verifier\model_v4.joblib
python scripts\verifier_eval.py --model build\verifier\model_v4.joblib --dir build\verifier\v3_eval `
  --targets 0.015,0.02 --out build\verifier\eval_v4.json
python scripts\verifier_cascade_eval.py --verifier build\verifier\model_v4.joblib `
  --verifier-v 0.011936064809560776 --out build\verifier\cascade_eval_v4.json
python scripts\day_video_cascade_eval.py --verifier build\verifier\model_v4.joblib `
  --verifier-v 0.011936064809560776
```
