# Gate results and experiment plan

Status: **Stage A only. No E0/E1 run has been launched. GATE 1 and GATE 2 have not been
evaluated.** This file will hold the measured results when they exist; it currently records the
Stage A gate, the exact E0/E1 recipe, and the GPU-hour estimate.

Rules: TEST/SEALED are never used for training or tuning. `Sample_videos` is the final held-out
check only. Numbers marked **M** are measured; **E** are estimates. No claim is made about an
unseen pilot camera.

---

## 1. Stage A gate (current)

| requirement | result | status |
|---|---|---|
| Free pool mined from P50's out-of-sample train negatives | see `docs/HARDNEG_ENGINE.md` | **M** |
| Licence-clean external images (Wikimedia + Open Images) | see source/licence table | **M** |
| >= 3,000 hard negatives | see composition table | **M / gate** |
| >= 8 causes, >= 100 per major cause | see composition table | **M / gate** |
| ~2,000 easy backgrounds | see composition table | **M** |
| Whole-source + >= 2 whole-cause holdout | Open Images V7 + `traffic_light` + `sun_glare_sky` | **M** |
| Independent negative set >= 5,000 | Open Images V7 + held-out causes | **M** |
| Leakage PASS (SHA-256 + pHash + id) | see leakage report | **M** |
| Review grids generated | `data/hard_negatives_v3/contact_sheets/` | **M** |

**Decision rule:** do not start E0/E1 until the composition table is shown and approved.
If the 3,000 target is short, the shortfall and the cause/source breakdown are reported; scaling
is a separate, costed step.

### Measured Stage A result (2026-10-29)

| requirement | measured | status |
|---|---|---:|
| hard negatives | **1,245** | **FAIL** (target 3,000) |
| hard train / hard holdout | 865 / 380 | - |
| easy backgrounds (train) | **2,797** | PASS (~2,000 target) |
| quarantine for review | 583 | review pending |
| independent negative eval set | **5,301** | PASS (>=5,000) |
| causes with >= 100 hard | dust 256, haze 133, camera artefacts 257 | **FAIL** (need >=8 causes at >=100) |
| named FP causes under 100 | warm light 18, traffic light 7, vehicle lights 26, screen 7, orange PPE 27, reflection 73 | **FAIL** |
| leakage verdict | **PASS** (166 exact + 22 pHash removed; 0 in final pool) | PASS |
| review grids | 22 cause grids + easy grid | PASS |
| whole-source holdout | Open Images V7 (209 hard, 4,373 easy, never trained) | PASS |
| whole-cause holdout | `sun_glare_sky` (67 hard) + `traffic_light` (7 hard) | PASS (traffic light denominator too small) |

**Decision: STOP before E0/E1.** Stage A fails on both the 3,000 count and the cause balance. The
pool is dominated by `camera_artifacts` (257) and `dust_plaster` (256), exactly the v1 failure
mode, while the measured user-facing FP drivers (warm light, traffic light, vehicle lights,
screens, orange PPE) are severely underfilled. Training on this pool would be underpowered and
cause-skewed. The next step is targeted collection for those causes (and review of the 583
quarantine images), not a training run.

The independent eval set is usable at 5,301 images, but the held-out `traffic_light` denominator
(7 hard) is too small to measure a cause-specific effect; it must be expanded before GATE 2.

---

## 2. Stage B/C: paired small-scale experiment

### 2.1 What is held fixed

| item | value |
|---|---|
| start weights | `yolo11s.pt` (pretrained, not P50) |
| positives | the 25% stratified list (14,747 images) |
| val / calibration | `val_early` for training, `val_cal` (2,000) for threshold/checkpoint selection |
| imgsz / batch / AMP | 640 / 64 / true |
| epochs | 30 (extend to 40 only if the learning curve is still moving) |
| patience | 10, on **recall at FPR <= 2% on val_cal**, not val loss |
| lr0 | 0.01, cosine, warmup 3 |
| seed / augmentation | identical in E0 and E1; only the train list differs |
| selection data | `val_cal` only; `val_report` is scored once after the run |
| forbidden | P50 fine-tune at lr 0.001-0.01 with early stop on val loss; TEST/SEALED tuning |

### 2.2 Arms

| arm | train list | negatives |
|---|---|---|
| **E0 control** | 14,747 positives | none |
| **E1 treatment** | 14,747 positives + Stage A negatives | 5-10% of images (8% default) |
| E2 optional | same as E1 | the other ratio (5% or 10%) |

Stage A negatives used in E1 exclude the held-out source (Open Images V7), the held-out causes
(`traffic_light`, `sun_glare_sky`) and anything quarantined. Synthetic is capped at 20% of the
negative list.

### 2.3 GPU-hour estimate (E)

Measured baseline from the previous FT_HN run: **72.1 images/s, 251.9 s/epoch** for 18,154
images on the Kaggle GPU. Extrapolating the same throughput:

| arm | images | 30 epochs | 40 epochs |
|---|---:|---:|---:|
| E0 | 14,747 | 1.7 h | 2.3 h |
| E1 @8% | ~15,930 | 1.8 h | 2.5 h |
| E2 optional | ~15,500 | 1.8 h | 2.4 h |
| evaluation/startup | - | ~0.5 h | ~0.5 h |
| **E0 + E1 total** | | **~4.0 h** | **~5.3 h** |
| **with E2** | | **~5.8 h** | **~7.7 h** |

One Kaggle GPU notebook at a time; checkpoint every epoch and resume from the last checkpoint.
These are estimates, not measurements of the new runs.

### 2.4 Evaluation protocol (identical images, paired)

1. Score `val_report` (5,119: 2,475 positive / 2,644 negative) and the independent negative set
   (>= 5,000, not used in training).
2. Recall at matched FPR **0.9% / 1.5% / 2.0% / 3.0%**; FPR at matched recall.
3. Wilson 95% CIs on every rate.
4. Paired McNemar on the identical image set for E1 vs E0.
5. Per-cause table with denominators; any cause with < 30 negatives gets **"too few to measure"**.
6. No point-estimate comparison across different image sets.

### 2.5 GATE 1

Pass only if **all** hold:

- E1 vs E0 at matched recall: FPR falls by >= 25% relative **and** McNemar p < 0.05;
- recall at fixed FPR is not significantly worse (McNemar p >= 0.05 against E1);
- the result is not driven by one cause or one source.

If GATE 1 fails: stop, report the likely reason (data quality, resolution, model size, label
noise, negative distribution) and propose the next experiment. Do not proceed to Stage D/E.

---

## 3. Stage D: generalisation gate

GATE 2 requires E1 to beat E0 on:

- the whole-source holdout (Open Images V7), and
- the whole-cause holdouts (`traffic_light`, `sun_glare_sky`),

with the same paired tests and Wilson CIs. If E1 only improves on seen causes, the fix is
**cause-specific**, not general; say so and stop.

---

## 4. Stage E: full run (only after Gates 1 and 2 and explicit approval)

- 50-100% positives, 30-40 epochs, same recipe and selection rule.
- GPU-hour estimate (E, from 72.1 img/s): 50% positives + 5-10% negatives ~31-32k images:
  **3.6-4.8 h** at 30-40 epochs. 100% positives + negatives ~61-62k images:
  **7.2-9.6 h** at 30-40 epochs. Add ~1 h evaluation/export.
- Compare to P50 on `val_report` (matched FPR, McNemar), independent negatives, bench_R DEV
  (one run), then the 11 held-out sample videos once, then **one SEALED read** logged in
  `bench_results/BENCH_USAGE.md`.
- Ship rule: at matched FPR recall >= P50, or at matched recall FPR significantly lower, and
  GATE 2 passed.

---

## 5. Defence in depth (parallel, CPU)

| item | status |
|---|---|
| Verifier retrained with >= 3,000 TP crops, class-agnostic IoU >= 0.3, 0.10-0.30 band | not done; previous verifier was NOT shipped (see `docs/VERIFIER_RESULT.md`) |
| Calibration bug (1.5%/2.0% both returned 0.91%) | acknowledged; must be fixed and unit-tested before the next verifier run |
| Video layer unit tests | existing 4/4 pass in `tests/test_video_filter.py` |
| Video layer measured on public clips | not measured in Stage A |
| Video layer on `Sample_videos` | only at the end, once, as held-out check |

---

## 6. What is measured vs estimated right now

| claim | label |
|---|---|
| Stage A source/licence/composition/leakage | **M** (see `HARDNEG_ENGINE.md`) |
| P50 free-pool false detections | **M** |
| E0/E1 effect | **not measured** |
| GPU hours | **E** |
| Pilot false alarms per camera-hour | **not measured** |
| Unseen-camera behaviour | **not claimed** |
