# Gate results and experiment plan

Status: **Stage A accepted for E0/E1 (with the 3,000 target corrected to the full run). E0/E1
are fully prepared and CPU-smoke-tested; no GPU run has been launched. GATE 1 and GATE 2 have
not been evaluated.** This file records the Stage A gate, the exact E0/E1 recipe/config diff,
composition, leakage, GPU estimate and the verifier fallback.

Rules: TEST/SEALED are never used for training or tuning. `Sample_videos` is the final held-out
check only. Numbers marked **M** are measured; **E** are estimates. No claim is made about an
unseen pilot camera.

---

## 1. Stage A gate (current)

| requirement | result | status |
|---|---|---|
| Free pool mined from P50's out-of-sample train negatives | see `docs/HARDNEG_ENGINE.md` | **M** |
| Licence-clean external images (Wikimedia + Open Images) | see source/licence table | **M** |
| >= 3,000 hard negatives (full run) | 1,245; E0/E1 need only 740-1,500 added | **M / full-run gap** |
| >= 8 causes, >= 100 per major cause (full run) | 3 causes >=100 | **M / full-run gap** |
| ~2,000 easy backgrounds | see composition table | **M** |
| Whole-source + >= 2 whole-cause holdout | Open Images V7 + `traffic_light` + `sun_glare_sky` | **M** |
| Independent negative set >= 5,000 | Open Images V7 + held-out causes | **M** |
| Leakage PASS (SHA-256 + pHash + id) | see leakage report | **M** |
| Review grids generated | `data/hard_negatives_v3/contact_sheets/` | **M** |

**Decision rule:** E0/E1 may start once this config diff, composition, leakage output and CPU
smoke test are approved. The 3,000-hard target is a full-run target, not an E0/E1 gate; the
full-run shortfall is handled by the separate targeted collection plan.

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

**Decision (corrected 2026-10-29): Stage A is ACCEPTED for E0/E1.** The 3,000-hard target applies
to the full 100% run, not to the 25% experiment. For E0/E1, 5-10% of the 14,747-image list is
740-1,500 added negatives; the prepared E1 uses 1,327 added negatives (865 hard + 462 easy,
9.0%). The pool is therefore sufficient for the paired experiment.

The **full-run** target is still unmet (1,245 < 3,000) and still cause-skewed (dust/camera
artefacts dominate; warm light, traffic light, vehicle lights, screens and orange PPE are under
100). Targeted collection for the full run is planned separately in
`docs/TARGETED_COLLECTION_PLAN.md` and is kept OUT of E1.

The independent eval set is 5,301 images. GATE 2 will report only held-out causes with enough
images (`sun_glare_sky` 67 hard, `skin_person` 26 hard, others as available); `traffic_light`
(7 hard) is flagged "too few to measure" and is not used as a gate.

---

## 2. Stage B/C: E0 vs E1 (prepared, not launched)

### 2.1 Measured correction to the base list

The "25% positive list" is **14,747 images total**, not 14,747 positives:

| base list | images | positive | dataset negative | negative share |
|---|---:|---:|---:|---:|
| `build/kaggle_dataset/lists/train_25.txt` | 14,747 | 6,532 | 8,215 | 55.7% |

E0 uses this list as-is, so it already contains the dataset's own negatives. The 5-10% rule is
therefore applied to the **added** hard/easy negatives: 1,327 = 9.0% of 14,747. If the intent was
8-10% *total* negatives, E0/E1 must be rebuilt from positives only; that is a different
experiment and is flagged, not silently changed.

### 2.2 Exact config diff

| item | E0 | E1 |
|---|---|---|
| start weights | `yolo11s.pt` fresh | `yolo11s.pt` fresh |
| train list | `train_25` (14,747) | `train_25` + 1,327 added (16,074) |
| added hard | 0 | 865 |
| added easy | 0 | 462 |
| epochs | 20 | 20 |
| patience | 10 | 10 |
| selection | max image recall at FPR<=2% on `val_cal` | same |
| train val | `val_early` | same |
| imgsz / batch | 640 / 64 | same |
| lr0 / lrf / schedule | 0.01 / 0.01 / cosine, warmup 3 | same |
| seed / deterministic | 0 / true | same |
| AMP | true on GPU | same |
| augmentation | mosaic 1.0, close_mosaic 10, mixup 0, copy_paste 0, translate 0.1, scale 0.5, fliplr 0.5, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, erasing 0.4 | same |
| difference | - | train list only |

Checkpoint selection: every epoch is scored on `val_cal`; the best `weights/best_recall.pt` is
kept by image-level recall at FPR<=2%, never by val loss. Patience 10 is applied to that metric.

### 2.3 E1 composition (measured)

| | count |
|---|---:|
| hard additions | 865 |
| easy additions | 462 |
| added total | 1,327 (9.0% of base) |
| E1 images | 16,074 |
| E1 total negatives | 9,542 (59.4%) |

Hard additions by cause (top): dust 206, camera artefacts 163, haze 117, reflection 50,
sunset 39, ordinary outdoor 38, stage/chandelier 37, orange vegetation 32, cave 26, orange PPE 24,
neon 21, skin 20, exhaust 20, snow 19, vehicle lights 18, warm light 14. Easy additions are
stratified across the same causes. Quarantine, `sun_glare_sky`, `traffic_light`, Open Images V7,
`Sample_videos` and synthetic are excluded from E1. The full table is
`build/stage_b/composition.csv`.

### 2.4 E0/E1 leakage (measured)

`build/stage_b/leakage_report.json`: **PASS**.

- 1,327 added negatives checked; 0 exact SHA-256 hits, 0 pHash<=2 hits, 0 id/path overlaps,
  0 duplicates, 0 quarantine, 0 held-out causes, 0 Open Images, 0 `Sample_videos`.
- Base package leakage report: **PASS** (bench_R / testbench100 / TEST).
- `eval_neg.txt` (5,301) is disjoint from `train.txt`; no synthetic, no Open Images training
  rows, no held-out-cause training rows.

### 2.5 GPU-hour estimate (E)

Measured pilot throughput: **72.1 images/s, 251.9 s/epoch** for 18,154 images (FT_HN run).

| arm | images | 20 epochs |
|---|---:|---:|
| E0 | 14,747 | **1.14 h** |
| E1 | 16,074 | **1.24 h** |
| **E0 + E1** | | **2.37 h** |
| +25% overhead | | **2.97 h** |

One Kaggle GPU notebook at a time; checkpoint every epoch and resume. These are estimates; the
E0/E1 runs themselves are not measured yet.

### 2.6 CPU smoke test (measured)

`kaggle/stage_b_train.py` was run on 16 train / 20 val / 20 val_cal images, CPU, 1 epoch, 320 px.
It trained, ran the recall-at-FPR selection callback, and saved `weights/best_recall.pt`
(tiny-set recall 0.30 at FPR 0.00). The full pipeline is exercised; this is not an accuracy result.

### 2.7 Evaluation protocol (identical images, paired)

1. Score `val_report` (5,119: 2,475 positive / 2,644 negative) and the independent negative set
   (5,301: Open Images V7 + held-out causes + low-confidence quarantine). The 5,301-image package
   is built at `build/kaggle_stage_b_eval_negatives/` (1.527 GB).
2. Recall at matched FPR **0.9% / 1.5% / 2.0% / 3.0%**; FPR at matched recall.
3. Wilson 95% CIs on every rate.
4. Paired McNemar on the identical image set for E1 vs E0.
5. Per-cause table with denominators; any cause with < 30 negatives/positives gets
   **"too few to measure"**.
6. No point-estimate comparison across different image sets.

### 2.8 GATE 1 (revised)

Pass only if **all** hold:

- E1 vs E0 at matched recall: FPR falls by >= 25% relative **and** McNemar p < 0.05 on
  **val_report**;
- the same matched-recall FPR reduction is significant on the **independent held-out negatives**;
- recall at fixed FPR is not significantly worse (McNemar p >= 0.05 against E1).

If it passes only on `val_report`, label it **in-distribution** and do not call it general.
If GATE 1 fails: stop, report the likely reason (data quality, resolution, model size, label
noise, negative distribution) and propose the next experiment.

---

## 3. Stage D: generalisation gate

GATE 2 requires E1 to beat E0 on the held-out causes that have enough images:

| held-out cause | hard images | usable for GATE 2? |
|---|---:|---|
| `sun_glare_sky` | 67 | yes |
| `skin_person` (held-out by split) | 26 | borderline; report CI |
| `traffic_light` | 7 | **too few to measure; not a gate** |
| other held-out rows | see composition | report only if >=30 |

Use the same paired tests and Wilson CIs. If E1 only improves on seen causes, the fix is
**cause-specific**, not general; say so and stop.

---

## 3.1 Verifier fallback (CPU, measured)

The verifier was retrained with **5,736 TP / 3,028 FP crops** using class-agnostic matching at
IoU>=0.3, and the calibration search was fixed to sweep every P50 threshold and every verifier
score (the old 101-quantile grid made 1.5% and 2.0% collapse to the same point).

| model | val_report recall | val_report FPR | notes |
|---|---:|---:|---|
| P50 calibrated | 0.9297 | 3.14% | baseline |
| verifier v3 @1.5% target | **0.9766** | **1.74%** | McNemar pos p=1.1e-31, neg p=1.7e-5 |
| verifier v3 @2.0% target | **0.9770** | **2.19%** | McNemar pos p=5.6e-33, neg p=0.0035 |

This is a **fallback**, not the primary fix; it was not shipped into the detector path. The model
is `build/verifier/model_v3.joblib`, features `features_train_v3.npz`, eval `eval_v3_mixed.json`.
The val_cal features were rebuilt with the fixed CLIP API and matched the old feature results to
within 1 FP, which is the compatibility check for the reused val_report features.

## 3.2 Quarantine review (measured)

One compact sheet contains only the 226 images with P50 conf >= 0.50 or CLIP fire/smoke >= 0.50:
`data/hard_negatives_v3/contact_sheets/quarantine_review.jpg` and
`quarantine_review.csv`. The other 357 quarantine rows remain excluded and are not in the sheet.

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
| Verifier retrained with >= 3,000 TP crops, class-agnostic IoU >= 0.3, 0.10-0.30 band | **DONE (M): 5,736 TP / 3,028 FP, CV AUC 0.977** |
| Calibration bug (1.5%/2.0% both returned 0.91%) | **FIXED (M): full threshold sweep; 1.5% and 2.0% now select distinct points** |
| Video layer unit tests | existing 4/4 pass in `tests/test_video_filter.py` |
| Video layer measured on public clips | not measured in Stage A |
| Video layer on `Sample_videos` | only at the end, once, as held-out check |
| Targeted collection for the full-run underfilled causes | **planned, NOT downloaded**; see `docs/TARGETED_COLLECTION_PLAN.md` |

---

## 6. What is measured vs estimated right now

| claim | label |
|---|---|
| Stage A source/licence/composition/leakage | **M** (see `HARDNEG_ENGINE.md`) |
| P50 free-pool false detections | **M** |
| E0/E1 effect | **not measured** (prepared + CPU smoke-tested) |
| E0/E1 GPU hours | **E**: 2.37 h raw, 2.97 h with 25% overhead |
| Verifier fallback effect | **M** on val_report (see 3.1) |
| Pilot false alarms per camera-hour | **not measured** |
| Unseen-camera behaviour | **not claimed** |
