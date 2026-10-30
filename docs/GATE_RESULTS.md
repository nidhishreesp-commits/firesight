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

### 2.6b Execution status

- `firesight-stage-b-code` private dataset uploaded.
- **E0 kernel completed** (20 epochs, 25% list, no added negatives). Training wall time
  **8,289.5 s (2.30 h)**, measured **68.65 img/s**, best epoch 19. The evaluation cell OOM'd on a
  large image with batch 16; E0 predictions were produced locally on CPU with batch 1. E1 was
  fixed to use batch 1.
- `firesight-stage-a-negatives` and `firesight-stage-b-eval-negatives` (5,301 images, 1.527 GB)
  private datasets uploaded.
- **E1 kernel completed** (20 epochs, 25% list + 1,327 added negatives). Training wall time
  **8,846.6 s (2.46 h)**, measured **65.62 img/s**, best epoch 19. E1 evaluation also OOM'd on
  the GPU prediction step; predictions were produced locally on CPU with batch 1.
- **Known evaluation bug:** `stage_b_predict.py` OOMs on one large val_report image on the T4
  even at batch 1 (11.72 GiB allocation inside `im.half()`). It does not affect training; the
  evaluation was completed locally with CPU batch-1 inference. Fix before reusing this cell.
- **Targeted collection Phase 1 is approved but not downloaded yet.** The verifier cascade and
  E0/E1 were completed first per the approved priority. The plan/size cap is in
  `docs/TARGETED_COLLECTION_PLAN.md`; no Phase 1 image bytes have been spent.

### 2.6c E0 result and P25 reproduction (measured)

E0 selection on val_cal: **89.66% recall @ 1.94% FPR** (epoch 19). val_report:

| val_cal target | fire/smoke thresholds | val_report recall | 95% CI | val_report FPR | FP / 2,644 |
|---:|---|---:|---|---:|---:|
| 0.9% | 0.27 / 0.67 | 79.35% | [77.7, 80.9] | 1.48% | 39 |
| 1.5% | 0.31 / 0.43 | 86.59% | [85.2, 87.9] | 1.89% | 50 |
| 2.0% | 0.27 / 0.39 | 88.20% | [86.9, 89.4] | 2.23% | 59 |
| 3.0% | 0.27 / 0.25 | 92.00% | [90.9, 93.0] | 2.84% | 75 |
| 3.25% | 0.13 / 0.29 | 92.85% | [91.8, 93.8] | 3.56% | 94 |
| 4.0% | 0.13 / 0.19 | 94.59% | [93.6, 95.4] | 4.24% | 112 |

P25 reference: **92.53% recall @ 3.25% FPR** on val_report. E0 sits between 92.00%/2.84% and
92.85%/3.56% around that operating point; interpolating to 3.25% FPR gives about **93.0%**
recall. The difference from P25 is within calibration-transfer noise (CIs overlap), so E0
**reproduces P25**, with a small operating-point shift explained by: 20 epochs instead of 15,
custom recall-at-FPR checkpoint selection instead of the old schedule, and recalibrating the
fire/smoke thresholds on val_cal. No material regression is established.

### 2.6d E1 result and GATE 1 (measured)

E1 selection on val_cal: **91.93% recall @ 1.84% FPR** (epoch 19) vs E0 **89.66% @ 1.94%**
(+2.27 pp recall at slightly lower FPR).

Gate 1 is evaluated at **matched recall**: E0's 1.5%-target operating point, recall **84.85%**
on val_report, applied as the same recall target to E1. Thresholds are image-level global
confidence calibrated on val_cal only; identical images are paired.

| set / metric | E0 | E1 | relative change | McNemar |
|---|---:|---:|---:|---|
| val_report FPR at 84.85% recall | 1.551% [1.15, 2.10] | **0.908% [0.61, 1.35]** | **-41.5%** | p=**0.0046** |
| independent-negative FPR at 84.85% recall | 0.491% [0.33, 0.72] | **0.208% [0.12, 0.37]** | **-57.7%** | p=**0.0026** |
| val_report recall | 84.85% | 84.85% | 0 | matched |
| independent n | 5,301 | 5,301 | - | - |

**GATE 1: PASS.** Both required sets show a >=25% relative FPR reduction at matched recall with
p<0.05 and no recall loss. The effect is not borderline; no second seed is needed for Gate 1.

At a matched-FPR 2.0% target, E1 also has higher val_report recall (90.34% vs 88.12%) with lower
val_report FPR (1.85% vs 2.08%); on the independent negatives E1 FPR is 0.811% vs 0.679% (not a
win). The matched-recall result is the Gate 1 evidence.

Per-cause independent FPR at matched recall (denominators; `too_few` = n<30):

| cause | n | E0 FPR | E1 FPR | too few |
|---|---:|---:|---:|---:|
| camera_artifacts | 1615 | 0.37% | **0.06%** | no |
| dust_plaster | 941 | 0.32% | **0.11%** | no |
| exhaust | 527 | 0.19% | 0.19% | no |
| sun_glare_sky (held out) | 323 | 0.62% | 0.62% | no |
| skin_person (held out) | 303 | 0.33% | 0.33% | no |
| stage_chandelier | 278 | 0.00% | 0.36% | no |
| orange_vegetation | 259 | 0.00% | 0.00% | no |
| reflection_lens_flare | 167 | 1.20% | **0.00%** | no |
| cave_warm_rock | 124 | 0.81% | **0.00%** | no |
| snow_white_overexposed | 121 | 1.65% | **0.00%** | no |
| ordinary_outdoor | 98 | 1.02% | 1.02% | no |
| traffic_light (held out) | 92 | 2.17% | **0.00%** | no |
| vehicle_lights | 73 | 1.37% | 1.37% | no |
| smoke_steam_nonfire | 67 | 1.49% | 1.49% | no |
| orange_ppe | 60 | 0.00% | 0.00% | no |
| haze_cloud_fog_mist | 54 | 3.70% | **0.00%** | no |
| ordinary_indoor | 52 | 0.00% | 0.00% | no |
| neon_lights | 51 | 0.00% | 0.00% | no |
| sunset_sunrise | 49 | 2.04% | 2.04% | no |
| screen_fire | 31 | 0.00% | 0.00% | no |
| warm_artificial_light | 16 | 0.00% | 0.00% | **yes** |

**GATE 2: NOT PASSED for the named held-out causes.** The overall independent set improves, but
`sun_glare_sky` (323; 67 hard) and `skin_person` (303; 26 hard) are unchanged at the matched-recall
point. `traffic_light` (92; only 7 hard) is **too few to measure** and is not used as a gate.
So the fix is **partly general** (overall independent FPR falls significantly) but **not
demonstrated as general on the named held-out causes**; do not claim cause-general improvement.

E1/E0 artifacts: `docs/stage_b/e1/`, `docs/stage_b/gate1.json`.

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

## 3.1 Verifier fallback (CPU, measured — v4 final)

The v3 verifier was **invalidated by leakage**: 149 of its training negatives were also in the
independent evaluation set. Those 149 source images were removed and the verifier was retrained
as **v4** (4,942 source images; 5,736 TP / 2,703 FP; CV AUC 0.9738). Leakage is now **PASS**
against val, TEST, bench_R DEV, independent negatives and 104 Sample_videos frames.

| model / set | recall | FPR | McNemar vs P50 calibrated |
|---|---:|---:|---|
| P50 calibrated, val_report | 92.97% | 3.14% | reference |
| v4 cascade @1.5% val_cal, val_report | **97.58%** | **1.85%** | pos p=3.4e-30, neg p=4.5e-5 |
| v4 cascade @2.0% val_cal, val_report | **97.70%** | **2.34%** | pos p=5.6e-33, neg p=0.014 |
| P50 calibrated, bench_R DEV (89/211) | 60.67% | 2.84% | reference |
| v4 cascade, bench_R DEV | 69.66% | 4.27% | improves recall, worsens FPR |
| P50 calibrated, independent 5,301 | - | 0.585% | reference |
| v4 cascade, independent 5,301 | - | **0.528%** | p=0.771 (not significant) |

Held-out independent negatives: `sun_glare_sky` 2.79% -> **1.86%** (n=323, 67 hard),
`skin_person` 0.33% -> **0.00%** (n=303, 26 hard), `traffic_light` 1.09% -> **0.00%**
(n=92, 7 hard; **too few hard images for a gate**). Per-cause table:
`docs/stage_b/verifier_v4_cascade.json`.

One authorized Sample_videos pass (2 fire / 9 nonfire): P50 calibrated gave 1 nonfire layer alarm
over 2.9 min; v4 cascade gave **0**, with 2/2 fire clips detected after onset (same first alarms
45.67 s and 10.2 s). Video coverage is too short to call a camera-hour rate.

Latency on CPU: P50 82.9 ms/frame + verifier 174.8 ms/frame = **257.7 ms/frame**; RSS peak
1.39 GB. This is a fallback, not a production latency claim.

Pilot-candidate configuration (not shipped): **P50@0.10 -> verifier v4 (v=0.011936, calibrated
on val_cal) -> temporal N-of-M -> video layer.** Still unmeasured: camera-hours, night/IR, rain,
cross-site generalisation, reviewer workload, and the full-run detector retrain.

## 3.2 Quarantine review (measured)

One compact sheet contains only the 226 images with P50 conf >= 0.50 or CLIP fire/smoke >= 0.50:
`data/hard_negatives_v3/contact_sheets/quarantine_review.jpg` and
`quarantine_review.csv`. The other 357 quarantine rows remain excluded and are not in the sheet.

---

## 4. Stage E: full run (BLOCKED: Gate 2 not passed)

**Status: STOP before Stage E.** Gate 1 passed (matched-recall FPR reduction on both val_report
and overall independent negatives). Gate 2 did **not** pass on the named held-out causes
(`sun_glare_sky`, `skin_person` unchanged; `traffic_light` too few). Stage E is not started.

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
| Verifier retrained with >= 3,000 TP crops, class-agnostic IoU >= 0.3, 0.10-0.30 band | **DONE (M): v4 5,736 TP / 2,703 FP, CV AUC 0.9738; 149 leaked negatives removed** |
| Calibration bug (1.5%/2.0% both returned 0.91%) | **FIXED (M): full threshold sweep; v4 1.5% and 2.0% select distinct points** |
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
| E0/E1 effect | **M**: E1 selection 91.93%/1.84% vs E0 89.66%/1.94% on val_cal |
| E0/E1 GPU hours | **M**: E0 2.30 h, E1 2.46 h = 4.76 h wall (not the 2.37 h estimate) |
| Verifier fallback effect | **M** on val_report (see 3.1) |
| Gate 1 | **M: PASS** (matched recall 84.85%; FPR -41.5% report p=0.0046, -57.7% independent p=0.0026) |
| Gate 2 | **M: NOT PASSED** on named held-out causes (sun_glare/skin unchanged; traffic too few) |
| Pilot false alarms per camera-hour | **not measured** |
| Unseen-camera behaviour | **not claimed** |
