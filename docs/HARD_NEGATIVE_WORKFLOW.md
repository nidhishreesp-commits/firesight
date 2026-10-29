# Hard-negative programme: method, workflow, and outcome

Status: **v1 complete, did not improve on P50.** This document records exactly how the
hard-negative work was selected, built, trained and evaluated, so the next attempt starts
from evidence instead of repeating the same mistakes.

---

## 1. Why hard negatives at all

The P50 detector (yolo11s, 29,493 images, 15 epochs) was the best model from the Phase 3b
learning-curve pilot:

| model | val recall @FPR<=2% | val FPR/neg | mAP50 fire/smoke | bench_R DEV recall (FPR) |
|---|---:|---:|---|---|
| best_fire_v3 (old) | 0.8012 | 2.34% | 0.522 / 0.616 | 40.0% (3.72%) |
| **P50** | **0.9297** | 3.14% | 0.633 / 0.679 | **56.5%** (2.33%) |

Two things stood out:

1. Recall saturates after 25% of the data (25% -> 50% = +0.7 pp), so **more of the same data
   is not the lever**.
2. The detector still raises **false alarms**, and the bench gap (93% on val vs 56% on the
   locked DEV set) is a *composition* gap: the bench contains controlled flames, sparks,
   smoke-only and hard lookalikes that D-Fire/FASDD barely contain.

Hard negatives target exactly that: teach the model what fire/smoke is *not*, using the
scenes it actually confuses.

---

## 2. Step 1 - measure the false positives (no guessing)

**Goal:** know which lookalikes cause the alarms before collecting anything.

**Method**

1. Take the saved P50 predictions on the `val.jsonl` report slice (5,119 images, 2,644 negatives).
2. Select negatives where the alert-level decision is positive -> **83 false positives**.
3. Add the **5 false positives** from bench_R DEV (300 images, 215 negatives) -> **88 total**.
4. Caption every FP with **Florence-2-base** (cached locally, CPU) using
   `<MORE_DETAILED_CAPTION>`.
5. Assign each caption to a category; visually inspect 8 to validate the captions.
6. Raw data: `build/fp_taxonomy/fps_categorized.csv`.

**Result (88 FPs)**

| category | count | share |
|---|---:|---:|
| other (roads, buildings, tunnels, misc) | 15 | 17% |
| warm artificial light (street lamps, lanterns, chandeliers, city lights) | 14 | 16% |
| sunset / orange dusk sky | 12 | 14% |
| haze / cloud / fog / overcast | 11 | 12% |
| cave interior / red-orange rock | 7 | 8% |
| person / skin tones | 7 | 8% |
| water / wet-road reflection | 5 | 6% |
| screen (game UI, poster, illustration of fire) | 5 | 6% |
| dust / construction dust | 3 | 3% |
| orange PPE / workwear / cones | 3 | 3% |
| smoke / steam plume (chimney, cooling tower) | 2 | 2% |
| orange flowers / autumn foliage | 2 | 2% |
| bright sky / sun / eclipse | 1 | 1% |
| suspected label error (real fire labelled negative) | 1 | 1% |

**Key finding:** dust/plaster - the category everyone assumes - is only **3/88**. The real
drivers are **warm light, sunset and haze**, plus screens, reflections and orange PPE.

---

## 3. Step 2 - design of the hard-negative pool

Existing tooling (`scripts/collect_hard_negatives.py`, `docs/HARD_NEGATIVE_COLLECTION.md`)
was extended with the three families the taxonomy found missing: `screen_fire`,
`cave_warm_rock`, `person_portrait`.

**Sources, in priority order**

1. **Own video captures** (`Sample_videos`) - best domain match, no rights risk; only
   `label == "nonfire"` clips are used, so fire clips are never mined as negatives.
2. **Train-negative mining** - existing D-Fire/FASDD negatives already in the training split;
   free, licensed, no download.
3. **Wikimedia Commons API** - only CC0 / public domain / CC BY / CC BY-SA; NC/ND rejected;
   per-image attribution recorded.

**QA signals per candidate**

| signal | purpose |
|---|---|
| CLIP zero-shot (`laion/CLIP-ViT-B-32`) | is this actually the requested category? (triage, not truth) |
| Detector hardness (`best_fire_v3.onnx`, max fire/smoke confidence) | prefer *hard* negatives (the ones the model already scores highly) |
| Quarantine at `max_conf >= 0.80` + CLIP fire/smoke >= 0.25 | possible real fire/smoke; never trained without human review |
| SHA-256 + dHash (Hamming <= 4) dedupe | no duplicates inside the pool |
| 10% deterministic holdout | never trained; used only for checks |
| Licence filter | every image has a recorded licence and attribution |

**Output layout:** `data/hard_negatives_v2/{images,labels,manifest.jsonl,attribution.jsonl,contact_sheets,lists,SUMMARY.md}`.
Negatives are background images with **empty YOLO label files** - no boxes to draw.

---

## 4. Step 3 - pilot collection and the leakage gate

**Pilot run:** 14 categories, 20 selected/category, 60 candidates/category, Wikimedia + own
video.

| stage | count |
|---|---:|
| candidates downloaded/scored | 1,108 |
| selected by CLIP+quota+dedupe | 276 |
| of which "hard" (max_conf >= 0.10) | 84 |
| quarantined as possible positives | 0 |
| **removed by the new leakage gate** | **47** |
| **final clean pool** | **229** (205 train / 24 holdout) |

**The leakage gate was the most valuable outcome.** A new check (`scripts/check_hardneg_leakage.py`)
compares 64-bit pHash against bench_R (1,000), testbench100 (100) and TEST (8,113).
It found **47 near-duplicates (Hamming <= 2), all against bench_R** - because the bench_R
negatives were themselves sourced from Wikimedia Commons. Categories affected: sunset 13,
neon 10, chandelier 7, clouds 5, exhaust 5, dust 4, vehicle lights 2, PPE 1. Those 47 were
marked `rejected_leakage`; the re-run reported **PASS** (0 near-duplicates, 0 path/id overlap).

Licence mix of the 276: CC BY-SA 4.0 (120), CC0 (29), public domain (24), CC BY-SA 3.0 (21),
CC BY 4.0 (20), CC BY 2.0 (19), and smaller CC variants; 20 own-video frames (proprietary internal).

**Contact-sheet review:** sunsets were excellent; `screen_fire` was mostly *generic* screens
(TVs, monitors) rather than screens showing fire - usable as screen negatives, but the query
needs tightening.

---

## 5. Step 4 - assembling the training pool

229 new negatives were only **1.8%** of the 14,747-image positive set (`build/ft_pilot`), far
too little to change anything. So the **free, already-licensed** train-negative mining was run:

| pool | count |
|---|---:|
| mined negatives (D-Fire + FASDD train) | 4,000 |
| of which hard (max_conf >= 0.10) | 306 |
| of which very hard (>= 0.30) | 103 |
| licences | CC BY 4.0: 3,596 / CC0-1.0: 404 |

Mined pool by category (top): dust/plaster 882, camera artifacts 563, red/orange neon 423,
orange PPE 288, ordinary 252, sunset 209, person 199, chandelier 179, flowers 158, clouds 137,
reflections 119, vehicle lights 82, exhaust 75, fog 69, cave 62, steam 47, screen 19.

**Dataset builder** (`scripts/build_finetune_dataset.py`):

- positives: the 25% stratified list (14,747 images);
- negatives: capped at 2,000, ordered **tier-1 first** (clouds, fog, dust, reflections, sunset),
  with tier-1 oversampled **x2** (3,407 negative lines);
- **18.8% negatives** of 18,154 training images;
- absolute paths, labels resolved by `/images/` -> `/labels/`; no copying.

Two provenance bugs were found and fixed on the way:
- the collector **overwrote** `manifest.jsonl` on each run (now merges);
- the mined pool included `local_negative` images excluded from the Kaggle package
  (now filtered to D-Fire/FASDD).

---

## 6. Step 5 - training recipe

| item | value | why |
|---|---|---|
| start weights | **P50 best.pt** | best pilot checkpoint; avoid retraining from scratch |
| data | 14,747 positives + 2,000 hard negatives (18.8%) | negatives as a minority signal |
| epochs | 15, patience 5 | fixed budget, early stop on val loss |
| imgsz / batch / AMP | 640 / 64 / true | same as the pilot |
| **lr0** | **0.001** | the earlier fine-tune (B) used the default 0.01 and degraded; low LR was the hypothesis |
| throughput | 72.1 img/s, 251.9 s/epoch | measured |
| outcome | **early stop after 6 epochs** | val loss stopped improving almost immediately |

Infrastructure needed for this run (all committed): `--data-yaml` and `--lr0` support in
`pilot_train.py`, a separate private code dataset so code fixes do not require a 3 GB
re-upload, and a hard-negative list (`negatives.jsonl`) shipped with the code.

---

## 7. Step 6 - evaluation protocol

1. Thresholds calibrated on `val_cal` (2,000 images) at **1.5%** and **2.0%** FPR/negative.
2. Scored on the disjoint `val_report` slice (5,119 images) - **Wilson 95% CIs**.
3. Confirmed once on **bench_R DEV** (300 images; 85 pos / 215 neg), thresholds from
   `val_mini + val_flames`.
4. **SEALED untouched.** TEST untouched. Every DEV use logged in `bench_results/BENCH_USAGE.md`.

---

## 8. Results

### val.jsonl report slice (thresholds calibrated at 2.0% on val_cal)

| model | accuracy | recall | 95% CI | FPR/neg | mAP50 fire/smoke |
|---|---:|---:|---|---:|---|
| best_fire_v3 | 0.8918 | 0.8012 | [0.785, 0.816] | 2.34% | 0.522 / 0.616 |
| P10 | 0.9037 | 0.8283 | [0.813, 0.843] | 2.57% | 0.544 / 0.577 |
| P25 | 0.9471 | 0.9253 | [0.914, 0.935] | 3.25% | 0.579 / 0.643 |
| **P50** | **0.9498** | **0.9297** | [0.919, 0.939] | 3.14% | 0.633 / 0.679 |
| FT25 (B, no negatives) | 0.9232 | 0.8727 | [0.859, 0.885] | 2.95% | 0.603 / 0.623 |
| **FT_HN (hard negatives)** | 0.9232 | **0.8703** | [0.856, 0.883] | **2.72%** | 0.598 / 0.610 |
| YOLO26-fire (HF) | 0.8625 | 0.7515 | [0.734, 0.768] | 3.37% | 0.374 / 0.316 |

FT_HN at the 1.5% calibration target: recall 0.8485 [0.834, 0.862], FPR 2.27%.

### bench_R DEV (300 images; 85 pos / 215 neg)

| model | accuracy | recall | 95% CI | FPR/neg |
|---|---:|---:|---|---:|
| best_fire_v3 | 0.8033 | 0.4000 | [0.302, 0.506] | 3.72% |
| **P50** | **0.8600** | **0.5647** | [0.459, 0.665] | **2.33%** |
| FT25 (B) | 0.8467 | 0.5412 | [0.436, 0.643] | 3.26% |
| **FT_HN** | 0.8333 | **0.4941** | [0.390, 0.598] | 3.26% |
| YOLO26-fire (HF) | 0.8400 | 0.5529 | [0.447, 0.654] | 4.65% |

### Verdict

Hard negatives **lowered the false-alarm rate slightly** (3.14% -> 2.72% on val) but **cost
recall**: -5.9 pp on val and **-7.1 pp on the locked bench**. P50 remains the best model.

---

## 9. Why it did not work (analysis)

1. **Early stopping fired at epoch 6.** Fine-tuning a converged model makes val loss rise
   almost immediately, so the best checkpoint is an early, half-adapted one. Patience on
   *val loss* is the wrong criterion here; patience should be on **mAP50-95 / recall at FPR**.
2. **The negative distribution did not match the measured FPs.** The mined pool is dominated
   by dust (882) and camera artifacts (563), while the real FPs were warm light (14), sunset
   (12) and haze (11). We trained hardest on the wrong lookalikes.
3. **18.8% negatives is too aggressive for a fine-tune.** With 2,000 negatives against 14,747
   positives, the model shifts towards "when in doubt, say no", which is exactly the recall
   drop we measured. The literature recipe is 5-10% for a fine-tune.
4. **Only 6 epochs of adaptation** - there was no chance for the new examples to be learned
   without damaging the old ones.

The one positive signal: FPR fell in the intended direction, so the *idea* is sound; the
*schedule and balance* are wrong.

---

## 10. What we actually achieved

**Knowledge**
- A measured FP taxonomy (88 FPs, 14 categories) instead of assumptions; dust is not the
  main problem.
- Proof that more epochs/data do not help: two independent fine-tunes both degrade.
- Proof that the external YOLO26 model is worse on our domain (75.2% vs 93.0% recall).

**Tooling (all committed and reusable)**
- `scripts/fp_taxonomy.py` - collect + caption FPs (Florence-2).
- `scripts/collect_hard_negatives.py` - licence-filtered collector with CLIP triage,
  detector hardness, quarantine, dedupe, holdout, contact sheets (now merges manifests).
- `scripts/check_hardneg_leakage.py` - pHash gate vs bench_R/testbench100/TEST; it caught 47
  real bench contaminations.
- `scripts/build_finetune_dataset.py` - positives + tiered negatives, oversampling, holdout.
- `kaggle/pilot_train.py` - `--data-yaml`, `--lr0`, resume, throughput, ONNX export.
- `kaggle/finetune_hardneg/` - the v1 fine-tune kernel (reusable with a new pool).

**Data (local, licensed)**
- 229 clean Commons/video negatives (leakage PASS) on disk, plus 4,000 mined negatives
  (CC BY 4.0 / CC0) in `data/hard_negatives_v2`.
- 47 bench_R-overlapping images quarantined (never trained on).

**Model**
- No regression shipped: **P50 remains the deployed candidate**.

---

## 11. Next attempt (recommended, not yet run)

1. **Rebalance the pool to the measured FPs**: warm light, sunset, haze/cloud, screen, PPE,
   reflections should dominate; cap dust at ~15%.
2. **5-10% negatives**, not 18.8%.
3. **Fresh training** (yolo11s from scratch, 25-50% positives + negatives, 30-40 epochs) or a
   **frozen-backbone head-only** fine-tune - not a full-network fine-tune from P50.
4. **Select on mAP50-95 and recall@FPR**, not val loss; patience 10.
5. Re-collect the Commons sunset/light pool with the fixed merge behaviour and the leakage
   gate in the loop.
6. Then: one DEV confirmation, one **SEALED** read, and report all 1,000.

---

## 12. Reproduction

```bash
# 1) FP taxonomy (needs P50 predictions)
python scripts/fp_taxonomy.py --caption

# 2) collect hard negatives (licence-filtered, contact sheets)
python scripts/collect_hard_negatives.py --sources wikimedia,video \
  --categories sunset_orange_sky,clouds,fog_haze_mist,reflections_lens_flare_glare,\
vehicle_lights_streetlights,red_orange_neon_lights,chandelier_stage_lights,\
dust_construction_plaster,orange_ppe_vests_cones,exhaust,steam,cave_warm_rock,\
person_portrait,screen_fire --per-category 20 --candidates-per-category 60 \
  --contact-sheets --out data/hard_negatives_v2

# 2b) free train-negative mining
python scripts/collect_hard_negatives.py --sources train_mine --per-category 1000 \
  --mine-limit 4000 --out data/hard_negatives_v2

# 3) leakage gate (must be PASS)
python scripts/check_hardneg_leakage.py --hard-neg data/hard_negatives_v2

# 4) build the training lists
python scripts/build_finetune_dataset.py --hard-neg data/hard_negatives_v2 \
  --positives build/kaggle_dataset/lists/train_25.txt \
  --val build/kaggle_dataset/lists/val_early.txt --out build/ft_v1 \
  --max-neg 2000 --oversample-tier1 2

# 5) train on Kaggle (kernel: kaggle/finetune_hardneg)
```

---

## 13. Caveats

- The 229 Commons/video negatives are on disk but their manifest/attribution rows were
  overwritten by the mining run before the merge fix; they were **not** used in FT_HN. The
  training pool was the 4,000 mined negatives.
- The `label_noise_suspect` FP must be reviewed; if it is a real fire labelled negative, the
  val label is wrong.
- bench_R DEV is only 85 positives, so DEV CIs are wide by construction; SEALED is the final
  arbiter and has not been read.
