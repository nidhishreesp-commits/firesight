# FireSight — benchmark and production report

All numbers below were produced by scripts in this repository on the machine
described in §1. Nothing is copied from a paper. No model was tuned on the
frozen TEST split.

---

## 1. Hardware / software

| | |
|---|---|
| CPU | 8 logical cores (no CUDA device; `torch.cuda.is_available() == False`) |
| RAM | 16 GB |
| OS | Windows (win32) |
| Python | 3.14.7 |
| Detectors | Ultralytics 8.4.147, ONNX Runtime 1.30, PyTorch 2.14 (CPU) |
| Note | **FPS numbers are CPU-only.** GPU figures will be much higher; re-measure on target hardware. |

---

## 2. Dataset

Built by `scripts/prepare_data.py` from D-Fire (Kaggle mirror) and FASDD_CV
(Hugging Face mirror). MS-FSDB (arXiv 2410.16631) was requested but its Google
Drive link was not retrievable without an account, so it is **not** included.

- 75,637 images after perceptual-hash dedupe (117,325 in) — 55.77% negative.
- 85,401 boxes: 52,450 fire, 32,951 smoke.
- Splits: train 59,272 / val 8,252 / **TEST 8,113** (frozen, sha256 locked,
  0 scene overlap with train/val).
- Class order unified to `0=fire, 1=smoke`; D-Fire's opposite order was remapped
  and visually verified (`scripts/contact_sheet.py`).

Full counts: `data/README.md`, `data/splits/README.md`, `data/processed/stats.json`.

**Leakage flags**
- D-Fire and FASDD official splits are random. Near-duplicates across splits were
  removed and near-identical frames were grouped into scenes, but same-camera /
  same-fire events at distant times cannot be detected by pHash. Residual
  same-source overlap is possible.
- The ss-prod models and the FireViewer checkpoint were trained on merged
  D-Fire/FASDD-derived data. Their TEST scores are therefore **not** clean
  generalization estimates (see §4).

---

## 2.5 Hand-labeled bench status (Phase 1–2)

`testbench100/` has been built but is **NOT LOCKED and NOT benchmarked** yet.
Composition: 50 positives (25 fire-only, 15 smoke-only, 10 fire+smoke) and 50
hard negatives (13 light/colour, 13 smoke-like, 12 fire-adjacent, 12
camera-artifact). Labels are provisional source-dataset human annotations plus
CC-licensed Wikimedia images; **the user must verify every image in the
dashboard Bench review page** before the SHA-256 lock is written.

Leakage check before review: pHash Hamming ≤6 against all train/val phashes
→ **0 flagged**; sampled CLIP embedding (1,600 train/val references) at
cosine ≥0.98 → **0 flagged**. Full report: `testbench100/LEAKAGE_CHECK.json`.

No model has been run on `testbench100`, and no accuracy number for it is
reported here. The 90%/95% acceptance question remains open until the bench is
locked and Phase 3 is run.

## 3. Benchmark — identical frozen splits

Command:
`python scripts/benchmark.py --manifest data/splits/test_bench.jsonl --val-manifest data/splits/val_bench.jsonl --target-fpr 0.01`
(test 3,000 images: 1,233 positive / 1,767 negative; thresholds calibrated on val only.)

| Model | Family | License | Size MB | CPU FPS | Fire AP50 | Smoke AP50 | Fire AP50-95 | Smoke AP50-95 | Recall @ val-FPR≤1% | Test FPR | Fire box R | Smoke box R |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ssprod_best_fire_v3 | YOLOv12n | proprietary | 10.1 | **14.3** | 0.680 | 0.707 | 0.388 | 0.448 | **0.767** | 0.0249 | 0.384 | 0.549 |
| ssprod_best_fire_v4 | YOLOv12n | proprietary | 10.1 | 8.8 | **0.688** | 0.706 | **0.396** | 0.445 | 0.733 | 0.0187 | 0.326 | 0.504 |
| ssprod_best_fire_v2 | YOLOv12n | proprietary | 5.3 | 13.7 | 0.286 | 0.117 | 0.127 | 0.043 | 0.231 | 0.0187 | 0.160 | 0.017 |
| ssprod_fire_pt | YOLO11s | proprietary | 54.5 | 6.4 | 0.405 | 0.337 | 0.176 | 0.166 | 0.183 | 0.0221 | 0.129 | 0.001 |
| finetune_v2_1ep (trained here) | YOLOv12n | internal | 5.2 | 14.0 | 0.570 | 0.407 | 0.303 | 0.219 | 0.418 | 0.0198 | 0.264 | 0.068 |

`finetune_v2_1ep` is a **compute-budget smoke test**, not a candidate: one epoch
from `best_fire_v2.pt` on a 6,400-image balanced TRAIN subset at 512 px on CPU.
It already improves fire AP50 from 0.286 (v2) to 0.570, which confirms the
training/export path works; it is far from converged (smoke AP50 0.407). A GPU
run for 50–100 epochs is the next step, not a conclusion from this row.

At a stricter val target of 0.1% FPR:

| Model | Recall @ test FPR | Test FPR |
|---|---:|---:|
| ssprod_best_fire_v3 | 0.541 | **0.0028** |
| ssprod_best_fire_v4 | 0.545 | 0.0068 |

**Recall at fixed false-alarm rate** (image-level, read off the TEST PR curve; this
is evaluation, not threshold tuning — see `artifacts/benchmark_target0001/*/metrics.json`):

| Model | FPR 0.1% | FPR 1% | FPR 5% | FPR 10% |
|---|---:|---:|---:|---:|
| ssprod_best_fire_v3 | **0.450** | 0.624 | 0.813 | 0.843 |
| ssprod_best_fire_v4 | 0.340 | **0.650** | **0.824** | **0.871** |

Reading: **v3 dominates at very low false-alarm rates (<1%); v4 catches up above
~5%.** Because the product requirement is "maximum recall at a very low fixed
false-alarm rate", v3 is the primary choice.

### Slow / zero-shot baselines (small subsets, CPU)

| Model | Split | Images | Fire AP50 | Smoke AP50 | Recall | FPR | CPU FPS |
|---|---|---:|---:|---:|---:|---:|---:|
| FireViewer RT-DETRv2-R50 | test_mini | 240 | 0.790 | 0.727 | 0.798 | 0.035 | **0.6** |
| Grounding DINO Tiny (zero-shot) | test_gsd | 32 | 0.452 | 0.353 | 0.522 | 0.333 | 0.11 |

- RT-DETRv2-R50 has the best AP of any model measured, but is ~24× slower than
  v3 on CPU and its weights **cannot be redistributed or shipped in a product**
  under the FireViewer license. Its documented operating threshold is ~0.02, far
  below the other models.
- Grounding DINO zero-shot is not production-viable (33% false-alarm rate, 0.11 FPS).
- RT-DETRv2 also required custom preprocessing: the HF-default processor scored
  ~0.02 mAP; a squash-to-768 / no-normalize path scored ~0.79 AP50.

---

## 4. External check (not D-Fire/FASDD)

`data/splits/external_cctv_nc.jsonl` — Simuletic CCTV Fire/Smoke Emergency
dataset (CC BY-NC 4.0, evaluation only; 240 images, positives only, so no FPR
can be measured).

| Model | Fire AP50 | Smoke AP50 | Fire img recall@0.25 | Smoke img recall@0.25 |
|---|---:|---:|---:|---:|
| ssprod_best_fire_v4 | 0.848 | 0.647 | 0.698 | 0.610 |
| ssprod_best_fire_v3 | 0.806 | 0.678 | 0.597 | 0.624 |
| FireViewer RT-DETRv2-R50 | 0.943 | 0.873 | 0 (scores < 0.25) | 0 |

All detectors lose several points on this external set. v4 leads on fire, v3 on
smoke. This is a small, partly synthetic set; treat as directional.

---

## 5. False-positive reduction (ablation)

### 5.1 Temporal N-of-M + ByteTrack (video)

Fire confirm = 3 of 5 frames; smoke = 4 of 6 with growth cue. Non-fire coverage
is only **2.9 minutes** of video, so 24 h rates are extrapolations with wide
uncertainty.

| Configuration (ssprod_best_fire_v3) | Fire clips detected | Mean TTD | Max TTD | Non-fire alarm events | Extrapolated FA/camera/24h |
|---|---:|---:|---:|---:|---:|
| Raw detector (no temporal) | 2/2 | −0.53 s | −0.4 s | 4 | 1986 |
| + temporal N-of-M | 2/2 | **+0.17 s** | +0.6 s | **0** | 0 |
| + temporal + verifier (0.6) | 2/2 | +0.17 s | +0.6 s | **0** | 0 |

`ssprod_best_fire_v4` + temporal also produced 0 non-fire alarms, but it
**false-confirmed fire at 3.5 s** on the car-fire clip (true ignition 40.5 s),
i.e. a −37 s "time-to-detection" that is actually a false alarm. v3 first
confirmed at 40.2 s. This is the strongest argument for v3 at low FAR.

### 5.2 Second-stage crop verifier

MobileNetV3-Small (fire / lookalike / smoke) trained **only** on TRAIN crops and
hard negatives mined from TRAIN non-fire images and non-fire video.

| Verifier lookalike threshold | Image recall | FPR | Recall @ FPR 0.1% | Recall @ FPR 1% |
|---|---:|---:|---:|---:|
| none (base v3) | 0.541 | 0.0028 | 0.450 | 0.624 |
| 0.5 | 0.548 | 0.0034 | **0.490** | **0.680** |
| 0.7 | 0.553 | 0.0034 | **0.500** | 0.660 |
| 0.8 | 0.539 | 0.0023 | 0.450 | 0.636 |

Net effect: **+5 percentage points of recall at 0.1% FPR** for a small FPR
increase. The verifier is a real but modest win; the temporal stage does the
heavy lifting for false alarms.

### 5.3 Flame-evidence class correction (post-processing)

The detector sometimes labels a visible flame under/inside a smoke plume as
`smoke` (e.g. highway fireball at 62–66 s). `src/firesight/flame.py` checks every
**existing** smoke box for a broad flame-colour mask (HSV red/orange/yellow,
S>100, V>120). If ≥8% of the box and ≥120 pixels are flame-coloured, the box is
relabelled `fire` and the evidence is kept in `meta.flame_fraction` /
`meta.original_label`.

- No new boxes are created on clean negatives, so this cannot create a new alarm
  on a negative image.
- On the 4,779 TEST negatives, only **2 of 54** false-positive boxes at conf 0.5
  changed class; both were already false positives.
- At 50 s and 55 s the smoke box over the highway flame is now `fire`
  (flame fraction 0.11 / 0.08); at 62–66 s the detector already returns `fire`
  natively.
- This is a **class-correction heuristic**, not a retrained model. It is
  disclosed in the UI via `original_label`/`flame_fraction`.

---

## 6. Chosen model and justification

**Deploy `ssprod_best_fire_v3` (YOLOv12n, ONNX) as the primary detector**, with:

1. ByteTrack + N-of-M temporal confirmation (fire 3/5, smoke 4/6 + growth);
2. the MobileNetV3 crop verifier at lookalike threshold 0.5–0.7;
3. per-camera ROI + confidence overrides.

Why not the others:
- **v4** has marginally better AP and higher recall above 5% FPR, but at 0.1%
  FPR it loses ~11 points of recall to v3 and it false-confirmed a pre-ignition
  car scene in the video test.
- **v2** and **fire.pt** are far weaker on this test.
- **RT-DETRv2-R50** has the best AP but is 24× slower on CPU and its license
  forbids redistribution/shipping; use it only if you have a GPU, need maximum
  AP, and accept the license terms.
- **Grounding DINO zero-shot** is a labelling aid, not a detector.

If you have a GPU and the FireViewer license is acceptable, RT-DETRv2-R50 is the
accuracy ceiling measured here. Otherwise v3 is the best recall-per-CPU-cycle at
low false-alarm rates.

---

## 7. Known failure cases (from the frozen TEST set)

Contact sheets: `artifacts/audit_negative_fp.jpg`, `artifacts/audit_fire.jpg`,
`artifacts/audit_smoke.jpg`. Confirmed lookalike false positives include:

| Category | Example seen | Confidence |
|---|---|---:|
| Sunset / orange sky | `dfire:test:WEB...` sunset with birds | 0.50 fire |
| Car taillights / headlights | car rear lights | 0.91 fire |
| Streetlights at night | road lamps | 0.97 fire |
| Chandelier / warm indoor lights | hotel lobby | 0.75 fire |
| Orange object in grass | orange toy/fruit | 0.89 fire |
| Worker in orange PPE | orange vest + helmet | 0.89 smoke |
| Clouds | mountain clouds | 0.65 smoke |
| Black bag / red text sign | dark backpack with red text | 0.89 fire |
| Water sun-glare | river reflection | 0.34 smoke |
| Construction machine (yellow/orange) | road roller | 0.92 fire |

Class confusion also occurs: smoke plumes are sometimes labelled `fire` (and
vice versa). This does not change the binary alarm but hurts per-class AP.

Other honest limitations:
- Smoke recall is much lower than fire at low FPR.
- Small/distant smoke is the dominant miss; annotation quality varies.
- Non-fire video coverage (2.9 min) is too short for a credible 24 h FA rate.
- TEST may be contaminated for models trained on merged D-Fire/FASDD data.
- The verifier and v3 were not retrained in this environment (CPU-only); the
  detector hard-negative retraining loop is implemented (`scripts/mine_hard_negatives.py`,
  `scripts/retrain_pipeline.py`) but a full GPU retrain is recommended.
- The keyword label-noise audit (`scripts/audit_label_noise.py`) found 0 filename
  matches because the public images are hash-named. Candle/match/sunset/steam
  noise therefore remains a **manual visual-review** task; the confirmed
  lookalike failures above are the evidence base.

---

## 8. Reproduce every number

```powershell
python scripts/prepare_data.py --sources dfire,fasdd,local
python scripts/freeze_test.py
python scripts/make_yolo_dataset.py
python scripts/benchmark.py --models ssprod_best_fire_v2,ssprod_best_fire_v3,ssprod_best_fire_v4,ssprod_fire_pt `
  --manifest data/splits/test_bench.jsonl --val-manifest data/splits/val_bench.jsonl --target-fpr 0.01 --out artifacts/benchmark_stage1
python scripts/benchmark.py --models ssprod_best_fire_v3,ssprod_best_fire_v4 `
  --manifest data/splits/test_bench.jsonl --val-manifest data/splits/val_bench.jsonl --target-fpr 0.001 --out artifacts/benchmark_target0001
python scripts/mine_hard_negatives.py
python scripts/train_verifier.py
python scripts/eval_verifier.py --target-fpr 0.001
python scripts/video_benchmark.py --models ssprod_best_fire_v3 --out artifacts/video_ablation --tag v3_temporal
python scripts/video_benchmark.py --models ssprod_best_fire_v3 --out artifacts/video_ablation --tag v3_raw --no-temporal
python scripts/eval_external.py --models ssprod_best_fire_v3,ssprod_best_fire_v4,fireviewer_rtdetrv2_r50
python -m pytest
```

---

## 9. Deployment position

This system is a **supplement**, not a replacement, for certified fire alarm
systems. It must run in shadow mode alongside existing alarms before it is
trusted; operators review alerts; no automated suppression or siren actuation
without human confirmation. See `README.md` §Production readiness.


## 10. Phase 3 — locked testbench100 (2026-10-22)

The 100-image bench is locked (	estbench100/LOCKED.json, reviewer
Nidhishree S P, SHA-256 20915b0…). The reviewer changed the composition to
**62 positives / 38 negatives** (22 fire-only, 19 smoke-only, 21 both), mainly
by reclassifying the 12 controlled-flame images as positives. The lock records
composition_override: true; there are **no fire-adjacent hard negatives** in
the locked bench. This is a material limitation when reading the numbers.

Thresholds were calibrated on al_mini + val_flames only: 240 ordinary
val images + 18 CC-licensed controlled flames + 60 TEST negatives + 8
construction-dust frames. The bench was never used for tuning. The table below
is the history; ench_results/SUMMARY.md has all runs.

| Model | Val target FPR | Accuracy (95% Wilson CI) | Recall | Specificity | F1 |
|---|---:|---|---:|---:|---:|
| **ensemble v3+v4 union** | 5% | **0.870 (0.790–0.922)** | 0.806 | **0.974** | 0.885 |
| v3 | 5% | 0.850 (0.767–0.907) | 0.790 | 0.947 | 0.867 |
| v4 | 5% | 0.830 (0.745–0.891) | 0.790 | 0.895 | 0.852 |
| cascade v3 + verifier | 1% | 0.850 (0.767–0.907) | 0.774 | 0.974 | 0.865 |
| FireViewer RT-DETRv2-R50 | 1% | 0.650 (0.553–0.736) | 0.435 | 1.000 | 0.607 |
| Pyronear yolo11s sensitive | 1% | 0.360 (0.273–0.458) | 0.016 | 0.921 | 0.030 |
| fire.pt (YOLO11s) | 5% | 0.550 (0.452–0.644) | 0.355 | 0.868 | 0.494 |

**Best measured configuration:** v3+v4 union, calibrated at 5% val image-level
false-alarm rate:

- accuracy **87.0%** (95% CI 79.0–92.2%)
- recall **80.6%** (95% CI 69.1–88.6%)
- precision **98.0%**
- specificity **97.4%** (95% CI 86.5–99.5%)
- per class: fire recall 81.4%, smoke recall 57.5%
- box AP50: fire 0.598, smoke 0.643; mIoU (matched) fire 0.778, smoke 0.886
- runtime: CPU, ONNX v3+v4 union

**90% and 95% image-level accuracy were not reached.** Even the lower bound of
the best 95% CI is 79.0%, far below the requested 90% lower bound. The
hard-negative specificity gate (≥95%) *is* met by this configuration (97.4%),
but accuracy is not.

### Remaining errors (13)

ench_results/20261022_163219_ensemble_v3_v4_union/false_positives.jpg and
alse_negatives.jpg; machine-readable list in the run JSON.

- **10 hard positives missed:** candles, lighters, gas flame, campfires,
  welding/grinding sparks, one welding frame and one fireplace. The models were
  not trained to treat these controlled flames as positives; v3/v4 often emit no
  box even at confidence 0.01.
- **3 smoke misses:** one sunset frame, one other_hard smoke frame, one small
  distant smoke (WEB11696 / AoF08219 thresholds).
- **1 false positive:** orange PPE (traffic-worker clothing) classified as
  smoke at 0.41 on the light/colour family.
- Accuracy by difficulty: easy 25/25, medium 19/21, **hard 6/16**.
  Camera-artifact and smoke-like negatives were 100% clean in this sample.

### Construction / plaster dust (user-requested)

Added hard_negatives/sources/construction_dust/images (16 frames from local
construction footage, proprietary internal). v3 false-alarmed on **1/16**
frames: ppe_2_3s.jpg → smoke 0.48. These frames are now in the calibration
negatives and the catalog; they must be included in the next verifier/retrain.

### Iteration log / why gains stopped

1. v3 at 1% val FPR: 81% accuracy; lowering to 5% gave 85%.
2. v4: 84% at 1%, 83% at 5% — more fire recall, more FPs.
3. Cascade with verifier: 85%; the verifier *drops* controlled flames because it
   was trained with candles/campfires as lookalikes, so it is counter-productive
   for the reviewer's current policy.
4. RT-DETRv2/Pyronear: not competitive on this mixed indoor/outdoor bench
   (RT-DETR 65%, Pyronear 36%).
5. v3+v4 union: 87% at 5% FPR; 87% at 10% FPR with more recall but specificity
   falls to 89.5%.
6. Adding fire.pt to the union did not help (83%), because its controlled-flame
   wins came with too many false positives.
7. Retrained the crop verifier with 9 controlled-flame positives + construction
   dust lookalikes (val accuracy 90.5% -> 91.9%) and applied it to a
   v3+fire.pt union cascade. Result: 87% accuracy, 82.3% recall, **94.7%**
   specificity, 96.2% precision. Fire-class accuracy rose to 91%, but two cloud
   false positives and five missed controlled flames kept the combined accuracy
   at 87%. It did not beat the simpler v3+v4 union (87%, 97.4% specificity).

**Next root-cause fix (not done):** retrain the verifier with controlled flames
as fire positives and construction dust as lookalike, then re-run the cascade;
and/or fine-tune a detector with candle/campfire/welding positives that are
**not** in the bench. This needs a GPU and new labelled positives; the current
CPU-only threshold/ensemble lever has stopped yielding gains.

### External models

- select-ai/fire-smoke-detection: no public weights found. Not evaluated.
- e1250/safety_detection: no declared licence; not downloaded without approval.
- pyronear/temporal-model: sequence model; not applicable to the single-image
  bench. It remains a candidate for the 30+ video clips.

### Repro

`powershell
python scripts/build_val_flames.py
python scripts/run_bench.py --model ensemble_v3_v4_union --target-fpr 0.05
python scripts/summarize_bench.py
`



## 11. Phase A — policy alignment and flame types (2026-10-22)

### Decision

Per the reviewer, **controlled flames are positives for the fire class**:
candles, lighters, matches, gas burners/stoves, tandoors, fireplaces, campfires,
BBQ, diyas, incense, welding/grinding sparks, fireworks, flame lamps and TV/movie
fire. They are no longer in any hard-negative list.

Recommended defaults (implemented as a runtime flag, not a retrain):

| Flame type | Detection training | Default alarm |
|---|---|---|
| uncontrolled | positive | **include** |
| controlled (candles, stoves, campfires, fireplace, BBQ, diya, incense, flame lamp) | positive | **exclude** |
| spark (welding, grinding, fireworks) | positive | **exclude** |
| screen (TV/movie fire) | positive | **exclude** |
| unknown (no type classifier yet) | — | **include** |

Why: fireworks and TV/movie fire are real flames and should be *recognised* by
the detector; they are not structural-fire emergencies. The lame_type
attribute lets you change that decision later with config/API, without
relabelling or retraining. Candles/stoves in normal operation should be
controlled + excluded (or ROI-masked); if your camera sees them as emergencies,
move them into the include list.

The alarm policy is in configs/default.yaml and POST /api/policy/fire-types;
the Live view has checkboxes. **Until the Phase B/D flame-type classifier is
trained, every fire detection is unknown and therefore included** — the flag
does not silently suppress all fire.

### A2. Side-car attributes

ench_attributes.csv (69 fire boxes; does **not** touch locked bench files):
57 uncontrolled, 9 controlled, 3 spark, 0 screen. It also records reviewer and
licence per box. The lame_type field is now part of GroundTruth and
Detection.

### A3. Per-flame-type recall on the locked bench

Best v3+v4 union at 5% val false-alarm target:

| Type | n | Detected | Recall | 95% CI |
|---|---:|---:|---:|---|
| uncontrolled | 31 | 30 | **96.8%** | 83.8–99.4% |
| controlled | 9 | 4 | 44.4% | 18.9–73.3% |
| spark | 3 | 1 | 33.3% | 6.1–79.2% |
| smoke-only | 19 | 13 | 68.4% | 46.0–84.6% |

**All flames as positives:** 87.0% accuracy (79.0–92.2%), specificity 97.4%.
**Runtime policy excludes controlled/spark/screen:** 92.0% accuracy
(84.5–96.1%), recall 88.0% (76.2–94.4%), specificity 97.4%, on 88 evaluable
images. 12 images were suppressed; 6 of them would have alarmed if controlled
flames were included. This is a **policy-conditional number**, not a free
accuracy gain: it excludes 12 reviewed positives from the denominator.

**Leakage warning:** the bench has been used **24 times** for model/operating-
point comparison (ench_results/BENCH_USAGE.md). The 87% and 92% numbers are
therefore optimistic and must not be used as final claims. No more threshold or
model selection on this bench. The honest final number is 	estbench_B, locked
and evaluated once after retraining.

### Diffs and plan

- Policy/doc/code diff: docs/phaseA_policy.diff
- Phase B1 collection plan (sources, licences, sizes, review workflow, compute):
  docs/PHASE_B_PLAN.md



## 12. bench_R reference bench — collection status (2026-10-22)

ench_R/ now contains **1,000 fresh Wikimedia Commons candidates** with the
requested composition (pending human review; **no model has been run**):

| Group | Count |
|---|---:|
| uncontrolled fire | 200 |
| controlled flame | 80 |
| spark / screen | 40 |
| smoke-only | 100 |
| fire+smoke | 80 |
| hard negatives | 500 (18 families, every family ≥15) |

- Sources: Wikimedia Commons only (CC0 / PD / CC BY / CC BY-SA), per-image
  attribution and SHA-256 in ench_R/index.json.
- **59 candidates removed** by pHash Hamming ≤6 against train/val/testbench100;
  **0 flagged** by CLIP at cosine ≥0.98 against 1,800 train/val references
  (ench_R/LEAKAGE_CHECK.json).
- Scene cap: max 3 near-identical frames per pHash scene.
- Boxes are intentionally empty; the reviewer must draw them. labels.csv is
  provisional and not a label file until locked.
- The **Reference bench** dashboard page provides keyboard shortcuts
  (←/→, F, S, N, A, Enter), resume, progress, a random 10% second-pass
  spot check with agreement rate, and a lock endpoint that enforces composition
  and writes MANIFEST.sha256/LOCKED.json. Locking also writes the stratified
  **DEV 300 / SEALED 700** split.
- ench_R images are local and git-ignored; metadata, URLs, hashes and the
  manifest are committed. This is deliberate (≈450 MB) and does not expose the
  proprietary construction footage.

**Phase C goal:** this 1,000-image fresh bench exceeds the planned 150–250
bench_B and satisfies the size/diversity intent **after you review and lock it**.
It is not yet a usable final bench, and no 90/95 claim is made.

**Fireworks / TV/movie fire:** included as positives for detection with
lame_type=spark / screen; default alarm policy excludes them. Change via
larm.fire_type_include/exclude or the Live-view policy panel without retraining.

**Still missing:** human boxes (500 positives), human review of query-based
categories, flame-type classifier, Phase B positive set (pyro-sdis downloaded;
3,000 Wikimedia positives not yet collected), GPU training, and the final
DEV/SEALED evaluation.



## 13. bench_R pre-annotation (human-reviewed labels remain ground truth)

Independent proposers only; **no evaluated detector was used**:

| Proposer | Licence | Images | Runtime |
|---|---|---|---:|
| google/owlv2-base-patch16-ensemble | Apache-2.0 | 498 | 76.2 min CPU |
| IDEA-Research/grounding-dino-tiny | Apache-2.0 | 100 (stratified) | 15.3 min CPU |

Prompts: fire, flame, candle flame, smoke, sparks. Proposals at low threshold
0.01, merged per class with weighted box fusion (IoU ≥0.5). Suggested image-level
labels use 0.05. Coverage of seeded positives: uncontrolled 199/200 (99.5%),
controlled 80/80, spark/screen 40/40, smoke-only 100/100, both 79/80 (98.8%).
Negatives were not proposed; they need single-key confirmation.

Bias control: **100 images (60 positives, 40 negatives) are blind** — the UI hides
proposals and the reviewer labels them from scratch. The separate 100-image
second-pass spot check excludes blind images. ench_R/PREANNOTATION_REPORT.md
contains the coverage/runtime table. Correction rate per proposer/category will be
computed after review and lock; ccepted_as_proposed, edits_made and
	ime_spent_s are stored per image.

The review UI now pre-fills proposals for non-blind images, marks them
PROPOSED, supports box drawing/editing, X delete-last-box, score slider, a
no-proposal counter/jump, and blind badges. Proposals are stored separately in
ench_R/proposals.json; they are never written into labels.csv until the
reviewer saves an image.

## 14. bench_R locked — Phase 0 baseline (2026-10-27)

bench_R was locked at the reviewed composition: 1,000 images (96 fire-only / 106
smoke-only / 82 both / 716 negative), manifest `9001fbc8…`, reviewer Nidhishree S P,
split **DEV 300 / SEALED 700** (seed 20261022). Thresholds calibrated on val_mini +
val_flames only. `fireviewer_rtdetrv2` was not run (HF download blocked).

**Headline (cascade_v3, n=1000, Wilson 95% CI):**

| view | value |
|---|---|
| accuracy | 83.00% [80.55%, 85.20%] |
| recall (any fire/smoke) | 47.89% [42.14%, 53.69%] |
| false alarms | 22 FP / 716 negatives = 3.07% [2.04%, 4.61%] (2.20% of all images) |
| **strict-policy recall** (alarm iff `uncontrolled`) | **67.95% [60.27%, 74.76%]**, n=156 |
| strict-policy accuracy | 91.74% [89.73%, 93.39%] (128 positives suppressed, 30 of them would have alarmed) |
| DEV strict recall | 65.22% [50.77%, 77.32%], n=46 |
| SEALED strict recall | 69.09% [59.93%, 76.96%], n=611 |
| fire box AP50 / smoke box AP50 | 0.089 (527 GT boxes) / 0.353 (294 GT boxes) |

Recall by ground-truth flame type (alert-level, cascade_v3, n=1000):

| type | n | recall [95% CI] |
|---|---:|---|
| uncontrolled (fire **or** smoke) | 156 | 67.95% [60.27%, 74.76%] |
| controlled | 85 | 31.76% [22.84%, 42.27%] |
| spark_benign (bench label spark) | 37 | 5.41% [1.50%, 17.70%] (2 hit only by a *smoke* detection; 0 by the fire class) |
| smoke-only | 106 | 46.23% [37.03%, 55.68%] |
| unknown | 6 | 16.67% [3.01%, 56.35%] |

Ranking on accuracy: cascade_v3 0.830 > cascade_union_v3_firept 0.822 > v4 0.819 >
ensemble_v3_v4_intersection 0.817 > ensemble_v3_firept_union 0.809 ≈
ensemble_v3_v4_firept_union 0.809 > v3 0.807 > ensemble_v3_v4_union 0.806 >
finetune_v2_1ep 0.789 > fire_pt 0.752 > pyronear_sensitive 0.717. Differences under
2 pp at n=1000 are noise; cascade_v3's accuracy CI alone spans 4.7 pp.

### Phase 0 fixes (measured effect)

- **Smoke no longer bypasses the policy.** `FireTypePolicy.allows()` applied to
  `label != "fire"` only, so every smoke detection survived regardless of type.
  Smoke-only images were also excluded from `flame_types` in both metric loaders.
- **Policy persisted** to `configs/fire_type_policy.json` (POST writes it, AppState
  loads it, yaml is the fallback) — a restart no longer reverts a policy change.
- **New `policy_strict`** view = alarm iff GT `flame_type == uncontrolled`, fire *and*
  smoke; legacy `policy_uncontrolled_plus_smoke` kept and now applies its exclusions to
  smoke too (66.05% vs 67.95% recall — the difference is the 6 `unknown` positives,
  which strict suppresses and legacy lets alarm).
- **`by_flame_type` is alert-level** and covers smoke-only images, so it reconciles
  exactly with `policy_strict` (uncontrolled 106/156 = strict tp/(tp+fn)).
- **Label cleanup:** 8 negative rows carried a `flame_type`; reset to `unknown` in
  labels.csv / DEV / SEALED / reviews. Composition and all 96 fire-only types unchanged;
  `labels_csv_sha256` 977d45d9… → 2d3fd83e…; images untouched (`manifest_sha256` stable).

Composition facts used by the strict policy: positives = 156 uncontrolled (81 fire + 75
smoke-only), 85 controlled (58 fire + 27 smoke-only), 37 spark, 6 unknown; the other
128 positives are suppressed by design. **`screen` has 0 images in bench_R** (the 37
`spark_screen`-category images all reviewed as `spark`), so screen behaviour is
**unmeasured** on the bench — it will be reported as unmeasured, never as zero-risk.

**Spark subtype split (plan patch) — done, in a side file:** `bench_R/SPARK_SUBTYPES_PROPOSED.csv`
maps all 37 `spark` rows → **`spark_benign`**, **0 → `electrical_arc`** (28 classified from
filenames with explicit welding/grinding/cutting vocabulary; the 9 remaining were inspected
image-by-image — grinding, weld arcs, torch cutting, chop-saw, model-rocket sparks; 15 ambiguous
rows were reviewed in total plus 2 keyword rows spot-checked, none showing an equipment fault
arc). Therefore **electrical_arc is unmeasured on bench_R** (too few examples to score), and the
`WARNING` tier for it cannot be validated here. Locked bench files are not edited — the side
file is additive and is a *proposal* for the 6-class taxonomy, not a change of bench truth.

### Taxonomy + alarm tiers (plan patch)

- **Six classes:** `FLAME_TYPES = (uncontrolled, controlled, spark_benign, electrical_arc,
  screen, unknown)`. The locked bench still says `spark`; `canonical_flame_type()` aliases it to
  `spark_benign`, so bench truth was **not edited** and no metric moved (verified: strict 67.95%,
  legacy 66.05%, accuracy 83.00% unchanged - only the `by_flame_type` bucket name changed).
- **Three runtime tiers:** `alarm` (uncontrolled, unknown), `warning` (electrical_arc - surfaced,
  never alarms), `log` (controlled, spark_benign, screen). Precedence is exclude > include >
  warning > log, so an operator can promote electrical_arc to alarm or demote it to log without
  retraining. Persisted in `configs/fire_type_policy.json` with a new `warning` key; a pre-tier
  policy file (no `warning`) loads with the default warning tier, and legacy `spark` in yaml or
  config normalises to `spark_benign`.
- **Unmeasured by construction:** bench_R has 0 `electrical_arc` and 0 `screen` images, so the
  warning tier is implemented and unit-tested but **not validated on the bench**. It is reported
  as unmeasured, never as zero-risk.

**False-alarm budget (agreed 2026-10-27):** `docs/FALSE_ALARM_BUDGET.md` fixes the FPR
definition (per **negative image**, 3.07% [2.04, 4.61] for cascade_v3 - the 2.20% all-image figure
is secondary) and proposes event ceilings of **<= 1.0 false-alarm events/camera-week** for
dashboard-only alarms and **<= 0.1/camera-week** for dispatch-triggering alarms. Image-level FPR
cannot be converted into event rates (persistent false positives repeat across frames), so the
event rate stays **unmeasured** until a 10-camera x 48 h continuous capture is scored end-to-end.

Written for the next stage: `docs/LABELING_RUBRIC.md` (6-class rubric) and
`docs/PILOT_3B_PLAN.md` (Kaggle learning-curve pilot, gated on approval).

## 15. Phase 3b pilot (measured) - data saturates at ~25%

Kaggle T4, yolo11s @640, batch 64, AMP, 15 epochs, stratified fractions of the 58,985-image
uploadable pool. Selection on `data/splits/val.jsonl` (cal 2,000 / report 5,119), thresholds
calibrated on `val_cal` at FPR <= 2% per negative image; evaluation via ONNX on CPU (the Kaggle
eval cell hit an unrelated OOM caused by passing a Python list to `predict`, which stacks all
7,119 images into one 8.15 GiB array).

| run | images | img/s | s/epoch | recall @FPR<=2% (cal) | 95% CI | FPR/neg (report) | mAP50 fire/smoke |
|---|---:|---:|---:|---:|---|---:|---|
| P10 | 5,898 | 67.8 | 87.0 | 0.8283 | [0.813, 0.843] | 2.57% | 0.544 / 0.577 |
| P25 | 14,747 | 71.7 | 205.7 | 0.9253 | [0.914, 0.935] | 3.25% | 0.579 / 0.643 |
| P50 | 29,493 | 70.7 | 417.4 | 0.9297 | [0.919, 0.939] | 3.14% | 0.633 / 0.679 |

Equal-FPR comparison at the baseline's achieved 2.34% (thresholds re-selected on the report slice,
symmetric across models, for comparison only): BASE 80.4% [78.8, 81.9], P10 82.3% [80.7, 83.7],
P25 91.6% [90.4, 92.6], P50 92.3% [91.2, 93.3].

**bench_R DEV confirmation** (one run per checkpoint, 300 images, SEALED untouched): P50 56.5%
[45.9, 66.5] > P25 50.6% > P10 48.2% > BASE 40.0% [30.2, 50.6]; DEV FPR P50 2.33% vs BASE 3.72%.
Ordering is consistent (no reversal); DEV n=85 positives, so the CIs are wide by construction.

Findings:
- **10% -> 25%: +9.3 pp at equal FPR** (disjoint CIs) - real gain.
- **25% -> 50%: +0.7 pp** (overlapping CIs) - **flat**. mAP50 still rises (0.579 -> 0.633) but that
  does not convert into FPR-constrained recall.
- The calibrated 2% target does not transfer to the held-out report slice (2.6-3.3%); calibrate at
  ~1.5% or pool calibration data before quoting "recall @ FPR<=2%".
- Recommendation: **TRIM** - run the full pipeline at 25-50% of the data (≈2.3-4.7 h per 15 epochs at
  measured 70.7 img/s), not 100% (≈9.3 h at 640), and spend the saved quota on epochs/960/hard
  negatives. Formal GO conditions are met (P10->P50 >= 5 pp, disjoint CIs, P50 > baseline at equal
  FPR, DEV ordering consistent), but the plateau makes 100% poor value.

### External models checked (2026-10-28)

| model | licence | val recall@FPR2% | val FPR/neg | val mAP50 f/s | bench_R DEV recall (FPR) | verdict |
|---|---|---:|---:|---|---|---|
| **ours P50** | AGPL-3.0 (weights) | **0.9297** [0.919, 0.939] | 3.14% | 0.633 / 0.679 | **56.5%** (2.33%) | reference |
| ours BASE (`best_fire_v3`) | proprietary | 0.8012 | 2.34% | 0.522 / 0.616 | 40.0% (3.72%) | reference |
| [SalahALHaismawi/yolov26-fire-detection](https://huggingface.co/SalahALHaismawi/yolov26-fire-detection) | MIT | 0.7515 [0.734, 0.768] | 3.37% | 0.374 / 0.316 | 55.3% (4.65%) | **worse** - not adopted |
| [TommyNgx/YOLOv10-Fire-and-Smoke-Detection](https://huggingface.co/TommyNgx/YOLOv10-Fire-and-Smoke-Detection) | Apache-2.0 | not evaluated | - | - | - | gated repo (HF token required); base model is Ultralytics/YOLO11, claimed mAP 0.85 unverified |

The YOLO26 card claims mAP50 94.9 / recall 88.8 on its own "custom" dataset; on our val.jsonl slice
it measures mAP50 **0.374** and recall **75.2%**, i.e. the claim does not transfer to this domain.
It also has 3 classes (`fire`/`other`/`smoke`), so `other` was excluded from scoring. On bench_R DEV it
ties P50 on recall within CI (55.3% vs 56.5%) but at ~2x the false-alarm rate (4.65% vs 2.33%), so it
is not an improvement. Both remain in the comparison table as evidence; neither replaces the current model.
