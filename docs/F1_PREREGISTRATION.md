# F1 pre-registration (prepared, not launched)

Status: **prepared. No GPU launch. SEALED remains unread for F1.** The one-shot SEALED reads for
P50, TommyNgx and SalahYOLOv26 are already logged; F1 must not use SEALED until it is a final
candidate.

## 1. Data composition (measured)

| component | images |
|---|---:|
| base 50% positives (D-Fire/FASDD, `train_50` non-empty labels) | **13,063** |
| Pyro-SDIS v2.0 positives, non-event cameras, train split | **7,259** |
| hard negatives (Stage A packaged pool) | **1,327** |
| **F1 total** | **21,649** |
| added negative share | **6.53% of positives** |

Event holdout: whole cameras **brison-110, courmettes-212, brison-200**; 2,673 images, 2,621 with
smoke, **167 sequences**. These cameras are excluded from F1 training. Training/event overlap: **0**.
Phase 1 lookalikes are represented by the available Stage A targeted hard negatives; the larger
Phase 1 Open Images lookalike set is still pending.

## 2. Pyro-SDIS audit (measured, 100 random images)

- 89 annotated, 11 empty (intentional false positives).
- Missing-box suspect: **1/100 (1.0%)**.
- Box-crop CLIP fog-as-smoke suspect: **41/89 (46%)** — automated, **not** confirmed human labels.
- Tiny boxes (<0.5% image area): **80% of annotated images**.
- The audit should be reviewed before training; the 46% flag is a triage rate, not a proven error
  rate. Steam/food/exhaust label noise must be excluded from positives.

## 3. Leakage / dedupe (measured)

Pyro candidates (7,259) checked against 23,004 reference pHashes and 6,535 reference SHA-256 hashes
from bench_R, val, TEST, independent negatives and 104 Sample_videos frames: **0 exact SHA, 0 pHash
<=2, 0 id/path**. Event-camera images are excluded from the training list; overlap check is 0.

## 4. Training recipe (unchanged E1 recipe)

| item | F1 |
|---|---|
| weights | fresh `yolo11s.pt` |
| epochs | 30 |
| patience / selection | 10 / recall@FPR<=2% on val_cal |
| imgsz / batch / AMP | 640 / 64 / true |
| lr0 / lrf / schedule | 0.01 / 0.01 / cosine, warmup 3 |
| seed / augmentation | 0 / identical to E1 |
| train val / selection | val_early / val_cal |

GPU estimate at measured **65-69 img/s**: 21,649 images x 30 epochs = **2.61-2.77 h**;
**~3.0-3.2 h** with startup/eval overhead. This is an estimate, not a run.

## 5. Pre-registered pass criteria

Compare F1 against **P50** and **E1** at matched FPR **0.9/1.5/2.0%** on:

1. `val_report`
2. independent negative set (clean subset)
3. bench_R **DEV only** (one run; no SEALED)
4. event set (recall within 10/30 s, false alarms per camera-hour)

Smoke and fire must be reported separately.

F1 passes only if:

- at matched FPR 2.0%, F1 recall is **>= max(P50, E1)** on val_report and the independent set,
  with paired McNemar p<0.05 on positives vs both;
- FPR at matched recall is **<= both** (or significantly lower);
- smoke recall improves by **>=10 pp** vs P50, and fire recall is not significantly worse;
- bench_R DEV shows no significant regression vs P50/E1 at matched FPR;
- event-level: recall within 10 s **>=85%**, within 30 s **>=95%**, and false alarms
  **<=0.05 per camera-hour**, with no regression vs P50/E1;
- held-out source/cause checks do not collapse (report them; no claim of generalisation if they do).

If F1 passes only on val_report, label it **in-distribution** and do not call it general. If the
result is borderline, propose a second seed before any conclusion.

## 6. Artifacts

- `build/f1/composition.json`, `leakage_report.json`, `lists/`
- `build/kaggle_f1_pyro/` (7,259 positives, private package, ~? MB)
- `build/kaggle_f1_code/` (training/eval code, chunked selection and eval)
- `kaggle/stage_f1/pilot_f1.ipynb` (prepared, not pushed)
- CPU smoke test: `build/f1_smoke3` (20 train / 20 val / 20 val_cal, 1 epoch, 320 px, passed;
  one selection row, `best_recall.pt` saved).

No GPU launch without approval.
