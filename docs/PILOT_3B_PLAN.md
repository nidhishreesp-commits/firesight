# Phase 3b — detector pilot on Kaggle (learning curve)

**Status:** plan for approval. **Nothing trains until the pilot results are shown and approved.**
Rule: measured results are marked ✅, estimates are marked ≈.

## 1. Question the pilot answers

Does *more training data* still buy recall at a fixed false-alarm rate?
If the curve is still rising at 50%, the full run is worth it. If it is flat, we stop and
report root cause instead of burning quota (Phase 3e stop-loss).

## 2. What is held fixed

| item | value | why |
|---|---|---|
| base weights | `yolo11s.pt` (plus one `yolo11n` sanity arm) | matches the intended full-run model, CPU-runnable for smoke tests |
| input size | 640 | 960 is evaluated in 3c, not here |
| epochs | 15, early-stop patience 5 on val loss | fixed epoch budget so the curve measures *data*, not schedule |
| batch / precision | 64, `amp=True` | T4-friendly |
| sampling | stratified by `has_fire`/`has_smoke`, **same class ratio at every fraction** | otherwise size and mix are confounded |
| metric thresholds | calibrated on `val_mini` + `val_flames` (326 imgs) | identical to how bench numbers are produced |
| **selection data** | `data/splits/val.jsonl` (8,252) | **never** `val_bench` |
| excluded from every pilot run | bench_R (DEV+SEALED), testbench100, `TEST`, `bench_attributes.csv` | exam and frozen test stay clean |

## 3. Fractions and runs

| run | fraction | images (≈) | epochs | est. GPU time ≈ |
|---|---:|---:|---:|---:|
| P10 | 10% | 5,927 | 15 | ~0.8 h |
| P25 | 25% | 14,818 | 15 | ~2.0 h |
| P50 | 50% | 29,636 | 15 | ~4.0 h |
| P10n | 10% with `yolo11n` | 5,927 | 15 | ~0.5 h (pipeline sanity check) |

≈ estimates: **to be measured**, not assumed. Full run ≈ 2× P50 (59,272 imgs, 30–60 epochs) →
roughly 8–20 h depending on what the pilot measures; that number is recomputed from real
`images/s` after the pilot.

Preceding all runs: **baseline arm** — current `best_fire_v3.onnx` scored on the same metric (✅ measured before any training, so "did we beat today's model" is never an estimate).

## 4. Metrics recorded per run

1. **`val.jsonl` recall @ FPR ≤ 2%** (headline; thresholds from val_mini+val_flames) with Wilson 95% CI.
2. `mAP50`, `mAP50-95` on `val.jsonl` at the best-val epoch.
3. Throughput: **images/s** and **wall-clock per epoch** (✅ measured — this is what makes the time extrapolation real).
4. Loss curves (train/val), best epoch, early-stop epoch.
5. Learning curve: metric vs training-set size, with the fitted shape and an explicit
   "still rising / flat" judgement.

## 5. Infrastructure (3a deliverables)

- `kaggle/` notebook + local `scripts/kaggle_train.py`: dataset packaging, pinned versions
  (`ultralytics`, `torch`, `opencv-python`, `onnx`, `numpy`), checkpoint to `/kaggle/working`
  **every epoch**, `--resume` across 12 h sessions, quota tracker, one GPU notebook at a time.
- Data upload: pack `images + labels + data.yaml` into a Kaggle dataset; creating/uploading it needs
  your Kaggle credentials or your click — **I will not download large external datasets without asking.**
- Compute: free Kaggle GPU only (T4 class, ~12 h/session, ~30 h/week). **No paid compute without approval.**
- Version pinning file committed so a rerun is reproducible.

## 6. Decision gate (before the full run)

I return one table + one plot:

| fraction | images | epochs | images/s | s/epoch | total h | val recall@FPR≤2% (CI) | mAP50 | vs baseline |

and state: (a) is the curve still rising, (b) predicted full-run time from **measured**
throughput, (c) predicted gain with uncertainty, (d) recommendation: **GO / TRIM / STOP**.
You approve, then 3c (full run: all 59,272 + new positives, 30–60 epochs, 640 and 960,
augmentations for blur/noise/low-light/compression/glare/small objects, hard-negative mining
rounds from bench_R DEV FPs).

## 7. Stop-loss

- Curve flat from 10% → 50% (within CI) → **STOP**, report root cause (label noise, task
  mismatch, augmentation, class balance) instead of training bigger.
- Pilot fails to beat the current model at equal FPR → **STOP**, diagnose first.
- Throughput implies full run > available quota → **TRIM** (fewer epochs / `yolo11n` / 25–50%) and re-approve.
- Any pipeline bug that would have silently leaked bench or `val_bench` → halt and fix.

## 8. Reading discipline

- Pilot decisions use `val.jsonl` only. bench_R **DEV** is used later for *error analysis and
  hard-negative mining* of an already-chosen model; **SEALED** is read once at the milestone end and logged in `BENCH_USAGE.md`.
- Differences under 2 pp at n≈8,252 are within CI noise; the curve is judged by the confidence intervals, not the point estimates.
