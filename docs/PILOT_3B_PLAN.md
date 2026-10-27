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
| weights licence | **AGPL-3.0** (Ultralytics default) | downloaded at runtime from `github.com/ultralytics/assets`; commercial distribution may require an Ultralytics Enterprise Licence - flag before shipping |
| input size | 640 | 960 is evaluated in 3c, not here |
| epochs | 15, early-stop patience 5 on val loss | fixed epoch budget so the curve measures *data*, not schedule |
| batch / precision | 64, `amp=True` | T4-friendly |
| sampling | stratified by `has_fire`/`has_smoke`, **same class ratio at every fraction** | otherwise size and mix are confounded |
| metric thresholds | calibrated on `val_mini` + `val_flames` (326 imgs) | identical to how bench numbers are produced |
| **selection data** | `data/splits/val.jsonl` (8,252) | **never** `val_bench` |
| excluded from every pilot run | bench_R (DEV+SEALED), testbench100, `TEST`, `bench_attributes.csv` | exam and frozen test stay clean |

## 3. Fractions and runs

Fractions are taken from the **uploadable pool only** (58,985 images: D-Fire + FASDD; the 287
proprietary/own-capture images are excluded from the Kaggle dataset, see §9), stratified so the
class ratio is identical at every fraction.

| run | fraction | images | epochs | est. @60 img/s | est. @100 img/s |
|---|---|---:|---:|---:|---:|
| P10 | 10% | 5,898 | 15 | 0.5 h | 0.3 h |
| P25 | 25% | 14,747 | 15 | 1.1 h | 0.7 h |
| P50 | 50% | 29,493 | 15 | 2.2 h | 1.4 h |
| **pilot total** | | | | **≈3.8 h** | **≈2.4 h** |
| P10n (optional) | 10% with `yolo11n` | 5,898 | 15 | ≈0.3 h | ≈0.2 h |

≈ T4 throughput band is **estimated** (60-100 img/s for yolo11s @640, batch 64, AMP) and is
replaced by the measured `images/s` the pilot itself logs. **Measured** on this machine for
calibration: yolo11n @320, batch 8, CPU = **12.35 images/s, 16.2 s/epoch** on 200 images
(dry run, §10) - a CPU floor, not a GPU prediction.

Full run ≈ 58,985 images x 40 epochs: **≈11 h @60 img/s, ≈6.6 h @100 img/s** at 640, roughly
double at 960. That fits 1-2 sessions of the ~12 h / ~30 h-per-week free quota; recompute from
measured `images/s` before approving.

Preceding all runs: **baseline arm** - current `best_fire_v3.onnx` scored on the same metric and
the same val slices, so "did we beat today's model" is never an estimate.

## 4. Metrics recorded per run

1. **`val.jsonl` recall @ FPR ≤ 2%** (headline; thresholds from val_mini+val_flames) with Wilson 95% CI.
2. `mAP50`, `mAP50-95` on `val.jsonl` at the best-val epoch.
3. Throughput: **images/s** and **wall-clock per epoch** (✅ measured — this is what makes the time extrapolation real).
4. Loss curves (train/val), best epoch, early-stop epoch.
5. Learning curve: metric vs training-set size, with the fitted shape and an explicit
   "still rising / flat" judgement.

## 5. Infrastructure (3a deliverables)

Everything below exists locally and is committed; **nothing has been uploaded**.

| file | role |
|---|---|
| `kaggle/pilot_firesight.ipynb` | the notebook to upload (preflight, train, score, summary, export) |
| `kaggle/pilot_train.py` | one learning-curve point: stratified fraction, checkpoint **every epoch**, `--resume`, measured throughput, ONNX export |
| `kaggle/pilot_eval.py` | val.jsonl scoring: calibrate on `val_cal`, report recall@FPR≤2%/negative + Wilson CI on `val_report`, bench_R-format JSON |
| `scripts/kaggle_package.py` | builds the PRIVATE dataset package, MANIFEST.csv, counts per source/licence, LEAKAGE_REPORT.json, CHECKLIST.md |
| `kaggle/kernel-metadata.json` | kernel spec (private, GPU, dataset source) - replace the username |
| `code/` inside the dataset | the exact `run_bench.py` + `firesight` package used for bench_R, so pilot metrics are computed identically |

Pinned versions (`build/kaggle_dataset/requirements-pinned.txt`): `ultralytics==8.4.147`,
`onnx==1.22.0`, `onnxruntime==1.30.0`. `torch` stays as provided by the Kaggle GPU image
(CUDA-matched); dry-run versions are recorded in `versions-reference.json`.

**Credentials:** the Kaggle token is read from `~/.kaggle/access_token` (or `~/.kaggle/kaggle.json`,
`KAGGLE_USERNAME` / `KAGGLE_KEY`) only; the CLI is invoked as `python -m kaggle` (no `kaggle` on PATH). It is never written to the repo, the notebook, logs or the dataset;
`.kaggle/`, `kaggle.json` and `build/` are git-ignored. If a token is ever pasted into a chat or
notebook, rotate it in Kaggle immediately.

**One GPU notebook at a time:** Kaggle may expose **2x T4 in one session**; the preflight requires
at least one GPU and training is pinned to `device 0`. Stop any other session first (quota ~30 h/week). Checkpointing every epoch to `/kaggle/working`
makes an interrupted session resumable, not lost.

**Compute:** free Kaggle GPU only (T4 class, ~12 h/session, ~30 h/week). **No paid compute
without approval.**

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

## 9. Upload package (built locally, NOT uploaded)

`python scripts/kaggle_package.py` writes `build/kaggle_dataset/` (manifest-only by default;
`--materialize` hardlinks images, `--zip` builds the archive to upload). Measured output:

| item | value |
|---|---|
| train pool (allowed) | **58,985** (D-Fire 4,829 + FASDD 54,156 of train.jsonl) |
| val slices (from `val.jsonl`) | early-stop 1,000 / threshold-calibration 2,000 / report 5,119 |
| total images in package | **67,104** |
| excluded sources | `local_seed` 19 (own captures), `local_negative` 268 (unclear-rights web images) |
| by source | FASDD 60,698 / D-Fire 6,406 |
| by licence | CC BY 4.0 (FASDD) 60,698 / CC0-1.0 (D-Fire) 6,406 |
| estimated size | ≈7.6 GB (mean 111 KB/image) |

**Leakage proof** (`LEAKAGE_REPORT.json`, verdict **PASS**): id overlap 0 vs bench_R,
testbench100 and TEST; path overlap 0 vs TEST and bench_R; pHash near-duplicates
(Hamming <= 2) 0 against all three; proprietary-path hits 0. This is in addition to the
scene isolation already done by `prepare_data.py`.

### Upload checklist (stop here until approved)

- [ ] `LEAKAGE_REPORT.json` verdict is PASS
- [ ] `dataset-metadata.json`: `isPrivate: true`, `id` set to your Kaggle username
- [ ] `MANIFEST.csv` present (16.8 MB; source + licence per image)
- [ ] `kaggle/kernel-metadata.json`: username replaced, `enable_gpu: true`, `dataset_sources` set
- [ ] Credentials only via env / `~/.kaggle/kaggle.json`; token rotated if ever exposed
- [ ] Upload command (only after go-ahead):
      `python -m kaggle datasets create -p build/kaggle_dataset --dir-mode zip`
      then `python -m kaggle kernels push -p kaggle/`
- [ ] Start **one** GPU session; re-run the train cell after any restart (it resumes)

## 10. Local dry run (measured, CPU)

`build/dry_run/`: 200 train / 50 val images, `yolo11n.yaml` built from scratch (no download),
320 px, batch 8, CPU, 2 epochs.

| step | result |
|---|---|
| training | 2 epochs, 200 images, **16.2 s/epoch**, **12.35 images/s** |
| checkpointing | `last.pt`, `best.pt`, `epoch0.pt` written each epoch (21.2 MB each) |
| interrupt + resume | stopped after epoch 1; `--resume` logged `Resuming training ... from epoch 2 to 2`, `resumed: true`, 2 rows in `results.csv` |
| ONNX export | `best.onnx`, 10.0 MB, opset 17, 0.6 s |
| scoring | `pilot_eval.py` on the same Kaggle `code/` layout: recall 0.0 [0, 0.176] @ FPR 0.0 [0, 0.243] (untrained model, expected) |

The dry run proves the training, per-epoch checkpoint, cross-session resume, ONNX export and
bench_R-format scoring paths. It is a pipeline test, not a result.

## 11. electrical_arc has zero examples

`electrical_arc` is a **configurable class** in the 6-class taxonomy, but there are **0**
`electrical_arc` images in bench_R and 0 in the training pool. It is therefore reported as
**unmeasured**, never as zero-risk. The warning tier exists and is unit-tested, but no metric on
it is meaningful yet.

Proposed sourcing (needs approval before any download):
1. Own captures of switchgear/arc-flash test footage, if available.
2. Public arc-flash demonstration videos from manufacturer/standards bodies (licence review per clip).
3. Electrical-panel/breaker thermal-camera datasets with arc events (check licence).
4. Label welding arcs as `spark_benign` and **only** true fault arcs as `electrical_arc`
   (the rubric already distinguishes them).
Target: 50-100 images before claiming any electrical_arc metric; until then the class stays in the
schema and out of the results table.

## 12. Decision rules (explicit)

Let `r10`, `r25`, `r50` be val recall @ FPR <= 2% per negative image (Wilson CI), and `b` the
current `best_fire_v3` scored on the same slices and calibration.

| decision | condition | next step |
|---|---|---|
| **GO** | `r50 - r10 >= 5 pp` **and** CIs of r10/r50 do not overlap **and** `r50 > b` at equal FPR **and** a single **bench_R DEV-only** confirmation run per checkpoint (300 images) reproduces the ordering (`r50 > b`, no reversal) | approve full run (all 58,985 + new positives, 30-60 epochs, 640 and 960), recompute hours from measured img/s |
| **TRIM** | curve rises but projected full run > ~20 GPU h or > remaining weekly quota | cut epochs / use 25-50% / yolo11n, re-approve |
| **STOP** | `r50 - r10` inside CI noise (flat curve) **or** no run beats `b` at equal FPR | stop; diagnose label noise, task mismatch, augmentation, class balance before more training |

**bench_R DEV condition (GO gate):** after the pilot, each checkpoint gets **exactly one** scoring
run on bench_R **DEV** (300 images) - never SEALED. DEV is used only for this confirmation and
error analysis; thresholds stay calibrated on `val_mini` + `val_flames`, and the run is logged in
`bench_results/BENCH_USAGE.md` as DEV-only. If the DEV ordering contradicts val.jsonl (P50 does not
beat the baseline, or P25/P50 reverse), the result is **STOP / diagnose**, not GO. SEALED remains
untouched until the milestone end.

Any leakage or benchmark contamination found in the pilot data halts the run and is fixed first.
Pilot numbers are **measured**; GPU-hour projections are **estimated** until the pilot logs
`images/s`.

## 13. How to run (after approval only)

```bash
# 1) build + inspect the package locally (no network)
python scripts/kaggle_package.py --materialize
python -c "import json;print(json.load(open('build/kaggle_dataset/LEAKAGE_REPORT.json'))['verdict'])"

# 2) set the username in both metadata files, then upload (needs your credentials)
python -m kaggle datasets create -p build/kaggle_dataset --dir-mode zip
python -m kaggle kernels push -p kaggle/

# 3) in the notebook: preflight -> install -> train -> score -> summary
#    after a session restart, re-run the train cell; it resumes from last.pt
```
