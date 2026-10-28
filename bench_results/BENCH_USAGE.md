# Frozen-bench usage log

## testbench100 (demoted — no longer an exam)

`testbench100` is frozen. **Recorded uses: 24.**

All rows below are evaluations that have already happened, including the Phase 3/4 model-and-operating-point comparison. That comparison used the bench for model selection, so the 87% all-flame and 92% policy-filtered numbers are **optimistic**, not clean generalization estimates.

**Status now:** testbench100 is *contaminated* (24 logged uses, several for model selection) and is kept only as a smoke test. **`bench_R` is the only exam** — there is deliberately no `testbench_B` / `bench_S` (per 2026-10-27 decision: no new test bench).

| UTC | run_id | model | val target FPR | accuracy | specificity | note |
|---|---|---|---:|---:|---:|---|
| 2026-10-22T10:23:45Z | `20261022_155344_v3` | v3 | 0.01 | 0.81 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:25:03Z | `20261022_155502_v4` | v4 | 0.01 | 0.84 | 0.921053 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:27:08Z | `20261022_155707_pyronear_sensitive` | pyronear_sensitive | 0.01 | 0.36 | 0.921053 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:28:10Z | `20261022_155809_cascade_v3` | cascade_v3 | 0.01 | 0.85 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:40:57Z | `20261022_161057_fireviewer_rtdetrv2` | fireviewer_rtdetrv2 | 0.01 | 0.65 | 1.0 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:41:49Z | `20261022_161149_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.01 | 0.81 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:41:50Z | `20261022_161150_ensemble_v3_v4_intersection` | ensemble_v3_v4_intersection | 0.01 | 0.86 | 0.947368 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:50:24Z | `20261022_162023_fire_pt` | fire_pt | 0.01 | 0.53 | 0.947368 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:58:25Z | `20261022_162824_v3` | v3 | 0.01 | 0.77 | 1.0 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:58:53Z | `20261022_162852_v4` | v4 | 0.01 | 0.8 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T10:59:39Z | `20261022_162938_fire_pt` | fire_pt | 0.01 | 0.52 | 0.947368 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:00:33Z | `20261022_163033_v3` | v3 | 0.05 | 0.85 | 0.947368 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:00:35Z | `20261022_163034_v4` | v4 | 0.05 | 0.83 | 0.894737 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:00:36Z | `20261022_163036_fire_pt` | fire_pt | 0.05 | 0.55 | 0.868421 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:02:16Z | `20261022_163216_ensemble_v3_firept_union` | ensemble_v3_firept_union | 0.05 | 0.78 | 0.894737 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:02:18Z | `20261022_163217_ensemble_v3_v4_firept_union` | ensemble_v3_v4_firept_union | 0.05 | 0.83 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:02:19Z | `20261022_163219_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.05 | 0.87 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:03:06Z | `20261022_163305_v3` | v3 | 0.1 | 0.87 | 0.921053 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:03:07Z | `20261022_163307_v4` | v4 | 0.1 | 0.84 | 0.868421 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:03:08Z | `20261022_163308_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.1 | 0.87 | 0.894737 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:20:35Z | `20261022_165035_cascade_union_v3_firept` | cascade_union_v3_firept | 0.05 | 0.87 | 0.947368 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:43:34Z | `20261022_171333_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.05 | 0.87 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:44:35Z | `20261022_171435_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.05 | 0.87 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |
| 2026-10-22T11:45:03Z | `20261022_171502_ensemble_v3_v4_union` | ensemble_v3_v4_union | 0.05 | 0.87 | 0.973684 | evaluation / model selection (pre-policy or policy diagnostic) |

## Policy diagnostic (not a new evaluation)

The 92.0% policy-filtered figure is a **re-scoring of the same 2026-10-22 17:15 run** with controlled/spark/screen images suppressed and excluded from the denominator. It is not a separate model evaluation. 12 images were suppressed; 6 of them would have alarmed if controlled flames were included.

## bench_R (the exam)

**Locked** 2026-10-27T02:44:04Z by Nidhishree S P — 1,000 images, manifest `9001fbc827a606168867ccf2151694f390ee3df34de16e1da2ee7a585d07993c`, composition 96 fire-only / 106 smoke-only / 82 both / 716 negative. Split: **DEV 300 / SEALED 700**, stratified random seed 20261022.

**Rules:** DEV may be used only for error analysis and hard-negative mining. SEALED is touched **once per milestone** and logged below. Thresholds are calibrated on `val_mini` + `val_flames`, never on bench_R. Models are selected on `data/splits/val.jsonl`, never on `val_bench`.

| UTC | milestone | what happened | SEALED? |
|---|---|---|---|
| 2026-10-27T09:20–11:17Z | **M0 baseline** | 11 models evaluated once (DEV + SEALED reported separately). `fireviewer_rtdetrv2` not run (HF download blocked → `failures.json`). Baseline: cascade_v3 acc 0.830, recall 0.479, FPR 0.022, strict-policy recall 0.6795. | yes (one-shot baseline) |
| 2026-10-27T12:18Z | **M1 metric recompute** | Policy/metric fixes re-scored from **cached predictions** (no new inference, no model change): smoke no longer bypasses the policy in `policy_metrics`; new `policy_strict` view; `by_flame_type` now alert-level and covers smoke-only images. | yes (re-read only, same predictions) |
| 2026-10-27 | **label cleanup** | 8 negative rows (no fire, no smoke) carried a `flame_type`; reset to `unknown` in `labels.csv`, `DEV.jsonl`, `SEALED.jsonl`, and `reviews.json` (note appended). Composition and all 96 fire-only types unchanged (12 uncontrolled / 52 controlled / 31 spark / 1 unknown). `labels_csv_sha256`: `977d45d9…` → `2d3fd83e…`; `manifest_sha256` unchanged (images untouched). | n/a (labels only) |
| 2026-10-27 | **spark subtype split** | Wrote additive side file `bench_R/SPARK_SUBTYPES_PROPOSED.csv` (37 rows: `spark` → `spark_benign` 37, `electrical_arc` 0; sha256 `74527f9450e22181fb5ec0b14b27bf697d06c0d5299f573eb6b64def39b4e2b5`). Locked bench files untouched; no scoring, no model change. **electrical_arc = unmeasured on bench_R.** | n/a (labels/metadata read, no scoring) |
| 2026-10-27T13:23Z | **M2 taxonomy re-score** | 6-class taxonomy (`spark` → `spark_benign` alias in the metric loader only; locked labels untouched). Re-scored from **cached predictions**, no inference, no model change. Verified no metric moved: cascade_v3 acc 0.8300, recall 0.4789, FPR 0.0220, strict 0.6795, legacy 0.6605; only the `by_flame_type` bucket name changed (`spark` → `spark_benign`, same 2/37). `electrical_arc` and `screen` remain unmeasured (0 images). | yes (re-read only, same predictions) |
| 2026-10-28 | **pilot DEV confirmation** | P10/P25/P50 + baseline scored **once** on bench_R **DEV** (300 images; 85 pos / 215 neg), thresholds calibrated on `val_mini`+`val_flames` (target FPR 2%). Ordering P50 56.5% > P25 50.6% > P10 48.2% > BASE 40.0%; DEV FPR P50 2.33% / BASE 3.72%. **SEALED untouched.** | no (DEV only) |
| 2026-10-28 | **external model check (DEV)** | YOLO26 HF model scored once on bench_R **DEV** (300; 85 pos / 215 neg): recall 55.3% [44.7, 65.4], FPR 4.65% vs P50 56.5% / 2.33%. Not adopted. **SEALED untouched.** | no (DEV only) |
| 2026-10-28 | **option B fine-tune (DEV)** | FT25 (P50 + 25 epochs at 25%, early-stopped at 6) scored once on bench_R **DEV**: recall 54.1% [43.6, 64.3], FPR 3.26% vs P50 56.5% / 2.33%. Worse; not adopted. **SEALED untouched.** | no (DEV only) |

**Known provenance caveat:** `labels_csv_sha256` and `manifest_sha256` are hashes of the *local working-copy bytes*. Git normalises CRLF→LF on commit, so a fresh clone's raw bytes hash differently (this was already true at lock time). Nothing verifies these at runtime — they are provenance tokens only. Fix: normalise to LF before hashing (deferred; would invalidate previously recorded tokens).
