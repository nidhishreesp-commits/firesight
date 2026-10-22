# Frozen-bench usage log

`testbench100` is frozen. **Recorded uses: 24.**

All rows below are evaluations that have already happened, including the Phase 3/4 model-and-operating-point comparison. That comparison used the bench for model selection, so the 87% all-flame and 92% policy-filtered numbers are **optimistic**, not clean generalization estimates. Going forward: evaluate only once per major retrain, and use `testbench_B` for final claims.

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

## bench_R

No model has been run on bench_R. It is **pending human review and lock**. Once locked, only the DEV split may be used for error analysis between iterations; SEALED runs must be logged here with date, model and reason.
