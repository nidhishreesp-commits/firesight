# Full-run positives plan (from DEV miss analysis)

Status: **plan only. Nothing downloaded. No GPU launch.** SEALED not used.

## 1. What is missing (from DEV miss analysis)

At the accepted P50 baseline (DEV only), 35/85 positives are missed. The dominant gap is
**thin/distant smoke (21/35)**; then low-light/spark (7), small/distant fire (4), unusual/controlled
(3). The current D-Fire/FASDD pool has only `fire`/`smoke` boxes; it does not provide labelled
subsets for these scene types, and the miss sheets show the gap directly.

| missing positive type | evidence | proposed target |
|---|---|---:|
| thin/distant early wildfire smoke | 21/35 misses; fixed-camera smoke plumes | 4,000 |
| fixed-camera wildfire sequences (time-ordered) | most misses are low-contrast smoke | 1,000 frames / 100 sequences |
| aerial/high-mounted fire/smoke | 4 DEV aerial-proxy positives only | 500 |
| low-light / night fire and welding sparks | 7/35 misses | 1,000 |
| indoor fire / room-scale | 1/35 miss plus no dedicated indoor subset | 500 |
| controlled fire (candles, burn barrels) | 11 controlled misses; policy suppresses them, but the detector still needs them | 1,000 |
| vehicle / ship / industrial fire at distance | missed vehicle/ship/industrial scenes in sheets | 1,000 |
| historical/black-and-white fire | several misses | **exclude** (not production-relevant) |
| steam/food/historical "smoke" label-noise | 5-7 tiles look like steam/food | **exclude from positives; use as hard negatives** |

## 2. Sources and licences

| source | content | licence | proposed use | size / status |
|---|---|---|---|---|
| **Pyronear `pyro-sdis`** | 33,636 fixed-camera wildfire images; 28,103 with smoke | **Apache-2.0** on HF; Licence Ouverte 2.0 on data.gouv | thin smoke, fixed-camera smoke, temporal sequences | HF card ~1.32 GB; data.gouv lists 33,636 images / ~3.28 GB. **Ask before download if >2 GB** |
| **Pyronear temporal / FP pools** (`pyronear-ds-03-2024`, `FP_2024`, `DS_fp`) | sequences + false positives | verify per dataset (Pyronear open data, likely Apache-2.0/LO 2.0) | event clips + hard negatives | size to be confirmed; ask first |
| **Wikimedia Commons** | fire/smoke and ordinary scenes, per-file | CC0 / PD / CC BY / CC BY-SA; NC/ND rejected | low-light, vehicle/ship/industrial, indoor, nonfire | small; per-file attribution |
| **Public CC BY video** (YouTube and similar) | fire/smoke events and ordinary scenes | per-video CC BY only | event clips + frames | list URLs/licences before download |
| **NASA / NOAA** | aerial/satellite wildfire | U.S. federal public domain (verify per item) | aerial/high-mounted positives | modest |
| **HPWREN / FIgLib** | 400+ fixed-camera ignition sequences with onset times | **CC BY-NC 4.0** | **rejected for commercial training**; only internal research if legal approves | ~30 GB; not proposed for download |
| **FLAME / MIVIA / VisiFire / FireNet** | research fire datasets | research/NC or unclear | **rejected** unless written commercial permission | - |

Do not use Open Images positives: Open Images is the held-out source for Stage D.

## 3. Dedupe and leakage

Every candidate goes through SHA-256 + pHash (Hamming <= 2) + id/path checks against:

- `bench_R` (DEV + SEALED, but no SEALED images opened for analysis)
- `val_cal`, `val_report`, `val` (never training)
- `TEST`, `testbench100`
- `C:\Users\Nidhi\My projects\Sample_videos` (frame pHash; final held-out)
- P50's `train_50` if reused

Exact/near duplicates are dropped; source/scene ids are recorded. The SEALED exam is never used
for training or selection.

## 4. Full-run mix and GPU estimate

Keep the E1 recipe: fresh `yolo11s.pt`, 5-10% **added** hard negatives from the Phase 1 lookalike
pools, recall-at-FPR checkpoint selection, patience 10, same seed/augmentation.

| arm | positives | new positives | added hard negatives (5-10%) | images | 30 epochs @67 img/s | 40 epochs @67 img/s |
|---|---:|---:|---:|---:|---:|---:|
| 50% + new | 13,062 | 10,000 | 1,150-2,310 | 24,200-25,400 | 3.0-3.2 h | 4.0-4.2 h |
| 100% + new | 26,125 | 10,000 | 1,810-3,610 | 37,900-39,700 | 4.7-4.9 h | 6.3-6.6 h |

Add ~0.5-1 h for evaluation/export. These are **estimates** from measured 65-69 img/s, not runs.

## 5. Config diff vs E1

| item | E1 | full run |
|---|---|---|
| weights | fresh yolo11s | same |
| positives | 25% list | 50% or 100% + new positives |
| added negatives | 1,327 (9.0%) | Phase 1 lookalikes at 5-10% of positives |
| epochs | 20 | 30-40 |
| patience / selection | 10 / recall@FPR<=2% val_cal | same |
| lr0 / cosine / warmup | 0.01 / yes / 3 | same |
| seed / augmentation | 0 / identical | same |
| validation | val_early; selection val_cal | same |
| excluded | SEALED, TEST, bench_R, Sample_videos | same |

## 6. Approval needed

Estimated Phase 1 positive download: Pyronear `pyro-sdis` (~1.3-3.3 GB) plus Wikimedia/CC video
frames. **If the Pyronear download exceeds 2 GB I will ask again before starting.** No download
or GPU run starts until approved.
