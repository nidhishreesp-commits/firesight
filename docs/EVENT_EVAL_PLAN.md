# Event-level evaluation plan and harness

Status: **source proposal + harness built; nothing downloaded; no GPU.** The 11 Sample_videos
remain the final held-out check and are not part of this event set.

## 1. Proposed clips (source, licence, size)

| source | fire clips / onset | nonfire hours | licence | size / status |
|---|---|---:|---|---|
| **Pyronear `pyro-sdis`** | 30-50 sequences; onset = first annotated smoke frame | many hours from empty-label sequences | **Apache-2.0** (HF) / Licence Ouverte 2.0 | HF ~1.32 GB; data.gouv 33,636 images ~3.28 GB. **Ask if >2 GB** |
| **Pyronear temporal / FP pools** (`pyronear-ds-03-2024`, `FP_2024`, `DS_fp`) | temporal sequences + hard negatives | yes | verify per dataset (Pyronear open data) | size to confirm; ask first |
| **Wikimedia Commons video** | fire/smoke clips; onset manual | traffic/street/construction/indoor | per-file CC0/PD/CC BY/CC BY-SA; NC/ND rejected | small; per-file list before download |
| **YouTube CC BY** | fire/smoke channels with CC BY | ordinary streets/construction | per-video CC BY only | URL/licence list before download |
| **NASA / NOAA** | aerial/satellite fire, no ground onset | - | U.S. federal public domain (verify) | optional, small |
| **HPWREN / FIgLib** | 400+ ignition sequences with onset offsets | - | **CC BY-NC 4.0** | **rejected for commercial**; only internal research with legal approval |
| **FLAME / MIVIA / VisiFire / FireNet** | research fire clips | - | research/NC/unclear | **rejected** unless written commercial permission |

The minimum viable set is **30-50 Pyronear fire sequences + >=20 hours of Pyronear/Wikimedia
nonfire**. FIgLib would be ideal for onset times but its NC licence blocks commercial use.

## 2. Harness

`scripts/event_eval.py` takes a JSONL manifest:

```json
{"id":"clip01","path":"data/event/clip01.mp4","label":"fire","onset_s":12.5,"camera_id":"cam01","source":"pyronear","licence":"Apache-2.0"}
```

It runs a detector (P50/E1) with optional verifier v4, temporal N-of-M and the video layer, then
reports:

- fire recall within **5/10/30 s** of onset (Wilson CI)
- time-to-detection: mean, median, p90, cluster-bootstrap CI
- non-fire alarm episodes per camera-hour (cluster-bootstrap CI)
- per-clip alarm times and TTD

Synthetic smoke test (DV frames, no SEALED): fire recall 1.0 within 5/10/30 s, TTD 0.4 s, 0
nonfire alarms; both with and without the video layer.

## 3. Run matrix after approval

| arm | detector | second stage | temporal | video layer |
|---|---|---|---|---|
| P50 | P50 | - | yes | yes |
| E1 | E1 best_recall.pt | - | yes | yes |
| P50 + verifier v4 | P50 | v4 | yes | yes |
| E1 + verifier v4 | E1 | v4 | yes | yes |

Thresholds: P50 0.29/0.33; E1 from its val_cal calibration. Verifier v4 threshold from val_cal
only. No event clip is used for training or threshold tuning.

## 4. What is still missing

- No public video has been downloaded yet.
- Onset times for Wikimedia/YouTube clips require manual annotation.
- Nonfire hours are not yet measured; the target is >=20 hours across day/night/weather.
- The NC-licensed FIgLib cannot be used for the commercial product; do not include it.
