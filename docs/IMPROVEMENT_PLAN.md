# Improvement plan: accuracy, precision and recall

Measured baseline: P50 on the full bench_R and the 42-clip public video set. The target is a
defensible **shadow/dashboard** and **dispatch** operating point, not a single “accuracy 90%”
claim.

## 1. What is actually failing (measured)

| failure | evidence |
|---|---|
| High raw recall, unacceptable false alarms | 42 clips: 90% recall, 59% FPR, 58% precision |
| Temporal/video layer over-suppresses smoke | smoke clips 0/5 layer alerts; recall 50% overall |
| Cloud/haze false alarms survive | 4/10 cloud/haze clips layer-alert |
| Image-level bench recall is low | 56.3% @2.1% FPR; max image accuracy ~86.5% |
| Smoke class is weak | bench smoke recall 47.9%; video smoke 0/5 after layer |
| Localization is weak | fire AP50 0.181, smoke AP50 0.372 |
| Indoor/industrial positives are missing | no labelled indoor/industrial subset in D-Fire/FASDD |
| Pyronear has tiny/label-noise risk | 80% tiny boxes; 41/89 automated fog-suspect crops |

## 2. Plan (ordered)

### A. Fix the temporal/video layer before retraining
- Smoke N-of-M is too strict for short clips (4/6). Use a lower, smoke-specific rule and require
  growth/flicker, not only static persistence.
- Do not suppress a static smoke plume merely because it is static; route to review, not drop.
- Calibrate per-camera and per-scene; separate dashboard and dispatch thresholds.
- Re-evaluate on the 42-clip set, split by source: 20 dev / 22 held-out test.
- Success: dashboard recall within 10 s >=85%, false alarms <=0.05/camera-hour; dispatch recall
  >=95%, false alarms <=0.005/camera-hour.

### B. Rebalanced F1 data (the main lever)
Target positive mix: **60% indoor/industrial, 25% thin/distant smoke, 15% controlled flames**.

| type | target | source plan |
|---|---:|---|
| indoor/industrial fire/smoke | 12,000 | Wikimedia Commons (CC0/PD/CC BY/CC BY-SA) + own/partner/licensed captures; video frames from CC BY sources |
| thin/distant smoke | 5,000 | Pyro-SDIS v2.0 after the audit/filter + sequence-consistent frames |
| controlled flames | 3,000 | Commons candle/campfire/fireplace/burn-barrel; own controlled burns |
| hard negatives | 1,000-2,000 (5-10%) | Phase 1 lookalikes + indoor/industrial steam, welding sparks, lamps, screens, vehicle lights, PPE, hot surfaces |

No NC licences. No steam/food/exhaust label noise in positives. Event cameras
(brison-110, courmettes-212, brison-200) stay out of training.

### C. Pyronear filter before training
- Min box size (e.g. max box area >=0.2% of image or min side >=24 px).
- Sequence-level consistency: require >=2 frames with a box in the same camera/date sequence.
- CLIP crop agreement + verifier agreement.
- User-marked 81-crop sample decides the thresholds; measure precision/recall of the filter on the
  marks before applying to 12,378.
- Report survivors per camera and tiny-box count. Do not train on unfiltered Pyronear.

### D. Verifier / mask fallback
- Keep verifier v4 as an optional second stage; integrate only if it improves at matched FPR with
  acceptable latency (v4 currently needs CLIP features; ~258 ms/frame CPU).
- SAM mask features showed promise at detection-level 5% FPR but MobileSAM is ~1.05 s/detection;
  research note only, not in the production path.

### E. Training recipe and GPU estimate
Fresh `yolo11s.pt`, 30 epochs, patience 10, recall@FPR<=2% on val_cal, same seed/augmentation as E1,
5-10% added hard negatives. For ~20,000 positives + 1,100-2,000 negatives:

| estimate | raw @65-69 img/s | 2x-low correction |
|---|---:|---:|
| 30 epochs | 2.5-2.8 h | **5.0-5.6 h** |
| 40 epochs | 3.3-3.7 h | **6.6-7.4 h** |

Use the 2x correction because the earlier 2.37 h estimate produced 4.76 h actual.

### F. Gates (pre-registered in `docs/F1_PREREGISTRATION.md`)
- Image-level: at 2% FPR, F1 recall >= max(P50, E1) on val_report and independent negatives;
  McNemar p<0.05; smoke recall +10 pp; fire not worse.
- bench_R DEV only for F1; no SEALED read until final candidate.
- Event-level: 42-clip public set + a larger event set; recall within 10/30 s, false alarms per
  camera-hour.
- Shadow pilot: >=4 weeks, >=5 cameras, >=500 camera-hours before autonomous dispatch.

## 3. Honest scarcity statement

Licence-clean **indoor/industrial fire** is scarce. Wikimedia Commons can supply perhaps a few
thousand usable stills after filtering, but not 12,000 surveillance-like positives. Reaching the
60% indoor/industrial target realistically requires one or more of:

- own captures (factory/warehouse/kitchen/electrical/plant/ship, with safety/consent);
- partner fire-department or industrial-safety footage with written permission;
- purchased/licensed stock (the current Downloads iStock/Getty clips are **not** licensed for
  training; they are test-only);
- CC BY public video frames (per-video verification; YouTube/Wikimedia).

Without that, the available pool can support a smaller rebalanced experiment, not the full 60/25/15
target. This must be decided before F1.

## 4. Immediate next actions

1. User marks the 81-crop Pyronear sheet → measure filter, then apply.
2. Approve indoor/industrial source list and data acquisition route.
3. Fix temporal/video-layer smoke rule and re-run the 42-clip dev/test split.
4. Build the rebalanced F1 composition and leakage report.
5. No GPU launch until the composition is approved.
