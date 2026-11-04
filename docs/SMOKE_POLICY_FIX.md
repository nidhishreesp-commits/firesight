# Smoke policy fix (Phase 0)

## What changed

Two independent assumptions were rejecting real smoke:

1. `TemporalConfirmer` required smoke growth; static plumes failed.
2. `VideoFalseAlarmFilter` treated all static detections as sun/cloud and routed them to review.

The fix is smoke-specific; fire behavior is unchanged.

### New smoke persistence rule

A smoke detection is confirmed if there are at least **3 smoke detections in 6 sampled frames**
(12 s at 2 fps), even if the boxes move and do not keep one track ID.

Config (`configs/default.yaml`):

```yaml
smoke_static_min_conf: 0.20
smoke_growth_bypass: true
smoke_trackless: true
smoke_persistence_min_conf: 0.20
smoke_persistence_iou: 0.05
smoke_persistence_n: 3
smoke_persistence_window: 6
```

The video filter now keeps static smoke above `smoke_static_min_conf` instead of suppressing it.
Fire and low-confidence static lookalikes keep the old behavior.

## Results

### Smoke-only set (6 unique clips)

| version | smoke layer recall | false alarms on 17 hard negatives |
|---|---:|---:|
| old policy | **1/6** | 0/17 |
| new policy | **5/6** | **0/17** |

Only the factory chimney clip is still missed (max smoke 0.219, below the 0.20 persistence floor
on enough frames).

### 42 public clips

| version | TP | FP | FN | precision | recall | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| old P50 layer | 10 | 6 | 10 | 62.5% | 50.0% | 27.3% | 61.9% |
| **new P50 layer** | **18** | **8** | **2** | **69.2%** | **90.0%** | 36.4% | **76.2%** |

Per category after the fix: fire **14/15**, smoke **4/5**, cloud/haze **6/10 false alarms**,
non-fire **2/12 false alarms**.

The extra false alarms are cloud/haze and two archival street scenes. The user's deployment
assumption excludes sky/cloud, but haze/dust can still be a risk; this is why a smoke verifier is
still needed before production.

### Sample_videos

Unchanged at **90.9% accuracy**, 2/2 fire, 1 sky false alarm (`drone.mp4`).

## Pyronear YOLO11s test

Downloaded `pyronear/yolov11s` v8.3.0 and `yolo11s_sensitive-detector` v1.0.0.

- Single class `item` = smoke.
- Checkpoint metadata says **AGPL-3.0**, not the card's Apache-2.0.
- On the 6 smoke clips at 0.20: **1/6 layer recall**, 1 false alarm (fire truck).
- At 0.10: 3/6 raw, **1/6 layer**.
- It misses chimney smoke, factory smoke, the Dragon Bravo plume and rubbish smoke.
- **Domain mismatch.** Not useful for these indoor/industrial/fixed-camera clips.

## Decision

- Keep the new P50 smoke policy; it is the best result so far and costs no GPU.
- Do not integrate Pyronear for these scenes.
- Do not call this production-ready: the smoke sample is only 6 clips and cloud/haze false alarms
  remain on the public set.
- Next: add 10-20 real smoke-only clips, then add a smoke verifier/fine-tune using Pyro-SDIS and
  smoke-heavy data.

## Tests

`python -m pytest -q` passes. New tests cover static-smoke keep and trackless smoke confirmation.


## New 5 stock clips (edge cases)

- 2 volcano fire clips: both detected.
- Incense smoke: missed (no detections).
- Colored smoke flare: raw detected, final layer missed.
- Mountain clouds: false alarm.

Details and segmentation test: `docs/NEW5B_AND_SEGMENTATION_TEST.md`.
