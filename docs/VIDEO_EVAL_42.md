# Public video evaluation (42 curated clips)

Sources: Wikimedia Commons (CC0/PD/CC BY/CC BY-SA, NC/ND rejected) + three proprietary
Downloads clips (stock; **test-only, not for training**). No SEALED. No training on these clips.

- Clips: **42** — 15 fire, 5 smoke, 10 cloud/haze, 12 non-fire.
- Sampling: 2 fps; non-fire clips capped at 120 s; fire/smoke full clip.
- Detector: P50 ONNX, temporal N-of-M, video layer. Thresholds fire 0.29 / smoke 0.33.

## Clip-level result

| mode | TP | FP | TN | FN | precision | recall | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw alert | 18 | 13 | 9 | 2 | 58.1% | **90.0%** | **59.1%** | 64.3% |
| temporal only | 10 | 6 | 16 | 10 | 62.5% | **50.0%** | **27.3%** | 61.9% |
| temporal + video layer | 10 | 6 | 16 | 10 | 62.5% | **50.0%** | **27.3%** | 61.9% |

Per category (layer alerts / clips):

| category | clips | layer alerts | raw alerts |
|---|---:|---:|---:|
| fire | 15 | 10 | 14 |
| smoke | 5 | **0** | 4 |
| cloud/haze | 10 | **4** | 7 |
| non-fire | 12 | 2 | 6 |

Non-fire coverage evaluated: **0.342 h**; layer false alarms **17.6 per camera-hour** on this
small set.

**Conclusion:** the earlier 11-clip result was optimistic. On a broader public set P50 has high
raw recall but unacceptable false alarms, and the temporal/video layer over-suppresses smoke
(0/5) while still passing cloud/haze false alarms. P50 is **not** a 90%-precision/recall production
detector.

Artifacts: `build/video_eval/results/summary.json`, `per_clip.csv`, `contact_sheet.jpg`,
`annotated/`. Curated manifest: `data/video_eval/public/curated_manifest.jsonl`.

## Extra: three Downloads stock clips (proprietary, test-only)

The three supplied stock clips (industrial interior, street waste collection, city crowd) are
**non-fire** and were run separately. P50 produced **0 raw alarms, 0 temporal alarms, 0 layer
alarms** on all three. They are **not** used for training or in the public benchmark; licence is
proprietary stock, test-only.
