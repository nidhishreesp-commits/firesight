# SalahYOLOv26: review and go/no-go

## Model kept locally

- Weights: `models/external/salah_yolov26_best.pt` (20,301,317 bytes)
- Source: `SalahALHaismawi/yolov26-fire-detection` on Hugging Face
- Licence: MIT
- Classes: `0 fire`, `1 other`, `2 smoke`
- Thresholds used: fire `0.51`, smoke `0.57`
- File is local only and is gitignored; it is not committed.

## 42-clip public video result

Same 42 curated Wikimedia clips, 2 fps, temporal + video layer.

| model | TP | FP | TN | FN | precision | recall | FPR | clip accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P50 | 10 | 6 | 16 | 10 | 62.5% | 50.0% | 27.3% | 61.9% |
| **SalahYOLOv26** | 11 | 3 | 19 | 9 | **78.6%** | **55.0%** | **13.6%** | **71.4%** |

Per category, Salah layer alerts: fire **11/15**, smoke **0/5**, cloud/haze **3/10 false alarms**,
non-fire **0/12 false alarms**.

Raw Salah (before layer): precision 55.0%, recall 55.0%, FPR 40.9%, accuracy 57.1%.

Three proprietary Downloads stock clips (non-fire, test-only): **0 alerts**.

## Bench_R image result (previous full run)

Salah: accuracy 82.2%, precision 77.6%, recall 52.46%, FPR 4.30%.
P50: accuracy 85.50%, precision 88.40%, recall 56.34%, FPR 2.10%.

So Salah wins on this small video set, but loses to P50 on the 1,000-image bench.
The 71.4% is **clip-level accuracy on 42 deliberately hard clips**, not a general
real-life accuracy estimate. Its recall CI is wide (55% [34.2%, 74.2%]).

## Decision

**Do not replace P50 with Salah as the sole production detector.** 71.4% clip accuracy,
0/5 smoke-only recall after the layer, and 13.6% FPR is not enough for autonomous real-life
alarm dispatch. A single public model winning by 9.5 points on 37 unique scenes is not a
production gate.

**Keep Salah as a candidate / second opinion.** It is small (20 MB), MIT-licensed, CPU-usable,
and its errors differ from P50: it catches more fire, has fewer cloud/non-fire layer false
alarms, but still misses every smoke-only clip. A P50+Salah ensemble is worth testing before
discarding either model.

## Ensemble result

P50+Salah was tested on the same 42 clips with several fusion rules. Full results:
`docs/ENSEMBLE_VIDEO_EVAL.md`.

- **Salah alone remains the best single model**: 73.8% clip accuracy, 55% recall, 9.1% FPR.
- **Intersect** is the low-false-alarm option: 90.9% precision, 4.6% FPR, but 50% recall.
- **Every ensemble still has 0/5 smoke recall.** The ensemble does not fix the smoke blocker.
- Counsel: keep Salah as the leading single candidate, but do not deploy it autonomously until
  smoke recall is fixed.

## Annotated videos for review

Open this gallery in a browser:

`build/video_eval/review_salah.html`

- 42 public annotated clips: `build/video_eval/results_salah_unique/annotated/`
- 3 stock annotated clips: `build/video_eval/downloads_salah/annotated/`
- Contact sheet: `build/video_eval/results_salah_unique/contact_sheet.jpg`
- Per-clip table: `build/video_eval/results_salah_unique/per_clip.csv`

## Next gate if we continue with Salah

1. Run a P50+Salah ensemble on the same 42 clips; require better smoke recall without raising
   false alarms.
2. Expand to 40+ event clips with real onset times.
3. Do not run/tune on `Sample_videos`; it stays the final held-out check.
4. Do not claim 90% from this 42-clip result.
