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

## Decision (updated after Sample_videos + 20 hard clips)

**Do not lock Salah as the base model.** Salah wins on the 42 public clips, but **P50 wins on
the user's real Sample_videos and on the 20 hard real clips**:

| set | P50 accuracy | Salah accuracy | P50 FPR | Salah FPR |
|---|---:|---:|---:|---:|
| 42 public clips | 64.3% | **73.8%** | 22.7% | **9.1%** |
| Sample_videos (11) | **90.9%** | 81.8% | **11.1%** | 22.2% |
| 20 hard real clips | **85.0%** | 65.0% | **6.3%** | 31.3% |

On `Sample_videos`, P50's only false alarm is the drone/sky clip (which the user says is not a
real deployment case); Salah false-alarms on construction and grinding sparks. On the 20 hard
clips, Salah false-alarms on neon, police/vehicle lights, construction and fire aftermath;
P50 false-alarms only on neon.

**Conclusion: P50 remains the base model. Salah stays as a secondary candidate, not the lock.**
Salah is small (20 MB), MIT-licensed, CPU-usable and useful as a second opinion, but it is not
more accurate on the real/industrial negatives that matter.

## Ensemble result

P50+Salah was tested on the same 42 clips with several fusion rules. Full results:
`docs/ENSEMBLE_VIDEO_EVAL.md`.

- **On the 42 public clips only, Salah alone is the best single model**: 73.8% clip accuracy, 55% recall, 9.1% FPR. It loses to P50 on Sample_videos and the 20 hard clips.
- **Intersect** is the low-false-alarm option: 90.9% precision, 4.6% FPR, but 50% recall.
- **Every ensemble still has 0/5 smoke recall.** The ensemble does not fix the smoke blocker.
- Counsel: do not deploy Salah autonomously; P50 remains the base model after the real-video tests.

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
3. `Sample_videos` was used for evaluation at the user's request; it must not be used for training or threshold tuning.
4. Do not claim 90% from this 42-clip result.
