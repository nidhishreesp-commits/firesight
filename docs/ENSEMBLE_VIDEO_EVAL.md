# P50 + Salah ensemble on the 42 public clips

Method: ran both models on the same frames at 2 fps, cached detections, then replayed each
fusion rule through the **same temporal confirmer and video false-alarm filter**. FPR is over
the 22 negative clips (10 cloud/haze + 12 non-fire), not only the 12 non-fire clips.

| fusion | TP | FP | FN | precision | recall | FPR | clip acc | FA/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P50 alone | 10 | 5 | 10 | 66.7% | 50.0% | 22.7% | 64.3% | 14.63 |
| **Salah alone** | 11 | 2 | 9 | 84.6% | **55.0%** | 9.1% | **73.8%** | 5.85 |
| union | 11 | 6 | 9 | 64.7% | 55.0% | 27.3% | 64.3% | 17.56 |
| split (Salah fire, P50 smoke) | 11 | 5 | 9 | 68.8% | 55.0% | 22.7% | 66.7% | 14.63 |
| split with fire union | 11 | 5 | 9 | 68.8% | 55.0% | 22.7% | 66.7% | 14.63 |
| **intersect** | 10 | **1** | 10 | **90.9%** | 50.0% | **4.6%** | **73.8%** | **2.93** |
| max-score, P50 thresholds | 11 | 7 | 9 | 61.1% | 55.0% | 31.8% | 61.9% | 20.48 |

Per-category layer alerts:

| fusion | fire | smoke | cloud/haze FA | non-fire FA |
|---|---:|---:|---:|---:|
| P50 | 10/15 | **0/5** | 4/10 | 1/12 |
| Salah | 11/15 | **0/5** | 2/10 | 0/12 |
| union | 11/15 | **0/5** | 5/10 | 1/12 |
| split | 11/15 | **0/5** | 4/10 | 1/12 |
| intersect | 10/15 | **0/5** | 1/10 | 0/12 |
| max-score | 11/15 | **0/5** | 6/10 | 1/12 |

## What this means

- **The ensemble does not fix smoke.** Every fusion rule has **0/5 smoke recall** after the
  temporal/video layer. The failure is before fusion: the smoke detections are not being
  confirmed/preserved by the current N-of-M + growth policy.
- **Union is worse**, not better: it adds false alarms without adding true positives.
- **Salah alone is the best single model here**: 73.8% clip accuracy, 55% recall, 9.1% FPR.
- **Intersect is the low-false-alarm option**: 90.9% precision, 4.6% FPR, 2.93 FA/hour, but it
  drops one fire and still has 0/5 smoke. It does not solve the product requirement.
- 42 clips include three versions of the same Dragon Bravo smoke footage. Deduping to 40 clips:
  Salah 77.5% accuracy / 61.1% recall / 9.1% FPR / **0/3 smoke**; intersect 77.5% / 55.6% /
  4.6% / **0/3 smoke**. The conclusion does not change.

## Decision

**Salah remains the best single candidate, but neither Salah nor the ensemble is production-ready.**
If the priority is fewer false alarms, intersect is the better ensemble. If the priority is
recall, Salah alone is better. For real life, **smoke recall is the blocker**, and it needs a
smoke-specific temporal/data fix, not model fusion.

## Annotated review

- Salah gallery: `build/video_eval/review_salah.html`
- Intersect gallery: `build/video_eval/review_intersect.html`
- Intersect videos: `build/video_eval/ensemble/intersect_annotated/`
- Per-clip data: `build/video_eval/ensemble/*_per_clip.csv`
- Comparison JSON: `build/video_eval/ensemble/comparison.json`

## Next smoke experiment

1. Diagnose why the 3 unique smoke scenes fail: N-of-M, track churn, or smoke-growth gate.
2. Test a smoke-specific policy on the same clips: longer window and/or growth gate only when
   confidence <0.60 (the current code rejects low-confidence static smoke).
3. Add more real smoke-only videos; do not tune on `Sample_videos`.
4. Re-run Salah alone + intersect with the new smoke policy, then move to 40+ event clips.
