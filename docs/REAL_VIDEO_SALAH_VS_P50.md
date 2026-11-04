# Salah vs P50 on real Sample_videos and 20 hard clips

This is the check requested after seeing Salah's 42-public-clip result. It uses the user's
`C:\Users\Nidhi\My projects\Sample_videos` and 20 additional real public-licence clips focused on
hard negatives: red lights, neon, vehicle/police lights, construction, welding sparks, kitchen,
sunset, steam, aftermath, campfires and a smoke plume.

**Important:** `Sample_videos` was used for evaluation only. It was not used for training or
threshold tuning. It is no longer a clean held-out set.

## Sample_videos: 11 clips (2 fire, 9 non-fire)

| model | TP | FP | FN | precision | recall | FPR | accuracy | false alarms |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **P50** | 2 | 1 | 0 | 66.7% | 100% | 11.1% | **90.9%** | drone/sky |
| **Salah** | 2 | 2 | 0 | 50.0% | 100% | 22.2% | 81.8% | ppe_3 construction, ppe_4 grinding sparks |

- Both detect `fire.mp4` (indoor/garage fire) and `fire_2.mp4` (vehicle fire).
- P50's only false alarm is the drone clip with sky; the user says sky is not a real deployment case.
  Excluding that, P50 has **0 false alarms** on the remaining 8 negatives.
- Salah false-alarms on construction/PPE and on grinding sparks. Those are directly relevant
  industrial lookalikes.
- Red light/beard: there is no beard clip in `Sample_videos`; no suitable beard video was found
  in the quick public search, so the beard case is **not tested**. Red-light clips are in the
  20-clip extra set; both models gave **0 alerts** on them.

## 20 additional real clips (4 fire/smoke, 16 non-fire)

| model | TP | FP | FN | precision | recall | FPR | accuracy | false alarms |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **P50** | 2 | 1 | 2 | 66.7% | 50.0% | 6.3% | **85.0%** | neon sign |
| **Salah** | 2 | 5 | 2 | 28.6% | 50.0% | 31.3% | 65.0% | neon, police/vehicle lights, construction x2, fire aftermath |

- Both detect the two campfires.
- Both miss the building-fire clip (no visible flame in the sampled frames) and the smoke plume
  (Salah max smoke 0.475 below its 0.57 threshold; P50 also misses after the layer).
- The 20-clip set does **not** include a usable beard video.
- P50 false-alarms once, on a neon sign. Salah false-alarms five times, including police lights
  and construction. These are exactly the "random stuff" cases that matter.

## Combined 31 real clips

| model | TP | FP | FN | precision | recall | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **P50** | 4 | 2 | 2 | 66.7% | 66.7% | 8.0% | **87.1%** |
| **Salah** | 4 | 7 | 2 | 36.4% | 66.7% | 28.0% | 71.0% |

## Verdict

**Salah is not better than P50 overall.** It wins only on the 42 public clips (73.8% vs 64.3%).
On the user's actual `Sample_videos` and on 20 hard real clips, **P50 wins clearly**:

- Sample_videos: P50 90.9% vs Salah 81.8%.
- 20 hard clips: P50 85.0% vs Salah 65.0%.
- Combined: P50 87.1% vs Salah 71.0%.
- P50 has fewer false alarms on both sets.

**Therefore: do not lock Salah as the base model.** Lock **P50** as the base; keep Salah as a
secondary candidate/ensemble member. Salah's industrial false alarms (construction, sparks,
police lights, neon, aftermath) are worse than P50's on the exact cases the user cares about.

If the user still wants Salah, it needs fine-tuning with industrial hard negatives and smoke;
but the current evidence says P50 is the better base. Neither model is production-ready:
- smoke-only recall is still 0/5 on the 42-clip public set;
- the building fire and smoke plume were missed on the 20-clip set;
- `Sample_videos` is only 11 clips and the extra set is 20 clips; confidence intervals are wide.

## Annotated review

- Browser review, Sample_videos: `build/sample_videos_review/review_sample.html`
- Browser review, 20 hard clips: `build/sample_videos_review/review_extra20.html`

- Sample_videos, Salah: `build/sample_videos_review/salah/annotated/`
- Sample_videos, P50: `build/sample_videos_review/p50/annotated/`
- 20 extra, Salah: `build/sample_videos_review/extra20_salah/annotated/`
- 20 extra, P50: `build/sample_videos_review/extra20_p50/annotated/`
- Contact sheets: `build/sample_videos_review/contact_sheet.jpg` (Sample_videos),
  `build/sample_videos_review/extra20_contact_sheet.jpg` (20 extra)

## Next

Do not spend time on a model switch. Keep P50 as base:
1. Add the industrial false alarms (construction, sparks, neon, police lights, aftermath) to the
   hard-negative set.
2. Fix smoke-only recall.
3. Fine-tune P50 (or a P50+Salah ensemble) on those hard negatives; then re-test on new clips.
