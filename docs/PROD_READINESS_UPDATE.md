# P50 production-readiness update (lead-facing)

## Bottom line

**P50 is not a 90%-accuracy fire detector on the hard bench, and it should not be pushed as an
autonomous alarm.** Its measured bench_R image accuracy is **85.5%** (max ~86.5% across the
threshold sweep); recall at 2% FPR is **56.3%**. Threshold tuning cannot reach 90% accuracy.
The honest release options are: (a) a **shadow/assistive** P50 candidate now, or (b) wait for the
rebalanced F1 retrain + event-level validation.

No SEALED read was made for this decision. The P50 baseline SEALED read is already logged.

## Measured facts

| metric | P50 |
|---|---:|
| bench_R all-image accuracy | **85.50%** |
| best image accuracy across global threshold sweep | **~86.5%** |
| recall at 2% FPR | **56.34%** |
| precision | **88.40%** |
| FPR/negative | **2.10%** |
| strict-policy recall (uncontrolled + smoke) | **69.87%** |
| fire / smoke recall | **60.11% / 47.87%** |
| DEV / SEALED recall | 58.82% / 55.28% |
| DEV / SEALED FPR | 2.00% / 2.14% |

To reach 90% image accuracy, P50 needs ~45 fewer errors (currently 21 FP + 124 FN), which means
recall would have to rise to roughly 72% at the same FPR. That is a data/model problem, not a
threshold problem.

## Video test (11 clips, final held-out set)

| system | correct clips | notes |
|---|---:|---|
| P50 raw alerts | 10/11 | false alarm on drone (smoke 0.63) |
| P50 + temporal + video layer | 10/11 | same drone false alarm |
| P50 + verifier v4 + temporal + video layer | **11/11** | 2/2 fire, 0/9 nonfire; verifier v4 is not integrated in production |

11/11 gives a Wilson 95% lower bound of only **~74%**. To support a 90% claim with zero errors you
need about **35+ clips**; we have 11. This is not enough evidence for a 90% production claim.

## Release decision

**Do not push autonomous.** The prepared shadow config is:

`configs/prod_candidate_p50_shadow.yaml`

- P50 ONNX SHA256 `4838218c…708849`
- thresholds fire 0.29 / smoke 0.33
- temporal N-of-M enabled
- verifier disabled (v4 is a separate fallback, not integrated)
- **dashboard escalation only**, no webhook/siren

This is suitable for **shadow-mode capture and reviewer triage**, not dispatch.

## What it would take to claim 90%

1. Rebalanced F1 positives (~60% indoor/industrial, 25% filtered thin/distant smoke, 15%
   controlled flames) + indoor/industrial hard negatives.
2. Event set: >=30-50 fire sequences with onset times and >=20-100 h non-fire, split by
   sequence/camera; no event sequence in training.
3. Shadow pilot: >=4 weeks / >=5 cameras / >=500 camera-hours, with measured event recall and
   false alarms per camera-week.
4. One final SEALED read for the chosen candidate, logged.
5. Lead sign-off on the exact image and event gates in `docs/ACCEPTANCE_PILOT.md`.

## Lead update (draft)

> P50 is the best measured detector, but it is not a 90%-recall model: bench_R recall is 56.3% at
> 2.1% FPR (accuracy 85.5%; max ~86.5%). On 11 held-out video clips it is 10/11 correct with the
> temporal/video layer; the only error is the drone smoke false alarm. The v4 cascade is 11/11 but
> is not production-integrated and has only 2.9 min of non-fire video. I recommend a shadow-mode
> release of P50 + temporal (dashboard only) now, and hold autonomous dispatch until the rebalanced
> F1 retrain and the event-level pilot gates pass. Pushing P50 as a 90% autonomous alarm would not
> be supported by the measurements.

## Rollback / monitoring if shadow is approved

- Shadow config only; alerts go to dashboard/review queue, no dispatch.
- Log every alert with frame, camera, confidence, model hash and reviewer decision.
- Weekly false alarms per camera and per cause; any camera >2x budget is investigated.
- Rollback is immediate by reverting to the current default config; P50 is not the default yet.
