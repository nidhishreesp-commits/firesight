# ACCEPTANCE_PILOT.md — proposed gates for pilot acceptance

Status: **proposal for agreement with the lead. None of these are met yet.** No model is
shipped. Numbers marked M are measured, P are proposed.

Rules: SEALED is not read again until a final candidate is chosen; every read is logged. The 11
`Sample_videos` remain the final held-out check. Thresholds are chosen on val_cal/val_mini only.

## 1. Image-level gates (bench_R and independent negatives)

Proposed starting points; the lead may change them. “Strict policy” = uncontrolled fire + smoke;
controlled/spark/screen are suppressed.

| gate | dashboard (human review) | dispatch (autonomous alarm) |
|---|---:|---:|
| Recall at FPR <= 2% per negative | >= 85% | >= 95% |
| Strict-policy recall | >= 85% | >= 95% |
| Precision at the operating point | >= 70% | >= 90% |
| FPR per negative image | <= 3.0% | <= 1.0% |
| Fire recall / smoke recall | >= 85% / >= 75% | >= 95% / >= 90% |
| Per-cause FPR (each cause with >=30 negatives) | <= 8% | <= 3% |
| Zero missed uncontrolled fire in the reviewed DEV/independent set | no | yes |

Current P50 (M): image recall **56.3%**, strict-policy **69.9%**, FPR **2.10%**, precision
**88.4%**. It **fails** the proposed recall gates and is only a shadow/assistive candidate.
E1 is not eligible until the full run and Gate 2 are complete.

## 2. Event-level gates (pilot cameras)

Measured by `scripts/event_eval.py` on licence-clean clips with ground-truth onset.

| gate | dashboard | dispatch |
|---|---:|---:|
| Recall within 10 s of onset | >= 85% | >= 95% |
| Recall within 30 s of onset | >= 95% | >= 99% |
| Time-to-detection median | <= 10 s | <= 5 s |
| Time-to-detection p90 | <= 30 s | <= 15 s |
| False alarms per camera-hour | <= 0.05 | <= 0.005 |
| False alarms per camera-week | <= 8 | <= 1 |
| Non-fire hours in the event set | >= 20 h | >= 100 h |

The event set must include day/night, haze/cloud, rain, traffic/street lights, construction,
screens, reflections and warm lights. No event clip is used for training or threshold tuning.
The current event set is **not built**; these gates are currently unmeasured.

## 3. Shadow-mode requirements

Before any autonomous dispatch:

1. **Duration and coverage:** at least 4 weeks and >= 5 cameras, >= 500 camera-hours, including
   at least 2 night cameras and 2 high-mounted cameras.
2. **No autonomy:** every alert goes to the review queue; no dispatch action. Log frame, camera,
   timestamp, confidence, model version and reviewer decision.
3. **Event review:** every alert is reviewed; every reviewed real fire is labelled with onset and
   outcome. Uncontrolled fires missed by the model must be identified from the review queue and
   incident logs, not only from model alerts.
4. **Measured gates:** image-level gates on bench_R/independent negatives **and** event-level
   gates on pilot data, with Wilson/bootstrap CIs; per-camera and per-week breakdowns.
5. **False-alarm budget:** report false alarms per camera-week separately for dashboard and
   dispatch; no camera may exceed 2x the site budget without investigation.
6. **Workload:** reviewer alerts per camera-week and minutes per alert; the dashboard gate must
   be met with the real review staffing.
7. **Stability:** no material drift across weeks; retraining only through the gated E0/E1 loop,
   with SEALED untouched until a final candidate.
8. **Rollback:** if any gate fails, remain in shadow mode; no autonomous alarm.

## 4. What must be true before the first autonomous alarm

- [ ] Full-run model (or E1) passes image-level gates on bench_R and independent negatives.
- [ ] Event-level gates pass on >=30 fire clips and >=100 non-fire hours (dispatch tier).
- [ ] Shadow mode passes all gates above for >=4 weeks / >=500 camera-hours.
- [ ] Verifier and video layer measured on pilot data, with latency and false-alarm budget.
- [ ] One final SEALED read for the chosen candidate, logged.
- [ ] Lead sign-off on the exact numbers.

Until then: **shadow/assistive only.**
