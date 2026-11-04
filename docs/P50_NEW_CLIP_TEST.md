# P50 on the supplied clips + fresh fire/smoke

Thresholds: P50 fire 0.29 / smoke 0.33, 2 fps, temporal + video layer. Videos are evaluated at
clip level. `Sample_videos` was not used here.

## 9 supplied clips — all non-fire

Traffic day, traffic getty, traffic night, crowd/stage, four industrial clips (including red
glow/machinery), and FRP construction.

**Result: 0 raw alerts, 0 temporal alerts, 0 layer alerts on all 9.**

## 5 additional supplied clips — all non-fire

City skyline at night, bearded man close-up, sunrise over clouds, red/yellow/green traffic light,
ambulance/police red-blue lights.

**Result: 0 raw alerts, 0 temporal alerts, 0 layer alerts on all 5.**

The specific cases the user cared about:

| clip | max fire | max smoke | raw | layer |
|---|---:|---:|---|---|
| bearded man | 0.149 | 0.176 | no | no |
| red traffic light | 0.000 | 0.000 | no | no |
| police/ambulance lights | 0.052 | 0.145 | no | no |
| industrial red glow | 0.000 | 0.000 | no | no |

**14 supplied non-fire clips: 0 false alarms.** This is a clean pass on the false-alarm cases the
user named.

## Fresh fire/smoke clips

Pulled from Wikimedia Commons (CC/PD), plus one unused public clip:

| clip | label | raw | temporal | layer | max fire | max smoke |
|---|---|---|---:|---:|---:|---:|
| Notre Dame fire/smoke | fire | yes | yes | **yes** | 0.631 | 0.857 |
| Burning building, St Petersburg | fire | yes | no | no | 0.066 | 0.340 |
| Burning rubbish from shrubs | fire | yes | no | no | 0.199 | 0.544 |

Raw recall **3/3**, but temporal + video layer keeps only **1/3**. The two missed clips are
smoke-dominant; the layer suppresses them.

Additional non-fire pulls:

| clip | raw | layer | notes |
|---|---|---:|---|
| Ashburton fire/ambulance scene | yes | no | fire trucks, no visible fire; suppressed |
| Chimney Bluffs landscape | no | no | — |
| Sprague train | yes | no | suppressed |

So among 17 non-fire clips total (14 supplied + 3 pulled), the final layer has **0 false alarms**.
Two raw false alarms (fire trucks, train) were suppressed by the temporal/video layer.

## What this means

- **P50 passes the false-alarm test on the supplied clips**, including beard, red light, police
  lights, industrial red glow, city, sunrise and construction. That was the main concern.
- **Smoke-only remains the weakness.** P50 raw-detects all 3 fresh fire/smoke clips, but the
  temporal/video layer drops 2 of 3 smoke-dominant clips.
- The fire/smoke sample is tiny (3 clips), so treat recall as a signal, not a final number.
- Do not call P50 production-ready until smoke-only recall is fixed. But the false-alarm
  behavior on the supplied negatives is good enough to proceed to local testing.

## Review

- Browser review: `build/sample_videos_review/review_p50_newclips.html`
- 9 supplied: `build/sample_videos_review/new9_p50/annotated/`
- 5 + fire/smoke: `build/sample_videos_review/extra5_fire_p50/annotated/`

## Next

1. Add 10-20 more real smoke-only clips (not fire-with-smoke).
2. Test a smoke-specific temporal rule: longer window, do not reject static low-confidence plumes.
3. Re-test P50 layer recall on new smoke clips and confirm false alarms stay at 0 on the 14
   supplied negatives.
4. Then local testing.
