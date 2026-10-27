# False-alarm budget and FPR definition (proposal — needs sign-off)

Status: **agreed 2026-10-27** (project owner). The FPR target is per **negative image**,
and the event-level ceilings below are adopted as operational budgets.

## 1. Which FPR definition

`bench_R` is negative-heavy: 716 negatives of 1,000 images. Two different numbers are
currently quoted for the same model, and they differ by ~30% relative:

| definition | cascade_v3 baseline | denominator |
|---|---|---|
| **FPR over negative images** (standard) | 22/716 = **3.07%** [2.04, 4.61] | 716 images with no fire and no smoke |
| FPR over all images | 22/1000 = **2.20%** | all 1,000 images |

**Proposal:** the target `≤ 0.01–0.02` refers to **FPR over negative images** — the
standard detection definition, and the one that does not move when bench composition
changes. Report FPR-over-all-images as a secondary line only, always labelled. Never
mix the two in a table without the denominator.

Implication: the current 3.07% is **above** the 1–2% target; 2.20% would look like it
is inside the band. The honest reading is that cascade_v3 does **not** meet the target
yet on either definition (2.20% > 2.00% too).

## 2. Event-level false-alarm ceiling

Image-level FPR does **not** convert into an event rate. A persistent false positive
(same tree, same reflection) fires on many consecutive frames, so the independent-frame
arithmetic badly over-counts, while a single-frame spike under-counts. Do not
extrapolate image FPR to camera-weeks; it is not identifiable from bench_R.

**Agreed ceilings:**

| alarm routing | ceiling | rationale |
|---|---|---|
| **Dashboard / review only** (level 1) | **≤ 1.0 false-alarm events per camera-week** | an operator can dismiss ~1 per week per camera without alarm fatigue |
| **Dispatch / notification** (level ≥ 2) | **≤ 0.1 false-alarm events per camera-week** | ≈ 1 dispatch-triggering false alarm per camera per 10 weeks; 100 cameras ≈ 10/week |

These are operational budgets, not measurements. They are stated per camera-week
because that is the unit operations teams staff against.

## 3. How it will be measured (not from bench_R)

bench_R is still images; it cannot produce event counts. Measurement protocol:

1. Pick 10 cameras across the main scenes (yard, indoor, gate, night).
2. Record **48 h continuous** per camera (720 h total).
3. Run the deployed policy end-to-end (detector → policy tier → temporal confirmation
   → alarm manager), log every `AlarmEvent`.
4. Count events with no real fire/smoke (reviewed from the saved clips), report
   **events per camera-week** with a Poisson 95% CI.
5. Report separately by tier (level 1 vs level ≥ 2) and by day/night.

Until that exists, event-level false-alarm rate is reported as **unmeasured**. Image-level
FPR remains the only measured number.

## 4. What is measured vs estimated (summary)

| quantity | status |
|---|---|
| image-level FPR (per negative image) | measured on bench_R |
| image-level FPR (per all images) | measured on bench_R |
| strict-policy uncontrolled recall | measured on bench_R |
| event-level false alarms per camera-week | **not measurable on bench_R — unmeasured** |
| electrical_arc / screen behaviour | **unmeasured** (0 bench_R images) |
