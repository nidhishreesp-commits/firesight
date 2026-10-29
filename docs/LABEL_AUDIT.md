# Label audit - P50 false positives (2026-10-28)

Purpose: find out how many of the false positives are actually **unlabelled fire/smoke**
(label noise), without touching any benchmark label.

## What was built

`scripts/fp_label_audit.py` produces, in `build/fp_audit/`:

| file | contents |
|---|---|
| `audit_index.csv` | one row per image: set, id, path, detector label+confidence, Florence-2 caption, provisional label-noise flag |
| `sheets/sheet_01..07.jpg` | visual grids, 6 x 5 tiles, detector box drawn + caption under each image |

Coverage:

- **all 88 false positives** from the taxonomy (83 `val_report` + 5 `bench_R DEV`) - verified: 0 missing;
- **top 200 highest-confidence detections on `val_report` negatives** (a superset of the 83, so the
  extra 117 are near-threshold candidates that did not alarm);
- total **205 images**.

## Provisional label-noise estimate

A keyword pass over the captions (`fire|flame|burning|smoke|...`) flags **17/205 = 8.3%**.
That number is an **upper bound**, because the keywords also match legitimate hard negatives
(chimneys, cooling towers) and captions that mention fire without showing an active fire.

Manual triage of the 17 flags suggests roughly **4-6 plausible true label-noise images**:

| image | caption evidence | likely |
|---|---|---|
| `dfire_val_AoF03113` | "a large fire burning on the right side of the road" | **real fire, labelled negative** |
| `fasdd_val_..._CV032004` | "a hand holding a lighter with a blue flame" | controlled flame (positive in our taxonomy), labelled negative |
| `benchR:73181771:Sunset_in_Thick_Smoke_Reno...` | filename + caption: sunset in thick smoke | smoke present; bench label says negative |
| `fasdd_val_..._CV005220` | factory chimney emitting white smoke | hard negative (industrial plume), **not** label noise |
| `fasdd_val_..._CV013281` | nuclear cooling tower | steam, hard negative, **not** label noise |
| `fasdd_val_..._CV014684` | "roof ... damaged by a fire" | no active fire; probably fine |
| others flagged (chandelier, portrait, game screenshot, landfill, fire trucks) | keyword false positives | **not** label noise |

**No benchmark labels were edited.** The `bench_R` rows are reported only; changing them would
require a new lock and would invalidate the recorded hashes.

## How to review

Open `build/fp_audit/sheets/sheet_01.jpg` ... `sheet_07.jpg` (7 images, 30 tiles each).
For each tile, the red box is the detection and the text under it is the Florence-2 caption.
Reply with the ids that are **wrong** (i.e. actually show fire/smoke, or are mislabelled for
another reason); everything else is accepted.

## Why this matters for the new plan

If ~5/205 (2.4%) of high-confidence negative detections are genuinely fire/smoke, then part of
the measured FPR is label noise rather than model error. That must be quantified before any
further training or before quoting an FPR target.
