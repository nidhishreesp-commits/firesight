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

---

## Symmetric audit (added 2026-10-28)

To avoid only fixing the model's mistakes, two more sets were built with
`scripts/symmetric_audit.py` (green = ground-truth box, red = P50 detection):

| group | images | source | sheets |
|---|---:|---|---|
| (A) random val negatives P50 did **not** flag | 200 of 2,561 | `build/audit_symmetric/sheets_negatives/` (7 sheets) | |
| (B) val positives P50 **missed** | **174 (all of them)** | `build/audit_symmetric/sheets_missed/` (6 sheets) | |

Missed-positive composition: **smoke-only 125, fire-only 33, fire+smoke 16**. Of the 174,
17 have no detection at all (max_conf = 0), 36 are below 0.10, 42 are 0.10-0.20, 79 are
0.20-threshold. Most are **small/distant smoke** from aerial or high-vantage cameras.

### Provisional label-noise rates (keyword pass over Florence-2 captions)

| group | keyword fire/smoke | share | interpretation |
|---|---:|---:|---|
| flagged FPs (205) | 17 | 8.3% | upper bound; manual triage suggests ~4-6 real cases (~2.4%) |
| random unflagged negatives (200) | 2 | **1.0%** | base-rate label noise among negatives is low |
| missed positives (174) | 104 | 59.8% | sanity check, not label noise: captions confirm fire/smoke in 60%; the rest are likely caption misses on tiny/distant smoke |

### Effect on the headline numbers (sensitivity, val_report)

| assumption | mislabelled FPs | FPR (was 3.14%) |
|---|---:|---:|
| all 17/205 keyword hits are positives (upper bound) | ~6.9 | **2.88%** |
| plausible 5/205 | ~2.0 | **3.06%** |
| random-negative base rate 2/200 | ~0.8 | **3.11%** |

Recall would move only slightly upward (each missed positive that is actually a negative is
removed from the denominator): a 10% label-noise rate among the 174 missed positives would take
recall from **92.97% to ~93.66%**; a 20% rate to ~94.35%.

**Conclusion so far:** label noise is real but small among negatives (1-3% of the FPs), so the
measured FPR is mostly genuine model error, not annotation error. The dominant recall loss is
**small/distant smoke**, not mislabelling. The reviewer's ids will replace these provisional
estimates; every metric will then be reported with original and corrected labels via
`scripts/label_corrections.py` (side file `label_corrections.csv`, no label files edited).
