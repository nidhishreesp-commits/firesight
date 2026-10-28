# Hard-negative collection master plan

Status: **pilot collection in progress.** Nothing enters a training run until the
review gates in section 7 pass.

## 1. Goal

Build a licence-clean, category-labelled hard-negative pool for the detector so
that clouds, sunset/orange sky, fog/haze, dust/plaster, reflections/glare, vehicle
lights, neon, PPE and the other lookalikes stop being scored as fire/smoke.

Non-goals:

- no Google/Bing image scraping (ToS + copyright; forbidden by
  `hard_negatives/CATALOG.md`);
- no bench_R or testbench100 images in training;
- no synthetic images unless explicitly marked and licensed.

## 2. Why not "3,000 per category" blindly

There are 23 review categories. 23 x 3,000 = **69,000 hard negatives**, more than
the entire 58,985-image training pool. That would:

- swamp the positives and reduce fire/smoke recall;
- spend days downloading and reviewing images whose marginal value is near zero;
- fail for categories where licensed supply is thin (plaster dust, real rain on
  lens, steam, screens).

Hard negatives are new *information*, not volume. The pilot showed extra copies
of the same data add nothing (25% -> 50% = +0.7 pp, `REPORT.md:646`).

Tiered targets instead:

| Tier | Families | Target / family | Why |
|---|---|---:|---|
| 1 | clouds, fog_haze_mist, dust_construction_plaster, reflections_lens_flare_glare, sunset_orange_sky | 3,000 | measured FP families on bench_R and the hardest to separate from smoke/fire |
| 2 | vehicle_lights_streetlights, red_orange_neon_lights, orange_ppe_vests_cones, exhaust, snow, overexposed_white, steam, chandelier_stage_lights, flowers_autumn_leaves | 1,000 | fire-like or smoke-like, lower FP counts, still important |
| 3 | camera_artifacts, ordinary_indoor, ordinary_streets_warehouses_offices, ordinary_other | 500 | context/easy negatives, already well covered locally |

Total ceiling: **27,500**, of which only ~15,000 need to be newly downloaded
(the rest can come from own video and train-negative mining). Phase the tiers:
pilot 20/family -> 500/family -> 3,000 only where residual FPs remain after the
first fine-tune.

## 3. Sources (licence-safe, in priority order)

1. **Own video captures** (`C:\Users\Nidhi\My projects\Sample_videos`): construction
   dust/PPE, vehicles/number plates, indoor retail, aerial. Licence: proprietary
   internal. Best domain match; no rights risk.
2. **Train-negative mining** (existing D-Fire + FASDD train negatives): already
   licensed and already in the training split. Mine the ones the detector scores
   highly, CLIP-categorise, then oversample. No new download, no bench leakage.
3. **Wikimedia Commons API**: CC0, public domain, CC BY, CC BY-SA only; reject
   NC/ND. This is the main new-image source for clouds, sunset, fog, steam,
   exhaust, lights, flowers, snow, camera artifacts.
4. **Openverse** (same licence filter) as a fallback if the API responds; currently
   times out from this machine.
5. **Local pools already on disk** (`fire_hardneg`, `fire-assets-v3/hardneg`,
   `ppe-bg-v2`): own/proprietary and unclear-rights web images; flagged, never
   uploaded to Kaggle without a rights review.

Explicitly rejected: Google Images, FireViewer corpus, HF "40K", Roboflow
projects, CC BY-NC datasets.

## 4. Category map and queries

Wikimedia queries per family (first pass, will be tuned by yield):

| Category | Queries |
|---|---|
| sunset_orange_sky | sunset, sunrise, orange sky, red sky |
| clouds | clouds, cumulus clouds, cloudscape, overcast sky |
| fog_haze_mist | fog, mist, haze, foggy landscape |
| steam | steam, water vapour, kettle steam, steam cooking |
| exhaust | exhaust smoke, chimney smoke, cooling tower, factory chimney |
| dust_construction_plaster | construction dust, plaster dust, sawdust, sandblasting, dust cloud |
| snow | snowfall, blizzard, snow landscape, blowing snow |
| overexposed_white | overexposed photo, white sky, bright sky, white wall |
| reflections_lens_flare_glare | lens flare, sun glare, water reflection, glass reflection |
| red_orange_neon_lights | neon sign, red neon sign, orange neon light, red lamp |
| vehicle_lights_streetlights | car headlights, street lights at night, brake lights, tail lights |
| chandelier_stage_lights | chandelier, stage lighting, concert stage lights, festival lights |
| orange_ppe_vests_cones | orange safety vest, traffic cone, high visibility vest, construction helmet |
| flowers_autumn_leaves | orange flowers, red flowers, autumn leaves, autumn foliage |
| camera_artifacts | motion blur, lens dirt, rain on lens, sensor noise |
| ordinary_indoor | office interior, warehouse interior, indoor hallway |
| ordinary_streets_warehouses_offices | street scene, warehouse exterior, office building |
| ordinary_other | landscape, park |

Video mapping (own footage):

| Clip | Candidate categories |
|---|---|
| ppe_1..ppe_4 | dust_construction_plaster, orange_ppe_vests_cones, ordinary_other |
| Car, number_plate | vehicle_lights_streetlights, ordinary_streets_warehouses_offices |
| Attendance, theft | ordinary_indoor |
| drone | ordinary_other, clouds |

## 5. Labels

Hard negatives are **background images with empty YOLO labels**. There is no box
to draw and no text prompt; the detector learns "this whole scene is not fire or
smoke" from empty-label examples. Class attribution is metadata only, used for
sampling and reporting.

An image is rejected as a negative if review shows real fire/smoke.

## 6. AI-assisted QA (triage, not truth)

For every candidate:

1. **CLIP category check** (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, cached):
   score against the 18 negative prompts; keep only if the requested family is
   the top-1 (video: top-1 among the clip's allowed families). CLIP is known
   noisy on fire-adjacent scenes, so it is a filter, not a label.
2. **Detector hardness** (`models/best_fire_v3.onnx`): record max fire/smoke
   confidence. This is how we find *hard* negatives instead of easy ones.
3. **Quarantine**: `max_conf >= 0.80` is marked `suspect_positive` and shown in
   the contact sheet; it never enters training without a human check.
4. **Human spot-check**: contact sheets per family; 100% review of quarantined
   images and >=10% random review of the rest. Reviewer writes to
   `data/hard_negatives/reviews.jsonl`.
5. **Dedupe**: SHA-256 exact plus dHash Hamming <= 4 against the existing pool,
   bench_R, testbench100 and TEST.

## 7. Gates before fine-tuning

- [ ] 0 ID/path/pHash overlap with bench_R, testbench100, TEST (report written).
- [ ] Every family >= pilot target with >=90% CLIP top-1 agreement.
- [ ] Quarantine reviewed; real fire/smoke removed.
- [ ] Licence table complete; no unclear-rights image is in the Kaggle package.
- [ ] Holdout (10%) never trained; calibration stays on `val_mini + val_flames`.
- [ ] Fine-tune comparison at equal FPR on val + bench_R DEV; SEALED untouched.

## 8. Output layout

```
data/hard_negatives/
  images/<category>/<id>.jpg          newly collected
  labels/<category>/<id>.txt          empty labels
  manifest.jsonl                      one row per candidate (selected/suspect/rejected)
  attribution.jsonl                   Wikimedia/capture attribution
  SUMMARY.md                          counts, licence mix, hardness, quarantine
  contact_sheets/<category>.jpg       visual review
  lists/train.txt                     paths for fine-tuning
  lists/holdout.txt                   10%, never trained
  leaked_report.json                  overlap check vs bench_R/testbench100/TEST
```

Train-negative-mined rows keep `reuse: true` and point at the existing processed
image; the fine-tune builder oversamples them rather than copying.

## 9. Commands

Pilot (what runs now):

```powershell
python scripts/collect_hard_negatives.py --sources wikimedia,video `
  --categories clouds,fog_haze_mist,dust_construction_plaster,reflections_lens_flare_glare,sunset_orange_sky `
  --per-category 20 --candidates-per-category 80 --contact-sheets
```

Scale one tier:

```powershell
python scripts/collect_hard_negatives.py --sources wikimedia,video,train_mine `
  --tier 1 --per-category 3000 --candidates-per-category 9000
```

Mine existing train negatives only (no download):

```powershell
python scripts/collect_hard_negatives.py --sources train_mine --per-category 1000
```

## 10. Fine-tune recipe (after the gates)

- Start from P25/P50, not yolo11s from scratch.
- Build the pool as positives + 10-20% hard negatives (oversample Tier-1 by 2x).
- Fine-tune 10-20 epochs at 640, low LR (~1e-3); 960 only if small/distant smoke
  is still the dominant miss.
- Recalibrate on `val_mini + val_flames`; iterate on bench_R DEV; one SEALED read
  at the milestone end.
- Re-run the verifier as a separate, cheap experiment; keep temporal N-of-M.
