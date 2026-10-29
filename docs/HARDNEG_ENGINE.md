# Hard-negative engine (Stage A)

Status: **Stage A in progress. No model has been trained or tuned. TEST/SEALED are not used for
training or tuning.** This document is the durable record of sources, licences, composition,
leakage checks and review grids. Numbers marked **M** are measured on this machine; numbers
marked **E** are estimates. Nothing here is a claim about an unseen pilot camera.

Rules inherited from the programme and enforced by the scripts:

- `C:\Users\Nidhi\My projects\Sample_videos` is a **final held-out check only**. It is not mined,
  not trained on, and not tuned on by any Stage A script.
- No image is used if it overlaps P50's training set (`train_50`), `val_cal`, `val_report`,
  `TEST`, `bench_R` (DEV or SEALED) or `testbench100`; the gate checks SHA-256, pHash (Hamming
  <= 2) and id/path.
- No NC/ND or unclear-rights image is accepted. Wikimedia Commons rows carry their own licence,
  licence URL, artist and page URL.
- Public video is proposed here but **not downloaded** until the source/licence list is approved.
- Synthetic images are training-only, clearly tagged, and capped at 20% of hard negatives.

---

## 1. Complete cause taxonomy

The engine covers every lookalike named in the brief and every family in
`docs/FP_TAXONOMY.md` / `docs/HARD_NEGATIVE_COLLECTION.md`. The taxonomy lives in
`scripts/hardneg_causes.py` so the collector, miner and documentation cannot drift apart.

| # | cause | covers | held out of training? |
|---|---|---|---:|
| 1 | `sun_glare_sky` | sun/sky glare, lens flare, overexposed sky, person/vehicle with sun behind them | **yes** (cause holdout) |
| 2 | `sunset_sunrise` | sunset/sunrise, orange/red/yellow dusk sky | no |
| 3 | `warm_artificial_light` | street lamps, lanterns, warm bulbs, city lights | no |
| 4 | `traffic_light` | red/orange traffic lights, railway/pedestrian signals | **yes** (cause holdout) |
| 5 | `vehicle_lights` | headlights, brake/tail lights, light trails | no |
| 6 | `haze_cloud_fog_mist` | clouds, haze, fog, mist, overcast, hazy horizon | no |
| 7 | `dust_plaster` | construction dust, plaster dust, sawdust, sandblasting, dust cloud/storm | no |
| 8 | `screen_fire` | TV/monitor/phone showing fire or orange graphics, posters | no |
| 9 | `reflection_lens_flare` | water/wet-road/glass/metal reflection, lens flare/glare | no |
| 10 | `orange_ppe` | hi-vis orange vest, workwear, traffic cone, helmet | no |
| 11 | `skin_person` | portrait/selfie/close-up skin | no |
| 12 | `cave_warm_rock` | cave interior, red/orange rock | no |
| 13 | `smoke_steam_nonfire` | steam/water vapour (cooling tower, kettle, geyser, vent) with no fire | no |
| 14 | `orange_vegetation` | orange/red flowers, autumn leaves/foliage | no |
| 15 | `neon_lights` | red/orange neon signs and lamps | no |
| 16 | `stage_chandelier` | chandelier, stage/concert/festival lighting | no |
| 17 | `snow_white_overexposed` | snow/blizzard, white sky, overexposed white wall | no |
| 18 | `exhaust` | vehicle/diesel exhaust fumes, no fire | no |
| 19 | `camera_artifacts` | motion blur, lens dirt, rain on lens, sensor noise | no |
| 20 | `ordinary_indoor` | office/warehouse/hallway/shop interiors, no fire | no (easy) |
| 21 | `ordinary_outdoor` | street/building/car park/landscape/park, no fire | no (easy) |

Two causes (`traffic_light`, `sun_glare_sky`) are **never trained on**. They are a direct Stage D
generalisation test, as are the whole Open Images V7 source and a 10% random holdout.

---

## 2. Sources and licences

### 2.1 Accepted sources

| source | how acquired | licence recorded per image | use |
|---|---|---|---|
| **Wikimedia Commons** | Commons API, `filetype:bitmap`, 1280 px thumbnails | per-file `LicenseShortName`; only CC0 / public domain / CC BY / CC BY-SA; NC and ND rejected; artist, credit and page URL stored | training + holdout (10%) |
| **Open Images V7 validation** | S3 image URLs from the official validation list | image listing: **CC BY 2.0**; annotations: CC BY 4.0 (official statement); source URL stored; Google warns to verify each image | **held-out source only** (never trained) |
| **D-Fire / FASDD train negatives** | already-licensed local training data, P50's out-of-sample `train_100 - train_50` | D-Fire CC0-1.0; FASDD CC BY 4.0 (as recorded by `kaggle_package.py`) | free pool, training-only after leakage gate |
| **Synthetic derivatives** | ordinary non-fire Wikimedia scenes with glare/sunset/bright-sky/haze | base licence + "synthetic derivative; internal research" | training-only supplement, capped at 20% |
| **Own pilot footage** | not downloaded in Stage A; `Sample_videos` is final held-out only | proprietary internal | pilot only, never Stage A training |

### 2.2 Rejected sources

Google Images/Bing scraping, Roboflow projects with unclear rights, CC BY-NC / CC BY-ND,
HF "40K" and FireViewer (unclear rights), and any local web image without a recorded licence.

### 2.3 Public video proposals (not downloaded until approved)

Stage A step 3 asks for a source/licence list *before* download. These are the candidates; none
has been downloaded in this session.

| proposed source | content | licence position | decision |
|---|---|---|---|
| Wikimedia Commons video (`filetype:video`) | traffic, streets, sunset, haze, construction, indoor | per-file CC0/PD/CC BY/CC BY-SA; NC/ND filter; artist/credit recorded | **primary**; accept after per-file check |
| Internet Archive moving images | street/traffic/construction/sunset/haze | per-item `licenseurl`; accept CC0/PD/CC BY/CC BY-SA only | **primary fallback** |
| NOAA / NASA image & video galleries | sun, sky, clouds, haze, weather | U.S. federal public domain (verify per item) | accept for sun/sky/haze |
| Pexels / Pixabay | ordinary streets, construction, indoor, sunset | site licence (free commercial use, attribution not required; not a CC licence) | hold; legal review needed before training |
| YouTube CC BY channels | traffic/streets/construction | per-video CC BY; must download only the licensed video and keep the URL | hold; per-video check required |
| State DOT traffic cameras | ordinary traffic, night lights | varies by state; often public domain but not always | hold; per-camera check required |

Video protocol when approved: <= 20 frames/clip, pHash dedupe, P50 at conf >= 0.05, same
leakage gate as images, and a max of 20 frames per source clip so a single clip cannot dominate.

---

## 3. Free pool first (no download)

`scripts/mine_free_pool.py` runs frozen P50 on the **29,492 images in `train_100` that are not in
`train_50`** (the exact out-of-sample set). It caches detections at conf 0.01 and writes false
detections on empty-label images at max_conf >= 0.05. This is the cheapest, highest-domain-match
source and needs no new licence.

### Measured free-pool result (M)

| item | value |
|---|---:|
| out-of-sample images (`train_100 - train_50`) | 29,492 |
| out-of-sample **negatives** scored by P50 | 16,430 |
| false detections at max_conf >= 0.05 | **1,091** |
| hard at >= 0.10 | 703 |
| very hard at >= 0.30 | 229 |
| quarantine-level at >= 0.50 | 100 |
| by source (false >= 0.05) | FASDD 816, D-Fire 275 |
| P50 throughput / wall time | 7.9 img/s, 34.8 min (M) |

This is the strongest source: it is in-domain, already licensed, and out-of-sample by
construction. It is also the source that reproduces the old failure mode - its CLIP distribution
is dominated by dust and camera artefacts, not by the warm-light/traffic-light/screen lookalikes
that the measured FP taxonomy found.

---

## 4. External image mining

`scripts/collect_stage_a_images.py` collects licence-clean Wikimedia Commons candidates.
`scripts/mine_stage_a.py` then applies, in order:

1. exact SHA-256 exclusion against every frozen reference image;
2. id/path exclusion;
3. pHash Hamming <= 2 exclusion against the frozen references (and <= 4 inside the pool);
4. P50 hardness at conf 0.01 (hard = max_conf >= 0.05; easy = < 0.02);
5. CLIP cause tag + CLIP fire/smoke **rank** (softmax probability is diluted across 23 prompts
   and was found to be uninformative);
6. quarantine if `max_conf >= 0.50` (human review), or `max_conf >= 0.30` and fire/smoke is in
   CLIP top-3, or `max_conf >= 0.10` and fire/smoke is CLIP top-2, or a sensitive cause
   (screen/steam/exhaust/stage) has fire/smoke in CLIP top-2 at `max_conf >= 0.05`;
7. split assignment (held-out source, held-out causes, 10% random holdout, else train);
8. synthetic cap.

### Measured composition (M)

Candidates: 10,168. After SHA/pHash/id leakage removal and dedupe: **9,924**.

| cause | hard all | hard train | hard holdout | easy all | quarantine |
|---|---:|---:|---:|---:|---:|
| `camera_artifacts` | 257 | 163 | 94 | 1859 | 125 |
| `cave_warm_rock` | 33 | 26 | 7 | 221 | 33 |
| `dust_plaster` | 256 | 206 | 50 | 1549 | 102 |
| `exhaust` | 41 | 20 | 21 | 621 | 38 |
| `haze_cloud_fog_mist` | 133 | 117 | 16 | 250 | 56 |
| `neon_lights` | 23 | 21 | 2 | 99 | 2 |
| `orange_ppe` | 27 | 24 | 3 | 171 | 6 |
| `orange_vegetation` | 43 | 32 | 11 | 357 | 22 |
| `ordinary_indoor` | 10 | 8 | 2 | 84 | 2 |
| `ordinary_outdoor` | 45 | 38 | 7 | 170 | 9 |
| `reflection_lens_flare` | 73 | 50 | 23 | 210 | 24 |
| `screen_fire` | 7 | 7 | 0 | 86 | 4 |
| `skin_person` | 46 | 20 | 26 | 450 | 43 |
| `smoke_steam_nonfire` | 13 | 6 | 7 | 88 | 24 |
| `snow_white_overexposed` | 27 | 19 | 8 | 167 | 14 |
| `stage_chandelier` | 49 | 37 | 12 | 530 | 32 |
| `sun_glare_sky` (held out) | 67 | 0 | 67 | 256 | 18 |
| `sunset_sunrise` | 44 | 39 | 5 | 225 | 16 |
| `traffic_light` (held out) | 7 | 0 | 7 | 85 | 3 |
| `vehicle_lights` | 26 | 18 | 8 | 136 | 5 |
| `warm_artificial_light` | 18 | 14 | 4 | 95 | 5 |
| **total** | **1,245** | **865** | **380** | **7,709** | **583** |

By source (hard / easy / quarantine):

| source | licence | hard | easy | quarantine |
|---|---|---:|---:|---:|
| Wikimedia Commons | CC0 / PD / CC BY / CC BY-SA (per image) | 204 | 1,348 | 117 |
| Open Images V7 validation (**held-out source**) | CC BY 2.0 image listing; annotations CC BY 4.0 | 209 | 4,373 | 192 |
| D-Fire out-of-sample train negatives | CC0-1.0 | 227 | 163 | 43 |
| FASDD out-of-sample train negatives | CC BY 4.0 | 592 | 1,825 | 223 |
| synthetic (training-only) | base licence + synthetic derivative tag | 13 | 0 | 8 |

Per-image provenance: Wikimedia rows carry title, Commons page URL, artist, credit, licence and
licence URL in `data/mined_pools/wikimedia_stage_a/attribution.jsonl` (2,034 rows) and in the pool
manifest. Open Images rows carry the source URL and the dataset-level image licence. The full
licence ledger is `docs/stage_a/source_licence.csv`; the dominant Wikimedia licences among
hard+easy rows are CC BY-SA 4.0 (785), CC0 (178), public domain (128), CC BY-SA 2.0 (116),
CC BY 2.0 (111), CC BY-SA 3.0 (100) and CC BY 4.0 (77). No NC/ND image was accepted.

**Gate status (corrected 2026-10-29):** the 3,000-hard target applies to the **full 100% run**,
not to the 25% E0/E1 experiment. For E0/E1, 5-10% of the 14,747-image list is 740-1,500 added
negatives, and the current pool (865 hard train + 2,797 easy train) is sufficient. Stage A is
therefore accepted for E0/E1.

For the **full run**, 1,245 hard is still below 3,000, and the balance is still wrong: only
`dust_plaster`, `haze_cloud_fog_mist` and `camera_artifacts` exceed 100 hard, while
`warm_artificial_light` 18, `traffic_light` 7, `vehicle_lights` 26, `screen_fire` 7,
`orange_ppe` 27 and `reflection_lens_flare` 73 are underfilled. Targeted collection for the full
run is planned separately and is **not** used in E1.

---

## 5. Leakage gate

The gate is mandatory and prints a PASS/FAIL verdict in
`data/hard_negatives_v3/leakage_report.json`. It compares against:

- P50 training set: `train_50` (29,493 images);
- all frozen val/test slices (`val`, `val_cal`, `val_report`, `val_bench`, `val_mini`,
  `val_flames`, `val_gsd`, `test`, `test_bench`, `test_mini`, `test_gsd`, external CCTV NC);
- `bench_R` (1,000, DEV + SEALED) and `testbench100` (100).

### Measured leakage result (M)

Reference set: 53,828 frozen rows (P50 `train_50`, all val/test slices, `bench_R` DEV+SEALED,
`testbench100`), 53,462 pHashes, plus 8,168 candidate SHA-256 values.

| check | result |
|---|---:|
| exact SHA-256 matches removed | **166** (all Wikimedia sunset/chandelier images already in `bench_R`) |
| pHash Hamming <= 2 removed | **22** |
| id/path overlap removed | 0 |
| final pool exact-SHA leaks | **0** |
| final pool id/path leaks | **0** |
| final pool pHash <= 2 leaks | **0** |
| **verdict** | **PASS** |

The high removal count is the gate working, not a failure. `eval_neg.txt` is disjoint from
`train.txt`; it contains no synthetic images, no Open Images training rows and no held-out-cause
training rows (checked programmatically).

---

## 6. Composition and review grids

Composition is written to `data/hard_negatives_v3/composition.csv` and
`summary.json`; the licence mix is in `source_licence.csv`. Review grids are in
`data/hard_negatives_v3/contact_sheets/<cause>.jpg` and `easy_backgrounds.jpg`. Each tile shows
the P50 box(es) and confidence so the reviewer only has to mark the wrong ones.

### Review grids (M)

Every tile is labelled `cause | status | source | P50 conf`; detector boxes are drawn. Files are
in `data/hard_negatives_v3/contact_sheets/`:

`camera_artifacts.jpg`, `cave_warm_rock.jpg`, `dust_plaster.jpg`, `exhaust.jpg`,
`haze_cloud_fog_mist.jpg`, `neon_lights.jpg`, `orange_ppe.jpg`, `orange_vegetation.jpg`,
`ordinary_indoor.jpg`, `ordinary_outdoor.jpg`, `reflection_lens_flare.jpg`, `screen_fire.jpg`,
`skin_person.jpg`, `smoke_steam_nonfire.jpg`, `snow_white_overexposed.jpg`,
`stage_chandelier.jpg`, `sunset_sunrise.jpg`, `sun_glare_sky.jpg`, `traffic_light.jpg`,
`vehicle_lights.jpg`, `warm_artificial_light.jpg`, and `easy_backgrounds.jpg`.

Reviewer procedure: open the grid, mark only the wrong tiles. `quarantine` tiles are **not** in
`lists/train.txt`. The dust grid shows why this matters: several high-confidence dust/smoke tiles
are visibly real fire/smoke and are now quarantined, not trained.

---

## 7. Synthetic supplement

`scripts/make_synthetic_hardneg.py` applies sun glare, lens flare, sunset grading, bright-sky
overexposure and haze to ordinary non-fire Wikimedia scenes, keeps only variants where P50 fires
at >= 0.05, and tags every row `source=synthetic`, `synthetic_effect=...`, `base_id=...`.
Synthetic rows are training-only and capped at 20% of hard negatives by the miner.

### Measured synthetic result (M)

| item | value |
|---|---:|
| base ordinary/non-fire Wikimedia scenes | 300 |
| variants generated (4 effects) | 1,200 |
| variants where P50 fired at >= 0.05 | 43 |
| variants kept after leakage/dedupe/cap | **13** |
| by effect (kept) | bright sky 7, sun glare 5, sunset grade 1, haze 0 |

The low yield is informative: simple glare/grading/bright-sky effects do not fool P50 often.
Synthetic is therefore a small supplement here, not a route to the 3,000 target. Synthetic rows
are training-only and excluded from every evaluation list.

---

## 8. Measured vs estimated

| item | status | value |
|---|---|---|
| Free-pool P50 inference | **M** | see free-pool section |
| Wikimedia licence filtering | **M** | every accepted row has licence + page URL |
| Leakage gate | **M** | SHA-256 + pHash + id/path |
| Hard-negative composition | **M** | `composition.csv` |
| Review grids | **M** | generated contact sheets |
| Download volume | **M** | reported in the run log; kept below the 2 GB approval cap |
| Pilot false-alarm rate on real cameras | **not measured** | pilot has not started |
| E0/E1 training effect | **not measured** | Stage B/C not run |
| GPU hours | **E** | see `docs/GATE_RESULTS.md` when Stage B is approved |

---

## 9. Reproduction

```powershell
# 1) free pool: P50 on out-of-sample train images (resumable)
python scripts/mine_free_pool.py --label-filter neg --out build/stage_a

# 2) targeted licence-clean Wikimedia collection (resumable, append-only)
python scripts/collect_stage_a_images.py --per-category 140 --max-pages 4 \
  --out data/mined_pools/wikimedia_stage_a

# 3) Open Images V7 validation (held-out source; resumable, append-only)
python scripts/download_openimages_subset.py --n 3000

# 4) mine + leakage gate + composition + grids
python scripts/mine_stage_a.py \
  --manifests data/mined_pools/wikimedia_stage_a/manifest.jsonl \
              data/mined_pools/openimages/manifest.jsonl \
              build/stage_a/free_pool_false.jsonl \
  --out data/hard_negatives_v3

# 5) optional synthetic supplement, then re-run step 4 with its manifest
python scripts/make_synthetic_hardneg.py --pool data/hard_negatives_v3/manifest.jsonl
```

The miner is idempotent: detection/CLIP caches are keyed by image id and reused on re-runs.
