# Hard-negative catalog (Phase 1)

Goal: every plausible fire/smoke lookalike, with source and licence, so the
detector and verifier can be trained against them **without touching the frozen
test bench**.

## Policy decisions (explicit)

1. **Controlled flames are hard negatives for the alarm task.** Candles, lighters,
   gas stoves, campfires/BBQ, fireplaces, fireworks, welding/grinding sparks and
   TV/movie fire are labelled `has_fire=0` / `has_smoke=0` in `testbench100`
   because the product alarm means an *uncontrolled* fire. They remain real
   flames and must be shown in the review UI so the reviewer can mark them
   `ambiguous` if they disagree.
2. **No copyrighted scraping.** Only own captures, open datasets and
   CC0 / Public Domain / CC BY / CC BY-SA Wikimedia Commons files are used.
   Per-image attribution is stored in
   `hard_negatives/sources/wikimedia/attribution.jsonl` and copied into
   `testbench100/index.json` (`license`, `origin`, `attribution`).
3. **D-Fire/FASDD labels are not automatically trusted.** The source datasets
   contain label noise (e.g. candles labelled fire, sunset frames with tiny
   fire boxes). Phase 2 rejects tiny/dark fire boxes and restricts "clear fire"
   candidates to D-Fire; every remaining label still requires human review.
4. **Bench images must not be in any training set.** They are drawn from the
   official D-Fire/FASDD **test** splits (already scene-isolated by pHash in
   `data/splits/test.jsonl`, 0 scene overlap with train/val), plus Wikimedia
   images and synthetic rain variants of test negatives. A separate leakage
   check is run before locking.

## Inventory (CLIP-assisted, 1,500 TEST negatives)

These counts are an **inventory aid, not ground truth**. CLIP zero-shot is noisy
(especially for fire-adjacent images), so only visually checked images are used
in the bench. Full rows: `hard_negatives/clip_inventory.json`.

| Family | Sub-category | CLIP count | In bench | Source | Licence |
|---|---|---:|---:|---|---|
| light_colour | sunset_sunrise | 98 | 3 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | street_head_lights | 73 | 3 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | lamps_neon_brake | 33 | 2 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | reflection_glare | 39 | 2 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | orange_clothing_ppe | 74 | 2 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | chandelier_festival | 21 | 1 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| light_colour | cones_machinery | 3 | 0 | not selected | — |
| light_colour | flowers_leaves | 10 | 0 | not selected | — |
| smoke_like | clouds | 172 | 6 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| smoke_like | fog_mist_haze | 91 | 5 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| smoke_like | dust_snow | 41 | 2 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| smoke_like | steam_cooking | 8 | 0 | CLIP candidates were food/objects, not steam | — |
| smoke_like | exhaust_chimney | 35 | 0 | CLIP candidates included car grilles, not exhaust | — |
| smoke_like | overexposed_white | 14 | 0 | candidates were easy flower fields | — |
| fire_adjacent | candle | 19 CLIP + 3 Wikimedia | 3 | Wikimedia Commons | CC0 / CC BY-SA |
| fire_adjacent | match_lighter | 2 Wikimedia | 2 | Wikimedia Commons | CC BY-SA / PD |
| fire_adjacent | gas_stove | 2 Wikimedia | 1 | Wikimedia Commons | CC BY / CC0 |
| fire_adjacent | campfire | 6 CLIP + 2 Wikimedia | 2 | Wikimedia Commons | CC BY-SA |
| fire_adjacent | welding_grinding | 6 CLIP + 2 Wikimedia | 2 | Wikimedia Commons | PD |
| fire_adjacent | fireworks | 5 CLIP + 2 Wikimedia | 1 | Wikimedia Commons | CC BY-SA |
| fire_adjacent | fireplace | 2 Wikimedia | 1 | Wikimedia Commons | CC BY-SA |
| fire_adjacent | screen_fire | 6 CLIP | 0 | CLIP candidates were cinema seats, not fire | — |
| camera_artifact | blur_noise_compression | 235 | 4 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| camera_artifact | infrared_night | 67 | 4 | D-Fire/FASDD test negatives | CC0 / CC BY 4.0 |
| camera_artifact | rain_on_lens | 5 CLIP + 4 synthetic | 4 | synthetic rain over TEST negatives | inherited (CC0/CC BY) |
| easy_negative | plain_scene / people_vehicles | 439 | 0 | excluded by design | — |

## Exhaustive checklist and status

Status: **B** = in testbench100; **C** = collected in the wider pool; **P** =
planned / not yet collected; **X** = policy-excluded from the alarm-negative
bench (still useful for verifier training).

### Light / colour
- B sunset, sunrise, orange sky
- B red/orange lamps, neon signs
- B brake/tail lights, headlights at night, streetlights
- B chandeliers / festival lights
- B reflections on glass/water/metal, lens flare, sun glare
- B red/orange clothing, PPE, safety vests
- P traffic cones, orange machinery (CLIP pool only 3; needs targeted collection)
- P flowers, autumn leaves (pool only 10; not selected)
- P LED screens / stage lighting / Christmas lights (not enough in current pool)

### Smoke-like
- B fog, mist, low clouds, haze
- B dust / sandstorm / blowing snow
- P steam (kitchens, showers, kettles, manholes)
- P vehicle exhaust / factory chimney water vapour / cooling towers
- P spray/water mist, breath in cold weather
- P snow / white overexposed walls
- P camera lens dirt/fogging (partly covered by synthetic rain)

### Fire-adjacent (policy: not an alarm event; must be reviewed)
- B candles, lighters, gas stove, campfire/BBQ, welding/grinding sparks,
  fireworks, fireplace
- P lit cigarettes, incense, matches (real flame images; Wikimedia match query
  returned book scans and was rejected)
- P tandoors / cooking flames / diyas
- P TV/movie fire scenes (CLIP candidates were cinema interiors, not screens)
- P industrial flares / decorative flame lamps

### Camera artifacts
- B motion blur, sensor noise, compression, synthetic rain
- B infrared/night mode
- P real rain-on-lens, flicker, exposure changes, real lens dirt

## Dedupe / leakage status

- Test bench comes from the official test split, which was built with a
  64-bit pHash: 0 scene overlap with train/val (`data/splits/TEST.lock.json`).
- The bench build additionally runs `scripts/check_bench_leakage.py`:
  pHash Hamming ≤6 against all train/val phashes, plus a sampled CLIP embedding
  check. Results are written to `testbench100/LEAKAGE_CHECK.json`.
- Any flagged image must be replaced before the bench is locked.

## What is still missing (honest)

- Steam, exhaust, real rain-on-lens, screen fire, candles/matches, and traffic
  cones are under-covered. They are marked P above; targeted collection needs
  more CC sources or your own captures.
- The CLIP inventory is noisy; counts are for sourcing only.
- The bench is 100 images, so 95% accuracy has a wide Wilson interval. Phase 5
  grows it to 500+.

### Construction / plaster dust (added after review)

| Category | Count | Source | Licence | Notes |
|---|---:|---|---|---|
| construction_site_dust_plaster | 16 frames | local Sample_videos/ppe_1..4 | proprietary internal | hard_negatives/sources/construction_dust/images |
| controlled-flame validation set | 18 images | Wikimedia Commons | CC0 / PD / CC BY / CC BY-SA | hard_negatives/sources/val_flames; used only for threshold calibration, never training or bench |

Confirmed false alarm: est_fire_v3 fired smoke 0.48 on
ppe_2_3s.jpg (workers + sand/plaster pile); 15/16 construction frames were
clean. This is now an explicit hard-negative family for retraining and is
included in the calibration negatives (data/splits/val_flames.jsonl).

**Reviewer note:** the locked bench uses an explicit composition override
62 positives / 38 negatives because the reviewer reclassified the 12
controlled-flame images as positives. The bench therefore has **no
fire-adjacent hard negatives**. That is recorded in 	estbench100/LOCKED.json
and must be considered when reading the 87% accuracy.
