# Hard-negative catalog (Phase 1)

Goal: every plausible fire/smoke lookalike, with source and licence, so the
detector and verifier can be trained against them **without touching the frozen
test bench**.

## Policy decisions (updated 2026-10-22)

1. **Controlled flames are POSITIVE for the fire class.** Candles, lighters,
   matches, gas stoves/burners, tandoors, fireplaces, campfires, BBQ, diyas,
   incense, welding/grinding sparks, fireworks, flame lamps and TV/movie fire are
   annotated as fire with a flame_type:
   - controlled — candle, lighter, stove, fireplace, campfire, BBQ, diya, incense, flame lamp
   - spark — welding, grinding, fireworks
   - screen — TV/movie fire
   - uncontrolled — ordinary fire
   - unknown — not yet classified
2. **Default runtime alarm policy:** include uncontrolled + unknown; exclude
   controlled, spark, screen. This is a config/API flag
   (alarm.fire_type_include/exclude, /api/policy/fire-types), so the decision
   can change without retraining. It requires the flame-type classifier trained in
   Phase B/D; until then all fire detections are unknown and included.
3. **Reason:** the locked bench shows the detector is strong on uncontrolled fire
   (96.8% recall) but weak on controlled flames (44.4%) and sparks (33.3%).
   Treating all flames as alarms gives 87% accuracy; excluding controlled/spark
   by policy gives 92.0% on the 88 evaluable images. That is a policy choice, not
   a free accuracy gain, and the bench has been used for model selection
   (see bench_results/BENCH_USAGE.md).
4. **Hard negatives remain:** B5 list — sunsets/orange sky, lamps/neon/brake/
   headlights/streetlights/chandeliers/stage lights/screens (non-fire), orange
   PPE/cones/clothing, flowers/leaves, reflections/lens flare/glare, clouds, fog,
   steam, exhaust, dust (including construction/plaster dust), snow/haze, and
   camera artifacts.
5. **No copyrighted scraping.** Only own captures, open datasets and CC0/PD/
   CC BY/CC BY-SA Commons files. Per-image attribution is recorded.
6. **Reviewer override:** the locked bench is 62 positives / 38 negatives; there
   are no fire-adjacent hard negatives. This is recorded in LOCKED.json.

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
| **controlled_flame_positive** | candle | 19 CLIP + 3 Wikimedia | 3 | Wikimedia Commons | CC0 / CC BY-SA |
| **controlled_flame_positive** | match_lighter | 2 Wikimedia | 2 | Wikimedia Commons | CC BY-SA / PD |
| **controlled_flame_positive** | gas_stove | 2 Wikimedia | 1 | Wikimedia Commons | CC BY / CC0 |
| **controlled_flame_positive** | campfire | 6 CLIP + 2 Wikimedia | 2 | Wikimedia Commons | CC BY-SA |
| **controlled_flame_positive** | welding_grinding | 6 CLIP + 2 Wikimedia | 2 | Wikimedia Commons | PD |
| **controlled_flame_positive** | fireworks | 5 CLIP + 2 Wikimedia | 1 | Wikimedia Commons | CC BY-SA |
| **controlled_flame_positive** | fireplace | 2 Wikimedia | 1 | Wikimedia Commons | CC BY-SA |
| **controlled_flame_positive** | screen_fire | 6 CLIP | 0 | CLIP candidates were cinema seats, not fire | — |
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

### Controlled flames (POSITIVE for fire; default excluded from alarm)

These are **not hard negatives**. They are annotated fire with flame_type
controlled, spark or screen; the runtime policy decides whether they alarm.
- B candles, lighters, gas stove, campfire/BBQ, welding/grinding sparks,
  fireworks, fireplace — now positives with flame_type
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
| controlled-flame validation set | 18 images | Wikimedia Commons | CC0 / PD / CC BY / CC BY-SA | flame_positives/sources/val_flames; used only for threshold calibration, never training or bench |

Confirmed false alarm: best_fire_v3 fired smoke 0.48 on
ppe_2_3s.jpg (workers + sand/plaster pile); 15/16 construction frames were
clean. This is now an explicit hard-negative family for retraining and is
included in the calibration negatives (data/splits/val_flames.jsonl).

**Reviewer note:** the locked bench uses an explicit composition override
62 positives / 38 negatives because the reviewer reclassified the 12
controlled-flame images as positives. The bench therefore has **no
fire-adjacent hard negatives**. That is recorded in testbench100/LOCKED.json
and must be considered when reading the 87% accuracy.
