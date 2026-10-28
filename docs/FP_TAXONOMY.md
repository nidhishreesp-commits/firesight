# False-positive taxonomy (P50 checkpoint, 2026-10-28)

Method: every false positive from the **val.jsonl report slice** (83) plus **bench_R DEV** (5) was
captioned with Florence-2 (cached, CPU) and assigned to a category; 8 were also inspected visually.
Raw data: `build/fp_taxonomy/fps_categorized.csv` (set, id, path, category, caption).
No TEST images and no SEALED images were used.

## Counts (88 FPs; 2,644 val negatives + 215 DEV negatives)

| category | count | share | what it is |
|---|---:|---:|---|
| other | 15 | 17% | roads, buildings, tunnels, misc scenes |
| **warm_light** | **14** | 16% | street lamps, lanterns, chandeliers, city lights |
| **sunset** | **12** | 14% | sunset/sunrise, orange dusk sky |
| **haze_cloud** | **11** | 12% | fog, mist, cloudscapes, hazy horizon |
| cave_warm_rock | 7 | 8% | cave interiors, red/orange rock |
| person | 7 | 8% | portraits/selfies (skin tones) |
| water_reflection | 5 | 6% | lake/puddle/wet-road reflections |
| **screen** | **5** | 6% | game UI, movie poster, illustration of fire |
| dust | 3 | 3% | dust storm, construction dust |
| orange_ppe | 3 | 3% | hi-vis vest, workwear, red crane |
| smoke_steam | 2 | 2% | cooling-tower / chimney plume |
| orange_vegetation | 2 | 2% | orange flowers, autumn foliage |
| bright_sky | 1 | 1% | sun / eclipse |
| label_noise_suspect | 1 | 1% | caption says a real fire is visible - needs review |

## What this means

1. **Dust/plaster is NOT the main cause** (3/88). The user-facing false alarms are dominated by
   **warm artificial light, sunset, and haze/cloud**, plus screens and wet reflections.
2. **Orange PPE and orange vegetation** matter (5/88 combined) and are cheap to fix.
3. One FP looks like **label noise** (a real fire labelled negative) - must be reviewed before it is
   used as a hard negative.
4. DEV FPs (5) are: snow plow, chandelier, solar eclipse, construction site, sunset in thick smoke -
   consistent with the val categories.

## Phase 1 budget (3,000 hard negatives)

| category | target images | query seeds |
|---|---:|---|
| warm_light | 400 | street lamp night, lantern festival, chandelier, neon sign, city lights |
| sunset | 350 | sunset, sunrise sea, orange dusk sky |
| haze_cloud | 350 | fog, mist, cloudscape, hazy horizon |
| cave_warm_rock | 250 | cave interior, red rock canyon, slot canyon |
| water_reflection | 250 | lake reflection, wet road, puddle |
| screen | 200 | screen displaying fire, video-game screenshot, poster |
| dust | 250 | dust storm, construction dust, plaster |
| orange_ppe | 250 | hi-vis vest, traffic cone, orange workwear |
| smoke_steam | 200 | cooling tower steam, chimney smoke, steam vent |
| orange_vegetation | 150 | orange flowers, autumn leaves |
| person | 200 | portrait, selfie (skin-tone FPs) |
| bright_sky | 100 | sun, solar eclipse |
| other | 300 | residual scenes from the FP list |
| **total** | **3,000** | |

Rules: Wikimedia Commons (CC0/PD/CC BY/CC BY-SA) + own captures; per-image attribution; pHash+CLIP
dedupe against bench_R (all 1,000) and TEST; never mine TEST; DEV mining allowed and logged.
Audit 50-100 per category at >=90% correctness before training.
