# Targeted collection plan for the full-run underfilled causes

Status: **plan only. No images have been downloaded.** The small Open Images annotation CSVs
(17-23 MB) were read to verify class names and counts; no image download has started. These
images are for the full 100% run, not for E1. The E0/E1 pool is already sufficient and is not changed.

## 1. Why this is needed

Stage A's full-run pool is still cause-skewed: dust 256, camera artefacts 257 and haze 133
dominate, while warm light 18, traffic light 7, vehicle lights 26, screen 7, orange PPE 27 and
reflection 73 are below the full-run target of >=100 hard per cause.

The Open Images V7 **validation** split is too small for the rare classes. Measured from the
official validation annotations (M):

| class / label | validation images | source |
|---|---:|---|
| Traffic light (box) | 18 | `validation-annotations-bbox.csv` |
| Street light (box) | 13 | same |
| Lantern (box) | 12 | same |
| Television (box) | 34 | same |
| Computer monitor (box) | 47 | same |
| Window (box) | 450 | same |
| Person (box) | 6,436 | same |
| Car (box) | 4,900 | same |
| Truck (box) | 269 | same |
| Sun (image label) | 19 | `validation-annotations-human-imagelabels.csv` |
| Sky (image label) | 291 | same |
| Cloud (image label) | 150 | same |
| Fog (image label) | 31 | same |
| Dust (image label) | 16 | same |
| Neon sign (image label) | 16 | same |
| Reflection (image label) | 124 | same |
| Haze (image label) | 27 | same |
| Sunset (image label) | 44 | same |
| Sun & Person | 1 | intersection |
| Sky & Person | 14 | intersection |
| Sky & Car | 5 | intersection |

Validation alone cannot produce >=100 hard per cause. The train split must be used.

## 2. Verified class names

Boxable classes (Open Images `class-descriptions-boxable.csv`):

`Traffic light` `/m/015qff`, `Street light` `/m/033rq4`, `Lantern` `/m/01jfsr`,
`Television` `/m/07c52`, `Computer monitor` `/m/02522`, `Window` `/m/0d4v4`,
`Person` `/m/01g317`, `Car` `/m/0k4j`, `Truck` `/m/07r04`, `Bus` `/m/01bjv`,
`Helmet` `/m/0zvk5`, `Laptop` `/m/01c648`, `Mobile phone` `/m/050k8`.

Image-level labels: `Sun` `/m/06m_p`, `Sky` `/m/01bqvp`, `Cloud` `/m/0csby`, `Fog` `/m/0g2z8`,
`Dust` `/m/015l_t`, `Neon sign` `/m/01s4b5`, `Sunset` `/m/01b2w5`, `Reflection` `/m/02l215`,
`Haze` `/m/03xxq7`, `Light` `/m/04k84`, `Vest` `/m/01sdv7`.

Open Images has **no boxable Vest / Traffic cone / High-visibility clothing / Headlight** class.
Orange PPE and some vehicle-light scenes therefore come from Wikimedia Commons categories.

## 3. Source, licence and size

Official Open Images licence: **annotations CC BY 4.0; images listed as CC BY 2.0** (Google warns
to verify each image). Wikimedia Commons: **per-image CC0 / public domain / CC BY / CC BY-SA**;
NC/ND rejected. Sizes are estimates from the measured 0.307 MB average of the 5,000 Open Images
validation images already downloaded (M).

| item | licence | size |
|---|---|---:|
| Open Images V7 train bbox annotations | annotations CC BY 4.0 | **1.194 GB** |
| Open Images V7 train human image labels | annotations CC BY 4.0 | 1.229 GB |
| Open Images class-filtered images, Phase 1 (2,100) | images CC BY 2.0 | **~0.65 GB** |
| Wikimedia Commons categories, Phase 1 (1,000) | per-image CC0/PD/CC BY/CC BY-SA | **~0.15 GB** |
| **Phase 1 total** | | **~2.0 GB** |
| Open Images class-filtered images, Phase 2 (+2,000) | images CC BY 2.0 | ~0.61 GB |
| **Full plan total** | | **~2.6 GB** |

Phase 1 can be kept at or just under the 2 GB cap by downloading the bbox annotations and 2,100
class-filtered images + 1,000 Wikimedia images. The full 8-cause x >=100-hard plan is ~2.6 GB and
**exceeds the cap**, so it needs explicit approval.

## 4. Collection mapping

| cause | Open Images filter | Wikimedia fallback |
|---|---|---|
| traffic_light | `Traffic light` boxes | `Category:Traffic lights`, `Category:Railway signals` |
| warm_artificial_light | `Street light`, `Lantern` boxes; `Light` image label | `Category:Street lamps`, `Lanterns`, `City lights` |
| vehicle_lights | `Car`, `Truck`, `Bus`, `Motorcycle` boxes + `Light` image label; mine night lights | `Car headlights`, `Tail lights`, `Light trails road` |
| screen_fire | `Television`, `Computer monitor`, `Laptop`, `Mobile phone` boxes | `Television screens`, `Computer monitors` |
| orange_ppe | none reliable (no Vest/Cone boxable class) | `High-visibility clothing`, `Traffic cones`, `Workwear` |
| reflection_lens_flare | `Window` boxes + `Reflection` image label | `Reflections`, `Lens flare`, `Water reflections` |
| sun/bright sky behind person/vehicle | `Person`/`Car`/`Truck` boxes intersected with `Sun`/`Sky`/`Cloud` image labels | `Sun glare`, `Backlit`, `Silhouettes`, `Contre-jour` |
| street lights | `Street light` boxes | `Street lamps` |

## 5. Protocol when approved

1. Download the Open Images train bbox annotations (1.194 GB) and select image IDs per class.
2. Sample the Phase 1 budget, preferring images with multiple relevant labels.
3. Download selected images, resize to <=1280 px JPEG, record image licence + source URL.
4. Mine with frozen P50 at conf >= 0.05; keep hard >= 0.05, tag cause with CLIP.
5. Dedupe SHA-256 + pHash <= 4 inside the pool, and apply the same leakage gate vs
   train_50/val/TEST/bench_R/testbench100.
6. Quarantine P50 >= 0.50 or CLIP fire/smoke >= 0.50 for review.
7. Report composition and review grids. Target >=100 hard per cause.
8. **Keep these images out of E1.** They are for the full run only.

## 6. Decision requested

- Approve **Phase 1 (~2.0 GB)** under the cap, or
- Approve the **full plan (~2.6 GB)**, which exceeds the 2 GB cap.

No download starts until this is answered.
