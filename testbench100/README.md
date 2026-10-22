# testbench100

**Status: PENDING HUMAN REVIEW. Do not run a model on this bench and do not report metrics until it is locked.**

Composition: 50 positives (25 fire, 15 smoke-only, 10 fire+smoke) and 50 hard negatives.
Provisional labels come from the source datasets' human annotations; every image must be
accepted/edited in the dashboard *Bench review* page. The reviewer name and date are stored
per image, and `POST /api/bench100/lock` writes `labels.csv`, YOLO labels and a SHA-256
manifest. Benchmarking is blocked while the bench is unlocked.

```json
{
  "seed": 20261022,
  "images": 100,
  "positives": 50,
  "negatives": 50,
  "positive_categories": {
    "fire": 25,
    "smoke": 15,
    "both": 10
  },
  "negative_families": {
    "camera_artifact": 12,
    "fire_adjacent": 12,
    "light_colour": 13,
    "smoke_like": 13
  },
  "negative_subcategories": {
    "blur_noise_compression": 4,
    "campfire": 2,
    "candle": 3,
    "chandelier_festival": 1,
    "clouds": 6,
    "dust_snow": 2,
    "fireplace": 1,
    "fireworks": 1,
    "fog_mist_haze": 5,
    "gas_stove": 1,
    "infrared_night": 4,
    "lamps_neon_brake": 2,
    "match_lighter": 2,
    "orange_clothing_ppe": 2,
    "rain_on_lens": 4,
    "reflection_glare": 2,
    "street_head_lights": 3,
    "sunset_sunrise": 3,
    "welding": 2
  },
  "review_status": "PENDING HUMAN REVIEW - not locked, not for benchmarking"
}
```

## Leakage check

- pHash Hamming <= 6 against all train/val phashes: **0 flagged** (exhaustive over cached phashes).
- CLIP embedding max cosine against a 1,600-image train/val sample: **0 flagged at >= 0.98** (sampled; not a proof).
- Full report: LEAKAGE_CHECK.json.

## Licences

D-Fire (CC0 Kaggle mirror / upstream per-source terms), FASDD_CV (CC BY 4.0),
Wikimedia Commons (CC0 / Public Domain / CC BY / CC BY-SA; per-image author and
URL in ../hard_negatives/sources/wikimedia/attribution.jsonl and in index.json).
Synthetic rain images are derived from TEST negatives and inherit their licence.

## Lock status

LOCKED by Nidhishree S P on 2026-10-22 (SHA-256 20915b0e21c…).
Composition override: **62 positives / 38 negatives** (22 fire-only, 19 smoke-only,
21 both) because the reviewer reclassified controlled flames as positives.
There are no fire-adjacent hard negatives. See LOCKED.json.
