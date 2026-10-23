# bench_R candidates

PENDING HUMAN REVIEW — no model has been run on these images.
Sources: Wikimedia Commons (CC0/PD/CC BY/CC BY-SA) only. Boxes are intentionally empty; the reviewer must draw them. `labels.csv` is provisional and not a label file until locked.

```json
{
  "selected": 1000,
  "positives": 500,
  "negatives": 500,
  "counts": {
    "negative|camera_artifacts": 40,
    "negative|chandelier_stage_lights": 25,
    "negative|clouds": 30,
    "negative|dust_construction_plaster": 25,
    "negative|exhaust": 25,
    "negative|flowers_autumn_leaves": 25,
    "negative|fog_haze_mist": 30,
    "negative|orange_ppe_vests_cones": 30,
    "negative|ordinary_indoor": 38,
    "negative|ordinary_other": 19,
    "negative|ordinary_streets_warehouses_offices": 33,
    "negative|overexposed_white": 20,
    "negative|red_orange_neon_lights": 30,
    "negative|reflections_lens_flare_glare": 25,
    "negative|snow": 20,
    "negative|steam": 25,
    "negative|sunset_orange_sky": 30,
    "negative|vehicle_lights_streetlights": 30,
    "positive|both": 80,
    "positive|controlled_fire": 80,
    "positive|smoke_only": 100,
    "positive|spark_screen": 40,
    "positive|uncontrolled_fire": 200
  },
  "removed": {
    "train_val_bench100_overlap": 59,
    "duplicate": 16,
    "scene_cap": 0
  },
  "pool": 1302
}
```

## Composition (selected candidates)

`json
{
  "selected": 1000,
  "positives": 500,
  "negatives": 500,
  "counts": {
    "negative|camera_artifacts": 40,
    "negative|chandelier_stage_lights": 25,
    "negative|clouds": 30,
    "negative|dust_construction_plaster": 25,
    "negative|exhaust": 25,
    "negative|flowers_autumn_leaves": 25,
    "negative|fog_haze_mist": 30,
    "negative|orange_ppe_vests_cones": 30,
    "negative|ordinary_indoor": 38,
    "negative|ordinary_other": 19,
    "negative|ordinary_streets_warehouses_offices": 33,
    "negative|overexposed_white": 20,
    "negative|red_orange_neon_lights": 30,
    "negative|reflections_lens_flare_glare": 25,
    "negative|snow": 20,
    "negative|steam": 25,
    "negative|sunset_orange_sky": 30,
    "negative|vehicle_lights_streetlights": 30,
    "positive|both": 80,
    "positive|controlled_fire": 80,
    "positive|smoke_only": 100,
    "positive|spark_screen": 40,
    "positive|uncontrolled_fire": 200
  },
  "removed": {
    "train_val_bench100_overlap": 59,
    "duplicate": 16,
    "scene_cap": 0
  },
  "pool": 1302
}
`

## Leakage

- pHash Hamming <=6 vs train/val/testbench100: **59 candidates removed** during collection.
- CLIP cosine >=0.98 vs 1,800 train/val references: **0 flagged** (LEAKAGE_CHECK.json).
- Scene cap: max 3 near-duplicate frames per pHash scene (0 additional drops in the final selection).

**Status: PENDING HUMAN REVIEW.** Boxes are empty; use the Reference bench page. No model has been run on bench_R. The images are local (Wikimedia source URLs in index.json) and are git-ignored; the metadata/manifest are committed.

## Pre-annotation

Independent proposers only: OWLv2 (Apache-2.0) on 500 positives, Grounding DINO tiny (Apache-2.0) on a stratified 100-positive subset. No evaluated detector (v3/v4/verifier/Pyronear/RT-DETR) was used. Proposals are not labels; human review is the ground truth. Coverage: 100% controlled/spark/smoke, 99.5% uncontrolled, 98.8% both. Runtime: OWLv2 76.2 min, DINO 15.3 min. Blind 10% (100 images) has proposals hidden; the separate 100-image second-pass spot check excludes blind images. See PREANNOTATION_REPORT.md.
