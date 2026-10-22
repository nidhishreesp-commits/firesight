# Phase B1 — flame-positive + hard-negative collection plan

Status: **PLAN ONLY. No dataset has been downloaded for Phase B yet.** Downloads
above 5 GB, restrictive licences, and paid compute need explicit approval.

## Target

- **3,000+ flame-positive images**, with boxes and `flame_type`.
- **1,500+ hard negatives** (negative:positive ratio at least 1:2; prefer 1:1
  in the training sampler).
- Diversity: candles, lighters, matches, gas burners/stoves, tandoors, fireplaces,
  campfires, BBQ, diyas, incense, welding/grinding, small indoor flames, distant
  small flames, night flames, cluttered scenes, early-stage uncontrolled fire.
- Licence + source recorded per image in a CSV/HF metadata table.

## Approved / already-present sources

| Source | Licence | Size | Use | Status |
|---|---|---|---:|---|
| Wikimedia Commons (Commons API, CC0/PD/CC BY/CC BY-SA only) | per-image | ~1–3 GB for ~1,500 images | controlled flames, fireplaces, welding, fireworks | approved category; download batches after plan approval |
| D-Fire | CC0 Kaggle mirror / upstream terms | already local | uncontrolled fire/smoke, early-stage | present |
| FASDD_CV | CC BY 4.0 | already local | uncontrolled fire/smoke, diverse scenes | present |
| Pyronear `pyro-sdis` | Apache-2.0 | 1.32 GB | smoke / wildfire | **ask before download** (under 5 GB but large; I will ask in the next message) |
| Own captures (you provide) | your terms | TBD | site-specific flames/dust/negatives | awaiting files |

## Candidates requiring licence verification / approval

| Source | Licence | Size | Decision |
|---|---|---:|---|
| HF `hiennguyen9874/fire-smoke-detection` ("40K" set) | **no declared licence** | 11.57 GB | **do not download** without approval; unclear rights |
| HF `fireviewer/fire-smoke-detection-corpus-v1` | "other" (rights matrix) | 17.74 GB | **do not download** without approval; rights are source-specific |
| HF `Simuletic/CCTV-Smoke-Fire-Emergency-Detection-Dataset` | CC BY-NC 4.0 | 0.34 GB | **exclude** — non-commercial |
| HF `e1250/safety_detection` | no declared licence | 15.7 MB | **exclude** unless you approve |
| Roboflow Universe fire/smoke projects | varies per project (CC BY 4.0 / unknown / restricted) | varies | requires an API key and per-project licence check; **not downloaded** |

I will not use the HF "40K" or FireViewer corpus without your explicit approval
and a rights review. The CC BY-NC dataset is not suitable for a commercial
product.

## Hard negatives to keep (B5)

Sunsets/orange sky, red/orange lamps, neon, brake/tail/headlights, streetlights,
chandeliers, festival/stage lights, non-fire screens, orange PPE/vests/cones/
clothing, flowers/leaves, reflections/glass/water/metal, lens flare, sun glare,
clouds, fog, mist, steam, exhaust, cooling towers, dust (including the existing
construction/plaster dust frames), blowing snow, haze, overexposed white,
camera artifacts (blur, noise, compression, IR, rain/fogged lens).

## Annotation and review

1. **Proposals:** Grounding DINO (offline, Apache-2.0 weights) on unlabeled
   images; optionally fire.pt/v3 proposals as a second proposal source for
   speed. DINO is weak on small candle flames on CPU (≈0.06 on a test candle
   crop), so its boxes are *suggestions only*.
2. **Human review:** the existing review UI is extended with a flame-positive
   page: draw/edit boxes, set `flame_type`, reviewer name, timestamp, ambiguous
   flag. **No auto-label enters training until reviewed.**
3. **Review stats:** report review rate (reviewed/total) and correction rate
   (boxes/labels changed by reviewer). Target 100% review; any unreviewed image is
   excluded from training.
4. **Dedupe:** pHash Hamming ≤6 and CLIP cosine ≥0.98 against `testbench100`,
   future `testbench_B`, and all training splits; zero overlap. Near-duplicates
   are dropped, not relabeled.
5. **Class balance:** report per-source/per-category counts and the licence
   table; keep the negative:positive ratio at ≥1:2 and monitor per-class
   frequency.

## Compute / cost estimate

- 3,000 images at 1024 px ≈ 3–6 GB (source-dependent).
- DINO auto-label on CPU ≈ 8–10 s/image → ~8 h for 3,000; a free Kaggle GPU
  would be ~20–40 min. **No paid compute without approval.**
- Human review is the dominant cost: roughly 10–20 s/image for easy boxes,
  longer for cluttered scenes. Plan several review sessions.
- Training (Phase D) on a free Kaggle GPU: YOLO11s/12s 100 epochs at 960 px on
  ~4–6k images ≈ 4–8 GPU-hours; RT-DETRv2-l ≈ 10–20 GPU-hours. Cost = 0 on
  free Kaggle quota, but quota/time must be approved.

## Deliverables for Phase B

- merged flame-positive dataset with boxes + `flame_type`
- licence/source table and per-source counts
- review stats (review rate, correction rate)
- dedupe report vs testbench100/B
- updated `CATALOG.md` and `data/README.md`
- no training or bench-B evaluation until you approve the data
