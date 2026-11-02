# Indoor/industrial positives and hard negatives: licence/source plan

Status: **plan only. No image/video download started.** Commons counts below are raw search hits,
not licence-clean counts; Commons was rate-limited during the licence sample. Estimates assume
1024 px JPEGs (~0.15 MB each) and heavy dedupe/relevance filtering.

## 1. Positives (indoor/industrial + controlled)

| type | proposed source | raw Commons hits | licence | estimated usable | estimated size |
|---|---|---:|---|---:|---:|
| factory/facility fire | Commons | factory 2,778; industrial 3,116; warehouse 2,490 | per-file CC0/PD/CC BY/CC BY-SA | 300-700 | 45-105 MB |
| refinery/chemical/plant fire | Commons | refinery 387; chemical 306; power plant 2,506 | same | 100-300 | 15-45 MB |
| indoor/room/house fire | Commons | house interior 3,595; building interior 6,202 | same | 300-800 | 45-120 MB |
| kitchen/electrical fire | Commons | kitchen 2,694; electrical 3,592 | same | 200-500 | 30-75 MB |
| vehicle/ship fire | Commons + CC BY video | vehicle/truck/ship searches unreliable; video frames better | same | 200-500 | 30-75 MB |
| controlled flames | Commons + own burns | fireplace 2,913; controlled burn 12,893; burn barrel 1,676; campfire raw huge/noisy | same | 500-1,000 | 75-150 MB |
| thin/distant smoke | Pyro-SDIS v2.0 after filter | 7,259 candidates (non-event train) | Apache-2.0 / LO 2.0 | 3,000-5,000 after filter | local, 1.1 GB |
| **target** | | | | **12,000 indoor/industrial + 5,000 thin + 3,000 controlled** | **images <2 GB; video may exceed** |

## 2. Hard negatives (indoor/industrial lookalikes)

| type | proposed source | raw Commons hits | licence | estimate |
|---|---|---:|---|---:|
| steam | Commons (sample 100% allowed) | 181,014 raw | per-file CC | 300-800 |
| welding sparks | Commons (94% allowed in 50-sample) | high | per-file CC/PD | 300-600 |
| street lamps | Commons (100% allowed in 50-sample) | high | per-file CC | 300-600 |
| neon signs | Commons (100% allowed in 50-sample) | high | per-file CC | 200-400 |
| vehicle lights | Commons | high | per-file CC | 200-400 |
| safety vests / PPE | Commons (38% allowed in 50-sample) | moderate | per-file CC/PD | 100-250 |
| hot metal / furnace | Commons (100% allowed in 50-sample) | high | per-file CC/PD | 200-400 |
| screens / monitors | Commons | moderate | per-file CC | 100-250 |
| **target** | | | | **1,000-2,000** |

## 3. Honest scarcity

**Licence-clean indoor/industrial fire is scarce.** Commons hits are inflated by historical photos,
aftermath, diagrams and irrelevant matches. A realistic Commons-only yield is a few thousand
stills, not 12,000 surveillance-like positives. Reaching the 60% target requires one or more of:

1. **Own captures** (factory, warehouse, kitchen, electrical panel, plant, ship) with safety/consent;
2. **Partner footage** from fire services / industrial safety teams with written permission;
3. **Purchased/licensed stock** — the current Downloads iStock/Getty clips are **proprietary and not
   licensed for ML training; test-only**;
4. **CC BY public video frames** (Wikimedia/YouTube), per-video verified, with onset/sequence
   metadata.

Without one of these, the honest option is a **smaller rebalanced experiment** using the available
few thousand indoor/industrial positives, not the full 12,000.

## 4. Download budget

Still-image Phase 1 estimate: 3,000-5,000 Commons images ≈ **0.45-0.75 GB**; with video frames
could approach 1-2 GB. **No download starts until you approve.** If the combined source plan
exceeds 2 GB I will ask again.

## 5. Dedupe

Every downloaded image/frame gets SHA-256 + pHash (Hamming <=2) + id/path against bench_R, val,
TEST, independent negatives and Sample_videos; exact/near duplicates dropped.
