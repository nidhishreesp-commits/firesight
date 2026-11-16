# External models on the full bench_R

Protocol for both: all 1,000 bench_R images (DEV 300 + SEALED 700), thresholds calibrated only on
`val_mini + val_flames` at target FPR 2%, no tuning on bench_R. The SEALED reads are logged in
`bench_results/BENCH_USAGE.md`.

## A. SalahALHaismawi/yolov26-fire-detection

- Status: **downloaded and evaluated.**
- Licence: **MIT** (model card); training dataset: Roboflow
  `personal-bodxv/fire-detection-sejra-fognw`, 8,939 images, **CC BY 4.0**.
- Weights: `models/external/salah_yolov26_best.pt`, **20.3 MB**.
- Loads with the installed **ultralytics 8.4.147** (no special `YOLOvv26` package needed).
- Classes: `{0: fire, 1: other, 2: smoke}`; `other` is ignored by the loader.
- Thresholds (val_mini+val_flames, target 2%): fire **0.51**, smoke **0.57**.
- CPU latency: **175 ms/image** measured over 5,119 images (P50: 82.9 ms).

| split | n | acc | precision | recall | specificity | FPR/neg | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEV | 300 | 82.67% | 77.97% | 54.12% | 93.95% | 4.33% | 63.89% |
| SEALED | 700 | 82.00% | 77.44% | 51.76% | 94.01% | 4.29% | 62.05% |
| **all** | **1,000** | **82.20%** | **77.60%** | **52.46%** | **93.99%** | **4.30%** | **62.61%** |

Per class (all): fire recall **58.99%**, smoke recall **26.60%**; fire precision 72.92%,
smoke precision 83.33%. Per flame type: uncontrolled 60.26%, controlled 56.47%, smoke-only
33.96%, spark 18.92%, unknown 0%. Box AP50: fire **0.2039**, smoke **0.2164**.

Per-cause FPs (all negatives): clouds 16.7%, flowers/autumn 24.0%, chandelier 20.0%, neon 13.3%,
both 11.5%, ordinary-other 10.5%, steam 8.7%, fog/haze 6.7%, PPE 6.7%, vehicle lights 3.3%,
sunset 3.3%; uncontrolled 3.0%, smoke-only 4.2%.

**Matched-FPR comparison (per-set thresholds; same images):**

| set | target FPR | P50 recall | E1 recall | Salah recall | McNemar P50 vs Salah (pos) | E1 vs Salah (pos) |
|---|---:|---:|---:|---:|---:|---:|
| bench_R DEV | 0.9% | 36.47% | 28.24% | 22.35% | p=0.017 | p=0.332 |
| bench_R DEV | 1.5% | 49.41% | 49.41% | 32.94% | p=0.009 | p=0.009 |
| bench_R DEV | 2.0% | 49.41% | 52.94% | 42.35% | p=0.238 | p=0.093 |
| val_report | 0.9% | 71.76% | 84.73% | 47.47% | p=5.2e-90 | p=1.9e-220 |
| val_report | 1.5% | 85.01% | 89.17% | 55.15% | p=3.9e-155 | p=1.9e-202 |
| val_report | 2.0% | 89.54% | 90.63% | 61.94% | p=5.3e-153 | p=1.7e-161 |
| independent 5,301 | 0.9/1.5/2.0% | - | - | - | p=1.0 (FPR matched by construction) | p=1.0 |

**Verdict:** SalahYOLOv26 is **worse than P50 and E1** on the full bench_R, on val_report, and
on DEV at matched FPR. It is not an upgrade.

## B. Roboflow Universe fire-and-smoke-segmentation

- Status: **blocked by access — not rejected, not evaluated.**
- Model card: instance segmentation, 201 images, **CC BY 4.0**; current best version is
  RF-DETR Nano, but there are YOLOv8n/YOLOv11n-seg versions.
- Weight download and hosted inference both require a **Roboflow API key**:
  `api.roboflow.com` returns 401; `detect.roboflow.com` returns 401; Universe pages are
  Cloudflare-403 to scripts.
- Dataset download (201 images or 7,110 augmented) also requires the Roboflow API key.
- No key is configured (`ROBOFLOW_API_KEY` absent). **I have not used any key.**

To proceed, set `ROBOFLOW_API_KEY` in the environment (do not paste it in chat) and confirm you
want me to use it. I will use it **only on bench_R and public-licence images**, never on
Sample_videos/proprietary footage, and will throttle to respect the account rate limit. The
fallback remains training a small `yolo11n-seg`/`yolov8n-seg` on the 201-image CC BY 4.0 dataset,
clearly labelled as our own model.

## Leakage

Salah's training data is not downloadable without a Roboflow key, so the hash/pHash overlap with
bench_R and the independent negatives is **unverified**. bench_R is sourced from Wikimedia, and
the training set is Roboflow, so overlap is unlikely but not proven. Model B's training data is also
key-gated. This is a blocker, not a pass.

## Measured vs estimated

- **Measured:** all Salah metrics above, thresholds, DEV/SEALED/all, AP50, per-cause FPs, latency,
  matched-FPR tables, and that ultralytics 8.4.147 loads the weights.
- **Not measured:** Model B (no key), Salah training-data overlap (no key), Model B mask stats.
