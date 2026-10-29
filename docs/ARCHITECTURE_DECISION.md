# Architecture decision: separate models vs one unified model

**Decision question (from the lead):** for production, should fire/smoke, PPE and vehicle counting stay as separate models, or be unified into one model?

**Bottom line:** **stay separate (Option B: shared pipeline, separate models).** Do not build a shared-backbone multi-task model (C) or one all-class detector (D) for production now. Keep fire in its own process/container with its own model and watchdog. Share the decode/preprocess layer and the person/vehicle detector between PPE and counting, and batch across cameras on the GPU. The current **PPE YOLO11m** is the cost driver; at 32 cameras and 5 fps PPE, it alone needs about **3 Orin NX 16GB or 2 AGX Orin 64GB**. If that hardware bill is unacceptable, the correct next step is to shrink/replace the PPE model (a future retrain, explicitly out of scope here), not to unify it with fire and counting.

- **Study type:** analysis + benchmarks. No model was trained, fine-tuned, exported or modified.
- **Target hardware given:** Jetson Orin NX / AGX Orin.
- **Site size given:** 32 cameras per site.
- **Required rates given:** fire 1-2 fps, PPE 5 fps, counting 10 fps; latency targets fire <2 s, PPE <1 s, counting <0.5 s.
- **Camera overlap given:** all three tasks run on the same cameras.

---

## 0. Evidence labels

Every number below is one of:

| label | meaning |
|---|---|
| **M** | **MEASURED** by this study on the stated hardware and bench. |
| **E** | **ESTIMATED** from published architecture/benchmarks, CPU-to-Jetson scaling, or stated cost assumptions. |
| **C** | **CLAIMED** by a vendor/author. Included only for the inventory; never used as evidence for the decision. |

No Jetson board was available on the study machine. **All Jetson numbers are E.** The CPU numbers are M on the machine described in §1.2. User-given requirements (32 cameras, 2/5/10 fps) are inputs, not measurements; they are not labelled M/E.

---

## 1. Scope, target hardware, measurement conditions

### 1.1 What was in scope

- Fire/smoke: deployed `best_fire_v2` and FireSight **P50** detector, plus the current and planned verifier.
- PPE: ss-prod `ppe-yolo11` (byte-identical to `ppe_yolo11.onnx`).
- Vehicle counting: `yolo11s` COCO + `autorickshaw` auxiliary.
- Options A-D, accuracy risk, operations, cost, migration, unknowns.

### 1.2 Measurement machine (CPU only)

| item | value | label |
|---|---|---|
| CPU | Intel Core i7-1165G7, 4 cores / 8 threads, 2.80 GHz | M |
| RAM | 15.7 GB | M |
| GPU | none (no NVIDIA GPU, no `nvidia-smi`) | M |
| ONNX Runtime | 1.26.0, CPUExecutionProvider, 4 intra-op threads | M |
| OpenCV | 4.11.0 | M |
| Python | 3.12.10 | M |
| Video used | `fire.mp4` 1280x720, `ppe_2.mp4` 768x432, `drone.mp4` 1920x1080, `Car.mp4` 3840x2160 | M |

The CPU is a laptop-class 4-core part. It is **not** the production target; it is the only measurement host available. Its numbers are used to (a) prove CPU-only is not viable, and (b) establish the relative cost of the models. Jetson numbers are estimated separately.

### 1.3 What was not measured

- No Jetson Orin NX or AGX Orin was available. TensorRT FP16 engine latency, VRAM, power and thermal behaviour on the target are **unknown**.
- No 32-camera RTSP/NVDEC load test was possible. Capacity is arithmetic from single-stream measurements and published per-device throughput, not a simultaneous 32-camera replay.
- No vehicle-counting ground truth exists in the repo, so **vehicle-counting accuracy is unmeasured**.
- P50 has not been run on bench_R SEALED; only DEV and val are measured.
- Event-level false alarms per camera-week are unmeasured (bench_R is still images).

---

## 2. INVENTORY

### 2.1 Fire / smoke

| field | Deployed `best_fire_v2` | FireSight **P50** (planned production) |
|---|---|---|
| File | `isight-inference/models/best_fire_v2.onnx` (`.pt` also present) | `firesight/build/kaggle_weights/runs/P50/weights/best.onnx` |
| SHA-256 | `f2b9bfed...f412bf` (ONNX, from inventory) | `4838218ce81d9d2f3bbb6c665e45d4c6cc2590699af75f0da159d3a127708849` (M) |
| Architecture | YOLOv12n | YOLO11s |
| Params | 2.56 M | 9.44 M |
| Input | 1x3x640x640 static | dynamic batch/height/width; trained/run at 640 |
| Classes | `fire`, `smoke` | `fire`, `smoke` |
| Size | 10.03 MiB ONNX; 5.27 MiB PT | 36.0 MiB ONNX |
| Licence | AGPL-3.0 (code + weights) | AGPL-3.0 (code + weights) |
| Measured accuracy | P 0.211 / R 0.217 / F1 0.214 / mAP50 0.077 on merged 2705-image test at conf 0.25 (M, EXP-004). FireSight test_bench: fire AP50 0.286, smoke AP50 0.117, recall@FPR1% 0.231, test FPR 0.0187 (M). | val: recall 0.9297 [0.919, 0.939] at 3.14% FPR/negative, mAP50 fire 0.633 / smoke 0.679 (M). bench_R DEV: recall 56.5% [45.9, 66.5] at 2.33% FPR/negative (M). SEALED: **not run** (unknown). |
| Claimed accuracy | training val mAP50 0.536 at epoch 100 (C, from the training run; not the deployed test) | none (C not applicable) |
| Inference pipeline | Ultralytics `.pt` in `production_pipeline.py:1856`; per-frame; raw conf 0.20; per-class floor fire 0.25 / smoke 0.20; min area 400 px; 4x4 grid cooldown 0.3 s; **no temporal N-of-M in the iSight path**. | ONNX/Ultralytics; letterbox 640; calibrated thresholds 0.29/0.33; ByteTrack + N-of-M 3/5 fire, 4/6 smoke with growth cue; optional crop verifier; `sample_fps` 5; per-camera ROI/conf/N/M/cooldown in `configs/cameras.yaml`; alarm tiers. |
| CPU latency (M) | 87.7 ms model-only; 102.9 ms end-to-end at 720p | 184.7 ms model-only; 193.1 ms end-to-end at 720p |
| CPU throughput (M) | 9.7 fps end-to-end | 5.2 fps end-to-end |

Note: `configs/default.yaml` currently points at `models/best_fire_v3.onnx`; P50 is the pilot's best checkpoint and the planned replacement, not the current default artifact. P50 is not yet in `models/registry.yaml`; promotion needs a registry entry, manifest and sha256.

**Reference candidates (not deployed):**

| model | file | size | fire AP50 | smoke AP50 | recall@FPR1% | test FPR | label |
|---|---|---:|---:|---:|---:|---:|---|
| ss-prod best_fire_v3 | `smartsight-models/best-fire-v3/model.onnx` | 10.1 MiB | 0.680 | 0.707 | 0.767 | 0.0249 | M |
| ss-prod best_fire_v4 | `smartsight-models/best-fire-v4/model.onnx` | 10.1 MiB | 0.688 | 0.706 | 0.733 | 0.0187 | M |
| older ss-prod fire.pt | `isight-inference/models/fire.pt` | 54.5 MiB | 0.405 | 0.337 | 0.183 | 0.0221 | M |

These are on the FireSight frozen `test_bench` (3,000 images), not bench_R. They are not directly comparable to P50's val/DEV numbers.

### 2.2 Planned verifier

| field | value |
|---|---|
| File | `firesight/models/verifier.onnx` + `verifier.onnx.data` |
| Architecture | MobileNetV3-Small, 3-class classifier (`fire`, `lookalike`, `smoke`) |
| Input | dynamic batch x 3 x 128 x 128 |
| Preprocess | crop around the detection with 15% margin, resize 128x128, BGR->RGB, ImageNet mean/std, NCHW |
| Size | 0.30 MiB graph + 6.06 MiB external data = **6.36 MiB** (M) |
| Licence | internal (trained here); no third-party weights |
| Status | `enabled: false` in `configs/default.yaml`; ablation exists |
| Measured effect | +5 pp recall at 0.1% FPR (base 0.450 -> 0.500 at lookalike threshold 0.7); FPR 0.0028 -> 0.0034. Base detector was **v3**, not P50; the verifier is not yet re-measured on P50 (M, `artifacts/verifier_ablation/ablation.md`) |
| CPU latency | 1.5 ms/crop at batch 1; 1.09 ms/crop at batch 20; 0.91 ms/crop at batch 50 (M) |
| Planned v2 | to be trained on P50 detections from 29,492 images unseen by P50 plus mined pools; **not yet trained or measured** (see `docs/CASCADE_AND_RESOLUTION.md`) |

The verifier is cheap and only runs on detections. It does not change the architecture decision, but it is part of the fire cascade gate.

### 2.3 PPE

| field | Deployed ss-prod `ppe-yolo11` |
|---|---|
| File | `smartsight-models/ppe-yolo11/model.onnx`; byte-identical to `isight-inference/models/ppe_yolo11.onnx` |
| SHA-256 | `f25949ee8673658b6f807dc005bc8f09f57fc530dc9282ca572d18bfff69362b` (M) |
| Architecture | YOLO11m (metadata says YOLO11m; 20.08 M params) |
| Input | 1x3x640x640 static |
| Classes | 13: Boots, Gloves, Goggles, Helmet, Mask, NO-Boots, NO-Gloves, NO-Goggles, NO-Helmet, NO-Mask, NO-Vest, Person, Vest |
| Size | 76.74 MiB ONNX (80.47 MB) |
| Licence | ONNX metadata **AGPL-3.0**; ss-prod manifest says "proprietary" -> **conflict must be resolved** |
| Measured accuracy | In-domain Roboflow test (482 images): mAP50 **0.842**, P **0.662**, R **0.881** (M). Cross-domain Ultralytics test (127): mAP50 **0.116**, P **0.562**, R **0.152** (M). Holdout (433, 4 classes): mAP50 **0.305**, P **0.336**, R **0.481** (M). iSight production-pipeline baseline on 127: P **0.371** / R **0.104** (M, EXP-004). Person-level exact-set accuracy: **95.3%** in-domain (204/214 matched people) but **26.7%** cross-domain (12/45) (M). Cross-domain false alarms: helmet **7/45 = 15.6%**, vest **29/45 = 64.4%** (M). In-domain false alarms: helmet 2/214 = 0.9%, vest 3/214 = 1.4% (M). |
| Claimed accuracy | "90-95%" stakeholder claim; Roboflow 5-class 97.6% (C, different class set and same-distribution validation; **not comparable**) |
| Inference pipeline | RTSP via GStreamer/NVDEC on Jetson or FFmpeg/OpenCV elsewhere; host `sample_fps` default **2.0**; letterbox 640; raw conf default **0.05**; person gate `person_conf >= 0.15`; centre-in-region; multi-scale 1.0 + 1.3; per-label floors; WBF + cross-NMS; tracker; 3-frame debounce; 60 s cooldown; evidence snapshot; TensorRT FP16 engine on Jetson (built on device) |
| CPU latency (M) | 524.6 ms model-only; 614.2 ms end-to-end at 768x432 |
| CPU throughput (M) | 1.6 fps end-to-end; full ss-prod pipeline 0.54 fps at 768x432 (M, audit) |
| Jetson latency | **unknown**; the 20.5 ms TensorRT figure in the ss-prod manifest is for RT-DETRv2-r18, **not** this YOLO11m |

Non-deployed local alternatives, for context only: `ppe.pt` 14-class, 197.9 MiB, mAP50 0.569 Ultralytics / 0.346 holdout (M); `PPE_1.pt` 11-class, val mAP50 0.560 (C).

### 2.4 Vehicle counting

| field | Deployed vehicle-counting stack |
|---|---|
| Models | `yolo11s.onnx` / `yolo11s.pt` (COCO 80) + `autorickshaw.onnx` (2-class aux) |
| SHA-256 | `yolo11s.onnx` `845fe625...41ea2d`; `autorickshaw.onnx` `9b9fabdd...c35f32` |
| Architecture | YOLO11s, 9.44 M; aux 3.05 M, exact arch unknown |
| Input | 1x3x640x640 static |
| Classes | COCO 80 (`person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, ...); aux `autorickshaw`, `object` |
| Size | 36.25 MiB ONNX / 18.42 MiB PT; aux 11.70 MiB |
| Licence | AGPL-3.0 + COCO terms (annotations CC BY 4.0; images Flickr ToU, mixed) |
| Measured accuracy | **Detection accuracy not measured** (no labelled local set). **Counting accuracy not measured** (no ground truth; all existing numbers are pipeline checks). |
| Claimed accuracy | COCO mAP50-95 47.0 for stock YOLO11s (C, public COCO benchmark; not project data) |
| Inference pipeline | OpenCV/FFmpeg decode; `_detect_multiscale` scale 1.0 + 1.5; optional 2x tiling; per-label NMS IoU 0.45; containment rule; aux merge with strict gates; ByteTrack; line crossing (tripwire/pass), count once per track, gap tolerance; per-label display floor 0.40 |
| CPU latency (M) | 189.2 ms model-only; 220.8 ms end-to-end at 1080p; 236.6 ms at 4K. Prior 1080p measurement: 146 ms main + 158 ms aux (M). |
| CPU throughput (M) | 4.5 fps end-to-end at 1080p; 4.2 fps at 4K; 4K full tiling 0.03 fps; 4K far-field tiling 0.08 fps |
| Live pipeline reality | End-to-end HTTP round-trip ~2.5 s -> ~0.4 fps effective; offline replay of the same clip/line gives 68 crossings vs 5 live at 0.3 fps (M). The gap is throughput, not the counter logic. |

**No vehicle-counting bench exists.** The repo has pipeline checks and a proposed validation protocol in `ml/counting/PILOT_PLAN.md`, but no frozen hand-counted set. §8 proposes `bench_V`.

### 2.5 What the current deployment looks like

The current code is **A-ish**: three separate models and mostly separate code paths.

- Fire is a `FireSmokeDetector` class in `production_pipeline.py:1841`.
- PPE is a separate ss-prod edge pipeline (`ppe-yolo11`) with its own runtime.
- Vehicle counting is in `pipeline.py` / `production_pipeline.py` with `yolo11s` + `autorickshaw`.
- There is **no shared decode**, **no shared person/vehicle detector**, and **no cross-camera GPU batching** today.
- ss-prod uses ONNX Runtime on x86 and TensorRT on Jetson; model rollout is manual `rsync` + restart; there is no model canary or automated rollback.

---

## 3. Options A-D

| option | description |
|---|---|
| **A. Three separate services/processes, no sharing** | Fire, PPE and counting each own their decode, preprocess, model and process. Maximum isolation, maximum duplicated work. |
| **B. Shared pipeline, separate models** | One decode + preprocess per camera; a shared person/vehicle detector feeds counting and PPE person-gating; fire/smoke stays a separate model; per-task frame rates; GPU batching across cameras (Triton / TensorRT / ONNX Runtime / DeepStream-style scheduling). |
| **C. Shared backbone with task-specific heads** | One backbone/neck, three heads (fire/smoke, PPE, vehicles). Joint multi-task training required. |
| **D. One unified detector with all classes** | One detector with fire, smoke, 13 PPE classes and vehicle classes. Joint training on a merged dataset. |

Options C and D were **not trained**. Their cost/accuracy are estimated from architecture/FLOPs and the measured per-model costs; no proxy model was trained. This is stated wherever C/D numbers appear.

---

## 4. Comparison table 1: architecture, accuracy and data risk

| metric | A. Separate, no sharing | B. Shared pipeline, separate models | C. Shared backbone + heads | D. One unified detector |
|---|---|---|---|---|
| Fire model | P50 / best_fire_v2 | P50 / best_fire_v2, isolated | shared backbone + fire head | same detector, fire classes |
| PPE model | YOLO11m 13-class | same YOLO11m 13-class | shared backbone + PPE head | same detector, 13 PPE classes |
| Counting model | YOLO11s COCO + aux | same YOLO11s COCO + aux, shared person/vehicle detector | shared backbone + vehicle head | same detector, vehicle classes |
| Shared decode/preprocess | no | **yes** | yes | yes |
| Cross-camera batching | no | **yes** (GPU) | yes | yes |
| Accuracy risk vs today | none (current models) | **none** (models unchanged; integration risk only) | **high** (joint training, partial labels, negative transfer, resolution conflict) | **very high** (partial labels, class imbalance, resolution conflict, label taxonomy conflicts) |
| Data requirement | existing per-task data | existing per-task data | a merged, multi-task labelled dataset or careful partial-label handling | a fully labelled all-class dataset; otherwise unlabeled objects become background |
| Fire recall/FPR risk | baseline | baseline | could regress; must pass bench_R gates | likely regress; must pass bench_R gates |
| PPE cross-domain risk | baseline (already weak: 0.116-0.305 mAP50) | baseline | could improve with shared features or degrade from negative transfer; unknown | likely degrade; 13 PPE classes compete with 80 COCO classes |
| Counting accuracy risk | unmeasured baseline | unmeasured baseline | unknown; needs bench_V | unknown; needs bench_V |
| Licence risk | isolated per model; all Ultralytics AGPL | isolated per model; all Ultralytics AGPL | combines AGPL weights + mixed datasets into one derivative | combines AGPL weights + COCO/Flickr + PPE + fire datasets into one derivative; hardest to clear |
| Failure isolation | **best** | good (separate model processes; shared decode is a single point) | poor (backbone/head coupling) | **worst** (one model for safety + business tasks) |
| Independent versioning/rollback | **best** | good (per-model artifacts; pipeline versioned separately) | poor (backbone update forces all heads) | **worst** (one version for all tasks) |
| Re-validation cost per update | 1 bench | 1 bench per model + integration | all 3 benches | all 3 benches + unified data audit |
| Engineering effort (one-time) | 1-2 person-weeks (E) | 4-6 person-weeks (E) | 12-20 person-weeks + data work (E) | 20-30 person-weeks + data work (E) |
| Pilot readiness | now | 2-4 weeks (E) | 3-6 months (E) | 6-12 months (E) |
| Verdict | safe but wasteful | **recommended** | research only | reject for production |

---

## 5. Comparison table 2: measured CPU and estimated Jetson performance

### 5.1 Measured on this CPU (M)

Required per site (32 cameras):

- fire 2 fps x 32 = **64 fps**
- PPE 5 fps x 32 = **160 fps**
- counting main detector 10 fps x 32 = **320 fps**
- counting aux (autorickshaw) 10 fps x 32 = **320 fps** (excluded from the table below; add ~1 device)
- sum including aux = **864 fps**; table below covers the 544 fps main path

| model path | model-only median | end-to-end median | end-to-end fps | cameras supported at required rate |
|---|---:|---:|---:|---:|
| P50 fire (640) | 184.7 ms | 193.1 ms | 5.2 | 2.6 cameras at 2 fps |
| best_fire_v2 (640) | 87.7 ms | 102.9 ms | 9.7 | 4.9 cameras at 2 fps |
| best_fire_v3 (640) | 85.8 ms | 91.8 ms | 10.9 | 5.4 cameras at 2 fps |
| PPE YOLO11m (640) | 524.6 ms | 614.2 ms | 1.6 | **0.33 cameras at 5 fps** |
| yolo11s COCO (640, 1080p) | 189.2 ms | 220.8 ms | 4.5 | 0.45 cameras at 10 fps |
| yolo11s COCO (640, 4K) | 189.2 ms | 236.6 ms | 4.2 | 0.42 cameras at 10 fps |
| verifier (128) | 1.5 ms/crop | - | - | negligible; 0.9 ms/crop at batch 50 |

Other measured costs:

| quantity | value | label |
|---|---:|---|
| Decode 720p | 163-225 fps (4.4-6.1 ms/frame) | M |
| Decode 1080p | 75.8 fps (13.2 ms/frame) | M |
| Decode 4K | 11.9 fps (84.0 ms/frame) | M |
| Letterbox 1080p -> 640 | 9.95 ms | M |
| Letterbox 4K -> 640 | 7.62 ms | M |
| Letterbox 1080p -> 1280 | 27.84 ms | M |
| Letterbox 4K -> 1280 | 29.47 ms | M |
| Session load, all 6 models | 1.35 s total (0.084-0.396 s each) | M |
| Cold import numpy+cv2+ORT+Ultralytics | 5.07 s | M |
| RSS, base | 61 MB | M |
| RSS, all models loaded | 270 MB | M |
| RSS peak during real inference | 1,618 MB | M |
| CPU load, 3 models concurrent | 83% on 4 threads | M |
| Mixed sequential inference | 2.62 inferences/s at 72% CPU | M |
| Concurrent mixed throughput | 4.42 inferences/s (PPE 0.75 + P50 1.75 + vehicle 1.92) | M |

The "cameras supported" column is arithmetic from single-stream throughput (fps / required fps) and assumes no contention. Real 32-camera contention will be lower, not higher.

**CPU conclusion:** this 4-core CPU cannot support 32 cameras at the required rates. PPE alone needs about **3 CPUs per camera** at 5 fps (1.6 fps measured). CPU-only is not a production option. A GPU is required.

Run-to-run note: CPU latency varies with thermal and background load. Earlier audits on the same class of machine recorded lower single-model values (PPE 255 ms, yolo11s 87.5 ms) than this run's 525 ms and 189 ms. The absolute CPU numbers are therefore approximate; the **relative** ranking (PPE >> yolo11s > P50 > YOLOv12n > verifier) and the conclusion (CPU is not viable for 32 cameras) are unchanged.

### 5.2 P50 batching on CPU (M)

P50 is the only detector with a dynamic ONNX batch dimension (the verifier is also dynamic, but it is a small classifier). Batching does **not** help on this CPU:

| batch | batch latency | per-frame latency | throughput |
|---:|---:|---:|---:|
| 1 | 179.0 ms | 179.0 ms | 5.6 fps |
| 2 | 385.1 ms | 192.6 ms | 5.2 fps |
| 4 | 770.7 ms | 192.7 ms | 5.2 fps |
| 8 | 1,566.7 ms | 195.8 ms | 5.1 fps |

End-to-end P50 batch (preprocess + infer + NMS): 224.6 ms/frame at batch 1, 201.4 ms/frame at batch 4, 198.2 ms/frame at batch 8. That is 8.9 -> 10.1 cameras at 2 fps: a ~13% gain, not a step change. **Batching is a GPU scheduling feature, not a CPU fix.** On Jetson, expect a modest 1.2-1.5x utilization gain (E); it does not reduce FLOPs.

### 5.3 Estimated Jetson capacity (E)

Basis: published Ultralytics Jetson TensorRT FP16, 640, batch 1: **YOLO11s = 188 fps on Orin NX 16GB, 274 fps on AGX Orin 64GB** (third-party measured; E for this pipeline). Scaling to other models uses measured CPU ratios and FLOPs:

- YOLO11m PPE: ~3.0x slower than YOLO11s -> **~63 fps NX, ~91 fps AGX** (E).
- YOLOv12n fire: ~2.2x faster than YOLO11s on CPU -> **~414 fps NX, ~603 fps AGX** (E).
- P50 is YOLO11s -> **188 fps NX, 274 fps AGX** (E).
- Verifier: <0.2 ms/crop (E); negligible.

| task at 32 cameras | required fps | Orin NX 16GB capacity | NX devices needed | AGX Orin 64GB capacity | AGX devices needed |
|---|---:|---:|---:|---:|---:|
| Fire (P50, 2 fps) | 64 | 188 fps | **1** | 274 fps | **1** |
| PPE (YOLO11m, 5 fps) | 160 | 63 fps | **3** | 91 fps | **2** |
| Counting (yolo11s, 10 fps) | 320 | 188 fps | **2** | 274 fps (356 INT8) | **2** FP16 / **1** INT8 |
| **A total (no batching)** | 544 | - | **6** | - | **5** |
| **B total (1.3x batching)** | 544 | - | **5** | - | **4** |
| **C (shared backbone + heads)** | backbone 320 + heads at 64/160/320 | ~2x YOLO11s effective | **2** | ~1x AGX | **1** |
| **D (one all-class detector)** | 320 | 188 fps | **2** | 274 fps (356 INT8) | **1-2** |

These device counts are a **lower bound**. They count one 640 forward pass per task and **exclude**:

- the **autorickshaw aux** (3.05 M params, ~2.9x faster than yolo11s on CPU): add ~1 NX or ~1 AGX at 32 cameras/10 fps;
- the current counting path's **1.0 + 1.5 multi-scale** (roughly 2x the main detector cost): add ~1-2 devices if kept;
- optional tiling (much more expensive; not viable at 10 fps on this class of hardware).

Add those before budgeting. The model-ranking conclusion does not change; the absolute hardware cost rises. The B total is the same model FLOPs as A: sharing the person/vehicle detector saves decode and gives consistent person boxes, but the PPE model still runs full-frame, so its 160 fps is still paid. B's gain is one decode per camera plus the 1.3x GPU batching/utilization factor.

C and D capacity is a **FLOPs estimate**, not a measurement. The estimate assumes a YOLO11s-scale shared backbone at 10 fps, heads cheap relative to the backbone, and no accuracy-driven need to run at 960/1280. If fire needs 1280 for distant smoke, C/D capacity gets worse (see §7.3).

**Per-frame latency on Jetson (E):**

| model | NX latency | AGX latency | required |
|---|---:|---:|---:|
| P50 fire | ~5.3 ms + preprocess/NMS = ~10-15 ms | ~3.6 ms + overhead = ~8-12 ms | <2,000 ms |
| PPE YOLO11m | ~15.9 ms + overhead = ~20-25 ms | ~11.0 ms + overhead = ~15-20 ms | <1,000 ms |
| yolo11s counting | ~5.3 ms + overhead = ~10-15 ms | ~3.6 ms + overhead = ~8-12 ms | <500 ms |
| fire v3 (YOLOv12n) | ~2.4 ms + overhead = ~6-10 ms | ~1.7 ms + overhead = ~5-8 ms | <2,000 ms |
| verifier | <0.2 ms/crop | <0.2 ms/crop | - |

The latency requirement is easy on Jetson. **The binding constraint is aggregate throughput and the PPE model size, not per-frame latency.**

### 5.4 Memory, CPU and startup on Jetson (E)

| quantity | A | B | C | D |
|---|---:|---:|---:|---:|
| Model weights (ONNX/FP16) | ~140 MB | ~140 MB | ~25-40 MB (one backbone + heads) | ~20-30 MB (one detector) |
| Runtime RAM/VRAM estimate | 3-5 GB | 2-4 GB | 1-2 GB | 1-2 GB |
| CPU load (NVDEC offload) | 40-70% | 30-60% | 20-40% | 20-40% |
| Cold start, cached engines | 6-10 s | 6-10 s | 4-8 s | 4-8 s |
| Cold start, first TensorRT build | minutes | minutes | minutes | minutes |

These are E. On NX 16GB and AGX 32/64GB, memory is not the constraint; throughput is.

---

## 6. Comparison table 3: cost

Assumptions (all **E**): 32 cameras/site; Orin NX 16GB node at **$900** (range $700-1,200, module + carrier/box + SSD/power); AGX Orin 64GB node at **$2,800** (range $2,500-3,500); engineering at **$2,000/person-week**; cameras, network, storage, installation and licences excluded. **Hardware is a lower bound**: it excludes the autorickshaw aux and the current multi-scale counting path (add roughly 1-3 devices: ~$900-2,700/site on NX, ~$2,800-8,400/site on AGX, depending on what is kept).

| option | NX nodes/site | AGX nodes/site | NX hardware/site | AGX hardware/site | NX hardware/camera | AGX hardware/camera | engineering (person-weeks) | engineering/site |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Separate, no sharing | 6 | 5 | $5,400 | $14,000 | **$169** | **$438** | 1-2 | $2k-4k |
| B. Shared pipeline, separate models | 5-6 | 4-5 | $4,500-5,400 | $11,200-14,000 | **$141-169** | **$350-438** | 4-6 | $8k-12k |
| C. Shared backbone + heads | 2 | 1 | $1,800 | $2,800 | **$56** | **$88** | 12-20 | $24k-40k |
| D. One unified detector | 2 | 1-2 | $1,800 | $2,800-5,600 | **$56** | **$88-175** | 20-30 | $40k-60k |

**Three-year TCO per camera at one 32-camera site** (hardware + engineering, no ops; E):

| option | NX | AGX |
|---|---:|---:|
| A | ~$231-294 | ~$500-563 |
| B | ~$391-544 | ~$600-813 |
| C | ~$806-1,306 | ~$838-1,338 |
| D | ~$1,306-1,931 | ~$1,338-2,050 |

C/D estimates assume one 640 pass with all classes; A/B totals exclude aux/multi-scale. If either side needs extra passes, its hardware rises. The comparison is directional; the engineering gap (tens of person-weeks) is larger than the hardware uncertainty.

At **one 32-camera site**, A is the cheapest on paper, but its duplicate decode is the scaling risk described in §10. B is the cheapest option that decodes each camera once. C/D are more expensive at one site because engineering dominates; they only win when the engineering is amortized across many sites or the camera count per site is much larger. At 128 cameras/site, hardware dominates more; with platform engineering shared across sites, C can become cheaper than B. That is the main economic argument for C, and it is a scale/amortization argument, not an accuracy argument.

Power is not a decision driver: Orin NX is 10-25 W and AGX Orin is 15-60 W; at $0.15/kWh the energy cost is roughly **$3-8 per camera per year** (E), small versus hardware and engineering.

**Hardware conclusion:** the PPE YOLO11m model is the single biggest hardware cost. If PPE moved to a YOLO11s/n-scale model at the same 5 fps (future retrain), B would need roughly **4 NX** (or **3 NX** if counting is also batched/INT8) or **3-4 AGX**, cutting hardware per camera to roughly **$84-113 (NX)** or **$263-350 (AGX)**. Rate-cutting PPE alone is weaker: at 2 fps the current YOLO11m still needs 2 NX unless batching is very effective.

---

## 7. Accuracy risk (analysis, no training)

### 7.1 Partial labels: the central risk for C and D

The datasets are annotated for one task only:

- Fire data (D-Fire + METU + FASDD-derived) labels **fire/smoke**; persons, vehicles and PPE in those images are **unlabelled background**.
- PPE data labels **PPE/person**; fire, smoke and vehicles are **unlabelled background**.
- COCO labels **80 classes**; fire, smoke and PPE are **not classes**.

If these are naively merged into one detector (D) or jointly trained (C), the loss treats every unlabelled object as background. A person in a fire image is a negative for "person"; an orange vest in a fire image is a negative for "vest"; a fire in a PPE image is a negative for "fire". The model learns contradictions.

Measured evidence of the conflict surface:

| evidence | value | label |
|---|---:|---|
| P50 fire false positives that are **orange PPE** | 3 / 88 = **3.4%** | M (`docs/FP_TAXONOMY.md`) |
| P50 fire false positives that are **persons/skin tones** | 7 / 88 = **8.0%** | M |
| Combined other-task objects in fire FPs | **11.4%** | M |
| P50 fire FPs from warm lights / sunset / haze | 14 + 12 + 11 = **37 / 88 = 42%** | M |
| High-confidence negative detections suspected label noise (fire/smoke present but labelled negative) | 17 / 205 = 8.3% upper bound; ~4-6 plausible (2.4%) | M (`docs/LABEL_AUDIT.md`) |
| PPE model cross-domain collapse | in-domain mAP50 **0.842** -> Ultralytics **0.116** (-86% relative); holdout **0.305** (-64% relative) | M |
| P50 fire recall: val vs bench_R DEV | **92.97%** -> **56.5%** (-36.5 pp) at similar FPR | M |

The PPE cross-domain collapse is the clearest warning: a model that fits its own label distribution can lose most of its accuracy when the distribution changes. A unified model trained on partially labelled data adds a **label-distribution change by construction**, not just a domain change.

The 11.4% other-task share in fire FPs is the exact region where a unified model must learn "orange vest is a vest, not fire" and "person is a person, not background". Without dense multi-task labels on the same images, it cannot learn that reliably.

**Small cross-task proxy (M; 23-30 frames per clip, CPU, conf 0.25; not a unified model, so this is directional only):**

| clip | P50 fire/smoke frames | PPE frames | yolo11s frames |
|---|---:|---:|---:|
| `ppe_2.mp4` (PPE scene) | 0/24 | 24/24 (expected) | 24/24 person/vehicle (expected) |
| `fire.mp4` (fire scene) | 5/30 smoke (expected) | 0/30 | 30/30 person/truck/bus (overlap) |
| `drone.mp4` (aerial traffic) | 6/23 smoke-ish | 12/23 **gloves** (cross-domain false positive) | 23/23 vehicle/person |
| `Car.mp4` (4K road) | 0/30 | 0/30 | 30/30 car/motorcycle/person |

The PPE model firing "gloves" on 52% of aerial drone frames is a concrete cross-domain false positive. The vehicle model firing on every fire and PPE frame is expected (people/vehicles are present), but it shows that the same pixels carry multiple task labels. A unified model must resolve these overlaps explicitly; the current per-task models do not have to.

### 7.2 Class imbalance

Dataset sizes:

| dataset | images | label |
|---|---:|---|
| COCO (vehicles/persons) | ~118,000 train | E (public dataset statistic) |
| Fire P50 pool | 29,493 | M |
| PPE train | 7,721 | M |
| PPE in-domain test | 482 | M |
| bench_R | 1,000 | M |

If sampled proportionally, COCO is ~76% of images and fire ~19%, PPE ~5%. COCO contributes ~1.2M instances (E); fire and PPE contribute far fewer. In a unified detector, the head classes (person, car, truck) dominate the loss, and fire/smoke/PPE become long-tail classes. The expected effect is lower recall on the rare classes unless the sampler is heavily rebalanced, which in turn over-samples the same fire/PPE images and increases overfitting. This is an E effect (instance counts for fire/PPE train are not on disk), but the direction is well established.

A unified model also has a **class-taxonomy conflict**: COCO `person` vs PPE `Person`; COCO `truck` vs autorickshaw; fire `smoke` vs construction dust/steam; PPE `mask` vs face mask/medical mask. Merging class names without a label audit creates ambiguous supervision.

### 7.3 Resolution conflict

Measured on P50's remaining misses (`docs/CASCADE_AND_RESOLUTION.md`):

| mode | throughput | misses recovered (of 53) | control positives | control negatives flagged (FPR) |
|---|---:|---:|---:|---:|
| 640 baseline | 6.67 img/s | 0 | 200/200 | 8/300 (2.67%) |
| 960 | 2.71 img/s | 5 (9.4%) | 197/200 | 16/300 (5.33%) |
| 1280 | 1.47 img/s | 14 (26.4%) | 188/200 | 29/300 (9.67%) |
| tiled 2x2 @640 | 2.42 img/s | 8 (15.1%) | 169/200 | 25/300 (8.33%) |

Vantage breakdown: at 1280, **46.2% of aerial/high-camera misses** were recovered vs **0% at 640**; ground/CCTV recovered 20.0%. But 1280 is **4.5x slower** and raises the negative-flag rate from 2.67% to 9.67% (**3.6x**). Tiling loses large-flame control positives (84.5% vs 100%).

The conflict is structural:

- **Distant smoke** needs 960-1280 or tiling; 640 misses it.
- **Gloves/goggles/helmets** on distant workers need high resolution and person crops.
- **Vehicles** need context and can tolerate 640, but small far vehicles benefit from 960.
- **PPE compliance** needs fine detail at the person level; a 640 full-frame detector already runs at 20.08 M params and is the compute bottleneck.

One unified input size cannot satisfy all three. Choosing 640 to keep cost down sacrifices fire recall; choosing 1280 to recover smoke multiplies compute and false alarms for every task. This is the strongest technical argument against D and, to a lesser extent, C.

### 7.4 Why C is not automatically safer than D

C keeps task-specific heads, which helps class separation and allows per-task losses. But the backbone is still trained on a merged, partially labelled dataset, so the same unlabelled-object problem applies to the shared features. C can be better than D if:

- the heads are trained with per-task losses that ignore unlabelled objects (masked loss), and
- the data loader samples tasks in a controlled way, and
- the backbone is not dominated by COCO, and
- fire is allowed a higher-resolution path or a separate high-res head.

Those are real research/engineering tasks, not a drop-in. Without them, C is a more complex way to get D's partial-label problem.

---

## 8. Accuracy gates any unified model (C or D) would have to meet

These gates are defined from the existing benches. A unified model must meet **all three task gates simultaneously**, with thresholds calibrated on val only and **not tuned on the benches**. "No regression" means the lower 95% CI bound is not worse than the current model's point estimate, or the difference is within a pre-registered non-inferiority margin (suggest 2 pp).

### 8.1 Fire gate (bench_R; the exam)

| gate | requirement | current evidence |
|---|---|---|
| F1 | On bench_R **DEV** (300 images, 85 pos / 215 neg), at FPR <= **2.33%** per negative image, recall >= **56.5%** | P50 measured 56.5% @ 2.33% (M) |
| F2 | On bench_R **SEALED** (700 images, one-shot), at FPR <= **2.20%**, recall >= **47.9%** and no worse than P50 on the same one-shot run | only cascade_v3 SEALED baseline: recall 47.9%, FPR 2.20% (M); P50 SEALED unknown |
| F3 | Event-level false alarms <= **1.0/camera-week** (dashboard) and <= **0.1/camera-week** (dispatch) | agreed budget, **unmeasured** (needs 10 cameras x 48 h) |
| F4 | Uncontrolled-fire recall >= **96.8%**; controlled/spark/screen do not alarm by default | measured on locked testbench100 labels (M); that bench is contaminated (24 logged uses) and is a smoke test, not the exam; electrical_arc/screen unmeasured (0 bench_R images) |
| F5 | Temporal path adds <= **+0.6 s** to time-to-detect vs raw | v3 temporal: mean +0.17 s, max +0.6 s (M) |
| F6 | With verifier at 0.5-0.7, recall@FPR1% >= **0.66-0.68** and FPR <= **0.0034** | v3 ablation: 0.660-0.680 at 0.0034 (M) |

A unified model that cannot meet F1/F2 on the frozen bench is not production-eligible for fire, regardless of its PPE or counting scores. Fire is the safety-critical task; it gates the whole decision.

### 8.2 PPE gate (bench_P; the ss-prod audit sets)

There is no file literally named `bench_P`. The ss-prod audit harness (`ppesight/audit_eval.py`) with its three frozen sets is the de facto PPE bench. Treat it as bench_P and freeze it.

These are **no-regression gates relative to the current deployed model**. Several current numbers are poor (for example 26.7% cross-domain person-level accuracy and 64.4% cross-domain vest false alarms). The gate prevents a unified model from making them worse; it does not declare the current numbers acceptable.

| gate | requirement | current evidence |
|---|---|---|
| P1 | In-domain Roboflow test 482 images: mAP50 >= **0.842**, P >= **0.662**, R >= **0.881** | deployed model measured (M) |
| P2 | Cross-domain Ultralytics test 127: mAP50 >= **0.116**, R >= **0.152**; holdout 433: mAP50 >= **0.305**, R >= **0.481** | deployed model measured (M) |
| P3 | Person-level helmet/vest compliance accuracy >= **95.3%** in-domain; cross-domain exact-set accuracy not below **26.7%** (current measured) | measured (M): 95.3% in-domain, 26.7% cross-domain |
| P4 | False "no-helmet" alerts on visibly helmeted workers: in-domain <= **1%**; cross-domain <= **15.6%** (current measured) | in-domain 2/214 = 0.9%; cross-domain 7/45 = 15.6% (M) |
| P5 | Per-class operating recall (in-domain): helmet >= **0.90**, vest >= **0.85**, gloves >= **0.75**, boots >= **0.70**, goggles >= **0.85** | measured op recall: helmet 0.928, vest 0.923, gloves 0.764, boots 0.729, goggles 0.891 (M) |
| P6 | Missing-lookup / `none` class handling: no silent "compliant" when the class is absent | policy gap; must be fixed in the product, not the model |

The cross-domain gates are deliberately set to the **current weak numbers**. A unified model is allowed to be no worse than today; it is not allowed to trade PPE cross-domain accuracy for fire or counting. P4 is where the current model's cross-domain behaviour is clearly unacceptable (15.6% false "no-helmet" on helmeted workers); it must be fixed or explicitly accepted before any pilot.

### 8.3 Vehicle-counting gate (bench_V; proposed)

No vehicle-counting bench exists. Propose a small, frozen bench:

- **bench_V:** 20 clips x 5 minutes, 1080p, from the pilot sites, covering day/night, near/far, dense/sparse, car/truck/bus/motorcycle/auto-rickshaw.
- Hand-count 3 x 60 s windows per clip (start/mid/end) by direction, using the `ml/video_eval/MANIFEST_SCHEMA.md` count-error schema.
- Freeze clip hashes and hand counts; never tune on it.

| gate | requirement | current evidence |
|---|---|---|
| V1 | Detection mAP50 >= **0.60** for car/truck/bus/motorcycle on a frozen 500-image Indian-site set; person/vehicle recall >= **0.90** at conf 0.40 | **unmeasured**; yolo11s is stock COCO |
| V2 | Count error per 60 s window: MAE <= **10%**, within-±1 <= **90%**; direction accuracy >= **95%** | **unmeasured** |
| V3 | Track ID switches <= **5%** per track; missed crossings <= **5%**; false crossings <= **2 per 10 min per camera** | **unmeasured** |
| V4 | At 10 fps, p95 end-to-end latency <= **500 ms**; frame-drop-induced undercount <= **5%** | CPU measured 4.5 fps (M); Jetson latency E; count undercount unmeasured |
| V5 | Auto-rickshaw aux does not relabel cars/motorcycles: false-AR rate <= **2%** on a no-AR clip | aux strict gates measured 11 -> 3 FPs on one clip (M); not a rate |

V2/V3 are the real product gates. A detector mAP is not a count accuracy claim.

---

## 9. Operations comparison

| dimension | A. Separate, no sharing | B. Shared pipeline, separate models | C. Shared backbone + heads | D. One unified detector |
|---|---|---|---|---|
| What happens if fire model crashes | PPE/counting unaffected | PPE/counting unaffected if fire is a separate process; shared decode failure affects all | fire head/backbone failure can affect all | all tasks down |
| What happens if PPE model regresses | fire/counting unaffected | fire/counting unaffected | backbone update may regress all heads | all tasks regress |
| Bad model rollback | per model, independent | per model, independent; pipeline version separate | coordinated backbone + all heads | one rollback for all |
| Versioning | simple | simple per model | complex, coupled | complex, coupled |
| Update cadence | fire (hard negatives), PPE (site/PPE changes), counting (vehicle mix/regions) independent | same, plus pipeline releases | every fire update re-validates PPE/counting | every update re-validates everything |
| Re-validation cost | 1 bench | 1 bench + integration | 3 benches | 3 benches + data audit |
| Monitoring | per-service metrics, easy attribution | per-model + pipeline; attribution still clear | per-head metrics, harder to attribute a regression | per-class metrics; hardest to attribute |
| Safety criticality | fire isolated from business tasks | fire isolated if separate process/container | fire coupled to non-safety backbone | fire coupled to counting/PPE |
| Licence implications | each model isolated; all Ultralytics AGPL; COCO terms on vehicle; PPE dataset CC BY 4.0 | same; fire model can be replaced independently | one derivative combines AGPL + mixed datasets | one derivative combines AGPL + COCO/Flickr + PPE + fire datasets; hardest to clear |
| Security/privacy | PPE worker images stay in PPE service | same, if data paths are separated | shared feature extractor may mix data | shared model mixes all data |
| Engineering effort | low | medium | high | very high |
| Pilot risk | low | medium | high | very high |
| Time to pilot | now | 2-4 weeks (E) | 3-6 months (E) | 6-12 months (E) |

**Licence note (AGPL vs Apache):** every current Ultralytics-trained detector (fire, PPE, vehicle, aux) is **AGPL-3.0 for code and weights**. The PPE ONNX metadata says AGPL-3.0 while the ss-prod manifest says "proprietary"; that conflict must be resolved. The vehicle model also carries COCO terms (annotations CC BY 4.0; images under Flickr ToU, some NC). The fire data sources are permissive (D-Fire CC0, METU CC BY 4.0), but the YOLO framework licence dominates the shipped weights. A closed commercial product therefore needs an **Ultralytics Enterprise License**, or the detectors must be replaced with Apache/MIT/BSD alternatives. Only PaddleOCR (Apache-2.0) is clearly permissive today; RT-DETR code is Apache-2.0 but the FireViewer weights are non-redistributable, and Grounding DINO's weights/data terms are not a clean commercial grant. A unified model does not fix this; it **concentrates** the licence problem into one artifact and makes incremental replacement harder. Keeping models separate lets you replace fire, PPE or counting independently.

**Operational conclusion:** fire alarms are safety-critical. Coupling fire to a model that also counts vehicles is an unacceptable failure mode for a production safety system. Even if C/D were more accurate on a bench, the shared failure domain and coupled rollback make them a poor operational choice for fire. If C is ever pursued, fire should remain a separate model/service; C would be a PPE+counting consolidation at most.

---

## 10. Recommendation

### 10.1 Decision

**Option B: shared pipeline, separate models.**

Concretely:

1. **One decode + preprocess per camera.** Read each RTSP stream once, decode with NVDEC/GStreamer, letterbox once, and publish frames to per-task queues. Do not decode the same camera three times.
2. **Separate model artifacts and processes:**
   - Fire: P50 (or the validated fire model) in its **own process/container**, with its own watchdog, model version, thresholds, temporal confirmer and alarm manager. Fire must not share a process with counting/PPE.
   - PPE: `ppe-yolo11` (or its validated successor) in the PPE service.
   - Counting: `yolo11s` + `autorickshaw` in the counting service.
3. **Share the person/vehicle detector** between counting and PPE person-gating. Use the same `yolo11s` output to produce person boxes for PPE association and vehicle tracks for counting. This saves decode and gives consistent person boxes, but **it does not remove the PPE model's own compute**: the current PPE model is still a full-frame YOLO11m.
4. **Per-task frame rates:** fire 2 fps, PPE 5 fps, counting 10 fps. Use timestamp-based sampling; never let a slow task back-pressure the decode.
5. **GPU batching across cameras** for fire and counting: P50 has a dynamic ONNX batch dimension; `yolo11s` is static batch 1 and needs a no-retrain re-export with batch>1 for efficient batching. On Jetson, TensorRT FP16, batch 1-8 with per-task CUDA streams / Triton / DeepStream-style scheduling. Build explicit TensorRT optimization profiles for P50 (640x640, batch 1-8); dynamic height/width is convenient for export but can select slower kernels. Batching on CPU gave no gain (M); on GPU expect 1.2-1.5x (E).
6. **Keep A's isolation where it matters:** fire's process, model and rollback are independent. The shared pipeline is a data-plane optimization, not a shared model.

### 10.2 What it costs

At 32 cameras, with the current YOLO11m PPE model (lower bound; excludes aux and multi-scale):

| item | NX | AGX |
|---|---:|---:|
| Devices | 5-6 Orin NX 16GB (E) | 4-5 AGX Orin 64GB (E) |
| Hardware/site | $4,500-5,400 (E) | $11,200-14,000 (E) |
| Hardware/camera | $141-169 (E) | $350-438 (E) |
| Engineering | 4-6 person-weeks (E) | 4-6 person-weeks (E) |
| 3-year TCO/camera, one site | ~$391-544 (E) | ~$600-813 (E) |

The PPE model is ~3 of the NX devices on this lower-bound basis. If PPE is replaced by a YOLO11s/n-scale model at the same 5 fps (future retrain), B drops to ~4 NX (or ~3 NX with counting batched/INT8) or ~3-4 AGX before aux/multi-scale, and hardware/camera falls to ~$84-113 (NX) / ~$263-350 (AGX) (E). That is a much better lever than unification.

**Why B rather than A, given the TCO table:** at one 32-camera site, A looks cheaper because its engineering estimate is lower. But A's duplicate decode is a serious scaling risk at this camera count. 32 x 1080p30 = **960 fps of decode per copy**. Measured CPU decode is **75.8 fps per stream-copy on one core** (M), so one copy alone needs ~13 cores; three copies need ~38 cores. Orin NX has one NVDEC, and duplicating decode three times may exceed it or force extra devices. A also needs 6-8 devices vs B's 5-7 (depending on whether the aux and multi-scale path are kept). B's single decode and fan-out is the safer data plane at 32 cameras. The fire process remains separate inside B, so A's most important isolation property is preserved without A's duplicated decode.

### 10.3 Why not C or D

- **D is rejected.** Partial labels, class imbalance and resolution conflict are not theoretical here: the fire model already false-alarms on orange PPE and persons (11.4% of its FPs), the PPE model collapses from 0.842 to 0.116 mAP50 cross-domain, and 640 recovers 0 of the 53 remaining misses (mostly small/distant smoke) while 1280 costs 4.5x throughput and 3.6x negative-flag rate. One all-class detector would have to solve all of that simultaneously, with a merged dataset that does not exist, for a safety-critical task. The accuracy risk is too high.
- **C is not production-ready.** It is the right long-term research direction if a fully labelled multi-task dataset and masked per-task losses exist. It is not a drop-in, and it still couples fire to non-safety tasks unless fire stays separate. A C pilot should be a PPE+counting consolidation, not fire.
- **A is safe but wasteful.** It duplicates decode and prevents batching. B keeps A's model isolation while removing the duplicate data plane.

---

## 11. What would change the recommendation

| change | effect |
|---|---|
| **Camera count per site < 8** | A or B fits on one NX/AGX. Unification has no hardware payoff; stay separate. |
| **Camera count > 128, or many sites share the same software** | C/D hardware savings become material; C becomes worth a pilot **if** the accuracy gates and a multi-task dataset exist. D remains high-risk. |
| **GPU budget < ~$3,000/site** | B's 5-7 NX is unaffordable. First lever: shrink/replace the PPE model or cut PPE rate. Second: C pilot (PPE+counting only). Not D. |
| **PPE model replaced by YOLO11s/n or a crop classifier** | B drops to ~4 NX / ~3-4 AGX (or ~3 NX / ~2-3 AGX with counting batched/INT8); unification becomes even less attractive. |
| **Fire accuracy/FPR requirement tightens, or certification is required** | Keep fire separate, on its own device/process, regardless of C/D economics. |
| **A fully labelled multi-task dataset (fire, smoke, PPE, vehicles on the same images) becomes available** | C/D risk drops. Still require all gates in §8; still keep fire separate if safety certification is needed. |
| **Fire must run at 960/1280 for distant smoke** | C/D capacity worsens (measured 4.5x slower at 1280); B with a dedicated fire device is safer. |
| **Ultralytics Enterprise License is not obtained** | All Ultralytics models must be replaced. Separate models allow incremental replacement; a unified model concentrates the licence problem. |
| **Power/thermal limit on NX (25 W)** | Measure; if throttling, use AGX or fewer cameras per device. This could make C's lower compute more attractive. |

---

## 12. Migration plan

### Phase 0 - freeze the baseline (week 0-2)

1. Record current model hashes and bench versions: P50, best_fire_v2, ppe-yolo11, yolo11s, autorickshaw, verifier.
2. Reproduce the CPU measurements in this doc on the actual Jetson Orin NX and AGX Orin with TensorRT FP16: P50, PPE, yolo11s, verifier; batch 1 and 4/8; p50/p95 latency, power, thermals.
3. Measure 32-camera NVDEC decode: 1080p30 and 4K; confirm NVDEC capacity and CPU load.
4. Run P50 on bench_R SEALED **once**, log it, and keep it as the fire baseline. Do not tune on SEALED.
5. Freeze bench_P (three ss-prod sets) and create bench_V (§8.3).

**Exit gate:** measured Jetson latency/throughput within 20% of the estimates; NVDEC can sustain 32 streams; fire SEALED baseline recorded.

### Phase 1 - implement B (week 2-6)

1. Build a shared decode/preprocess service: one capture per camera, NVDEC, letterbox once, timestamped frame bus.
2. Split the runtime into three model services (fire, PPE, counting) with per-task queues and frame-rate governors.
3. Fire service: P50 + verifier + temporal confirmer + alarm manager, own process/container, own model version, watchdog, shadow mode first. No dependency on PPE/counting.
4. PPE service: `ppe-yolo11`, person boxes from the shared `yolo11s` detector (or its own person class), per-camera ROI/floors.
5. Counting service: `yolo11s` + `autorickshaw`, ByteTrack, line crossing; per-camera lines/zones.
6. GPU scheduling: TensorRT FP16 engines; P50 dynamic batch; re-export `yolo11s` with batch>1 (no retrain); per-task CUDA streams or Triton/DeepStream; batch across cameras.
7. Add per-model metrics: latency p50/p95, queue depth, dropped frames, detections/frame, alarm rate, model version. Fire gets its own alerting and audit log.

**Exit gate:** B runs 32 simulated cameras at 2/5/10 fps on the target device(s) with p95 latency within targets and no frame-drop-induced counting loss >5%.

### Phase 2 - validate and pilot (week 6-10)

1. Fire: bench_R DEV + SEALED (one-shot), verifier ablation, shadow mode on 10 cameras for 2 weeks, event-level false alarms per camera-week.
2. PPE: bench_P three sets, person-level compliance, false "no-helmet" audit on helmeted workers, shadow mode.
3. Counting: bench_V, per-window count error, direction accuracy, track ID switches, false crossings.
4. Canary one site; keep A as rollback for 2 weeks.

**Exit gate:** all §8 gates met; fire event-level budget met or explicitly accepted as unmeasured; rollback tested.

### Phase 3 - future (only if justified)

- **PPE cost reduction:** retrain/replace PPE with a YOLO11s/n-scale model or a person-crop classifier (out of scope for this study). Re-run bench_P. Target: 4 NX / 3-4 AGX for B (3 NX / 2-3 AGX with counting batched/INT8).
- **C research pilot (PPE+counting only):** build a masked multi-task dataset, train a shared backbone with two heads, and run a small proxy experiment. Do not include fire. Gate on bench_P + bench_V. Only proceed if B's hardware cost at scale is the binding constraint.
- **D:** not recommended. Revisit only if a fully labelled all-class dataset, a multi-task training method that handles partial labels, and a resolution strategy (e.g., high-res fire head) all exist and pass the gates.

---

## 13. Numbers still unknown

| unknown | why it matters | how to close it |
|---|---|---|
| Jetson Orin NX/AGX TensorRT FP16 latency for the exact P50, PPE, yolo11s and verifier artifacts | all capacity/cost estimates are scaled from published YOLO11s numbers | build engines on the target and re-run this benchmark |
| 32-camera NVDEC decode capacity and CPU load | decode can become the bottleneck before the models do | 32-stream 1080p30/4K soak test |
| P50 on bench_R SEALED | the only fire exam that has not been run for P50 | one-shot SEALED run, logged |
| Event-level fire false alarms per camera-week | image FPR does not convert to event rate | 10 cameras x 48 h end-to-end shadow run |
| Vehicle-counting accuracy and count error | no bench exists; all counting numbers are pipeline checks | build bench_V, hand-count 3 x 60 s per clip |
| PPE cross-domain accuracy on real site footage | current cross-domain numbers are public datasets, not the customer's site | site-representative frozen set, shadow mode |
| PPE false "no-helmet" rate on helmeted workers | the audit found real false alerts; not quantified | audit 200-500 person-frames per site |
| Verifier v2 accuracy on the independent >=5,000-image negative set | planned verifier is not trained/measured | train and evaluate per `docs/CASCADE_AND_RESOLUTION.md` |
| Fire performance at 960/1280 on the target camera mix | resolution is the main recall lever and the main cost lever | per-camera resolution study in shadow mode |
| TensorRT engine sizes and VRAM per model | memory estimate is E | measure on target |
| Power/thermal throttling on Orin NX at 25 W with 32 streams | could reduce capacity below the estimate | soak test with `tegrastats` |
| Ultralytics Enterprise License cost | a hard production blocker independent of architecture | legal/commercial decision |
| Actual camera resolutions, codecs, bitrates, reconnect behaviour | decode and preprocess costs depend on them | site survey |
| Whether a fully labelled multi-task dataset can be built | the gate for C/D | data audit and labelling plan |

---

## 14. Appendix: raw measurement notes

- Benchmark host: Intel i7-1165G7, 4c/8t, 15.7 GB RAM, no NVIDIA GPU, Windows, ONNX Runtime 1.26.0 CPU, 4 intra-op threads.
- Models timed with random/real tensors after 3-5 warm-up runs; medians and p95 over 15-30 runs. End-to-end includes letterbox, inference and YOLO decode/NMS.
- Video decode measured with OpenCV `grab()`/`retrieve()` on the actual sample videos, 60 frames per clip.
- Model load times measured as cold `InferenceSession` creation within one process; cold import measured in a fresh subprocess.
- CPU load measured with `psutil` while three models ran concurrently for 12 s, and while a mixed sequential loop ran for 8 s.
- All accuracy numbers are copied from the cited repo artifacts (`ml/eval/baselines`, `ml/eval/EXP-004`, `ppesight/artifacts`, `bench_results`, `REPORT.md`, `docs/CASCADE_AND_RESOLUTION.md`, `docs/LABEL_AUDIT.md`, `docs/FP_TAXONOMY.md`); they were not re-run for this study except where marked as this study's measurement.
- Jetson scaling basis: published Ultralytics Jetson TensorRT FP16 640 b1 (YOLO11s 188 fps Orin NX 16GB, 274 fps AGX Orin 64GB), plus measured CPU ratios for YOLO11m and YOLOv12n. These are E, not M.
- No model was trained, fine-tuned, exported or modified. The only writes were this document and temporary benchmark scripts outside the repo.

---

### Final answer to the lead

**Keep them separate (B).** Share the data plane and the person/vehicle detector; keep fire, PPE and counting as independent models and independent failure domains. The measured data does not support unification: the PPE model is already fragile cross-domain (0.842 -> 0.116 mAP50), fire loses 36.5 pp recall from val to bench_R DEV, 11.4% of fire false positives are other-task objects, and 640 recovers 0 of the 53 remaining misses (mostly small/distant smoke). A unified model would have to solve partial labels, class imbalance and resolution conflict at once, for a safety-critical alarm, with a dataset that does not exist. The cost problem is real, but the right fix is to shrink the PPE model, not to merge it with fire.
