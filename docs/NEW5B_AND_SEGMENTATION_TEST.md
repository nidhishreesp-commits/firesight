# 5 new clips + segmentation test

## 1. P50 smoke policy on the 5 new clips

| clip | label | raw | layer | max fire | max smoke |
|---|---|---|---:|---:|---:|
| Incense smoke (`200929`) | smoke | no | **no** | 0.000 | 0.000 |
| Colored smoke flare (`167889`) | smoke | yes | **no** | 0.000 | 0.598 |
| Volcano 1 (`152174`) | fire | yes | **yes** | 0.831 | 0.657 |
| Mountain clouds (`37079`) | non-fire | yes | **false alarm** | 0.000 | 0.614 |
| Volcano 2 (`152085`) | fire | yes | **yes** | 0.633 | 0.728 |

- Fire: **2/2 detected**.
- Smoke: **0/2 detected** at the final layer. The flare was visible raw but not persisted; incense was not detected at all.
- Cloud clip false-alarmed; the user says sky/cloud is not a deployment case, but it confirms the cloud weakness.

These are not normal CCTV fire cases (incense, smoke flare, volcano, mountain fog), so treat them as edge cases, not a general accuracy estimate.

## 2. Fire segmentation

Model: `sreeharivp23/fire-segmentation-yolo11n` (YOLO11n-seg, 1 class `fire`, 6 MB).

Checkpoint metrics: mask mAP50 **0.550**, mask mAP50-95 **0.287**, mask recall **0.507**.
Licence: **AGPL-3.0** (Ultralytics).

4 sampled frames per clip:

| clip | frames with fire mask | max conf | max mask area |
|---|---:|---:|---:|
| Volcano 1 | **4/4** | 0.845 | 11.6% |
| Volcano 2 | **4/4** | 0.956 | 6.6% |
| Household fire | 1/4 | 0.324 | 2.7% |
| Vehicle fire | 1/4 | 0.617 | 4.2% |
| Incense smoke | 0/4 | — | — |
| Colored smoke flare | 0/4 | — | — |
| Chimney smoke | 0/4 | — | — |
| Mountain clouds | 0/4 | — | — |

It segments lava/fire well on the volcano clips, but **over-segments smoke/cloud as fire** on those
frames, misses most of the household/vehicle fire frames, and does **not** segment smoke at all.
Not production material.

## 3. Smoke segmentation

### SmokeSegmenter (`fl0wxr/SmokeSegmenter`, MIT code)

- The repo reports BiSeNet-R18 and PIDNet-S fine-tuned on S-Smoke (2,400 train / 287 test).
- The provided `cityscapes-bisenet-R18.pth` downloaded from Drive is a **19-class Cityscapes**
  checkpoint, not a smoke-fine-tuned 2-class model. Loading it into the 2-class network fails with
  classifier shape mismatch.
- `PIDNet_S_Camvid_Test.pt` is **access-blocked** on Google Drive (“cannot retrieve public link”).
- No usable smoke-segmentation weights were obtained. Training S-Smoke would be required.

### HF `Pravesh-21/fire-segmentation-dataset`

- 1,348 images, **fire only**, YOLO-seg polygons; 250 negatives.
- Labels are **machine-generated** (YOLOv8 boxes → SAM 2.1 masks → polygons).
- Licence: `other / mixed-source`, based on CAIR Fire-Detection and DeepQuestAI Fire-Smoke.
- This is a training dataset, not a smoke model. No smoke class.

### Roboflow Universe fire-and-smoke-segmentation

- Page returns **403** without a Roboflow account/API key.
- Hosted project; no weights tested. Do not send private CCTV to a hosted API.

## Verdict

**Segmentation is not a shortcut right now.**

- Fire segmentation is available but weak (mask mAP50 0.55) and fire-only.
- Smoke segmentation has no usable pretrained weights in these sources; it needs training on
  S-Smoke/D-Fire or another smoke-mask dataset.
- The P50 smoke policy fix remains the best current improvement.

If we pursue segmentation, the right use is a **second-stage smoke verifier / growth estimator**,
not a primary detector: train RF-DETR-Seg or YOLO11n-seg on fire+smoke masks with camera-disjoint
splits, then use mask area/growth to separate smoke from cloud. That is a Phase 2/3 task after more
smoke-only validation clips.

## Review artifacts

- New clips P50: `build/sample_videos_review/new5b_p50/`
- Fire segmentation contact sheet: `build/segmentation_review/fire_seg_contact_sheet.jpg`
- Fire segmentation metrics: `build/segmentation_review/fire_seg_metrics.json`
- Fire segmentation model: `models/external/fire_seg_yolo11n.pt`
