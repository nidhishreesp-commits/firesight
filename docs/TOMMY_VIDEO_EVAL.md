# TommyNgx YOLOv10: check on Sample_videos

## What was actually downloaded

- User file: `C:\Users\Nidhi\Downloads\best.pt`
- Copied to: `models/external/tommy_yolov10_best.pt`
- Size: 64,111,730 bytes
- Architecture: `yolov10x.yaml`, 31.8M parameters, 2 classes `{0: Fire, 1: Smoke}`
- Checkpoint date: 2024-07-22; Ultralytics version 8.2.59
- **Checkpoint licence: AGPL-3.0**, despite the HF card saying Apache-2.0.
- `C:\Users\Nidhi\Downloads\pytorch_model.bin` is a separate HF/Transformers weight
  archive, not the `best.pt` used here. It was not used.

The downloaded model's own validation metrics:

| metric | value |
|---|---:|
| precision | 0.549 |
| recall | 0.640 |
| mAP50 | 0.601 |
| mAP50-95 | 0.404 |

The HF card's `mAP 0.85` is **unverified** and does not match the checkpoint's recorded
validation metrics. It is also mAP, not video/clip accuracy.

## Sample_videos: 11 clips (2 fire, 9 non-fire)

Tommy thresholds: fire 0.25, smoke 0.25 (model-card default), 2 fps, temporal + video layer.

| model | TP | FP | FN | precision | recall | FPR | accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **P50** | 2 | 1 | 0 | 66.7% | 100% | 11.1% | **90.9%** |
| **Salah** | 2 | 2 | 0 | 50.0% | 100% | 22.2% | 81.8% |
| **Tommy YOLOv10** | 2 | 2 | 0 | 50.0% | 100% | 22.2% | 81.8% |

Tommy detects both fires (`fire.mp4`, `fire_2.mp4`). Its false alarms:

- `Attendance.MOV` — office scene, smoke max 0.648
- `ppe_4.mp4` — grinding sparks, fire max 0.679

Salah false-alarms on `ppe_3` construction and `ppe_4` sparks. P50 false-alarms only on
`drone.mp4` sky (which the user says is not a deployment case).

## Accuracy difference

- **P50 is 9.1 percentage points better than Tommy** on Sample_videos (90.9% vs 81.8%).
- Tommy and Salah are tied at 81.8%.
- P50 precision is 16.7 points higher; P50 FPR is 11.1 points lower.
- Tommy is **not better than P50**, and only ties Salah.

## Verdict

**Do not lock Tommy.** On the only set where all three were actually tested (Sample_videos),
Tommy is tied with Salah and worse than P50. Its own checkpoint metrics are weak (mAP50 0.601),
the 85% card claim is unverified, and the checkpoint is AGPL-3.0 (a legal risk for proprietary
deployment).

The 20 hard clips and 42 public clips were not completed for Tommy: the 31.8M YOLOv10x CPU run
timed out after 40 minutes. The Sample_videos result is enough to reject it as the base model.
If a full run is required, export to ONNX or use GPU, then run the same manifests.

## Annotated review

- `build/sample_videos_review/tommy/annotated/`
- Summary: `build/sample_videos_review/tommy/summary.json`
- Per clip: `build/sample_videos_review/tommy/per_clip.csv`

## Recommendation

Keep **P50 as the base model**. Keep Salah as a secondary candidate. Tommy adds no benefit on the
real sample clips and brings an AGPL licence issue; do not spend remaining time on it.
