# Sample_videos: P50 annotated run

Input: `C:\Users\Nidhi\My projects\Sample_videos` (11 clips). Model: P50 ONNX, 5 fps sampling,
boxes drawn at conf >=0.10, alert summary at P50 thresholds fire 0.29 / smoke 0.33. No temporal
filter and no verifier in this pass. This is the one final held-out video check.

| video | actual | P50 alert | max fire | max smoke | first fire | first smoke | annotated |
|---|---|---:|---:|---:|---:|---:|---|
| fire.mp4 | fire | **yes** | 0.8645 | 0.6544 | 42.33 s | 1.67 s | `build/annotated/P50/fire_annotated.mp4` |
| fire_2.mp4 | fire | **yes** | 0.8791 | 0.8673 | 7.40 s | 7.20 s | `build/annotated/P50/fire_2_annotated.mp4` |
| drone.mp4 | nonfire | **yes (false alarm)** | 0.0 | 0.634 | - | 3.80 s | `build/annotated/P50/drone_annotated.mp4` |
| Attendance.MOV | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/Attendance_annotated.mp4` |
| Car.mp4 | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/Car_annotated.mp4` |
| number_plate.mp4 | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/number_plate_annotated.mp4` |
| ppe_1.mp4 | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/ppe_1_annotated.mp4` |
| ppe_2.mp4 | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/ppe_2_annotated.mp4` |
| ppe_3.mp4 | nonfire | no | 0.0 | 0.0 | - | - | `build/annotated/P50/ppe_3_annotated.mp4` |
| ppe_4.mp4 | nonfire | no | 0.0775 | 0.0 | - | - | `build/annotated/P50/ppe_4_annotated.mp4` |
| theft.mp4 | nonfire | no | 0.0 | 0.2074 | - | - | `build/annotated/P50/theft_annotated.mp4` |

Result: **2/2 fire clips detected; 1/9 nonfire clips false-alarmed** (drone: smoke box over the
background, max 0.634). The annotated drone video shows the false positive directly.

TommyNgx: **not run.** The HF repo is gated (`gated=auto`) and returns HTTP 401 without an
`HF_TOKEN`. No annotated Tommy video exists. The other public model (rabahdev) was benchmarked on
bench_R DEV in `docs/EXTERNAL_MODELS.md` and was worse than P50.
