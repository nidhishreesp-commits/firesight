# SAM mask-feature experiment (CPU, no detector retraining)

## Model / licence

- Segmenter: **MobileSAM** checkpoint `mobile_sam.pt` from Ultralytics assets, 38.8 MB.
- Licence position: Ultralytics SAM wrapper/code is **AGPL-3.0**; MobileSAM upstream is
  Apache-2.0. Verify the weight licence before shipping. This experiment is internal only.
- P50 boxes at conf >= 0.10 were used as SAM box prompts.

## What was measured

- Samples: 500 detections from val_cal, 500 from val_report, all 278 DEV detections.
- Features per mask: mask area, mask area / image area, fill = mask area / box area,
  compactness (4*pi*area/perimeter^2).
- TP/FP: class-agnostic IoU >= 0.3 with any GT.
- Logistic regression trained on the val_cal sample, evaluated on val_report and DEV.
- **Detection-level** matched FPR, not image-level; SAM is too slow at CPU scale to run all
  val_report/bench_R detections.
- Latency: **~1.05 s per detection** on CPU (MobileSAM). This is the main blocker.

## Results

### Detection-level AUC

| set | verifier v4 | mask only | **verifier + mask** |
|---|---:|---:|---:|
| val_cal | 0.779 | 0.676 | **0.795** |
| val_report | 0.805 | 0.676 | **0.827** |
| DEV | 0.728 | 0.808 | **0.777** |

Mask features add signal: +0.022 AUC on val_report and +0.049 on DEV; mask-only is strongest on
DEV (0.808).

### Matched detection FPR

| set | target FPR | verifier recall | verifier+mask recall | McNemar (positives) |
|---|---:|---:|---:|---:|
| val_report | 2% | 24.22% | 19.88% | p=0.098 (worse) |
| val_report | 5% | 36.65% | **45.96%** | **p=0.0018** |
| val_report | 10% | 52.80% | 58.39% | p=0.073 |
| DEV | 2% | 11.11% | **25.00%** | **p=0.0059** |
| DEV | 5% | 17.59% | **40.74%** | **p=0.00011** |
| DEV | 10% | 37.96% | 46.30% | p=0.211 |

Negative McNemar was not significant (p=1.0 at most points) except the DEV 2% result is driven by
the positives. At 5% detection FPR the gain is consistent and significant on both sets.

### Controlled vs uncontrolled separation (DEV positives)

- Mask **area ratio**: controlled median 0.0715 vs uncontrolled 0.1263; AUC (uncontrolled larger)
  **0.627**. So mask area has some controlled-vs-uncontrolled signal.
- Fill ratio: medians 0.594 vs 0.614; AUC **0.470** — no separation.

## Verdict

Promising but **not production-ready**:

- Mask features improve the verifier at **detection-level matched FPR 5%** (val_report
  +9.3 pp recall, p=0.0018; DEV +23.2 pp, p=0.00011) and improve AUC.
- They do **not** help at 2% on val_report in this sample.
- MobileSAM at ~1.05 s/detection is far too slow for production on CPU. A GPU/batched version or
  a smaller prompt-based segmenter would be needed; latency is unmeasured there.
- Frame-to-frame mask growth was **not** measured (still images only); it belongs in the event/video
  harness.
- Model B masks were unavailable (Roboflow key), so no comparison was possible.

Measured: AUC, detection-level matched-FPR recall, McNemar, latency, controlled/uncontrolled
separation. Estimated/unmeasured: image-level impact, GPU SAM latency, mask growth on video.
