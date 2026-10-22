# testbench100 results history

Locked bench: 100 images, composition override 62 positives / 38 negatives. Thresholds calibrated on `val_mini + val_flames` only; the bench was never tuned on.

| Model | Val target FPR | Accuracy (95% CI) | Recall | Specificity | F1 | Fire R | Smoke R | Run |
|---|---:|---|---:|---:|---:|---:|---:|---|
| cascade_union_v3_firept | 0.05 | 0.870 (0.790 to 0.922) | 0.823 | 0.947 | 0.887 | 0.814 | 0.450 | `20261022_165035_cascade_union_v3_firept` |
| ensemble_v3_v4_union | 0.05 | 0.870 (0.790 to 0.922) | 0.806 | 0.974 | 0.885 | 0.814 | 0.575 | `20261022_163219_ensemble_v3_v4_union` |
| ensemble_v3_v4_union | 0.1 | 0.870 (0.790 to 0.922) | 0.855 | 0.895 | 0.891 | 0.814 | 0.725 | `20261022_163308_ensemble_v3_v4_union` |
| v3 | 0.1 | 0.870 (0.790 to 0.922) | 0.839 | 0.921 | 0.889 | 0.791 | 0.725 | `20261022_163305_v3` |
| ensemble_v3_v4_intersection | 0.01 | 0.860 (0.779 to 0.915) | 0.806 | 0.947 | 0.877 | 0.791 | 0.675 | `20261022_161150_ensemble_v3_v4_intersection` |
| cascade_v3 | 0.01 | 0.850 (0.767 to 0.907) | 0.774 | 0.974 | 0.865 | 0.767 | 0.450 | `20261022_155809_cascade_v3` |
| v3 | 0.05 | 0.850 (0.767 to 0.907) | 0.790 | 0.947 | 0.867 | 0.791 | 0.575 | `20261022_163033_v3` |
| v4 | 0.1 | 0.840 (0.756 to 0.899) | 0.823 | 0.868 | 0.864 | 0.814 | 0.725 | `20261022_163307_v4` |
| ensemble_v3_v4_firept_union | 0.05 | 0.830 (0.745 to 0.891) | 0.742 | 0.974 | 0.844 | 0.698 | 0.600 | `20261022_163217_ensemble_v3_v4_firept_union` |
| v4 | 0.05 | 0.830 (0.745 to 0.891) | 0.790 | 0.895 | 0.852 | 0.791 | 0.650 | `20261022_163034_v4` |
| ensemble_v3_v4_union | 0.01 | 0.810 (0.722 to 0.875) | 0.710 | 0.974 | 0.822 | 0.651 | 0.650 | `20261022_161149_ensemble_v3_v4_union` |
| v4 | 0.01 | 0.800 (0.711 to 0.867) | 0.694 | 0.974 | 0.811 | 0.744 | 0.500 | `20261022_162852_v4` |
| ensemble_v3_firept_union | 0.05 | 0.780 (0.689 to 0.850) | 0.710 | 0.895 | 0.800 | 0.581 | 0.600 | `20261022_163216_ensemble_v3_firept_union` |
| v3 | 0.01 | 0.770 (0.678 to 0.842) | 0.629 | 1.000 | 0.772 | 0.488 | 0.575 | `20261022_162824_v3` |
| fireviewer_rtdetrv2 | 0.01 | 0.650 (0.553 to 0.736) | 0.435 | 1.000 | 0.607 | 0.442 | 0.250 | `20261022_161057_fireviewer_rtdetrv2` |
| fire_pt | 0.05 | 0.550 (0.452 to 0.644) | 0.355 | 0.868 | 0.494 | 0.256 | 0.250 | `20261022_163036_fire_pt` |
| fire_pt | 0.01 | 0.520 (0.423 to 0.615) | 0.258 | 0.947 | 0.400 | 0.256 | 0.125 | `20261022_162938_fire_pt` |
| pyronear_sensitive | 0.01 | 0.360 (0.273 to 0.458) | 0.016 | 0.921 | 0.030 | 0.000 | 0.025 | `20261022_155707_pyronear_sensitive` |

## Notes

- `pyronear_sensitive` is a single-class wildfire-smoke detector (`item` mapped to smoke); it cannot separate fire.
- `pyronear/temporal-model` is a sequence model, not applicable to the single-image bench.
- `select-ai/fire-smoke-detection` had no public weights found; `e1250/safety_detection` has no declared licence and was not downloaded.
- 90% and 95% image-level accuracy were **not reached**; the best is 87% with a 95% CI of 79.0 to 92.2%.
- See `bench_results/<run_id>/false_positives.jpg` and `false_negatives.jpg`.
