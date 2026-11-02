# Resolution test on DEV high-mounted/aerial proxy

Scope: DEV only, keyword-proxy subset from the miss analysis (31 images: 4 positives / 27
negatives). No SEALED. This is **not** a global 960/1280 change.

| imgsz | recall | 95% CI | FPR | 95% CI | mean ms/img | median ms/img |
|---:|---:|---|---:|---|---:|---:|
| 640 | 50.0% | [15.0, 85.0] | 0.0% | [0, 12.5] | 137.8 | 89.5 |
| 960 | 50.0% | [15.0, 85.0] | 0.0% | [0, 12.5] | 197.3 | 199.3 |
| 1280 | 50.0% | [15.0, 85.0] | 0.0% | [0, 12.5] | 313.9 | 324.3 |

**Result: no recall gain at 960 or 1280; latency increases ~1.4x and ~2.3x.** The subset has only
**4 positives**, so this is inconclusive, not evidence that higher resolution never helps.

Recommendation: do not change the global inference size. If aerial/high-mounted cameras become a
real deployment target, collect a dedicated >=100-positive subset and re-run this test; otherwise
keep 640.
