# bench_R pre-annotation report

**Proposals are not labels.** Human review is the ground truth. No evaluated detector was used.

| Proposer | Licence | Images | Runtime |
|---|---|---:|---:|
| google/owlv2-base-patch16-ensemble | Apache-2.0 | 525 | 78.9 min |
| IDEA-Research/grounding-dino-tiny | Apache-2.0 | 100 | 15.3 min |

## Coverage (positives)

| Family | Positives | With proposal | Rate |
|---|---:|---:|---:|
| both | 80 | 79 | 98.8% |
| controlled_fire | 80 | 80 | 100.0% |
| smoke_only | 100 | 100 | 100.0% |
| spark_screen | 40 | 40 | 100.0% |
| uncontrolled_fire | 200 | 199 | 99.5% |

Blind 10%: 100 images (60 positives, 40 negatives); proposals hidden in the review UI.
Second-pass spot check is separate (100 images, not the blind sample).
Correction rate per proposer/category is pending human review and will be appended after locking.

Licences: OWLv2 and Grounding DINO are Apache-2.0; model cards checked. No restricted model was used.
