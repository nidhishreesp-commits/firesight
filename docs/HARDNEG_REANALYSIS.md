# Hard-negative experiment: re-analysis

Status: **correction of the earlier write-up.** The previous conclusion ("hard negatives cost
recall, keep P50") used the wrong control. This document re-does the comparison properly,
adds paired tests on identical images, audits the mined pool's provenance and hardness under
**P50**, and states plainly which earlier claims are unsupported.

Rules unchanged: SEALED/TEST untouched; bench_R/testbench100/TEST never trained on; every new
image goes through the leakage gate; Wilson CIs + paired McNemar; one DEV evaluation per candidate.

---

## 1. What was actually compared

| run | start | data | negatives | lr0 | epochs | stopped |
|---|---|---|---|---:|---:|---|
| FT25 ("B") | P50 | 14,747 positives (25%) | **none** | 0.01 | 25 | epoch 6 |
| FT_HN | P50 | 14,747 positives + 2,000 mined | **18.8%** | 0.001 | 15 | epoch 6 |
| P50 | yolo11s | 29,493 positives (50%) | none | 0.01 | 15 | epoch 15 |

**The correct control for FT_HN is FT25** - same starting point, same positives, same
fine-tune mechanism, no negatives. The earlier document compared FT_HN against P50, which
conflates "adding negatives" with "fine-tuning at all".

**Confound that must be stated:** FT25 used `lr0=0.01`, FT_HN used `lr0=0.001`. Both early
stopped at epoch 6, so the effective schedule length matches, but the learning rate differs
10x. A perfectly matched control (lr 0.001, 15 epochs, no negatives) has **not** been run.
Conclusions below are therefore "no detectable effect of the negatives **as tested**", not
"negatives never help".

---

## 2. Paired tests - val.jsonl report slice

Identical images, thresholds calibrated at 2.0% on `val_cal`; 5,119 images = 2,475 positives +
2,644 negatives.

| model | false positives / 2,644 | true positives / 2,475 | accuracy |
|---|---:|---:|---:|
| P50 | 83 (3.14%) | 2,301 (92.97%) | 0.9498 |
| FT25 | 78 (2.95%) | 2,160 (87.27%) | 0.9232 |
| FT_HN | 72 (2.72%) | 2,154 (87.03%) | 0.9232 |

Exact McNemar (two-sided binomial on discordant pairs):

| comparison | outcome | discordant (A right/B right) | p |
|---|---|---|---:|
| **FT_HN vs FT25 - negatives** | no-alarm correct | 33 vs 27 | **0.519** |
| **FT_HN vs FT25 - positives** | alarm correct | 98 vs 104 | **0.725** |
| **FT_HN vs FT25 - overall** | correct | 131 vs 131 | **1.000** |
| FT_HN vs P50 - negatives | no-alarm correct | 40 vs 29 | 0.228 |
| FT_HN vs P50 - positives | alarm correct | 28 vs 175 | <1e-4 |
| FT25 vs P50 - positives | alarm correct | 29 vs 170 | 1.8e-25 |
| FT25 vs P50 - overall | correct | 57 vs 193 | 1.8e-18 |

**Reading:** the 6-FP reduction (78 -> 72) and the 6-TP loss (2,160 -> 2,154) between FT25 and
FT_HN are **both inside noise**. The large recall loss is between the *fine-tunes* and P50, and
FT25 (no negatives) shows the same loss.

---

## 3. Paired tests - bench_R DEV

300 images = 85 positives + 215 negatives; thresholds calibrated on `val_mini + val_flames`.

| model | false positives / 215 | true positives / 85 |
|---|---:|---:|
| **P50** | **5** (2.33%) | **48** (56.5%) |
| FT25 | **7** (3.26%) | 46 (54.1%) |
| FT_HN | **7** (3.26%) | 42 (49.4%) |

Exact McNemar:

| comparison | discordant | p |
|---|---|---:|
| FT_HN vs FT25 - negatives (5 vs 7 FPs) | 4 vs 4 | **1.000** |
| FT_HN vs FT25 - positives | 1 vs 5 | **0.219** |
| FT_HN vs FT25 - overall | 5 vs 9 | **0.424** |
| FT_HN vs P50 - negatives | 1 vs 3 | 0.625 |
| FT_HN vs P50 - positives | 2 vs 8 | 0.109 |
| FT_HN vs P50 - overall | 3 vs 11 | 0.057 |
| FT25 vs P50 - overall | 8 vs 12 | 0.503 |

**Reading:** on the locked bench nothing reaches significance. The direction (P50 best) is
consistent but DEV has only 85 positives, so it cannot resolve a 5-6 image difference. The
earlier claim that FT_HN is clearly worse on DEV is **not supported at the 5% level**.

---

## 4. Provenance and hardness of the 4,000 mined negatives

The pool was selected by **best_fire_v3** hardness (`max_conf >= 0.10`), then reused for a
**P50** experiment. That is a measurement mismatch.

| property | value |
|---|---:|
| mined negatives selected | 4,000 |
| **already in P50's training set** (25% list) | **1,001 (25.0%)** |
| not in P50's training set | 2,999 |
| "hard" under **best_fire_v3** (>=0.10) | 306 |
| "very hard" under **best_fire_v3** (>=0.30) | 103 |
| **"hard" under P50 (>=0.10)** | **162 (4.1%)** |
| **"very hard" under P50 (>=0.30)** | **48 (1.2%)** |
| P50 >= 0.50 | 17 |

**Reading:** 25% of the "hard negatives" were images P50 had already trained on, and only 4.1%
are actually hard for P50. The pool was, in effect, mostly *easy* negatives plus a quarter of
duplicates - which explains why the intervention could not move the needle.

---

## 5. Earlier conclusions that are NOT supported

| earlier claim | verdict | evidence |
|---|---|---|
| "Hard negatives cost 5.9 pp recall on val" | **unsupported** | vs FT25: 6 TPs, p=0.725. The drop is from fine-tuning, and FT25 shows it too (p=1.8e-25 vs P50) |
| "Hard negatives lowered the false-alarm rate" | **not established** | 72 vs 78 FPs, p=0.519 (direction only) |
| "FT_HN is clearly worse than P50 on the locked bench" | **not established** | DEV overall p=0.057, positives p=0.109 |
| "The mined pool is hard for the detector" | **false for P50** | only 162/4,000 >=0.10; 25% already in training; 306/103 were v3 numbers |
| "Fine-tuning from P50 with hard negatives is the right lever" | **not established** | the fine-tune itself is the confound; no matched no-negative control at lr 0.001 |
| "DEV 5 vs 7 FPs proves degradation" | **unsupported** | McNemar p=1.000 for FT_HN vs FT25; p=0.625 vs P50 |

## 6. What still stands

- **P50 is the best checkpoint measured so far**: 92.97% recall @ 3.14% FPR on val, 56.5%
  recall @ 2.33% FPR on bench_R DEV; the old model is 80.1% / 40.0%.
- **The FP taxonomy is descriptive and valid**: of 88 FPs, warm light 14, sunset 12, haze/cloud
  11, cave/rock 7, person 7, water 5, screen 5, **dust 3**, PPE 3, steam 2, orange vegetation 2.
- **The leakage gate works and is mandatory**: it removed 47 images that were literally the
  same files as bench_R negatives.
- **The learning curve is flat after 25%** (separate experiment, unaffected).
- **External YOLO26 is worse on our domain** (75.2% vs 93.0% recall, mAP 0.374 vs 0.633).

## 7. Consequences for the new plan

1. **Stop using v3 hardness to select negatives.** Every future pool must be scored by the
   frozen model that will use it (P50), at a low threshold, and duplicates of the training set
   must be removed (1,001/4,000 were already seen).
2. **Do not fine-tune the whole network from a converged checkpoint.** Both attempts
   early-stopped at epoch 6 and lost 5-6 pp recall on val. Use a frozen backbone + head, or a
   fresh training run.
3. **The measurement was underpowered**, not the idea: DEV has 85 positives and 215 negatives;
   a 0.3 pp FPR difference needs thousands of negatives. Hence the >=5,000-image held-out
   negative set (item 5 of the new plan).
4. **Judge candidates with paired tests on identical images**, not by comparing point estimates
   from different runs.
