# E0 vs E1 config diff

**Measured correction:** the 25% list is **14,747 images** = **6,532 positive + 8,215 dataset negative**. It is not 14,747 positives.

E0 train: 14,747 images, no added negatives.

E1 train: 16,074 images = E0 + 865 hard + 462 easy = 1,327 added negatives (9.00% of the base list).

Everything else is identical (model, 20 epochs, 640, batch 64, lr0 0.01, cosine, seed 0, deterministic, patience 10, augmentation, val_early/val_cal). Checkpoint selection is image-level recall at FPR<=2% on val_cal, never val loss.

The 8-10% rule is applied to **added** negatives. The base list already contains 55.7% negatives; E1 total negatives are 59.4%. If you intended 8-10% total negatives, the base must be positives-only.
