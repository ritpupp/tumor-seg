# LEARNING

## Step 1 — Inspect data
- What: counted 110 patients / 3929 slices (all 256x256 RGB uint8, masks binary 0/255), confirmed every image has a mask, parsed patient ID (e.g. TCGA_CS_6667) from folder names; 65.1% of slices have empty masks.
- Why: split must be by patient (step 2) to avoid leakage, and the 65/35 empty/tumor imbalance means pixel accuracy is meaningless and empty slices need care in Dice.
- Wrong result looks like: masks offset from the bright FLAIR lesion in overlays, image/mask count mismatch, mask values other than {0,255}, or ~0% / ~100% empty slices (would mean mis-paired files).

## Step 2 — Patient-level split
- What: shuffled the 110 patients (seed 42) into 77/16/17 train/val/test (2719/621/589 slices) and saved patient IDs + folder names to splits.json; asserted the sets are disjoint and cover everyone.
- Why: slices from one patient look almost the same, so a slice-level split would leak the same patient into train and test and inflate Dice.
- Wrong result looks like: the disjointness assert failing, a split with 0 patients, or a test Dice far above val Dice (a sign of leakage).

## Step 3 — Dataset, dataloader, augmentation
- What: LGGDataset reads slices from splits.json folders; train aug = h/v flips, ±15° rotation, ±15% brightness (albumentations, applied jointly to image+mask), then ImageNet normalization; batches are x (16,3,256,256) float32, y (16,1,256,256) in {0,1}.
- Why: joint transforms keep masks aligned; ImageNet stats match the pretrained resnet34 encoder; augmentation stretches 77 training patients further.
- Wrong result looks like: mask outline drifting off the lesion after rotation/flip, mask values other than {0,1} (interpolated masks), or normalized x with mean≈0.45 (normalize skipped) / huge ranges.

## Step 4 — Model + overfit test
- What: smp.Unet(resnet34, imagenet) with BCE + soft-Dice loss; trained (Adam 1e-3, no aug) on 16 tumor-only train slices from 16 different patients; eval-mode mean Dice hit 0.927 at step 50 on MPS.
- Why: if the model can't memorize 16 images, something is broken (labels, loss, normalization, threshold) and a full run would just waste GPU time; tumor-only slices stop the "predict nothing" shortcut from scoring well.
- Wrong result looks like: Dice stuck near 0 (mask/label misaligned or wrong sign on loss), Dice ≈ 1 at step 1 (empty masks counted as perfect), or train-mode loss falling while eval Dice stays low (BatchNorm/eval mismatch).

## Step 5 — train.py (smoke test)
- What: train.py (--data-dir required, AdamW 1e-3 + cosine LR, fp16 autocast+GradScaler on CUDA, best.pt saved by val per-patient Dice, curves + history json every epoch); smoke run 3 epochs × 50 slices on MPS ran end to end and best.pt reloads with all keys matched.
- Why: per-patient Dice (pixels pooled across a patient's slices) is the selection metric because ~65% of slices are empty and per-slice Dice rewards predicting nothing on them; the smoke run catches crashes before paying for GPU time.
- Wrong result looks like: val Dice flat at 0 after many full epochs, train loss falling while val loss climbs from early on (overfitting / leakage of aug into val), NaN loss under AMP, or per-slice Dice ≫ per-patient Dice (model just predicting empty).
