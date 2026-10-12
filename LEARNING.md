# LEARNING

## Step 1 — Inspect data
- What: counted 110 patients / 3929 slices (all 256x256 RGB uint8, masks binary 0/255), confirmed every image has a mask, parsed patient ID (e.g. TCGA_CS_6667) from folder names; 65.1% of slices have empty masks.
- Why: split must be by patient (step 2) to avoid leakage, and the 65/35 empty/tumor imbalance means pixel accuracy is meaningless and empty slices need care in Dice.
- Wrong result looks like: masks offset from the bright FLAIR lesion in overlays, image/mask count mismatch, mask values other than {0,255}, or ~0% / ~100% empty slices (would mean mis-paired files).

## Step 2 — Patient-level split
- What: shuffled the 110 patients (seed 42) into 77/16/17 train/val/test (2719/621/589 slices) and saved patient IDs + folder names to splits.json; asserted the sets are disjoint and cover everyone.
- Why: slices from one patient look almost the same, so a slice-level split would leak the same patient into train and test and inflate Dice.
- Wrong result looks like: the disjointness assert failing, a split with 0 patients, or a test Dice far above val Dice (a sign of leakage).
