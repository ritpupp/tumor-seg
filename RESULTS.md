# RESULTS

## Data (step 1)
| Item | Value |
|---|---|
| Patients | 110 |
| Slices | 3929 (20–88 per patient, mean 35.7) |
| Slice size | 256x256x3 uint8; mask 256x256 {0,255} |
| Empty-mask slices | 2556 (65.1%) |
| Tumor slices | 1373 (34.9%) |
| Patients with no tumor | 0 |
| Note | 3 patients (FG_5962, FG_7637, FG_A4MU; 138 slices) have a non-black background in one channel |

## Split (step 2, seed 42, by patient)
| Split | Patients | Slices | % slices with tumor |
|---|---|---|---|
| train | 77 | 2719 | 34.4% |
| val | 16 | 621 | 33.7% |
| test | 17 | 589 | 39.0% |

## Data pipeline (step 3)
Batch: x (16,3,256,256) float32, ImageNet-normalized; y (16,1,256,256) float32 in {0,1}. Tumor pixels ≈ 0.7–1.6% per batch (heavy class imbalance).

## Overfit test (step 4)
U-Net resnet34 (ImageNet), Dice+BCE, Adam 1e-3, 16 tumor-only train slices (16 patients), MPS.
Mean Dice 0.927 / IoU 0.871 at step 50 (target > 0.9) — PASS. Per-image Dice 0.70–0.97.

## Model metrics

### Training (step 5, Kaggle GPU, AMP)
40 epochs, AdamW 1e-3 cosine, batch 16. Best checkpoint: epoch 27, val per-patient Dice 0.8546 (all slices).

### Test set (step 6) — 17 patients, 589 slices, `checkpoints/best.pt` (epoch 27)
Dice/IoU computed per patient with pixels pooled over that patient's slices; mean ± std over patients.

| Metric | Dice | IoU |
|---|---|---|
| **Tumor-containing slices only (headline)** | **0.778 ± 0.271** | **0.695 ± 0.267** |
| All slices | 0.770 ± 0.273 | 0.684 ± 0.266 |
| Tumor-only, median over patients | 0.890 | 0.802 |
| Tumor-only, excluding 2 failure patients (n=15) | 0.871 ± 0.096 | 0.783 ± 0.128 |

- False-positive slices: 30/359 empty slices have some predicted tumor.
- Missed slices: 31/230 tumor slices have no prediction at all.
- Two failure patients drive the large std: TCGA_FG_A60K (Dice 0.12, over-segments: 73,861 predicted px vs 7,719 GT px on tumor slices, plus 24,475 FP px on empty slices) and TCGA_HT_A5RC (Dice 0.04, under-segments: 312 predicted px vs 17,188 GT px).
- Val cross-check with evaluate.py: all-slices per-patient Dice 0.8546 (matches checkpoint exactly); tumor-only 0.873 ± 0.077.

Per-patient test results (`outputs/test_per_patient.csv`):

| Patient | Slices | Tumor slices | Dice (tumor) | IoU (tumor) | Dice (all) | IoU (all) |
|---|---|---|---|---|---|---|
| TCGA_CS_4944_20010208 | 20 | 9 | 0.890 | 0.802 | 0.868 | 0.767 |
| TCGA_CS_5393_19990606 | 20 | 8 | 0.920 | 0.852 | 0.876 | 0.780 |
| TCGA_CS_6665_20010817 | 24 | 11 | 0.565 | 0.394 | 0.565 | 0.394 |
| TCGA_CS_6667_20011105 | 20 | 8 | 0.933 | 0.874 | 0.933 | 0.874 |
| TCGA_CS_6668_20011025 | 28 | 7 | 0.880 | 0.785 | 0.844 | 0.730 |
| TCGA_DU_5851_19950428 | 40 | 13 | 0.921 | 0.854 | 0.921 | 0.854 |
| TCGA_DU_6404_19850629 | 53 | 16 | 0.930 | 0.869 | 0.929 | 0.867 |
| TCGA_DU_6408_19860521 | 56 | 27 | 0.928 | 0.865 | 0.922 | 0.855 |
| TCGA_DU_7014_19860618 | 60 | 30 | 0.908 | 0.832 | 0.908 | 0.832 |
| TCGA_DU_A5TP_19970614 | 38 | 12 | 0.829 | 0.709 | 0.827 | 0.705 |
| TCGA_FG_7634_20000128 | 27 | 11 | 0.944 | 0.894 | 0.944 | 0.894 |
| TCGA_FG_A60K_20040224 | 73 | 24 | 0.121 | 0.064 | 0.093 | 0.049 |
| TCGA_HT_7616_19940813 | 30 | 15 | 0.741 | 0.588 | 0.741 | 0.588 |
| TCGA_HT_7692_19960724 | 20 | 6 | 0.881 | 0.788 | 0.881 | 0.788 |
| TCGA_HT_7879_19981009 | 28 | 11 | 0.910 | 0.835 | 0.910 | 0.835 |
| TCGA_HT_8107_19980708 | 20 | 7 | 0.888 | 0.798 | 0.888 | 0.798 |
| TCGA_HT_A5RC_19990831 | 32 | 15 | 0.036 | 0.018 | 0.036 | 0.018 |

## Error analysis (step 7)
Figures: `outputs/patients_best_median_worst.png`, `outputs/dice_vs_size.png`, `outputs/failure_cases.png`.
Per-patient intensity stats: `outputs/patient_intensity_stats.csv`.

**Best / median / worst test patient (tumor-only Dice):** FG_7634 0.944 · CS_4944 0.890 · HT_A5RC 0.036.

**Dice vs tumor size (230 tumor slices):** strongly size-dependent (Spearman ρ = 0.65).

| GT tumor area | Slices | Mean slice Dice | Slices missed entirely |
|---|---|---|---|
| < 250 px | 24 | 0.04 | 10 |
| 250–500 px | 28 | 0.48 | 4 |
| 500–1k px | 35 | 0.53 | 7 |
| 1k–2k px | 72 | 0.76 | 9 |
| > 2k px | 71 | 0.89 | 1 |

**Three failure modes**
1. **Over-segmentation of a whole hemisphere (FG_A60K, Dice 0.12).** Predicts 73,861 px on tumor slices vs 7,719 px GT and 24,475 px on empty slices; this one patient accounts for 21 of the 30 false-positive empty slices in the test set. The scan is grayscale (all three channels identical), and the model outlines a broad, mildly bright region across one hemisphere instead of the small midline lesion. Grayscale alone is *not* the explanation: 4 train patients are fully grayscale and the grayscale val patient HT_A616 scores 0.90. More likely the diffuse hemispheric brightness looks like tumor when colour (multi-sequence) cues are missing.
2. **Whole tumor missed (HT_A5RC, Dice 0.04).** 14 of 15 tumor slices get no prediction at all (312 px predicted vs 17,188 px GT), even though the lesions are large (1.5–2.3k px). The tumor is brighter than surrounding brain on FLAIR (contrast 1.48, dataset median 1.57), but the whole FLAIR channel is unusually dark relative to the other two (balance 0.40 vs median 0.94; 4th lowest of 110 patients). This is not sufficient on its own: val patient FG_6688 has balance 0.36 and scores 0.85. **Cause unresolved**; channel imbalance is the leading suspect.
3. **Small tumors, mostly at the top/bottom edge of a tumor.** Slices with < 250 px of tumor average Dice 0.04 and 10/24 are missed outright. Of the 31 missed tumor slices, 14 are the first or last tumor slice of their patient, although such edge slices are only 15% of tumor slices. Pooled per-patient Dice hides much of this because small slices contribute few pixels.

Minor: a few false positives on FLAIR-bright non-tumor regions (e.g. periventricular brightness in CS_5393, 4 slices).

## Demo (step 8)
`python src/app.py` → http://127.0.0.1:7860. Upload a slice → red overlay + tumor area (pixels and % of head area; no mm² because pixel spacing isn't stored in the images). Threshold slider, 3 example slices in `examples/`. Checked: demo pixel counts match `evaluate.py` exactly on 3 test slices (TIF and PNG), and the running server answers through its API. Output examples: `outputs/demo_examples.png`.

## Packaging (step 9)
README.md written with the results above; requirements.txt pinned; data/, .venv/, checkpoints/ and *.pt are git-ignored.

## v2 (branch v2-norm): per-image normalization — comparison only, NOT the headline model
v2 = v1 with one change: `--norm per-image` (each channel standardized over the head region; `--aug basic` as v1).
Same splits.json, seed 42, 40 epochs, resnet34 U-Net, Dice+BCE. Best epoch 37 by val (val curve flat from epoch ~21).
Test evaluated once (`logs/v2_eval_test.log`); v1 test numbers are the committed step-6 CSVs (not re-run).
**v1 remains the headline model**: it was chosen by val, and v2 is slightly *worse* on val.

| Metric | v1 | v2 |
|---|---|---|
| Test Dice, tumor-only (mean ± std) | 0.778 ± 0.271 | 0.814 ± 0.226 |
| Test IoU, tumor-only (mean ± std) | 0.695 ± 0.267 | 0.729 ± 0.225 |
| Test Dice, tumor-only (median) | 0.890 | 0.902 |
| Test IoU, tumor-only (median) | 0.802 | 0.821 |
| Test Dice, all slices (mean ± std) | 0.770 ± 0.273 | 0.805 ± 0.224 |
| Test IoU, all slices (mean ± std) | 0.684 ± 0.266 | 0.714 ± 0.224 |
| Test Dice HT_A5RC (tumor-only) | 0.036 | 0.676 |
| Test Dice FG_A60K (tumor-only) | 0.121 | 0.000 |
| Test false-positive empty slices | 30/359 | 13/359 |
| Test tumor slices with no prediction | 31/230 | 37/230 |
| Test slice Dice, tumor <250 px (n=24) | 0.042 (missed 10) | 0.219 (missed 16) |
| Test slice Dice, tumor 250-500 px (n=28) | 0.482 (missed 4) | 0.536 (missed 9) |
| Test slice Dice, tumor 500-1k px (n=35) | 0.530 (missed 7) | 0.527 (missed 11) |
| Test slice Dice, tumor 1k-2k px (n=72) | 0.763 (missed 9) | 0.858 (missed 1) |
| Test slice Dice, tumor >2k px (n=71) | 0.888 (missed 1) | 0.899 (missed 0) |
| Val Dice, tumor-only (mean ± std) | 0.873 ± 0.077 | 0.864 ± 0.087 |
| Val IoU, tumor-only (mean ± std) | 0.782 ± 0.111 | 0.770 ± 0.120 |
| Val Dice, tumor-only (median) | 0.893 | 0.893 |
| Val Dice, all slices (mean ± std) [selection metric] | 0.855 ± 0.106 | 0.844 ± 0.120 |

Size buckets **excluding HT_A5RC and FG_A60K** (so the two outliers don't drive them):

| Tumor area | n | v1 Dice (missed) | v2 Dice (missed) |
|---|---|---|---|
| < 250 px | 12 | 0.077 (8) | 0.367 (5) |
| 250–500 px | 22 | 0.592 (4) | 0.683 (3) |
| 500–1k px | 23 | 0.722 (3) | 0.750 (2) |
| 1k–2k px | 64 | 0.853 (2) | 0.876 (0) |
| > 2k px | 70 | 0.901 (0) | 0.906 (0) |

- Paired per-patient tumor-only Dice: v2 better on 5, worse on 2, within ±0.01 on 10 of 17; median diff +0.004.
- Most of the +0.036 test mean comes from **HT_A5RC: 0.036 → 0.676** (alone ≈ +0.038 on the mean). Excluding HT_A5RC and FG_A60K: 0.871 → 0.878.
- **FG_A60K: 0.121 → 0.000.** The failure changed type: v1 over-segmented (73,861 px predicted vs 7,719 GT); v2 predicts nothing on 23/24 tumor slices (160 px, 0 overlap). FP px on its empty slices 24,475 → 5,375.
- Test false-positive empty slices 30 → 13; tumor slices with no prediction 31 → 37 (the increase is FG_A60K).
- Val (selection metric, all slices): v1 0.855 ± 0.106 vs v2 0.844 ± 0.120; tumor-only 0.873 vs 0.864. Val does not favour v2.
- Caveats: one training run per model, 17 test patients; the test gain is dominated by one patient, so this does not show v2 is better in general. Full table: `outputs/v2/v1_vs_v2.md`.
