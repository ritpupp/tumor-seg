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
| **Tumor-containing slices only (headline)** | **0.778 ± 0.271** | **0.695 ± 0.268** |
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
