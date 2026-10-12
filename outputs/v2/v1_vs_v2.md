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

Paired per-patient tumor-only Dice, v2 - v1: v2 better on 5, worse on 2, within ±0.01 on 10 of 17 patients; median diff +0.004.
Excluding HT_A5RC, FG_A60K: v1 0.871 vs v2 0.878 (n=15).

| Patient | v1 | v2 | diff |
|---|---|---|---|
| TCGA_FG_A60K | 0.121 | 0.000 | -0.121 |
| TCGA_CS_5393 | 0.920 | 0.902 | -0.018 |
| TCGA_CS_6668 | 0.880 | 0.872 | -0.008 |
| TCGA_DU_6404 | 0.930 | 0.925 | -0.006 |
| TCGA_HT_7879 | 0.910 | 0.906 | -0.004 |
| TCGA_DU_6408 | 0.928 | 0.926 | -0.001 |
| TCGA_HT_7692 | 0.881 | 0.880 | -0.001 |
| TCGA_CS_6665 | 0.565 | 0.568 | +0.002 |
| TCGA_DU_5851 | 0.921 | 0.925 | +0.004 |
| TCGA_FG_7634 | 0.944 | 0.948 | +0.004 |
| TCGA_HT_8107 | 0.888 | 0.892 | +0.005 |
| TCGA_CS_6667 | 0.933 | 0.941 | +0.008 |
| TCGA_CS_4944 | 0.890 | 0.902 | +0.012 |
| TCGA_DU_7014 | 0.908 | 0.922 | +0.014 |
| TCGA_DU_A5TP | 0.829 | 0.870 | +0.040 |
| TCGA_HT_7616 | 0.741 | 0.789 | +0.048 |
| TCGA_HT_A5RC | 0.036 | 0.676 | +0.640 |
