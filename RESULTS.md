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
Training (step 5): train.py smoke-tested (3 epochs, 50 slices) — pipeline OK. **Full training run pending (user GPU run).**
