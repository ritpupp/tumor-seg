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

## Model metrics
Not yet trained.
