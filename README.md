# Brain tumor segmentation on LGG MRI (U-Net)

> **Research demo, not for clinical use.** This model has not been validated for diagnosis or treatment
> decisions and must not be used on real patients.

A U-Net that outlines lower-grade glioma (LGG) on single brain MRI slices, trained and evaluated with a
patient-level split, plus an error analysis and a Gradio demo.

**Headline result (17 held-out test patients, tumor-containing slices only):
Dice 0.778 ± 0.271, IoU 0.695 ± 0.268 (median patient Dice 0.890).**

![Best, median and worst test patients](outputs/patients_best_median_worst.png)
*Green = ground truth, red dashed = prediction. Rows: best, median and worst test patient by Dice.*

## Problem

Given one axial MRI slice, predict a pixel mask of the tumor. Manual outlining of gliomas is slow and
varies between readers; an automatic first draft is a common research target. Tumor is a small target:
about 65% of slices contain no tumor, and on slices that do, the tumor usually covers only a few
percent of the image.

## Data

[LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Buda et al., 2019),
from The Cancer Imaging Archive / TCGA.

| | |
|---|---|
| Patients | 110 |
| Slices | 3,929 (20–88 per patient) |
| Image | 256×256, 3 channels = pre-contrast, FLAIR, post-contrast |
| Mask | 256×256 binary, FLAIR abnormality outlined by a radiologist |
| Slices with tumor | 1,373 (34.9%) |

**Split by patient**, 70/15/15 (seed 42) → 77 / 16 / 17 patients (2,719 / 621 / 589 slices), stored in
[`splits.json`](splits.json). Slices from one patient are near-duplicates, so a slice-level split would leak
the same patient into training and test and inflate the scores.

![Sample slices with masks](outputs/samples.png)

## Method

- **Model:** U-Net with a ResNet-34 encoder pretrained on ImageNet
  ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)), 24.4M parameters, one output channel.
- **Loss:** binary cross-entropy + soft Dice (Dice counters the heavy background/tumor imbalance).
- **Input:** RGB slice normalized with ImageNet mean/std.
- **Augmentation (train only):** horizontal/vertical flips, rotation ±15°, brightness ±15%, applied jointly to
  image and mask ([preview](outputs/aug_preview.png)).
- **Training:** AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, batch 16, 40 epochs, fp16 mixed precision
  on a Kaggle GPU (~22 s/epoch). Checkpoint kept from the epoch with the best validation Dice (epoch 27).
- **Sanity check before training:** the model fits 16 tumor slices to Dice 0.927 in 50 steps
  ([figure](outputs/overfit.png)).

![Training curves](outputs/train_curves.png)

## Results

Dice and IoU are computed **per patient**, pooling pixels over that patient's slices, then averaged over the
17 test patients (mean ± std). Prediction threshold 0.5.

| Test set (17 patients) | Dice | IoU |
|---|---|---|
| **Tumor-containing slices only (headline)** | **0.778 ± 0.271** | **0.695 ± 0.268** |
| All slices (also penalizes false alarms on tumor-free slices) | 0.770 ± 0.273 | 0.684 ± 0.266 |
| Tumor-only, median patient | 0.890 | 0.802 |
| Tumor-only, excluding the 2 failure patients (n=15) | 0.871 ± 0.096 | 0.783 ± 0.128 |
| Validation (16 patients), all slices | 0.855 ± 0.106 | 0.759 ± 0.141 |

- 30 of 359 tumor-free test slices receive some false-positive prediction (21 of them from one patient).
- 31 of 230 tumor slices receive no prediction at all.
- The large standard deviation comes from two patients (Dice 0.12 and 0.04); see below.

Full per-patient table: [RESULTS.md](RESULTS.md) and [`outputs/test_per_patient.csv`](outputs/test_per_patient.csv).

## Error analysis

**Small tumors are the main weakness.** Slice Dice rises steadily with tumor size (Spearman ρ = 0.65):

| Tumor area on slice | < 250 px | 250–500 | 500–1k | 1k–2k | > 2k |
|---|---|---|---|---|---|
| Slices | 24 | 28 | 35 | 72 | 71 |
| Mean slice Dice | 0.04 | 0.48 | 0.53 | 0.76 | 0.89 |

![Dice vs tumor size](outputs/dice_vs_size.png)

**Three failure modes** ([figure](outputs/failure_cases.png)):

1. **Over-segmentation of a whole hemisphere** (patient FG_A60K, Dice 0.12). The model predicts about 9.6× the
   true tumor area, outlining a broad, mildly bright region instead of the small midline lesion. This scan is
   grayscale (one sequence copied into all three channels), but grayscale alone does not explain it:
   4 training patients are also grayscale, and the grayscale validation patient scores 0.90.
2. **Whole tumor missed** (patient HT_A5RC, Dice 0.04). 14 of 15 tumor slices get no prediction, although
   the lesions are large. The FLAIR channel of this scan is unusually dark (4th lowest of 110 patients), but a
   validation patient with an equally dark FLAIR channel scores 0.85, so **the cause is unresolved**.
3. **Small tumors missed**, especially on the first/last slice of a tumor. 10 of 24 slices with < 250 px of
   tumor are missed entirely; 14 of the 31 completely missed slices are tumor-edge slices, although those
   are only 15% of tumor slices.

![Failure cases](outputs/failure_cases.png)

## Demo

```bash
python src/app.py            # then open http://127.0.0.1:7860   (--share for a public link)
```

Upload a slice (.tif/.png/.jpg) → predicted tumor in red, plus tumor area in pixels and as % of the head area
(area is not given in mm² because the images do not store pixel spacing). Three example slices are in
[`examples/`](examples/). The demo uses the same preprocessing as evaluation and reproduces its pixel counts exactly.

![Demo outputs](outputs/demo_examples.png)

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Data: download from Kaggle into data/ (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation -p data --unzip
#   -> data/kaggle_3m/TCGA_*/ ...   (point --data-dir at the folder holding the TCGA_* folders)

python src/inspect_data.py                 # 1. counts, empty-mask %, outputs/samples.png
python src/make_splits.py                  # 2. patient-level split -> splits.json (already committed)
python src/check_dataset.py                # 3. batch shapes, outputs/aug_preview.png
python src/overfit_test.py                 # 4. overfit 16 tumor slices, Dice > 0.9
python src/train.py --data-dir data/kaggle_3m --max-samples 50 --epochs 3   # 5a. smoke test
python src/train.py --data-dir data/kaggle_3m --epochs 40                   # 5b. full run (GPU recommended)
python src/evaluate.py                     # 6. test Dice/IoU per patient -> outputs/test_per_*.csv
python src/error_analysis.py               # 7. figures + failure-mode stats
python src/app.py                          # 8. Gradio demo
```

Every script takes `--data-dir` (default `data/kaggle_3m`). On Kaggle, attach the dataset and run e.g.
`python src/train.py --data-dir /kaggle/input/lgg-mri-segmentation/kaggle_3m` (the exact path depends on the
dataset version). The trained checkpoint (`checkpoints/best.pt`, ~98 MB) is not in the repo; train it or
place your own there.

**Repository layout**

```
src/            inspect_data, make_splits, dataset, check_dataset, model, overfit_test,
                train, evaluate, error_analysis, app
splits.json     patient-level train/val/test split
outputs/        figures and per-patient / per-slice result CSVs
examples/       3 test slices for the demo
RESULTS.md      all metrics;  LEARNING.md  step-by-step notes
```

## Limitations

- **Not a clinical tool.** Research demo only; no regulatory review and no clinical validation.
- **Small test set.** 17 test patients from one public dataset; the ± numbers are wide, and two patients
  move the mean by about 0.09 Dice. Results may not transfer to other scanners, protocols or hospitals.
- **2D, slice by slice.** The model sees no neighboring slices, which likely contributes to misses at the top
  and bottom of a tumor.
- **Small tumors are unreliable** (mean Dice 0.04 below 250 px), and one test tumor was missed entirely for
  reasons we could not pin down.
- **Single run, single seed, fixed threshold 0.5.** No hyperparameter search, cross-validation, or test-time
  augmentation; numbers may vary between training runs.
- **Labels** are one annotation per slice of FLAIR abnormality; inter-reader variability is not measured.
- **Tumor area is in pixels**, not mm², because pixel spacing is not available in this dataset's images.

## Reference

Buda M., Saha A., Mazurowski M. A. *Association of genomic subtypes of lower-grade gliomas with shape features
automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine, 2019.
