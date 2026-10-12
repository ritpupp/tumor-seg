"""Step 1: inspect the LGG MRI segmentation dataset.

Counts patients/slices, checks every image has a mask, parses patient IDs,
reports % empty-mask slices, and saves image+mask overlays.
"""
import argparse
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Folder names look like TCGA_CS_6667_20011105 -> patient ID TCGA_CS_6667 (matches data.csv)
FOLDER_RE = re.compile(r"^(TCGA_[A-Z0-9]{2}_[A-Z0-9]{4})_(\d{8})$")


def parse_patient_id(folder_name):
    m = FOLDER_RE.match(folder_name)
    if m is None:
        raise ValueError(f"Unexpected folder name: {folder_name}")
    return m.group(1)


def collect_pairs(data_dir):
    pairs = []  # (patient_id, image_path, mask_path)
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        pid = parse_patient_id(folder.name)
        tifs = list(folder.glob("*.tif"))
        masks = {p for p in tifs if p.stem.endswith("_mask")}
        images = [p for p in tifs if p not in masks]
        for img in images:
            mask = img.with_name(img.stem + "_mask.tif")
            assert mask in masks, f"Missing mask for {img}"
            pairs.append((pid, img, mask))
        orphan = masks - {img.with_name(img.stem + "_mask.tif") for img in images}
        assert not orphan, f"Masks without images: {orphan}"
    return pairs


def overlay(ax, img, mask, title):
    ax.imshow(img)
    if mask.any():
        ax.contour(mask > 0, levels=[0.5], colors="lime", linewidths=1)
        red = np.zeros((*mask.shape, 4))
        red[mask > 0] = [1, 0, 0, 0.35]
        ax.imshow(red)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="outputs/samples.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pairs = collect_pairs(Path(args.data_dir))
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    patients = sorted({p for p, _, _ in pairs})
    per_patient = Counter(p for p, _, _ in pairs)
    print(f"Patients: {len(patients)}")
    print(f"Slices (image+mask pairs): {len(pairs)}")
    print(f"Slices per patient: min={min(per_patient.values())} "
          f"max={max(per_patient.values())} mean={np.mean(list(per_patient.values())):.1f}")
    print("All images have a mask: True (asserted)")
    print(f"Example patient IDs: {patients[:3]}")

    shapes, img_dtypes, mask_vals = Counter(), Counter(), set()
    empty = 0
    empty_by_patient = Counter()
    nonempty_pairs = []
    for pid, img_p, mask_p in pairs:
        img = np.array(Image.open(img_p))
        mask = np.array(Image.open(mask_p))
        assert img.shape[:2] == mask.shape[:2], f"Shape mismatch {img_p}"
        shapes[(img.shape, mask.shape)] += 1
        img_dtypes[str(img.dtype)] += 1
        mask_vals.update(np.unique(mask).tolist())
        if mask.max() == 0:
            empty += 1
            empty_by_patient[pid] += 1
        else:
            nonempty_pairs.append((pid, img_p, mask_p))

    print(f"Image/mask shapes: {dict(shapes)}")
    print(f"Image dtypes: {dict(img_dtypes)}  Mask unique values: {sorted(mask_vals)}")
    print(f"Empty-mask slices: {empty}/{len(pairs)} = {100 * empty / len(pairs):.1f}%")
    print(f"Slices with tumor: {len(nonempty_pairs)} = {100 * len(nonempty_pairs) / len(pairs):.1f}%")
    no_tumor_patients = [p for p in patients if empty_by_patient[p] == per_patient[p]]
    print(f"Patients with no tumor in any slice: {len(no_tumor_patients)} {no_tumor_patients}")

    # 6 overlays: 4 with tumor, 2 without, so we see both cases
    rng = random.Random(args.seed)
    empty_pairs = [x for x in pairs if x not in set(nonempty_pairs)]
    chosen = rng.sample(nonempty_pairs, min(4, len(nonempty_pairs)))
    chosen += rng.sample(empty_pairs, min(6 - len(chosen), len(empty_pairs)))
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
    for i, (pid, img_p, mask_p) in enumerate(chosen):
        img = np.array(Image.open(img_p))
        mask = np.array(Image.open(mask_p))
        axes[0, i].imshow(img)
        axes[0, i].set_title(img_p.stem, fontsize=7)
        axes[0, i].axis("off")
        overlay(axes[1, i], img, mask, f"overlay ({int((mask > 0).sum())} tumor px)")
    fig.suptitle("Top: raw MRI (RGB = pre-contrast / FLAIR / post-contrast). Bottom: mask overlay (red fill, green outline)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
