"""v2 check: strong intensity augmentation + per-image normalization. Uses train slices only.

Prints: v1 pipeline unchanged, masks untouched by intensity aug, per-image norm stats.
Saves outputs/aug_preview_v2.png.
"""
import argparse
import random
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset import (IMAGENET_MEAN, IMAGENET_STD, LGGDataset, PerImageStandardize, get_augs,
                     get_transforms, head_mask)


def load(pair):
    img_p, mask_p = pair
    return np.array(Image.open(img_p).convert("RGB")), (np.array(Image.open(mask_p)) > 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="outputs/aug_preview_v2.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = LGGDataset(args.data_dir, "train", max_samples=args.max_samples)
    tumor = [p for p in ds.pairs if load(p)[1].any()]
    print(f"train slices: {len(ds)}, with tumor: {len(tumor)}")

    # 1. v1 pipeline (defaults) is unchanged: plain ImageNet normalization
    img, mask = load(tumor[0])
    got = get_transforms(False)(image=img, mask=mask)["image"].numpy().transpose(1, 2, 0)
    want = (img / 255.0 - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    print(f"v1 eval transform == ImageNet normalize: max abs diff {np.abs(got - want).max():.2e}")

    # 2. Intensity aug never touches the mask (geometric aug moves image and mask together)
    intensity_only = A.Compose(get_augs("strong")[3:])
    print("strong intensity transforms:", [type(t).__name__ for t in intensity_only.transforms])
    changed_img, n = 0, 0
    for pair in tumor[:: max(1, len(tumor) // 40)]:
        im, m = load(pair)
        for _ in range(5):
            out = intensity_only(image=im, mask=m)
            assert np.array_equal(out["mask"], m), "intensity aug changed the mask!"
            changed_img += not np.array_equal(out["image"], im)
            n += 1
    print(f"mask identical after intensity-only aug: {n}/{n} draws (image changed in {changed_img}/{n})")
    full = get_transforms(True, norm="per-image", aug="strong")
    vals = set()
    for pair in tumor[:20]:
        vals |= set(np.unique(full(image=load(pair)[0], mask=load(pair)[1])["mask"].numpy()).tolist())
    print(f"mask values after full strong pipeline: {sorted(vals)}")

    # 3. Per-image normalization stats inside the head mask
    std = PerImageStandardize()
    means, stds, fracs = [], [], []
    for pair in ds.pairs[:: max(1, len(ds) // 200)]:
        im = load(pair)[0]
        hm = head_mask(im)
        z = std.apply(im)[hm]
        means.append(z.mean(0)); stds.append(z.std(0)); fracs.append(hm.mean())
    means, stds = np.array(means), np.array(stds)
    print(f"per-image norm over {len(means)} train slices, inside head: mean per channel "
          f"{means.mean(0).round(3)} (max |mean| {np.abs(means).max():.3f}), std {stds.mean(0).round(3)}; "
          f"head fraction median {np.median(fracs):.2f}")
    odd = [p for p in tumor if p[0].parent.name.startswith(("TCGA_FG_7637", "TCGA_FG_5962", "TCGA_FG_A4MU"))]
    for p in odd[:: max(1, len(odd) // 3)][:3]:
        im = load(p)[0]
        print(f"  bright-background slice {p[0].stem}: head fraction {head_mask(im).mean():.2f} "
              f"(simple sum>30 threshold would give {(im.astype(int).sum(-1) > 30).mean():.2f})")

    # 4. Figure: original | head mask | 3 strong augs | last aug after per-image norm
    rows = tumor[:: max(1, len(tumor) // 3)][:3] + (odd[len(odd) // 2:len(odd) // 2 + 1] if odd else [])
    augs = A.Compose(get_augs("strong"))
    fig, axes = plt.subplots(len(rows), 6, figsize=(18, 3.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, pair in enumerate(rows):
        im, m = load(pair)
        panels = [(im, m, "original"), (head_mask(im), None, "head mask (for per-image stats)")]
        for k in range(3):
            out = augs(image=im, mask=m)
            panels.append((out["image"], out["mask"], f"strong aug {k + 1}"))
        z = std.apply(panels[-1][0])
        panels.append(((z * 0.25 + 0.5).clip(0, 1), panels[-1][1], "aug 3, per-image normalized"))
        for c, (pim, pm, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(pim, cmap="gray" if pim.ndim == 2 else None)
            if pm is not None and pm.any():
                ax.contour(pm, levels=[0.5], colors="lime", linewidths=1)
            ax.set_title(title if c else f"{pair[0].stem[5:]}", fontsize=8)
            ax.axis("off")
    fig.suptitle("v2 strong intensity augmentation (green = mask outline; must stay on the lesion)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=85)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
