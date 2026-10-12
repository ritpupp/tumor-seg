"""Step 3 check: print batch shapes/stats and save outputs/aug_preview.png."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD, LGGDataset, get_loaders, get_transforms


def denorm(t):
    img = t.permute(1, 2, 0).numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return img.clip(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="outputs/aug_preview.png")
    args = ap.parse_args()
    torch.manual_seed(0)

    loaders = get_loaders(args.data_dir, batch_size=args.batch_size, num_workers=2,
                          max_samples=args.max_samples)
    for split, dl in loaders.items():
        x, y = next(iter(dl))
        print(f"{split:5s}: {len(dl.dataset):4d} slices, {len(dl):3d} batches | "
              f"x {tuple(x.shape)} {x.dtype} mean={x.mean():.2f} std={x.std():.2f} "
              f"range=[{x.min():.2f},{x.max():.2f}] | y {tuple(y.shape)} {y.dtype} "
              f"unique={torch.unique(y).tolist()} tumor_px={y.mean():.3%}")

    # Preview: same 4 tumor slices, original (no aug) + 4 random augmentations each.
    raw = LGGDataset(args.data_dir, "train", transform=get_transforms(False), max_samples=args.max_samples)
    aug = LGGDataset(args.data_dir, "train", transform=get_transforms(True), max_samples=args.max_samples)
    tumor_idx = [i for i in range(len(raw)) if raw[i][1].sum() > 0]
    picks = tumor_idx[:: max(1, len(tumor_idx) // 4)][:4]
    n_aug = 4
    fig, axes = plt.subplots(len(picks), n_aug + 1, figsize=(3 * (n_aug + 1), 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(picks):
        samples = [raw[i]] + [aug[i] for _ in range(n_aug)]
        for c, (x, y) in enumerate(samples):
            ax = axes[r, c]
            ax.imshow(denorm(x))
            ax.contour(y[0].numpy(), levels=[0.5], colors="lime", linewidths=1)
            ax.set_title("original" if c == 0 else f"aug {c}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Augmentation preview (green = mask outline; should stay on the lesion)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
