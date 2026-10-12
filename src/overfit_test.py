"""Step 4 check: overfit the U-Net on 16 tumor-only train slices (no augmentation).

Passes when mean per-image Dice on those 16 images > 0.9 (model in eval mode).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from check_dataset import denorm
from dataset import LGGDataset, get_transforms
from model import DiceBCELoss, build_model, dice_iou_per_image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_tumor_slices(ds, n):
    """Middle tumor slice of each of the first n patients, so all n come from different scans."""
    by_patient = {}
    for i, (img_p, mask_p) in enumerate(ds.pairs):
        if np.array(Image.open(mask_p)).max() > 0:
            by_patient.setdefault(img_p.parent.name, []).append(i)
    assert len(by_patient) >= n, f"Only {len(by_patient)} patients with tumor"
    return [idxs[len(idxs) // 2] for idxs in list(by_patient.values())[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--n-images", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target-dice", type=float, default=0.9)
    ap.add_argument("--out", default="outputs/overfit.png")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = get_device()
    print(f"Device: {device}")

    ds = LGGDataset(args.data_dir, "train", transform=get_transforms(False))
    idx = pick_tumor_slices(ds, args.n_images)
    x = torch.stack([ds[i][0] for i in idx]).to(device)
    y = torch.stack([ds[i][1] for i in idx]).to(device)
    tumor_px = y.sum((1, 2, 3))
    assert len(set(idx)) == args.n_images and (tumor_px > 0).all(), "All overfit images must contain tumor"
    print(f"Batch x {tuple(x.shape)}, y {tuple(y.shape)}; all {len(idx)} slices have tumor "
          f"(tumor px per slice: min={int(tumor_px.min())} median={int(tumor_px.median())} max={int(tumor_px.max())}); "
          f"from {len({ds.pairs[i][0].parent.name for i in idx})} patients")

    model = build_model().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    loss_fn = DiceBCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses, dices = [], []
    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                logits = model(x)
            dice, iou = dice_iou_per_image(logits, y)
            dices.append((step, dice.mean().item()))
            print(f"step {step:4d} loss={loss.item():.4f} eval Dice={dice.mean():.4f} IoU={iou.mean():.4f}")
            if dice.mean() > args.target_dice and step >= 50:
                break

    final_dice = dices[-1][1]
    print(f"Per-image Dice: {[round(d, 3) for d in dice.tolist()]}")
    print(f"FINAL mean Dice on {len(idx)} images = {final_dice:.4f} -> "
          f"{'PASS' if final_dice > args.target_dice else 'FAIL'} (target > {args.target_dice})")

    # Figure: loss/Dice curves + 4 predictions vs ground truth
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(2, 4, (1, 2))
    ax.plot(losses, label="train loss")
    ax.set_xlabel("step"); ax.set_ylabel("Dice+BCE loss"); ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(*zip(*dices), "r.-", label="eval Dice")
    ax2.axhline(args.target_dice, color="r", ls="--", lw=0.8)
    ax2.set_ylabel("Dice", color="r"); ax2.set_ylim(0, 1)
    ax.set_title(f"Overfit {len(idx)} tumor slices: final Dice {final_dice:.3f}")
    pred = (torch.sigmoid(logits) > 0.5).float().cpu()
    for j in range(4):
        a = fig.add_subplot(2, 4, 5 + j)
        a.imshow(denorm(x[j].cpu()))
        a.contour(y[j, 0].cpu().numpy(), levels=[0.5], colors="lime", linewidths=1)
        a.contour(pred[j, 0].numpy(), levels=[0.5], colors="red", linewidths=1, linestyles="--")
        a.set_title(f"Dice {dice[j]:.3f} (green=GT, red=pred)", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=90)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
