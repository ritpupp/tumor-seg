"""Step 5: train the U-Net. Same script runs locally and on Kaggle (--data-dir).

Selection metric: mean per-patient Dice on val (tumor pixels pooled over all slices of a
patient), which is not inflated by the ~65% empty slices. Slice-level Dice is logged too.

Kaggle example:
  python src/train.py --data-dir /kaggle/input/lgg-mri-segmentation/kaggle_3m --epochs 40
  v2: add --norm per-image --aug strong --ckpt-dir checkpoints_v2 --out-dir outputs/v2
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import AUGS, NORMS, get_loaders
from model import DiceBCELoss, build_model, dice_iou_per_image


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, use_amp):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = loss_fn(logits.float(), y)  # loss in fp32 for stability
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp):
    """Returns val loss, slice-level Dice/IoU, and per-patient Dice/IoU (pooled pixels)."""
    model.eval()
    patients = [img.parent.name for img, _ in loader.dataset.pairs]  # loader must not shuffle
    stats = defaultdict(lambda: np.zeros(3))  # patient -> [intersection, pred_sum, target_sum]
    total, slice_dice, slice_iou, i = 0.0, [], [], 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        total += loss_fn(logits, y).item() * x.size(0)
        d, iou = dice_iou_per_image(logits, y)
        slice_dice += d.tolist()
        slice_iou += iou.tolist()
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter = (pred * y).sum((1, 2, 3)).cpu().numpy()
        ps, ts = pred.sum((1, 2, 3)).cpu().numpy(), y.sum((1, 2, 3)).cpu().numpy()
        for j in range(x.size(0)):
            stats[patients[i + j]] += (inter[j], ps[j], ts[j])
        i += x.size(0)
    pd_, pi_ = [], []
    for inter, ps, ts in stats.values():
        if ps + ts == 0:
            pd_.append(1.0); pi_.append(1.0)
        else:
            pd_.append(2 * inter / (ps + ts)); pi_.append(inter / (ps + ts - inter))
    return {"loss": total / i, "slice_dice": float(np.mean(slice_dice)),
            "slice_iou": float(np.mean(slice_iou)), "patient_dice": float(np.mean(pd_)),
            "patient_iou": float(np.mean(pi_)), "n_patients": len(stats)}


def plot_curves(history, out):
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    a.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    a.plot(ep, [h["val"]["loss"] for h in history], "o-", label="val")
    a.set_xlabel("epoch"); a.set_ylabel("Dice+BCE loss"); a.set_title("Loss"); a.legend()
    b.plot(ep, [h["val"]["patient_dice"] for h in history], "o-", label="val per-patient Dice")
    b.plot(ep, [h["val"]["slice_dice"] for h in history], "o-", label="val per-slice Dice")
    best = max(history, key=lambda h: h["val"]["patient_dice"])
    b.axvline(best["epoch"], color="gray", ls="--", lw=0.8, label=f"best epoch {best['epoch']}")
    b.set_xlabel("epoch"); b.set_ylabel("Dice"); b.set_ylim(0, 1); b.set_title("Validation Dice"); b.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing the TCGA_* patient folders")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-samples", type=int, default=None, help="cap slices per split (smoke test)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (CUDA only anyway)")
    ap.add_argument("--norm", choices=NORMS, default="imagenet",
                    help="imagenet = v1; per-image = standardize each channel over the head region")
    ap.add_argument("--aug", choices=AUGS, default="basic",
                    help="basic = v1; strong = + contrast, gamma, per-channel gain (image only)")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing best.pt in --ckpt-dir")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    use_amp = device.type == "cuda" and not args.no_amp  # fp16 autocast + GradScaler on CUDA
    print(f"Device: {device} | AMP: {use_amp} | args: {vars(args)}")

    ckpt_dir, out_dir = Path(args.ckpt_dir), Path(args.out_dir)
    if (ckpt_dir / "best.pt").exists() and not args.overwrite:
        raise SystemExit(f"{ckpt_dir / 'best.pt'} already exists; use another --ckpt-dir or pass --overwrite")
    # Test split is never loaded during training; model selection uses val only.
    loaders = get_loaders(args.data_dir, args.splits, args.batch_size, args.num_workers, args.max_samples,
                          norm=args.norm, aug=args.aug, splits=("train", "val"))
    print({k: len(v.dataset) for k, v in loaders.items()}, "slices")

    model = build_model().to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loaders["train"], loss_fn, opt, scaler, device, use_amp)
        val = evaluate(model, loaders["val"], loss_fn, device, use_amp)
        lr = opt.param_groups[0]["lr"]
        sched.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val, "lr": lr})
        improved = val["patient_dice"] > best
        if improved:
            best = val["patient_dice"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val, "args": vars(args)},
                       ckpt_dir / "best.pt")
        print(f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train {train_loss:.4f} | val {val['loss']:.4f} "
              f"| val patient Dice {val['patient_dice']:.4f} IoU {val['patient_iou']:.4f} "
              f"| slice Dice {val['slice_dice']:.4f} | {time.time() - t0:.0f}s{'  *best*' if improved else ''}",
              flush=True)
        plot_curves(history, out_dir / "train_curves.png")
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))

    print(f"Best val per-patient Dice {best:.4f} -> {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
