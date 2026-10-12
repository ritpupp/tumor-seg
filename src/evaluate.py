"""Step 6: evaluate best.pt on the test set.

Per-patient Dice/IoU with pixels pooled across a patient's slices, two ways:
  - tumor-only: only slices whose ground-truth mask contains tumor (headline)
  - all slices: every slice, so false positives on empty slices count against the model
Reported as mean +- std over test patients. Also writes per-slice and per-patient CSVs.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LGGDataset, get_transforms
from model import build_model
from train import get_device


def pooled(inter, pred, gt):
    """Dice and IoU from summed pixel counts; both-empty counts as perfect."""
    if pred + gt == 0:
        return 1.0, 1.0
    return 2 * inter / (pred + gt), inter / (pred + gt - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(encoder_weights=None)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"Device: {device} | checkpoint epoch {ckpt['epoch']} "
          f"(val per-patient Dice {ckpt['val']['patient_dice']:.4f})")

    ds = LGGDataset(args.data_dir, args.split, args.splits, get_transforms(False, norm=ckpt["args"].get("norm", "imagenet")), args.max_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows, i = [], 0
    with torch.no_grad():
        for x, y in dl:
            pred = (torch.sigmoid(model(x.to(device))) > 0.5).float().cpu()
            inter = (pred * y).sum((1, 2, 3))
            ps, gs = pred.sum((1, 2, 3)), y.sum((1, 2, 3))
            for j in range(x.size(0)):
                img_p = ds.pairs[i + j][0]
                d, iou = pooled(inter[j].item(), ps[j].item(), gs[j].item())
                rows.append({"patient": img_p.parent.name, "slice": img_p.stem,
                             "gt_px": int(gs[j]), "pred_px": int(ps[j]), "inter_px": int(inter[j]),
                             "dice": d, "iou": iou})
            i += x.size(0)

    # Pool pixel counts per patient: all slices vs tumor-containing slices only
    acc = defaultdict(lambda: {"all": np.zeros(3), "tumor": np.zeros(3), "n": 0, "n_tumor": 0})
    for r in rows:
        a = acc[r["patient"]]
        c = np.array([r["inter_px"], r["pred_px"], r["gt_px"]])
        a["all"] += c
        a["n"] += 1
        if r["gt_px"] > 0:
            a["tumor"] += c
            a["n_tumor"] += 1

    per_patient = []
    for pid, a in sorted(acc.items()):
        dt, it = pooled(*a["tumor"]) if a["n_tumor"] else (float("nan"), float("nan"))
        da, ia = pooled(*a["all"])
        per_patient.append({"patient": pid, "n_slices": a["n"], "n_tumor_slices": a["n_tumor"],
                            "dice_tumor": dt, "iou_tumor": it, "dice_all": da, "iou_all": ia,
                            "fp_px_on_empty": int(a["all"][1] - a["tumor"][1])})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in ((f"{args.split}_per_slice.csv", rows), (f"{args.split}_per_patient.csv", per_patient)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    print(f"\n{'patient':24s} {'slices':>6s} {'tumor':>5s} {'Dice_t':>7s} {'IoU_t':>7s} "
          f"{'Dice_all':>8s} {'IoU_all':>8s} {'FPpx_empty':>10s}")
    for p in per_patient:
        print(f"{p['patient']:24s} {p['n_slices']:6d} {p['n_tumor_slices']:5d} {p['dice_tumor']:7.4f} "
              f"{p['iou_tumor']:7.4f} {p['dice_all']:8.4f} {p['iou_all']:8.4f} {p['fp_px_on_empty']:10d}")

    def ms(key):
        v = np.array([p[key] for p in per_patient if not np.isnan(p[key])])
        return v.mean(), v.std(), len(v)

    print(f"\n{args.split.upper()} SET, per-patient mean +- std (pixels pooled per patient):")
    for label, k in (("Tumor-only slices [HEADLINE]", "tumor"), ("All slices", "all")):
        (dm, ds_, n), (im, is_, _) = ms(f"dice_{k}"), ms(f"iou_{k}")
        print(f"  {label:30s} Dice {dm:.4f} +- {ds_:.4f} | IoU {im:.4f} +- {is_:.4f} | n={n} patients")

    empty = [r for r in rows if r["gt_px"] == 0]
    fp_slices = sum(r["pred_px"] > 0 for r in empty)
    tumor = [r for r in rows if r["gt_px"] > 0]
    missed = sum(r["pred_px"] == 0 for r in tumor)
    print(f"  Empty slices with any predicted tumor (false positive slices): {fp_slices}/{len(empty)}")
    print(f"  Tumor slices with no prediction at all (missed slices): {missed}/{len(tumor)}")
    print(f"Saved {out_dir / f'{args.split}_per_slice.csv'} and {out_dir / f'{args.split}_per_patient.csv'}")


if __name__ == "__main__":
    main()
