"""Step 7: error analysis on the test set.

Uses outputs/test_per_patient.csv + test_per_slice.csv (from evaluate.py) and re-runs the model
on selected slices to draw overlays. Saves:
  outputs/patients_best_median_worst.png  - tumor-only Dice ranking: best / median / worst patient
  outputs/dice_vs_size.png                - per-slice and per-patient Dice vs tumor size
  outputs/failure_cases.png               - examples of each failure mode
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from check_dataset import denorm
from dataset import get_transforms
from model import build_model
from train import get_device


class Predictor:
    def __init__(self, ckpt, data_dir):
        self.device = get_device()
        self.model = build_model(encoder_weights=None)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
        self.model.to(self.device).eval()
        self.tf = get_transforms(False)
        self.data_dir = Path(data_dir)

    @torch.no_grad()
    def __call__(self, patient, slice_stem):
        img_p = self.data_dir / patient / f"{slice_stem}.tif"
        image = np.array(Image.open(img_p).convert("RGB"))
        gt = np.array(Image.open(img_p.with_name(f"{slice_stem}_mask.tif"))) > 0
        x = self.tf(image=image)["image"][None].to(self.device)
        prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        return image, gt, prob > 0.5


def draw(ax, image, gt, pred, title):
    ax.imshow(image)
    if gt.any():
        ax.contour(gt, levels=[0.5], colors="lime", linewidths=1.2)
    if pred.any():
        ax.contour(pred, levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def slice_title(r):
    return f"{r.slice.split('_')[-1]}: GT {r.gt_px}px pred {r.pred_px}px Dice {r.dice:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--n-slices", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out_dir)
    pp = pd.read_csv(out / "test_per_patient.csv").sort_values("dice_tumor", ascending=False)
    ps = pd.read_csv(out / "test_per_slice.csv")
    ps["slice_idx"] = ps.slice.str.rsplit("_", n=1).str[-1].astype(int)
    tum = ps[ps.gt_px > 0].copy()
    emp = ps[ps.gt_px == 0]
    predict = Predictor(args.ckpt, args.data_dir)

    # ---- 1. Best / median / worst patient overlays (tumor-only Dice) ----
    picks = {"BEST": pp.iloc[0], "MEDIAN": pp.iloc[len(pp) // 2], "WORST": pp.iloc[-1]}
    fig, axes = plt.subplots(3, args.n_slices, figsize=(3.2 * args.n_slices, 10))
    for r, (label, p) in enumerate(picks.items()):
        t = tum[tum.patient == p.patient].sort_values("slice_idx")
        sel = t.iloc[np.linspace(0, len(t) - 1, args.n_slices).round().astype(int)]  # spread through tumor
        for c, s in enumerate(sel.itertuples()):
            draw(axes[r, c], *predict(s.patient, s.slice), slice_title(s))
        axes[r, 0].text(-0.08, 0.5, f"{label}\n{p.patient[:12]}\nDice {p.dice_tumor:.3f}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Test patients ranked by tumor-only Dice (green = ground truth, red dashed = prediction)")
    fig.tight_layout(rect=(0.06, 0, 1, 0.97))
    fig.savefig(out / "patients_best_median_worst.png", dpi=90)
    plt.close(fig)
    print("Best/median/worst:", {k: (v.patient, round(v.dice_tumor, 4)) for k, v in picks.items()})

    # ---- 2. Dice vs tumor size ----
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
    worst2 = list(pp.patient.iloc[-2:])
    for name, g in tum.groupby("patient"):
        hl = name in worst2
        a.scatter(g.gt_px, g.dice, s=14 if hl else 10, alpha=0.9 if hl else 0.45,
                  label=name[:12] if hl else None, zorder=3 if hl else 2,
                  color=None if hl else "tab:blue")
    a.set_xscale("log"); a.set_xlabel("ground-truth tumor area on slice (pixels, log)")
    a.set_ylabel("slice Dice"); a.set_title("Per tumor slice (n=%d)" % len(tum)); a.legend(fontsize=8)
    bins = [0, 250, 500, 1000, 2000, 1e9]
    labels = ["<250", "250-500", "500-1k", "1k-2k", ">2k"]
    tum["size_bin"] = pd.cut(tum.gt_px, bins, labels=labels)
    summary = tum.groupby("size_bin", observed=False).agg(
        n=("dice", "size"), mean_dice=("dice", "mean"), missed=("pred_px", lambda v: (v == 0).sum()))
    for x, row in zip(np.logspace(np.log10(80), np.log10(4000), len(labels)), summary.itertuples()):
        a.text(x, -0.13, f"{row.Index}\nn={row.n} μ={row.mean_dice:.2f}", fontsize=7, ha="center",
               transform=a.get_xaxis_transform())
    pt = tum.groupby("patient").gt_px.mean().rename("mean_gt_px")
    pp2 = pp.set_index("patient").join(pt)
    b.scatter(pp2.mean_gt_px, pp2.dice_tumor, s=40)
    for name, r in pp2.iterrows():
        b.annotate(name[5:12], (r.mean_gt_px, r.dice_tumor), fontsize=7, xytext=(3, 3), textcoords="offset points")
    b.set_xscale("log"); b.set_xlabel("mean tumor area per tumor slice (pixels, log)")
    b.set_ylabel("patient Dice (tumor-only, pooled)"); b.set_title("Per patient (n=%d)" % len(pp2)); b.set_ylim(0, 1)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "dice_vs_size.png", dpi=100)
    plt.close(fig)
    print("\nTumor slices by size:\n" + summary.to_string())
    rho = pd.Series(tum.gt_px).corr(tum.dice, method="spearman")
    print(f"Spearman(slice tumor area, slice Dice) = {rho:.3f}")

    # ---- 3. Failure-mode stats ----
    missed = tum[tum.pred_px == 0]
    print(f"\nMissed tumor slices: {len(missed)}/{len(tum)}; median GT area {missed.gt_px.median():.0f}px "
          f"(vs {tum.gt_px.median():.0f}px for all tumor slices); by patient: "
          f"{missed.patient.str[:12].value_counts().to_dict()}")
    # position of each tumor slice within its patient's tumor extent (0 = first tumor slice, 1 = last)
    ext = tum.groupby("patient").slice_idx.agg(["min", "max"])
    tum = tum.join(ext, on="patient")
    tum["edge"] = (tum.slice_idx == tum["min"]) | (tum.slice_idx == tum["max"])
    print(f"Missed slices at the top/bottom edge of the tumor: "
          f"{tum[tum.pred_px == 0].edge.sum()}/{len(missed)} (edge slices are {tum.edge.mean():.0%} of tumor slices)")
    fp = emp[emp.pred_px > 0]
    print(f"False-positive empty slices: {len(fp)}/{len(emp)}; FP px total {fp.pred_px.sum()}; by patient: "
          f"{fp.patient.str[:12].value_counts().to_dict()}")
    for name in worst2:
        t = tum[tum.patient == name]
        print(f"{name}: {len(t)} tumor slices, GT {t.gt_px.sum()}px, pred {t.pred_px.sum()}px, "
              f"overlap {t.inter_px.sum()}px, slices missed {int((t.pred_px == 0).sum())}, "
              f"FP px on empty slices {emp[emp.patient == name].pred_px.sum()}")

    # ---- 4. Failure-case figure: one row per mode ----
    a60k, a5rc = worst2[0], worst2[1]
    small_missed = tum[~tum.patient.isin(worst2) & (tum.dice < 0.1) & (tum.gt_px < 500)]
    rows = [
        ("1. Over-segmentation\n" + a60k[:12], tum[tum.patient == a60k].sort_values("pred_px", ascending=False).head(4)),
        ("2. Whole tumor missed\n" + a5rc[:12], tum[tum.patient == a5rc].sort_values("gt_px", ascending=False).head(4)),
        ("3. Small tumors missed\n(other patients)", small_missed.sort_values("gt_px", ascending=False)
         .drop_duplicates("patient").head(4)),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10))
    for r, (label, sel) in enumerate(rows):
        for c in range(4):
            ax = axes[r, c]
            if c < len(sel):
                s = sel.iloc[c]
                draw(ax, *predict(s.patient, s.slice), f"{s.patient[5:12]} " + slice_title(s))
            else:
                ax.axis("off")
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    fig.suptitle("Failure modes (green = ground truth, red dashed = prediction)")
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    fig.savefig(out / "failure_cases.png", dpi=90)
    plt.close(fig)

    # Hypothesis checks over ALL patients (train/val/test): are the failure patients' scans unusual?
    #  - grayscale: all three channels identical (one sequence copied into RGB)
    #  - FLAIR balance: green (FLAIR) brightness inside the head / mean of red+blue
    #  - FLAIR contrast: median over tumor slices of FLAIR(tumor) / FLAIR(rest of brain)
    splits = json.loads(Path(args.splits).read_text())["folders"]
    stats = []
    for split, folders in splits.items():
        for folder in folders:
            imgs = sorted(p for p in (Path(args.data_dir) / folder).glob("*.tif") if not p.stem.endswith("_mask"))
            gray, head_px, ratios = 0, [], []
            for p in imgs:
                a = np.array(Image.open(p).convert("RGB")).astype(float)
                m = np.array(Image.open(p.with_name(p.stem + "_mask.tif"))) > 0
                gray += np.abs(a[..., 0] - a[..., 1]).mean() < 1 and np.abs(a[..., 1] - a[..., 2]).mean() < 1
                head = a.sum(-1) > 30
                head_px.append(a[head])
                if m.sum() > 100:
                    ratios.append(a[..., 1][m].mean() / a[..., 1][head & ~m].mean())
            mean = np.concatenate(head_px).mean(0)
            stats.append({"patient": folder, "split": split, "gray_frac": gray / len(imgs),
                          "flair_balance": mean[1] / ((mean[0] + mean[2]) / 2),
                          "flair_contrast": np.median(ratios) if ratios else np.nan})
    st = pd.DataFrame(stats)
    st.to_csv(out / "patient_intensity_stats.csv", index=False)
    g = st[st.gray_frac == 1]
    print(f"\nFully grayscale patients: {len(g)} -> {g.groupby('split').patient.apply(lambda v: [x[:12] for x in v]).to_dict()}")
    print(f"FLAIR balance: median {st.flair_balance.median():.2f}; lowest 5: "
          f"{[(r.patient[:12], r.split, round(r.flair_balance, 2)) for r in st.nsmallest(5, 'flair_balance').itertuples()]}")
    print(f"FLAIR tumor/brain contrast: median {st.flair_contrast.median():.2f}, 5th pct {st.flair_contrast.quantile(0.05):.2f}")
    for name in worst2:
        r = st[st.patient == name].iloc[0]
        print(f"  {name[:12]}: grayscale {r.gray_frac:.0%}, FLAIR balance {r.flair_balance:.2f}, "
              f"FLAIR contrast {r.flair_contrast:.2f}")
    print(f"\nSaved {out}/patients_best_median_worst.png, dice_vs_size.png, failure_cases.png")


if __name__ == "__main__":
    main()
