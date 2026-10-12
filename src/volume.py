"""Per-patient tumor volume (v1) from the saved per-slice test results. No inference.

Volume = tumor pixels summed over ALL of a patient's slices (so false positives on tumor-free slices
count). Units are pixels: these images carry no mm spacing, so this is a relative measure, not cm^3.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FOCUS = ("TCGA_HT_A5RC", "TCGA_FG_A60K")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-slice", default="outputs/test_per_slice.csv", help="from evaluate.py (v1 test)")
    ap.add_argument("--out", default="outputs/volume_v1.csv")
    args = ap.parse_args()

    s = pd.read_csv(args.per_slice)
    assert {"patient", "pred_px", "gt_px"} <= set(s.columns), "per-slice CSV must hold pred_px and gt_px"
    v = s.groupby("patient").agg(n_slices=("slice", "size"), pred_px=("pred_px", "sum"), true_px=("gt_px", "sum"))
    v["error_px"] = v.pred_px - v.true_px
    v["pct_error"] = 100 * v.error_px / v.true_px
    v = v.reset_index()
    print(f"{len(v)} test patients, {len(s)} slices (from {args.per_slice}; no inference)\n")
    print(f"{'patient':14s} {'predicted':>10s} {'true':>8s} {'error':>9s} {'% error':>8s}")
    for r in v.itertuples():
        flag = "  <-- " + r.patient[5:12] if r.patient.startswith(FOCUS) else ""
        print(f"{r.patient[:12]:14s} {r.pred_px:10,d} {r.true_px:8,d} {r.error_px:+9,d} {r.pct_error:+7.1f}%{flag}")

    ape = v.pct_error.abs()
    over, under = (v.error_px > 0).sum(), (v.error_px < 0).sum()
    pear = np.corrcoef(v.pred_px, v.true_px)[0, 1]
    spear = v.pred_px.rank().corr(v.true_px.rank())
    rest = v[~v.patient.str.startswith(FOCUS)]
    print(f"\nAbsolute % error: mean {ape.mean():.1f}%, median {ape.median():.1f}%")
    print(f"Bias: total predicted {v.pred_px.sum():,} vs true {v.true_px.sum():,} px "
          f"({100 * (v.pred_px.sum() / v.true_px.sum() - 1):+.1f}%); median signed % error {v.pct_error.median():+.1f}%; "
          f"over-estimated {over}, under-estimated {under} of {len(v)} patients")
    print(f"Correlation predicted vs true volume: Pearson r = {pear:.3f}, Spearman rho = {spear:.3f}")
    print(f"Excluding {', '.join(f[5:] for f in FOCUS)} (n={len(rest)}): mean abs % error {rest.pct_error.abs().mean():.1f}%, "
          f"median {rest.pct_error.abs().median():.1f}%, median signed {rest.pct_error.median():+.1f}%, "
          f"Pearson r = {np.corrcoef(rest.pred_px, rest.true_px)[0, 1]:.3f}")
    for f in FOCUS:
        r = v[v.patient.str.startswith(f)].iloc[0]
        print(f"  {f[5:]}: predicted {r.pred_px:,} vs true {r.true_px:,} px ({r.pct_error:+.1f}%)")
    # Is there a size effect? Spearman between true volume and signed % error.
    print(f"Size effect: Spearman(true volume, signed % error) = {v.true_px.rank().corr(v.pct_error.rank()):.3f}; "
          f"excluding the two = {rest.true_px.rank().corr(rest.pct_error.rank()):.3f}")
    v.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
