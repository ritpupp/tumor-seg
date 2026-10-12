"""Compare two runs from their saved evaluate.py CSVs (no inference, no test re-run).

Usage: python src/compare_runs.py --a outputs --b outputs/v2 --names v1 v2
"""
import argparse
from pathlib import Path

import pandas as pd

BINS = [0, 250, 500, 1000, 2000, 1e9]
LABELS = ["<250", "250-500", "500-1k", "1k-2k", ">2k"]
FOCUS = ["TCGA_HT_A5RC", "TCGA_FG_A60K"]


def summarize(d):
    pp = pd.read_csv(d / "test_per_patient.csv") if (d / "test_per_patient.csv").exists() else None
    ps = pd.read_csv(d / "test_per_slice.csv") if (d / "test_per_slice.csv").exists() else None
    vp = pd.read_csv(d / "val_per_patient.csv")
    tum = ps[ps.gt_px > 0].copy()
    tum["bin"] = pd.cut(tum.gt_px, BINS, labels=LABELS)
    ms = lambda s: f"{s.mean():.3f} ± {s.std(ddof=0):.3f}"
    out = {
        "Test Dice, tumor-only (mean ± std)": ms(pp.dice_tumor),
        "Test IoU, tumor-only (mean ± std)": ms(pp.iou_tumor),
        "Test Dice, tumor-only (median)": f"{pp.dice_tumor.median():.3f}",
        "Test IoU, tumor-only (median)": f"{pp.iou_tumor.median():.3f}",
        "Test Dice, all slices (mean ± std)": ms(pp.dice_all),
        "Test IoU, all slices (mean ± std)": ms(pp.iou_all),
    }
    for f in FOCUS:
        out[f"Test Dice {f[5:]} (tumor-only)"] = f"{pp[pp.patient.str.startswith(f)].dice_tumor.iloc[0]:.3f}"
    emp = ps[ps.gt_px == 0]
    out["Test false-positive empty slices"] = f"{(emp.pred_px > 0).sum()}/{len(emp)}"
    out["Test tumor slices with no prediction"] = f"{(tum.pred_px == 0).sum()}/{len(tum)}"
    for b, g in tum.groupby("bin", observed=False):
        out[f"Test slice Dice, tumor {b} px (n={len(g)})"] = f"{g.dice.mean():.3f} (missed {(g.pred_px == 0).sum()})"
    out["Val Dice, tumor-only (mean ± std)"] = ms(vp.dice_tumor)
    out["Val IoU, tumor-only (mean ± std)"] = ms(vp.iou_tumor)
    out["Val Dice, tumor-only (median)"] = f"{vp.dice_tumor.median():.3f}"
    out["Val Dice, all slices (mean ± std) [selection metric]"] = ms(vp.dice_all)
    return out, pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="outputs")
    ap.add_argument("--b", default="outputs/v2")
    ap.add_argument("--names", nargs=2, default=["v1", "v2"])
    ap.add_argument("--out", default="outputs/v2/v1_vs_v2.md")
    args = ap.parse_args()
    (sa, pa), (sb, pb) = summarize(Path(args.a)), summarize(Path(args.b))
    na, nb = args.names
    lines = [f"| Metric | {na} | {nb} |", "|---|---|---|"] + [f"| {k} | {sa[k]} | {sb[k]} |" for k in sa]

    m = pa[["patient", "dice_tumor"]].merge(pb[["patient", "dice_tumor"]], on="patient", suffixes=("_a", "_b"))
    m["diff"] = m.dice_tumor_b - m.dice_tumor_a
    better, worse = (m["diff"] > 0.01).sum(), (m["diff"] < -0.01).sum()
    rest = m[~m.patient.str.startswith(tuple(FOCUS))]
    lines += ["", f"Paired per-patient tumor-only Dice, {nb} - {na}: {nb} better on {better}, worse on {worse}, "
                  f"within ±0.01 on {len(m) - better - worse} of {len(m)} patients; median diff {m['diff'].median():+.3f}.",
              f"Excluding {', '.join(f[5:] for f in FOCUS)}: {na} {rest.dice_tumor_a.mean():.3f} vs {nb} "
              f"{rest.dice_tumor_b.mean():.3f} (n={len(rest)}).", "",
              "| Patient | " + na + " | " + nb + " | diff |", "|---|---|---|---|"]
    lines += [f"| {r.patient[:12]} | {r.dice_tumor_a:.3f} | {r.dice_tumor_b:.3f} | {r.diff:+.3f} |"
              for r in m.sort_values("diff").itertuples()]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
