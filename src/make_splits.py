"""Step 2: patient-level 70/15/15 train/val/test split -> splits.json.

Stores patient IDs and folder names (relative), so the split works with any --data-dir.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_data import collect_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/kaggle_3m")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--out", default="splits.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs = collect_pairs(Path(args.data_dir))
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    folders = defaultdict(set)
    for pid, img_p, _ in pairs:
        folders[pid].add(img_p.parent.name)

    patients = sorted(folders)
    random.Random(args.seed).shuffle(patients)
    n = len(patients)
    n_train, n_val = round(0.70 * n), round(0.15 * n)
    split_ids = {
        "train": sorted(patients[:n_train]),
        "val": sorted(patients[n_train:n_train + n_val]),
        "test": sorted(patients[n_train + n_val:]),
    }

    # No patient may appear in two sets, and every patient must be in exactly one.
    s = {k: set(v) for k, v in split_ids.items()}
    assert not (s["train"] & s["val"]), s["train"] & s["val"]
    assert not (s["train"] & s["test"]), s["train"] & s["test"]
    assert not (s["val"] & s["test"]), s["val"] & s["test"]
    assert s["train"] | s["val"] | s["test"] == set(patients)
    print("Assert passed: no patient in two sets; all patients assigned.")

    split_of = {pid: k for k, ids in split_ids.items() for pid in ids}
    stats = defaultdict(lambda: [0, 0])  # split -> [slices, tumor slices]
    for pid, _, mask_p in pairs:
        st = stats[split_of[pid]]
        st[0] += 1
        st[1] += int(np.array(Image.open(mask_p)).max() > 0)
    for k in ("train", "val", "test"):
        sl, tu = stats[k]
        print(f"{k:5s}: {len(split_ids[k]):3d} patients ({100 * len(split_ids[k]) / n:.1f}%), "
              f"{sl:4d} slices, {100 * tu / max(sl, 1):.1f}% with tumor")

    out = {
        "seed": args.seed,
        "patients": split_ids,
        "folders": {k: sorted(f for pid in v for f in folders[pid]) for k, v in split_ids.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
