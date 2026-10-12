"""Step 3: LGG slice dataset, augmentation, and dataloaders.

Images are RGB uint8 256x256; masks are {0,255} -> float {0,1} with shape (1, H, W).
Normalization: 'imagenet' (v1, ImageNet mean/std) or 'per-image' (v2, each channel standardized by
its own mean/std over the head region).
"""
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


NORMS = ("imagenet", "per-image")
AUGS = ("basic", "strong")


def head_mask(img, margin=20, border=8, min_frac=0.05):
    """Boolean HxW head mask. Subtracts each channel's background level (median of the image border)
    first, because 3 patients have a non-black background in one channel. Falls back to the whole
    image if the mask is tiny (e.g. a nearly empty top-of-head slice)."""
    a = img.astype(np.float32)
    edge = np.concatenate([a[:border].reshape(-1, 3), a[-border:].reshape(-1, 3),
                           a[:, :border].reshape(-1, 3), a[:, -border:].reshape(-1, 3)])
    mask = (a - np.median(edge, axis=0)).max(-1) > margin
    return mask if mask.mean() >= min_frac else np.ones(mask.shape, bool)


class PerImageStandardize(A.ImageOnlyTransform):
    """Standardize each channel by its own mean/std over the head region (applied to the whole image).

    A channel that is (near-)constant inside the head (std < min_std grey levels, ~5% of slices, mostly
    a sequence missing at the top/bottom of a scan) is treated as missing and set to 0; dividing by its
    ~0 std gave values up to 1e7. Output is clipped to +-clip to bound the remaining rare outliers.
    """

    def __init__(self, min_std=1.0, clip=10.0, p=1.0):
        super().__init__(p=p)
        self.min_std = min_std
        self.clip = clip

    def apply(self, img, **params):
        a = img.astype(np.float32)
        head = a[head_mask(img)]
        mean, std = head.mean(0), head.std(0)
        missing = std < self.min_std
        z = (a - mean) / np.where(missing, 1.0, std)
        z[..., missing] = 0.0
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def get_transform_init_args_names(self):
        return ("min_std", "clip")


def get_augs(aug="basic"):
    """Train-time augmentation (before normalization). Geometric part is identical for both settings;
    intensity transforms are image-only (albumentations never applies them to the mask)."""
    assert aug in AUGS, aug
    geometric = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
    ]
    if aug == "basic":  # v1
        intensity = [A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=0.5)]
    else:
        intensity = [
            # Ranges chosen so the darkest 5% of draws keep ~1/3 of head brightness; wider ranges
            # (0.25/0.25, gamma 70-150, gain 0.75-1.25) crushed ~5% of images to near-black uint8.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.RandomGamma(gamma_limit=(80, 125), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=True, elementwise=False, p=0.5),  # per-channel gain
        ]
    return geometric + intensity


def get_normalize(norm="imagenet"):
    assert norm in NORMS, norm
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) if norm == "imagenet" else PerImageStandardize()


def get_transforms(train, norm="imagenet", aug="basic"):
    return A.Compose((get_augs(aug) if train else []) + [get_normalize(norm), ToTensorV2()])


def list_slices(data_dir, folders):
    """Return sorted (image_path, mask_path) pairs for the given patient folders."""
    pairs = []
    for folder in folders:
        for img in sorted((Path(data_dir) / folder).glob("*.tif")):
            if img.stem.endswith("_mask"):
                continue
            pairs.append((img, img.with_name(img.stem + "_mask.tif")))
    return pairs


class LGGDataset(Dataset):
    def __init__(self, data_dir, split, splits_path="splits.json", transform=None, max_samples=None):
        folders = json.loads(Path(splits_path).read_text())["folders"][split]
        self.pairs = list_slices(data_dir, folders)
        if max_samples:
            self.pairs = self.pairs[:max_samples]
        assert self.pairs, f"No slices found for split={split} in {data_dir}"
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_p, mask_p = self.pairs[i]
        image = np.array(Image.open(img_p).convert("RGB"))
        mask = (np.array(Image.open(mask_p)) > 0).astype(np.float32)
        if self.transform:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask.unsqueeze(0).float()


def get_loaders(data_dir, splits_path="splits.json", batch_size=16, num_workers=4, max_samples=None,
                norm="imagenet", aug="basic", splits=("train", "val", "test")):
    loaders = {}
    for split in splits:
        ds = LGGDataset(data_dir, split, splits_path, get_transforms(split == "train", norm, aug), max_samples)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                    drop_last=(split == "train"), persistent_workers=num_workers > 0)
    return loaders
