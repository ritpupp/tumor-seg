"""Step 3: LGG slice dataset, augmentation, and dataloaders.

Images are RGB uint8 256x256; masks are {0,255} -> float {0,1} with shape (1, H, W).
Normalization uses ImageNet mean/std because the encoder is ImageNet-pretrained.
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


def get_transforms(train):
    aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=0.5),
    ] if train else []
    return A.Compose(aug + [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


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


def get_loaders(data_dir, splits_path="splits.json", batch_size=16, num_workers=4, max_samples=None):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = LGGDataset(data_dir, split, splits_path, get_transforms(split == "train"), max_samples)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"),
                                    num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                                    drop_last=(split == "train"), persistent_workers=num_workers > 0)
    return loaders
