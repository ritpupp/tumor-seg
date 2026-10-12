"""Step 4: U-Net (resnet34 encoder, ImageNet weights), Dice+BCE loss, Dice/IoU metrics."""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(encoder_weights="imagenet"):
    return smp.Unet(encoder_name="resnet34", encoder_weights=encoder_weights,
                    in_channels=3, classes=1)  # outputs logits (N,1,H,W)


class DiceBCELoss(nn.Module):
    """BCE on logits + soft Dice loss (computed over the whole batch)."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum()
        dice = (2 * inter + self.smooth) / (probs.sum() + target.sum() + self.smooth)
        return self.bce(logits, target) + (1 - dice)


@torch.no_grad()
def dice_iou_per_image(logits, target, threshold=0.5, eps=1e-7):
    """Hard Dice and IoU per image, shape (N,). Empty pred + empty target counts as 1."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (1, 2, 3)
    inter = (pred * target).sum(dims)
    total = pred.sum(dims) + target.sum(dims)
    union = total - inter
    both_empty = total == 0
    dice = torch.where(both_empty, torch.ones_like(inter), 2 * inter / (total + eps))
    iou = torch.where(both_empty, torch.ones_like(inter), inter / (union + eps))
    return dice, iou
