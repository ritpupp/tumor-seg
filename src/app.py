"""Step 8: Gradio demo. Upload an MRI slice -> tumor mask overlay + tumor area.

Research demo, not for clinical use.
Run: python src/app.py [--ckpt checkpoints/best.pt] [--port 7860]
"""
import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from dataset import get_transforms
from model import build_model
from train import get_device

SIZE = 256  # training resolution


class Segmenter:
    def __init__(self, ckpt):
        self.device = get_device()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model = build_model(encoder_weights=None)
        self.model.load_state_dict(state["model"])
        self.model.to(self.device).eval()
        self.tf = get_transforms(False, norm=state["args"].get("norm", "imagenet"))  # match training

    @torch.no_grad()
    def __call__(self, image, threshold=0.5):
        """image: PIL image (any size/mode). Returns (overlay PIL RGB, mask bool HxW, summary str)."""
        if image is None:
            raise gr.Error("Please upload an MRI slice.")
        rgb = image.convert("RGB")
        orig_size = rgb.size
        arr = np.array(rgb.resize((SIZE, SIZE), Image.BILINEAR)) if orig_size != (SIZE, SIZE) else np.array(rgb)
        x = self.tf(image=arr)["image"][None].to(self.device)
        prob = torch.sigmoid(self.model(x))[0, 0].cpu().numpy()
        if orig_size != (SIZE, SIZE):
            prob = np.array(Image.fromarray(prob).resize(orig_size, Image.BILINEAR))
        mask = prob > threshold

        base = np.array(rgb).astype(np.float32)
        overlay = base.copy()
        overlay[mask] = 0.55 * base[mask] + 0.45 * np.array([255, 0, 0])
        tumor_px = int(mask.sum())
        head_px = int((base.sum(-1) > 30).sum())
        if tumor_px == 0:
            summary = "No tumor detected on this slice."
        else:
            summary = (f"Tumor area: {tumor_px:,} pixels ({100 * tumor_px / max(head_px, 1):.1f}% of the head "
                       f"area on this slice). Pixel spacing is not stored in these images, so area is "
                       f"reported in pixels, not mm².")
        if orig_size != (SIZE, SIZE):
            summary += f"\nNote: image was resized from {orig_size[0]}x{orig_size[1]} to {SIZE}x{SIZE} for the model."
        return Image.fromarray(overlay.clip(0, 255).astype(np.uint8)), mask, summary


def build_ui(segment, examples):
    def run(image, threshold):
        overlay, _, summary = segment(image, threshold)
        return overlay, summary

    with gr.Blocks(title="LGG tumor segmentation demo") as demo:
        gr.Markdown("# Brain MRI tumor segmentation (LGG)\n"
                    "Upload a 3-channel MRI slice (pre-contrast / FLAIR / post-contrast, as in the "
                    "Kaggle LGG dataset). The U-Net marks predicted tumor in red.\n\n"
                    "**Research demo, not for clinical use.**")
        with gr.Row():
            inp = gr.Image(type="pil", label="MRI slice (.tif / .png / .jpg)", image_mode="RGB")
            out = gr.Image(type="pil", label="Predicted tumor (red)")
        thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Probability threshold")
        txt = gr.Textbox(label="Result", lines=3)
        btn = gr.Button("Segment", variant="primary")
        btn.click(run, [inp, thr], [out, txt], api_name="segment")
        if examples:
            gr.Examples(examples=[[e, 0.5] for e in examples], inputs=[inp, thr])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--examples-dir", default="examples", help="folder of example slices (optional)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()

    segment = Segmenter(args.ckpt)
    ex_dir = Path(args.examples_dir)
    examples = sorted(str(p) for p in ex_dir.glob("*.png")) if ex_dir.is_dir() else []
    build_ui(segment, examples).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
