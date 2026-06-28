"""text_counterfactual.py — Tests whether the image attention pattern
actually changes when the input text changes, with the SAME X-ray image
held fixed, as opposed to the image pathway being purely image-driven and
ignoring text content.

For each selected pathology, runs the model twice on the same X-ray:
  1. With its real radiology report
  2. With a blank/neutral text ("" → just the [CLS] token, no information)

Then compares the learned image-pooling attention map between the two
runs. If the heatmap changes meaningfully when the only difference is the
text input, that's direct evidence the cross-attention conditions image
processing on text content. If it barely changes, the image pathway is
largely text-independent.

Usage:
    python text_counterfactual.py \
        --checkpoint checkpoints/multimodal_fusion.pt \
        --tokenizer  ../text_classification/checkpoints/tokenizer.json \
        --csv        /content/drive/MyDrive/dataset_labeled.csv \
        --image_dir  /content/Png \
        --labels Pneumothorax Cardiomegaly Effusion Normal
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df              # noqa: E402
from visualize_attention import (                            # noqa: E402
    load_model, forward_with_attention, patch_grid_size, VAL_TF,
)


def run_with_text(model, tok, pil_img, text, device):
    pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
    enc = tok.encode(text)
    ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)
    out = forward_with_attention(model, ids, pixel_values)
    probs = out["logits"].sigmoid()[0].cpu()
    pool_w = out["pool_w"][0].cpu()      # (197,)
    return probs, pool_w


def grid_from_pool(pool_w, grid):
    patches = pool_w[1:].reshape(grid, grid)            # drop CLS token
    if patches.max() > 0:
        patches = patches / patches.max()
    return patches.numpy()


def overlay(ax, pil_image, grid_2d, title, cmap="jet", vmin=0, vmax=1):
    heat = F.interpolate(
        torch.tensor(grid_2d).unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1], mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    ax.imshow(pil_image)
    ax.imshow(heat, cmap=cmap, alpha=0.45, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.axis("off")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    grid = patch_grid_size(model.vit)
    df = load_paired_df(args.csv)

    n = len(args.labels)
    fig, axes = plt.subplots(n, 4, figsize=(17, 4.4 * n))
    if n == 1:
        axes = axes[None, :]

    for i, label in enumerate(args.labels):
        matches = df.index[df[label] == 1].tolist()
        if not matches:
            print(f"No sample found for '{label}', skipped.")
            continue
        idx = matches[0]
        row = df.iloc[idx]
        img_path = os.path.join(args.image_dir, row["image_id"])
        pil_img = Image.open(img_path).convert("RGB")
        real_text = row["findings"]
        label_idx = LABEL_COLS.index(label)

        probs_real, pool_real = run_with_text(model, tok, pil_img, real_text, device)
        probs_blank, pool_blank = run_with_text(model, tok, pil_img, "", device)

        grid_real = grid_from_pool(pool_real, grid)
        grid_blank = grid_from_pool(pool_blank, grid)
        grid_diff = grid_real - grid_blank   # signed difference, real - blank
        delta_l1 = np.abs(grid_diff).mean()

        axes[i, 0].imshow(pil_img)
        axes[i, 0].set_title(
            f"{label}  (idx={idx})\n"
            f"P real text={probs_real[label_idx]:.2f}  |  "
            f"P blank text={probs_blank[label_idx]:.2f}",
            fontsize=11.5, fontweight="bold")
        axes[i, 0].axis("off")

        overlay(axes[i, 1], pil_img, grid_real,
               "Image pooling — with real report")
        overlay(axes[i, 2], pil_img, grid_blank,
               "Image pooling — with blank text")

        heat_diff = F.interpolate(
            torch.tensor(grid_diff).unsqueeze(0).unsqueeze(0),
            size=pil_img.size[::-1], mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        axes[i, 3].imshow(pil_img)
        im = axes[i, 3].imshow(heat_diff, cmap="RdBu_r", alpha=0.55,
                              vmin=-1, vmax=1)
        axes[i, 3].set_title(
            f"Difference (real − blank)\nmean |Δ| = {delta_l1:.3f}",
            fontsize=11.5, fontweight="bold")
        axes[i, 3].axis("off")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

    fig.suptitle(
        "Text counterfactual test — same image, real vs blank report\n"
        "(if image attention barely changes, the image pathway is largely text-independent)",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = args.out or os.path.join(ROOT, "outputs", "text_counterfactual.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--labels", nargs="+",
                        default=["Pneumothorax", "Cardiomegaly", "Effusion", "Normal"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args)
