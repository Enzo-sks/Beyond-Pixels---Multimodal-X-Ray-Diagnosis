"""compare_attention_samples.py — Checks whether the attention heatmaps are
genuinely conditioned by the image/text content or whether they always
fall back on the same patches ("attention sink").

Selects one sample per target pathology (each expected to have a different
anatomical location, e.g. Pneumothorax = apical, Effusion = basal) and
displays their heatmaps side by side. If the hotspots stay at the same
position across samples despite different pathologies, the attention is
probably not content-specific.

Usage:
    python compare_attention_samples.py \
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
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df              # noqa: E402
from mmmia.multimodal_fusion.architecture_1.interpretability.visualize_attention import (                            # noqa: E402
    load_model, forward_with_attention, patch_grid_size,
    heatmap_overlay, VAL_TF,
)

from PIL import Image                                        # noqa: E402


def pick_sample(df, label):
    """Returns the index of the first row where `label` is positive."""
    matches = df.index[df[label] == 1].tolist()
    if not matches:
        raise ValueError(f"No sample found for label '{label}'")
    return matches[0]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    grid = patch_grid_size(model.vit)

    df = load_paired_df(args.csv)

    n = len(args.labels)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    if n == 1:
        axes = axes[None, :]

    for row_i, label in enumerate(args.labels):
        idx = pick_sample(df, label)
        row = df.iloc[idx]

        img_path = os.path.join(args.image_dir, row["image_id"])
        pil_img = Image.open(img_path).convert("RGB")
        text = row["findings"]

        pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
        enc = tok.encode(text)
        ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)

        out = forward_with_attention(model, ids, pixel_values)
        probs = out["logits"].sigmoid()[0].cpu().numpy()
        pred_label_idx = LABEL_COLS.index(label)

        t2i_cls = out["t2i_w"][0, 0]

        axes[row_i, 0].imshow(pil_img)
        axes[row_i, 0].set_title(
            f"{label}  (idx={idx})\nP({label})={probs[pred_label_idx]:.2f}",
            fontsize=12, fontweight="bold")
        axes[row_i, 0].axis("off")

        heatmap_overlay(axes[row_i, 1], pil_img, out["pool_w"][0], grid,
                         "Learned image pooling")

        heatmap_overlay(axes[row_i, 2], pil_img, t2i_cls, grid,
                         "Text[CLS] → Image attention")

    fig.suptitle(
        "Attention heatmap comparison across pathologies\n"
        "(hotspots should move if attention is content-specific)",
        fontsize=14, fontweight="bold", y=1.0,
    )
    fig.tight_layout()

    out_path = args.out or os.path.join(ROOT, "outputs", "attention_comparison.png")
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
                        default=["Pneumothorax", "Cardiomegaly", "Effusion", "Normal"],
                        help="One pathology per row of the figure")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    main(args)