"""representation_mediation.py — Decomposes the text counterfactual effect
into "explained by text_repr" vs "explained by image_repr", averaged over
many samples per label.

text_counterfactual.py showed that removing the report text causes large
drops in predicted probability, while the image's spatial pooling WEIGHTS
barely move. This script checks whether the prediction shift is instead
explained by a change in REPRESENTATION CONTENT (not just attention
weights): for each sample, we extract text_repr and image_repr under both
the real report and a blank text, then feed all 4 combinations through
the classification head:

    (text_R, image_R)  -> should reproduce P(real text)
    (text_B, image_B)  -> should reproduce P(blank text)
    (text_R, image_B)  -> isolates the effect of text_repr alone
    (text_B, image_R)  -> isolates the effect of image_repr alone

If (text_R, image_B) ~ P(real text), the prediction shift is driven almost
entirely by the text branch. If (text_B, image_R) ~ P(real text), the
image branch's CONTENT (not just its attention weights) is doing the work
instead. Cosine similarity between the real/blank representations of each
branch quantifies how much each one actually moved.

This is averaged over N positive samples per label (not just one), since
single-sample results can be noisy or unrepresentative.

Usage:
    python representation_mediation.py \
        --checkpoint checkpoints/multimodal_fusion.pt \
        --tokenizer  ../text_classification/checkpoints/tokenizer.json \
        --csv        /content/drive/MyDrive/dataset_labeled.csv \
        --image_dir  /content/Png \
        --labels Pneumothorax Cardiomegaly Effusion Normal \
        --n_samples 30 \
        --out results.csv
"""

import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df              # noqa: E402
from visualize_attention import load_model, VAL_TF           # noqa: E402


@torch.no_grad()
def get_representations(model, input_ids, pixel_values):
    """Replicates MultimodalFusion.forward but returns the pooled
    text_repr and image_repr separately, before the classification head."""
    text_tokens = model.text_encoder(input_ids)
    image_tokens = model.vit(pixel_values=pixel_values).last_hidden_state

    ca = model.cross_attn
    T = ca.text_proj(text_tokens)
    I = ca.image_proj(image_tokens)
    T_ca, _ = ca.t2i(T, I, I)
    I_ca, _ = ca.i2t(I, T, T)
    T = ca.norm_t1(T + T_ca)
    I = ca.norm_i1(I + I_ca)
    T = ca.norm_t2(T + ca.ffn_t(T))
    I = ca.norm_i2(I + ca.ffn_i(I))

    text_repr = T[:, 0]
    pool_w = F.softmax(model.image_pool_w(I), dim=1)
    image_repr = (pool_w * I).sum(dim=1)
    return text_repr, image_repr


def encode(tok, text, device):
    enc = tok.encode(text)
    return torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)


def run_one_sample(model, tok, image_dir, row, label_idx, device):
    img_path = os.path.join(image_dir, row["image_id"])
    pil_img = Image.open(img_path).convert("RGB")
    pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)

    ids_real = encode(tok, row["findings"], device)
    ids_blank = encode(tok, "", device)

    text_R, image_R = get_representations(model, ids_real, pixel_values)
    text_B, image_B = get_representations(model, ids_blank, pixel_values)

    def predict(t, i):
        logits = model.head(torch.cat([t, i], dim=1))
        return logits.sigmoid()[0, label_idx].item()

    p_real_real = predict(text_R, image_R)
    p_blank_blank = predict(text_B, image_B)
    p_realT_blankI = predict(text_R, image_B)
    p_blankT_realI = predict(text_B, image_R)

    cos_text = F.cosine_similarity(text_R, text_B).item()
    cos_image = F.cosine_similarity(image_R, image_B).item()

    return [p_real_real, p_blank_blank, p_realT_blankI,
            p_blankT_realI, cos_text, cos_image]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    df = load_paired_df(args.csv)
    rng = random.Random(args.seed)

    metric_names = ["P_real_real", "P_blank_blank",
                    "P_realText_blankImage", "P_blankText_realImage",
                    "cos_text_real_vs_blank", "cos_image_real_vs_blank"]

    per_sample_rows = []
    summary_rows = []

    for label in args.labels:
        matches = df.index[df[label] == 1].tolist()
        if not matches:
            print(f"No sample found for '{label}', skipped.")
            continue

        chosen = rng.sample(matches, min(args.n_samples, len(matches)))
        label_idx = LABEL_COLS.index(label)

        values = []
        for idx in chosen:
            row = df.iloc[idx]
            result = run_one_sample(model, tok, args.image_dir, row,
                                    label_idx, device)
            values.append(result)
            per_sample_rows.append([label, idx] + result)

        values = np.array(values)               # (n_used, 6)
        means = values.mean(axis=0)
        stds = values.std(axis=0)

        print(f"\n{label}  (n={len(chosen)} samples)")
        for name, m, s in zip(metric_names, means, stds):
            print(f"  {name:<26} {m:.3f} ± {s:.3f}")

        summary_rows.append([label, len(chosen)] +
                            [v for pair in zip(means, stds) for v in pair])

    if args.out:
        import csv
        base, ext = os.path.splitext(args.out)
        summary_path = args.out
        detail_path = f"{base}_per_sample{ext}"

        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label", "n_samples"]
            for name in metric_names:
                header += [f"{name}_mean", f"{name}_std"]
            writer.writerow(header)
            writer.writerows(summary_rows)

        with open(detail_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "sample_index"] + metric_names)
            writer.writerows(per_sample_rows)

        print(f"\nSaved summary:    {summary_path}")
        print(f"Saved per-sample: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--labels", nargs="+",
                        default=["Pneumothorax", "Cardiomegaly", "Effusion", "Normal"])
    parser.add_argument("--n_samples", type=int, default=30,
                        help="Number of positive samples to average per label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None,
                        help="Optional path to save summary CSV "
                             "(a second file with '_per_sample' suffix is "
                             "also saved with the raw per-sample values)")
    args = parser.parse_args()
    main(args)