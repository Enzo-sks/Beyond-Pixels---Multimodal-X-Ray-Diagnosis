"""mediation_auc.py — Full-test-set, all-21-label, AUC-based version of
representation_mediation.py.

representation_mediation.py averages mean predicted probability over 30
positive samples for 4 pathologies. That shows whether probabilities shift,
but not whether the model still *discriminates* positives from negatives
under each counterfactual condition, and it only covers a fifth of the
label set.

This script reconstructs the exact test split used in train.py (same
MultilabelStratifiedShuffleSplit, same random_state=42), runs every test
sample once through the backbone, and assembles 4 conditions per label by
recombining (text_repr, image_repr) and feeding them through the shared
head:

    real_real      = (text_R, image_R)   -> reproduces the reported test AUC
    blank_blank    = (text_B, image_B)   -> floor: no information at all
    realT_blankI   = (text_R, image_B)   -> text branch's contribution alone
    blankT_realI   = (text_B, image_R)   -> image branch's contribution alone

For each of the 21 labels, computes ROC-AUC under each condition over the
full test set (968 samples), plus a bootstrap 95% CI on each AUC and on the
delta AUC(real_real) - AUC(blankT_realI). A small delta means the image
representation alone (after being shaped by cross-attention with the real
report) already explains most of the discriminative power — i.e. text
content leaked into the image branch rather than the model learning
independent visual evidence.

Usage:
    python mediation_auc.py \
        --checkpoint ../checkpoints/multimodal_fusion.pt \
        --tokenizer  ../../text_classification/checkpoints/tokenizer.json \
        --csv        /content/drive/MyDrive/dataset_labeled.csv \
        --image_dir  /content/Png \
        --out        results/mediation_auc.csv
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import LABEL_COLS, load_paired_df               # noqa: E402
from visualize_attention import load_model, VAL_TF            # noqa: E402


@torch.no_grad()
def get_representations(model, input_ids, pixel_values):
    """Same as representation_mediation.py: replicates MultimodalFusion.forward
    but returns pooled text_repr / image_repr separately, before the head."""
    text_tokens = model.text_encoder(input_ids)
    image_tokens = model.vit(pixel_values=pixel_values).last_hidden_state

    ca = model.cross_attn
    T = ca.text_proj(text_tokens)
    I = ca.image_proj(image_tokens)
    T_ca, _ = ca.t2i(T, I, I)
    if ca.use_i2t:
        I_ca, _ = ca.i2t(I, T, T)
        I = ca.norm_i1(I + I_ca)
    else:
        I = ca.norm_i1(I)
    T = ca.norm_t1(T + T_ca)
    T = ca.norm_t2(T + ca.ffn_t(T))
    I = ca.norm_i2(I + ca.ffn_i(I))

    text_repr = T[:, 0]
    pool_w = F.softmax(model.image_pool_w(I), dim=1)
    image_repr = (pool_w * I).sum(dim=1)
    return text_repr, image_repr


def rebuild_test_split(df):
    """Reproduces the exact 70/15/15 split from train.py."""
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(msss.split(df, df[LABEL_COLS]))
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(msss2.split(df.iloc[temp_idx], df.iloc[temp_idx][LABEL_COLS]))
    return df.iloc[temp_idx].iloc[test_idx].reset_index(drop=True)


@torch.no_grad()
def collect_probs(model, tok, image_dir, test_df, device):
    """Runs the 4-condition forward pass for every test sample.

    Returns a dict of {condition_name: (N, 21) prob array} plus the
    (N, 21) ground-truth label array.
    """
    n = len(test_df)
    n_labels = len(LABEL_COLS)
    conditions = ["real_real", "blank_blank", "realT_blankI", "blankT_realI"]
    probs = {c: np.zeros((n, n_labels), dtype=np.float32) for c in conditions}
    labels = test_df[LABEL_COLS].values.astype(np.float32)

    blank_ids = torch.tensor(tok.encode("").ids, dtype=torch.long).unsqueeze(0).to(device)

    for i in range(n):
        row = test_df.iloc[i]
        img_path = os.path.join(image_dir, row["image_id"])
        pil_img = Image.open(img_path).convert("RGB")
        pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)

        ids_real = torch.tensor(tok.encode(row["findings"]).ids,
                                 dtype=torch.long).unsqueeze(0).to(device)

        text_R, image_R = get_representations(model, ids_real, pixel_values)
        text_B, image_B = get_representations(model, blank_ids, pixel_values)

        def predict(t, i_):
            return model.head(torch.cat([t, i_], dim=1)).sigmoid()[0].cpu().numpy()

        probs["real_real"][i]    = predict(text_R, image_R)
        probs["blank_blank"][i]  = predict(text_B, image_B)
        probs["realT_blankI"][i] = predict(text_R, image_B)
        probs["blankT_realI"][i] = predict(text_B, image_R)

        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} samples processed")

    return probs, labels


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap CI on a single AUC, resampling rows with replacement."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return np.nan, np.nan
    lo = np.percentile(aucs, 100 * alpha / 2)
    hi = np.percentile(aucs, 100 * (1 - alpha / 2))
    return lo, hi


def bootstrap_delta_ci(y_true, score_a, score_b, n_boot=1000, seed=42, alpha=0.05):
    """Bootstrap CI on AUC(score_a) - AUC(score_b), same resampled indices
    used for both so the delta accounts for paired variance."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auc_a = roc_auc_score(y_true[idx], score_a[idx])
        auc_b = roc_auc_score(y_true[idx], score_b[idx])
        deltas.append(auc_a - auc_b)
    if not deltas:
        return np.nan, np.nan
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return lo, hi


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device, use_i2t=not args.no_i2t)

    df = load_paired_df(args.csv)
    test_df = rebuild_test_split(df)
    print(f"Test set: {len(test_df)} samples (rebuilt with random_state=42, "
          f"matches train.py split)")

    probs, labels = collect_probs(model, tok, args.image_dir, test_df, device)

    rows = []
    for j, label in enumerate(LABEL_COLS):
        y_true = labels[:, j]
        n_pos = int(y_true.sum())

        auc_real_real    = safe_auc(y_true, probs["real_real"][:, j])
        auc_blank_blank  = safe_auc(y_true, probs["blank_blank"][:, j])
        auc_realT_blankI = safe_auc(y_true, probs["realT_blankI"][:, j])
        auc_blankT_realI = safe_auc(y_true, probs["blankT_realI"][:, j])

        delta = auc_real_real - auc_blankT_realI if not np.isnan(auc_real_real) else np.nan

        if n_pos >= 2 and n_pos <= len(y_true) - 2:
            ci_real_real    = bootstrap_auc_ci(y_true, probs["real_real"][:, j], args.n_boot)
            ci_blankT_realI = bootstrap_auc_ci(y_true, probs["blankT_realI"][:, j], args.n_boot)
            ci_delta = bootstrap_delta_ci(
                y_true, probs["real_real"][:, j], probs["blankT_realI"][:, j], args.n_boot,
            )
        else:
            ci_real_real = ci_blankT_realI = ci_delta = (np.nan, np.nan)

        rows.append({
            "label": label,
            "n_pos": n_pos,
            "auc_real_real": auc_real_real,
            "auc_real_real_ci_lo": ci_real_real[0],
            "auc_real_real_ci_hi": ci_real_real[1],
            "auc_blank_blank": auc_blank_blank,
            "auc_realT_blankI": auc_realT_blankI,
            "auc_blankT_realI": auc_blankT_realI,
            "auc_blankT_realI_ci_lo": ci_blankT_realI[0],
            "auc_blankT_realI_ci_hi": ci_blankT_realI[1],
            "delta_real_minus_blankTrealI": delta,
            "delta_ci_lo": ci_delta[0],
            "delta_ci_hi": ci_delta[1],
        })

        print(f"{label:<20} n_pos={n_pos:<4} "
              f"AUC real={auc_real_real:.4f}  "
              f"AUC blankT+realI={auc_blankT_realI:.4f}  "
              f"AUC realT+blankI={auc_realT_blankI:.4f}  "
              f"delta={delta:.4f}")

    result_df = pd.DataFrame(rows)
    print(f"\nMean AUC (real_real)       : {result_df['auc_real_real'].mean():.4f}")
    print(f"Mean AUC (blankT+realI)    : {result_df['auc_blankT_realI'].mean():.4f}")
    print(f"Mean AUC (realT+blankI)    : {result_df['auc_realT_blankI'].mean():.4f}")
    print(f"Mean delta (real - blankT) : {result_df['delta_real_minus_blankTrealI'].mean():.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        result_df.to_csv(args.out, index=False)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--n_boot", type=int, default=1000,
                         help="Bootstrap resamples for CIs")
    parser.add_argument("--no_i2t", action="store_true",
                         help="Set this when --checkpoint was trained with "
                              "train.py --no_i2t (leakage-resistant variant)")
    parser.add_argument("--out", default=None,
                         help="Optional path to save per-label results CSV")
    args = parser.parse_args()
    main(args)
