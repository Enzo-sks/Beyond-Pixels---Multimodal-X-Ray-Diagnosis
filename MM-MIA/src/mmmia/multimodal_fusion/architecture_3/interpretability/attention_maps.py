"""attention_maps.py — Interprétabilité du Q-Former label-aligné (architecture_3).

Chaque pathologie a SA propre requête. La cross-attention requête_j → patches
image montre *où* le modèle regarde sur la radiographie pour décider du label j.
C'est l'intérêt principal du design label-aligné : une carte d'attention par
pathologie, directement interprétable (pas de pooling qui mélangerait tout).

Pour un échantillon (image + rapport), produit une figure :
  - la radiographie originale,
  - puis une heatmap de cross-attention par label affiché (top-k prédits),
    titrée « label (proba) », superposée sur l'image.

Le checkpoint Lightning (`qformer_*.ckpt`) contient les hyperparamètres
(`save_hyperparameters`), donc on reconstruit le modèle exactement — `last`/`deep`,
`n_layers`, `pad_id` sont relus du checkpoint, aucun risque de mismatch.

Usage (sur le cluster, l'inférence est légère — CPU suffit) :
    python attention_maps.py \
        --ckpt ../checkpoints/qformer_14_last.ckpt \
        --image_dir ~/data/Png \
        --index 0 --topk 4

    # plusieurs échantillons d'un coup (premiers du test ayant ≥1 label positif) :
    python attention_maps.py --ckpt ../checkpoints/qformer_14_last.ckpt \
        --image_dir ~/data/Png --n_samples 8
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")                    # batch SLURM : pas d'affichage interactif
import matplotlib.pyplot as plt

# ── Chemins (même convention sys.path que train.py) ─────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.dirname(ROOT)                              # architecture_3/
MM_ROOT = os.path.dirname(ARCH)                           # multimodal_fusion/
REPO_ROOT = os.path.abspath(os.path.join(ARCH, "..", "..", "..", ".."))
sys.path.insert(0, ARCH)        # model.py
sys.path.insert(0, MM_ROOT)     # common/

from model import FusionQFormer, build_cxr_tokenizer                  # noqa: E402
from common.data import (load_paired_df, resolve_label_cols, build_texts,  # noqa: E402
                         build_groups, grouped_train_val_test)
from common.transforms import VAL_TF                                  # noqa: E402

DEFAULT_CSV = os.path.join(REPO_ROOT, "data", "shared", "dataset_labeled_major.csv")


# ======================================================================
#  Chargement du modèle depuis un checkpoint Lightning
# ======================================================================

def load_model(ckpt_path, device):
    """Reconstruit FusionQFormer d'après les hyperparamètres du checkpoint,
    puis charge les poids entraînés (préfixe Lightning `model.` retiré)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    n_labels = hp.get("n_labels", 14)
    model = FusionQFormer(
        n_labels=n_labels,
        n_layers=hp.get("n_layers", 3),
        pad_id=hp.get("pad_id", 0),
        text_feature_mode=hp.get("text_feature_mode", "last"),
        norm=hp.get("norm", "post"),
        radio_only=hp.get("radio_only", False),
        attn_pooled_head=hp.get("attn_pooled_head", False),
        freeze_text=True, freeze_image=True,
    ).to(device)

    # state_dict Lightning : clés préfixées `model.` (LitFusionQFormer.model).
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("model.")}
    # Les checkpoints entraînés avant add_pooling_layer=False portent un pooler
    # ViT (image_encoder.pooler.*) jamais utilisé (on ne lit que last_hidden_state).
    # On le retire pour matcher le modèle actuel sans relâcher strict=True.
    sd = {k: v for k, v in sd.items() if not k.startswith("image_encoder.pooler.")}
    model.load_state_dict(sd)
    model.eval()
    return model, hp


def patch_grid_size(vit) -> int:
    return vit.config.image_size // vit.config.patch_size        # 224 / 16 = 14


def heatmap_overlay(ax, pil_image, weights_197, grid, title, cmap="jet"):
    """weights_197 : tenseur (197,) avec le token [CLS] en position 0 (ignoré)."""
    patches = weights_197[1:].reshape(grid, grid)
    patches = patches / patches.max().clamp(min=1e-8)
    heat = F.interpolate(
        patches.unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1], mode="bilinear", align_corners=False,
    )[0, 0].cpu().numpy()
    ax.imshow(pil_image)
    ax.imshow(heat, cmap=cmap, alpha=0.45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


# ======================================================================
#  Une figure pour un échantillon
# ======================================================================

@torch.no_grad()
def make_figure(model, tok, label_cols, grid, row, image_dir, block, topk, device):
    pil_img = Image.open(os.path.join(image_dir, row["image_id"])).convert("RGB")
    text = build_texts(row.to_frame().T)[0]                 # indication + findings
    gt = row[label_cols].values.astype(float)

    pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
    ids = torch.tensor(tok.encode(text).ids, dtype=torch.long).unsqueeze(0).to(device)

    logits, aux = model(ids, pixel_values, return_attn=True)
    probs = logits.sigmoid()[0].cpu().numpy()
    ca = aux["ca"][block][0]               # (14, 197) : attention requête_j → image

    # Labels à afficher : top-k par proba prédite.
    show = np.argsort(probs)[::-1][:topk]

    n = 1 + len(show)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(pil_img)
    axes[0].set_title("Radiographie originale", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for k, j in enumerate(show, start=1):
        mark = "✓" if gt[j] == 1 else "·"      # ✓ = label positif dans la vérité terrain
        heatmap_overlay(axes[k], pil_img, ca[j], grid,
                        f"{label_cols[j]} {mark}\np={probs[j]:.2f}")
    for k in range(n, len(axes)):
        axes[k].axis("off")

    gt_labels = [label_cols[i] for i in range(len(label_cols)) if gt[i] == 1]
    fig.suptitle(f"Q-Former label-aligné — cross-attention requête→image  "
                 f"(bloc {block})\nVérité terrain : {', '.join(gt_labels) or 'Normal'}",
                 fontsize=13)
    fig.tight_layout()
    return fig


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hp = load_model(args.ckpt, device)
    grid = patch_grid_size(model.image_encoder)
    tok, _ = build_cxr_tokenizer(max_len=args.max_len)

    df = load_paired_df(args.csv)
    label_cols = resolve_label_cols(df, hp.get("n_labels", 14))
    mode_tag = hp.get("text_feature_mode", "last")
    if hp.get("radio_only"):
        mode_tag += "_radio"
    print(f"Checkpoint : {args.ckpt}  (mode={mode_tag}, n_layers={hp.get('n_layers')})")
    print(f"Device     : {device} | grille patches {grid}x{grid}")

    out_dir = args.out_dir or os.path.join(ROOT, "results", f"attention_{mode_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Choix des échantillons : --index unique, ou --n_samples premiers du TEST
    # ayant au moins un label positif (split groupé identique à l'entraînement).
    if args.n_samples:
        _, _, test_idx = grouped_train_val_test(build_groups(df), train=0.70, val=0.15)
        pos = [i for i in test_idx if df.iloc[i][label_cols].values.astype(float).sum() > 0]
        indices = pos[:args.n_samples]
    else:
        indices = [args.index]

    for idx in indices:
        row = df.iloc[idx]
        fig = make_figure(model, tok, label_cols, grid, row,
                          args.image_dir, args.block, args.topk, device)
        out_path = os.path.join(out_dir, f"attention_{mode_tag}_idx{idx}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx}] {row['image_id']} → {out_path}")

    print(f"=> {len(indices)} figure(s) dans {out_dir}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="checkpoints/qformer_14_{last,deep}.ckpt")
    p.add_argument("--image_dir", required=True, help="Répertoire des PNG (ex. ~/data/Png)")
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--index", type=int, default=0, help="Index de l'échantillon (df)")
    p.add_argument("--n_samples", type=int, default=0,
                   help="Si >0 : N premiers échantillons du TEST avec ≥1 label positif")
    p.add_argument("--topk", type=int, default=4, help="Nombre de labels affichés (top proba)")
    p.add_argument("--block", type=int, default=-1, help="Bloc Q-Former dont on lit la CA (-1=dernier)")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--out_dir", default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
