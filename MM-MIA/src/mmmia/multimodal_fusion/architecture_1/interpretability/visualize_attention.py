"""visualize_attention.py — Interprétabilité du modèle MultimodalFusion.

Pour un couple (image, texte) du dataset, affiche :
  1. La radiographie originale
  2. La heatmap du pooling image appris (les patches qui pèsent le plus
     dans la représentation finale de l'image, utilisée par la tête de
     classification)
  3. La heatmap d'attention Texte[CLS] → Image (les régions de l'image
     que le résumé textuel regarde)
  4. Les mots du rapport les plus influents, pondérés par l'attention
     Image → Texte et par l'importance de pooling de chaque patch image

Seuls deux fichiers de checkpoint sont nécessaires : `tokenizer.json` et
`multimodal_fusion.pt` (ce dernier contient déjà les poids fine-tunés du
texte ET de l'image — `bert_pretrained.pt` n'est donc pas requis ici).

Usage:
    python visualize_attention.py \
        --checkpoint checkpoints/multimodal_fusion.pt \
        --tokenizer  ../text_classification/checkpoints/tokenizer.json \
        --csv        ../../../data/shared/dataset_labeled.csv \
        --image_dir  /content/Png \
        --index 0
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# ── Chemins vers les modules existants (même convention que train.py) ──────
ROOT = os.path.dirname(os.path.abspath(__file__))
TEXT_ROOT = os.path.join(ROOT, "..", "text_classification")
sys.path.insert(0, ROOT)
sys.path.insert(0, TEXT_ROOT)

from models import BERTForMLM                                    # noqa: E402
from model import MultimodalFusion                               # noqa: E402
from dataset import LABEL_COLS, load_paired_df                   # noqa: E402

from transformers import ViTModel                                # noqa: E402
from tokenizers import Tokenizer                                 # noqa: E402
from torchvision import transforms                                # noqa: E402


VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device,
                use_i2t: bool = True):
    """Reconstruit l'architecture puis charge les poids entraînés.

    use_i2t doit correspondre à la variante utilisée à l'entraînement
    (False pour un checkpoint produit avec `train.py --no_i2t`)."""
    tok = Tokenizer.from_file(tokenizer_path)
    bert = BERTForMLM(tok.get_vocab_size(), 256, 8, 6, 512)
    vit = ViTModel.from_pretrained("codewithdark/vit-chest-xray")

    model = MultimodalFusion(
        text_encoder=bert.encoder, vit=vit,
        n_labels=len(LABEL_COLS), d_model=512, n_heads=8, dropout=0.1,
        use_i2t=use_i2t,
    ).to(device)

    # Le checkpoint contient TOUS les poids (texte + image + fusion + tête) :
    # ceci écrase les poids HuggingFace/aléatoires chargés ci-dessus.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tok


@torch.no_grad()
def forward_with_attention(model: MultimodalFusion, input_ids, pixel_values):
    """Réplique MultimodalFusion.forward en conservant les poids d'attention."""
    text_tokens = model.text_encoder(input_ids)
    image_tokens = model.vit(pixel_values=pixel_values).last_hidden_state

    ca = model.cross_attn
    T = ca.text_proj(text_tokens)
    I = ca.image_proj(image_tokens)

    T_ca, t2i_w = ca.t2i(T, I, I)   # t2i_w : (B, L_text, N_image)
    I_ca, i2t_w = ca.i2t(I, T, T)   # i2t_w : (B, N_image, L_text)

    T = ca.norm_t1(T + T_ca)
    I = ca.norm_i1(I + I_ca)
    T = ca.norm_t2(T + ca.ffn_t(T))
    I = ca.norm_i2(I + ca.ffn_i(I))

    text_repr = T[:, 0]
    pool_w = F.softmax(model.image_pool_w(I), dim=1)        # (B, N_image, 1)
    image_repr = (pool_w * I).sum(dim=1)

    logits = model.head(torch.cat([text_repr, image_repr], dim=1))

    return {
        "logits": logits,
        "t2i_w": t2i_w,
        "i2t_w": i2t_w,
        "pool_w": pool_w.squeeze(-1),   # (B, N_image)
    }


def patch_grid_size(vit) -> int:
    return vit.config.image_size // vit.config.patch_size   # 224 / 16 = 14


def heatmap_overlay(ax, pil_image, weights_197, grid, title, cmap="jet"):
    """weights_197 : tenseur (197,) incluant le token CLS en position 0."""
    patches = weights_197[1:].reshape(grid, grid)
    patches = patches / patches.max().clamp(min=1e-8)
    heat = F.interpolate(
        patches.unsqueeze(0).unsqueeze(0),
        size=pil_image.size[::-1], mode="bilinear", align_corners=False,
    )[0, 0].cpu().numpy()

    ax.imshow(pil_image)
    ax.imshow(heat, cmap=cmap, alpha=0.45)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(args.checkpoint, args.tokenizer, device)
    grid = patch_grid_size(model.vit)

    # ── Échantillon ──────────────────────────────────────────────────────
    df = load_paired_df(args.csv)
    row = df.iloc[args.index]
    img_path = os.path.join(args.image_dir, row["image_id"])
    pil_img = Image.open(img_path).convert("RGB")
    text = row["findings"]
    gt = row[LABEL_COLS].values.astype(float)

    pixel_values = VAL_TF(pil_img).unsqueeze(0).to(device)
    enc = tok.encode(text)
    ids = torch.tensor(enc.ids, dtype=torch.long).unsqueeze(0).to(device)
    tokens = enc.tokens

    out = forward_with_attention(model, ids, pixel_values)
    probs = out["logits"].sigmoid()[0].cpu().numpy()

    # ── Importance des mots ──────────────────────────────────────────────
    # Pondère l'attention Image→Texte de chaque patch par son poids de
    # pooling : les mots regardés par les patches qui comptent vraiment.
    pool_w = out["pool_w"][0]              # (197,)
    i2t_w = out["i2t_w"][0]                # (197, L_text)
    word_importance = (pool_w.unsqueeze(1) * i2t_w).sum(dim=0).cpu().numpy()
    word_importance[0] = -np.inf           # exclut [CLS] du classement

    t2i_cls = out["t2i_w"][0, 0]           # (197,) texte[CLS] → image

    # ── Figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    axes[0, 0].imshow(pil_img)
    axes[0, 0].set_title("Radiographie originale", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    heatmap_overlay(axes[0, 1], pil_img, out["pool_w"][0], grid,
                     "Pooling image appris\n(ce que le modèle retient de l'image)")

    heatmap_overlay(axes[1, 0], pil_img, t2i_cls, grid,
                     "Attention Texte[CLS] → Image\n(régions liées au rapport)")

    order = np.argsort(word_importance)[::-1][:12]
    ax = axes[1, 1]
    ax.barh(range(len(order)), word_importance[order][::-1], color="#007A8A")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([tokens[i] for i in order][::-1], fontsize=12)
    ax.set_xlabel("Importance (pondérée par pooling image)")
    ax.set_title("Mots les plus influents du rapport", fontsize=13, fontweight="bold")

    top5 = np.argsort(probs)[::-1][:5]
    pred_txt = "  |  ".join(f"{LABEL_COLS[i]} ({probs[i]:.2f})" for i in top5)
    gt_labels = [LABEL_COLS[i] for i in range(len(LABEL_COLS)) if gt[i] == 1]
    fig.suptitle(
        f"Top-5 prédictions : {pred_txt}\n"
        f"Vérité terrain : {', '.join(gt_labels) or 'Normal'}",
        fontsize=12.5,
    )

    fig.tight_layout()
    out_path = args.out or os.path.join(ROOT, "outputs", f"attention_{args.index}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    print(f"Texte analysé : {text}")
    print(f"Sauvegardé    : {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Chemin vers multimodal_fusion.pt")
    parser.add_argument("--tokenizer", required=True,
                        help="Chemin vers tokenizer.json")
    parser.add_argument("--csv", required=True,
                        help="Chemin vers dataset_labeled.csv")
    parser.add_argument("--image_dir", required=True,
                        help="Répertoire contenant les fichiers PNG")
    parser.add_argument("--index", type=int, default=0,
                        help="Index de l'échantillon dans le dataset")
    parser.add_argument("--out", default=None,
                        help="Chemin de sortie du PNG (optionnel)")
    args = parser.parse_args()
    main(args)
