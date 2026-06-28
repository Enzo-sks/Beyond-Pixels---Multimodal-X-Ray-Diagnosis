"""query_collapse.py — Mesure le « collapse » des requêtes label-alignées.

Les 14 requêtes sont des paramètres INDÉPENDANTS, mais leurs cartes d'attention
se ressemblent (cf. attention_maps.py). Hypothèse : la self-attention sur le
texte partagé + la FFN homogénéisent les requêtes couche après couche, si bien
qu'au dernier bloc les vecteurs H_j sont quasi colinéaires → mêmes cartes.

Ce script teste l'hypothèse quantitativement : il capture les requêtes
- à l'état initial (paramètre `queries`, avant tout bloc),
- après chaque bloc Q-Former (hooks forward),
puis calcule la matrice de similarité cosinus des 14 requêtes, moyennée sur un
batch d'échantillons. Une similarité hors-diagonale proche de 1 = collapse.

Sorties :
  - tableau console : cosinus hors-diagonale moyen par bloc (init → dernier),
  - heatmap 14×14 de la similarité au dernier bloc (results/collapse_<mode>/),
  - courbe du cosinus moyen vs profondeur (init, bloc 0, …, bloc N-1).

Usage :
    python query_collapse.py --ckpt ../checkpoints/qformer_14_last.ckpt \
        --image_dir ~/data/Png --n_samples 32
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)                         # attention_maps (réutilisé)

from attention_maps import load_model, ARCH, MM_ROOT, DEFAULT_CSV   # noqa: E402,F401
from common.data import (FusionDataset, fusion_collate, load_paired_df,   # noqa: E402
                         resolve_label_cols, build_groups, grouped_train_val_test)
from common.transforms import VAL_TF                                 # noqa: E402
from model import build_cxr_tokenizer                                # noqa: E402


def cos_matrix(q):
    """q : (B, n, d) → matrice (n, n) de cosinus moyenne sur le batch."""
    qn = F.normalize(q, dim=-1)
    return torch.einsum("bid,bjd->bij", qn, qn).mean(0)             # (n, n)


def mean_offdiag(S):
    n = S.size(0)
    return ((S.sum() - S.diagonal().sum()) / (n * (n - 1))).item()


def effective_rank(q):
    """Participation ratio des valeurs singulières de q (B, n, d) ∈ [1, n],
    moyenné sur le batch. ≈1 = toutes les requêtes sur une direction (collapse) ;
    ≈n = n directions indépendantes (pleinement distinctes)."""
    s = torch.linalg.svdvals(q)                            # (B, min(n,d))
    s2 = s ** 2
    pr = s2.sum(-1) ** 2 / (s2 ** 2).sum(-1).clamp(min=1e-12)
    return pr.mean().item()


def decompose(q):
    """q_j = q̄ + r_j (composante partagée + résidu par-label). Retourne, moyenné
    sur le batch : fraction d'énergie partagée ‖q̄‖²/⟨‖q_j‖²⟩, cosinus hors-diag
    des RÉSIDUS (le vrai collapse d'identité), rang effectif des résidus."""
    qbar = q.mean(1, keepdim=True)                         # (B,1,d) composante commune
    r = q - qbar                                           # (B,n,d) résidus par-label
    e_q = (q ** 2).sum(-1).mean(1)                         # (B,) ⟨‖q_j‖²⟩
    e_shared = (qbar.squeeze(1) ** 2).sum(-1)             # (B,) ‖q̄‖²
    shared_frac = (e_shared / e_q.clamp(min=1e-8)).mean().item()
    return shared_frac, mean_offdiag(cos_matrix(r)), effective_rank(r)


def text_align(q, t_pooled):
    """cos(q̄, texte poolé) moyenné : la composante commune EST-elle le texte ?"""
    if q.size(0) != t_pooled.size(0):
        return float("nan")                               # init (B=1) : non comparable
    return F.cosine_similarity(q.mean(1), t_pooled, dim=-1).mean().item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, hp = load_model(args.ckpt, device)
    mode_tag = hp.get("text_feature_mode", "last")
    tok, _ = build_cxr_tokenizer(max_len=args.max_len)

    df = load_paired_df(args.csv)
    label_cols = resolve_label_cols(df, hp.get("n_labels", 14))

    # Échantillons : N premiers du TEST avec ≥1 label positif (split identique
    # à l'entraînement) — représentatif sans fuite.
    _, _, test_idx = grouped_train_val_test(build_groups(df), train=0.70, val=0.15)
    pos = [i for i in test_idx
           if df.iloc[i][label_cols].values.astype(float).sum() > 0]
    sample_idx = pos[:args.n_samples]
    print(f"Checkpoint : {args.ckpt}  (mode={mode_tag})")
    print(f"Batch      : {len(sample_idx)} échantillons (test, ≥1 positif)")

    ds = FusionDataset(df, label_cols, tok, args.image_dir, VAL_TF)
    ids, px, _ = fusion_collate([ds[i] for i in sample_idx])
    ids, px = ids.to(device), px.to(device)

    # Hooks : capturer la requête en sortie de chaque bloc (out[0] = q).
    captured = []
    handles = [blk.register_forward_hook(lambda m, i, o: captured.append(o[0].detach()))
               for blk in model.qformer.blocks]
    with torch.no_grad():
        model(ids, px, return_attn=True)
    for h in handles:
        h.remove()

    # État initial : le paramètre `queries` (1, n, d), commun à tous les échantillons.
    q_init = model.qformer.queries.detach()                        # (1, n, d)

    # Texte poolé (moyenne des tokens non-pad) pour tester l'alignement q̄↔texte.
    with torch.no_grad():
        tmask = (ids != model.pad_id).long()
        T = model.text_encoder(input_ids=ids, attention_mask=tmask).last_hidden_state
        w = tmask.unsqueeze(-1).float()
        t_pooled = (T * w).sum(1) / w.sum(1).clamp(min=1e-8)        # (B, d)

    all_q = [q_init] + captured
    names = ["init"] + [f"bloc {i}" for i in range(len(captured))]
    mats = [cos_matrix(q) for q in all_q]

    # Le cosinus BRUT confond « composante texte commune » (voulu) et « perte
    # d'identité par-label » (à éviter). On centre pour les séparer.
    print("\n--  diagnostic de collapse des 14 requêtes (moyenné sur le batch)  --")
    print("  cos brut : confond commun+résidu | cos résid : VRAI collapse d'identité")
    print("  rang eff ∈ [1,14] (→14 = distinctes) | %part. : énergie composante commune")
    print(f"  {'prof.':8s} {'cos brut':>9s} {'%part.':>7s} {'cos résid':>10s} "
          f"{'rang eff':>9s} {'cos(q̄,txt)':>11s}")
    for name, q in zip(names, all_q):
        raw = mean_offdiag(cos_matrix(q))
        sf, rc, er = decompose(q)
        ta = text_align(q, t_pooled)
        print(f"  {name:8s} {raw:+9.3f} {sf * 100:6.1f}% {rc:+10.3f} "
              f"{er:9.2f} {ta:+11.3f}")

    out_dir = args.out_dir or os.path.join(ROOT, "results", f"collapse_{mode_tag}")
    os.makedirs(out_dir, exist_ok=True)

    # (1) Heatmap de la similarité au dernier bloc.
    S_last = mats[-1].cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(S_last, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xticks(range(len(label_cols)))
    ax.set_yticks(range(len(label_cols)))
    ax.set_xticklabels(label_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(label_cols, fontsize=8)
    ax.set_title(f"Similarité cosinus des requêtes — {mode_tag}, {names[-1]}\n"
                 f"hors-diag. moyen = {mean_offdiag(mats[-1]):.3f}", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p1 = os.path.join(out_dir, f"cosine_matrix_{mode_tag}.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # (2) Courbe-clé : cos BRUT vs cos RÉSIDUEL (+ rang effectif) par profondeur.
    raw_curve = [mean_offdiag(S) for S in mats]
    decs = [decompose(q) for q in all_q]
    res_curve = [d[1] for d in decs]
    er_curve = [d[2] for d in decs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(range(len(names)), raw_curve, "o-", color="#bbbbbb", linewidth=2,
            label="cos BRUT (confond commun+résidu)")
    ax.plot(range(len(names)), res_curve, "o-", color="#C0392B", linewidth=2.5,
            label="cos RÉSIDUEL (vrai collapse d'identité)")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("cosinus hors-diagonale moyen"); ax.set_ylim(-0.1, 1.05)
    ax.grid(alpha=0.3); ax.legend(loc="center left", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(range(len(names)), er_curve, "s--", color="#2471A3", linewidth=1.8,
             label="rang effectif (→14 = distinctes)")
    ax2.set_ylabel("rang effectif des requêtes"); ax2.set_ylim(0, len(label_cols) + 0.5)
    ax2.legend(loc="center right", fontsize=9)
    ax.set_title(f"Collapse des requêtes par profondeur — {mode_tag}")
    fig.tight_layout()
    p2 = os.path.join(out_dir, f"collapse_curve_{mode_tag}.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)

    print(f"\n=> {p1}\n=> {p2}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--out_dir", default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
