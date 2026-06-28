"""cggm_curves.py — Figure 3-panneaux à la CGGM (Fig. 3 du papier).

Trace, au fil des epochs : (a) accuracy unimodale, (b) magnitude de gradient
(norme du gradient de la matrice de poids de chaque tête), (c) direction du
gradient = cos(w_fusion, w_modalité). Les données viennent de l'historique
sauvegardé par train.py (`interpretability/results/cggm/<tag>.json`).

Usage :
    python cggm_curves.py results/cggm/<tag>.json [--out fig.png]
"""

import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cggm_history(records, out_path):
    """records : liste de dicts (un par epoch) issue de history['cggm']."""
    if not records:
        raise ValueError("historique CGGM vide")
    ep = [r["epoch"] for r in records]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(ep, [r["acc_txt"] for r in records], "-o", label="text")
    ax[0].plot(ep, [r["acc_img"] for r in records], "-s", label="image")
    ax[0].set(title="(a) unimodal accuracy", xlabel="Epoch", ylabel="acc")
    ax[0].legend()

    ax[1].plot(ep, [r["gnorm_txt"] for r in records], "-o", label="text")
    ax[1].plot(ep, [r["gnorm_img"] for r in records], "-s", label="image")
    ax[1].plot(ep, [r["gnorm_fus"] for r in records], "-^", label="fusion")
    ax[1].set(title="(b) gradient magnitude", xlabel="Epoch", ylabel=r"$\|\nabla W\|$")
    ax[1].legend()

    ax[2].plot(ep, [r["cos_txt"] for r in records], "-o", label="text")
    ax[2].plot(ep, [r["cos_img"] for r in records], "-s", label="image")
    ax[2].set(title=r"(c) gradient direction  $\cos(w_F, w_i)$",
              xlabel="Epoch", ylabel="cosine")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("history_json", help="JSON produit par train.py (--cggm)")
    ap.add_argument("--out", default=None, help="PNG de sortie (défaut : même nom)")
    a = ap.parse_args()
    with open(a.history_json) as f:
        data = json.load(f)
    records = data["cggm"] if isinstance(data, dict) else data
    out = a.out or a.history_json.replace(".json", ".png")
    plot_cggm_history(records, out)
    print(f"=> {out}")
