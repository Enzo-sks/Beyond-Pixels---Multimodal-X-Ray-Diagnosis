"""
generate_result_figures.py — Publication-quality result figures for Beyond Pixels
Produces 3 PNG files in outputs/figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NAVY  = "#0D1B3E"
TEAL  = "#007A8A"
GREEN = "#1A7A4A"
GOLD  = "#D4A017"
RED   = "#C0392B"
GRAY  = "#6B7280"
WHITE = "#FFFFFF"
LIGHT = "#F4F7FA"
ACCENT = "#00B4CC"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": WHITE,
    "figure.facecolor": WHITE,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

fig1, ax = plt.subplots(figsize=(12, 6))

models = [
    "TF-IDF +\nLogistic Reg.",
    "Custom BERT\n(MLM pretrain)",
    "CXR-BERT\n(Microsoft)",
    "DenseNet-121",
    "ViT\nchest-xray",
    "Multimodal\nFusion ★",
]
aucs   = [0.910, 0.940, 0.970, 0.780, 0.780, 0.9886]
modalities = ["Text", "Text", "Text", "Image", "Image", "Text + Image"]

colors = {
    "Text":         TEAL,
    "Image":        "#1A5276",
    "Text + Image": GREEN,
}
bar_colors = [colors[m] for m in modalities]

x = np.arange(len(models))
bars = ax.bar(x, aucs, color=bar_colors, width=0.55,
              edgecolor=WHITE, linewidth=1.2, zorder=3)

# Value labels
for bar, auc, mod in zip(bars, aucs, modalities):
    fw = "bold" if mod == "Text + Image" else "normal"
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.004,
            f"{auc:.3f}",
            ha="center", va="bottom", fontsize=13,
            fontweight=fw, color=NAVY)

# Highlight multimodal bar
bars[-1].set_edgecolor(GREEN)
bars[-1].set_linewidth(3)

# Baseline reference line
ax.axhline(0.780, color="#1A5276", linewidth=1.5,
           linestyle=":", alpha=0.6)
ax.text(5.45, 0.782, "Image baseline", fontsize=10,
        color="#1A5276", va="bottom", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12.5)
ax.set_ylabel("Mean AUC-ROC (21 labels)", fontsize=13, color=NAVY)
ax.set_ylim(0.70, 1.02)
ax.set_title("Model Comparison — Mean AUC on IU X-Ray Test Set",
             fontsize=16, fontweight="bold", color=NAVY, pad=12)

legend_handles = [
    mpatches.Patch(facecolor=TEAL,    label="Text only"),
    mpatches.Patch(facecolor="#1A5276", label="Image only"),
    mpatches.Patch(facecolor=GREEN,   label="Text + Image (ours)"),
]
ax.legend(handles=legend_handles, fontsize=12, loc="lower right",
          framealpha=0.9)

ax.yaxis.set_tick_params(labelsize=12)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, "fig1_model_comparison.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig1_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-class AUC (sorted)
# ═══════════════════════════════════════════════════════════════════════════

labels_raw = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Infiltration", "Mass", "Nodule",
    "Hernia", "Fracture", "Pleural_Thickening", "Opacity", "Consolidation",
    "Granuloma", "Calcinosis", "Scoliosis", "Atherosclerosis", "Normal",
]
aucs_raw = [
    0.9875, 0.9887, 0.9731, 0.9670, 0.9969, 0.9922, 0.9604, 0.9550,
    0.9839, 0.9946, 0.9918, 0.9997, 0.9922, 1.0000, 0.9999, 0.9936,
    0.9999, 0.9974, 0.9950, 1.0000, 0.9920,
]

# Sort ascending
pairs = sorted(zip(aucs_raw, labels_raw))
aucs_sorted  = [p[0] for p in pairs]
labels_sorted = [p[1] for p in pairs]

bar_cols = [GREEN  if a >= 0.99 else
            TEAL   if a >= 0.97 else
            GOLD   if a >= 0.95 else
            RED
            for a in aucs_sorted]

fig2, ax2 = plt.subplots(figsize=(10, 9))

y_pos = np.arange(len(labels_sorted))
bars2 = ax2.barh(y_pos, aucs_sorted, color=bar_cols,
                 edgecolor=WHITE, linewidth=0.8, height=0.72, zorder=3)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels_sorted, fontsize=12.5)
ax2.set_xlim(0.935, 1.010)
ax2.set_xlabel("AUC-ROC", fontsize=13, color=NAVY)
ax2.set_title("Per-Pathology AUC — Multimodal Fusion (Test Set, n=968)",
              fontsize=15, fontweight="bold", color=NAVY, pad=12)

# Mean line
mean_auc = np.mean(aucs_sorted)
ax2.axvline(mean_auc, color=RED, linewidth=2, linestyle="--", alpha=0.85)
ax2.text(mean_auc + 0.001, -0.8,
         f"Mean = {mean_auc:.4f}",
         fontsize=11.5, color=RED, fontweight="bold", va="top")

# Value labels
for bar, auc in zip(bars2, aucs_sorted):
    ax2.text(auc + 0.0005,
             bar.get_y() + bar.get_height() / 2,
             f"{auc:.4f}",
             va="center", fontsize=10.5, color=NAVY, fontweight="bold")

legend_handles2 = [
    mpatches.Patch(facecolor=GREEN, label="AUC ≥ 0.99"),
    mpatches.Patch(facecolor=TEAL,  label="0.97 ≤ AUC < 0.99"),
    mpatches.Patch(facecolor=GOLD,  label="0.95 ≤ AUC < 0.97"),
    mpatches.Patch(facecolor=RED,   label="AUC < 0.95"),
]
ax2.legend(handles=legend_handles2, fontsize=11.5, loc="lower right",
           framealpha=0.9)

ax2.xaxis.set_tick_params(labelsize=12)
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig2_per_class_auc.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig2_per_class_auc.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Training Curve (dual axis)
# ═══════════════════════════════════════════════════════════════════════════

epochs = list(range(1, 23))
val_auc = [0.6247, 0.8218, 0.9206, 0.9596, 0.9741, 0.9790, 0.9829,
           0.9879, 0.9868, 0.9860, 0.9868, 0.9867, 0.9878, 0.9877,
           0.9881, 0.9858, 0.9860, 0.9872, 0.9865, 0.9870, 0.9855, 0.9855]
val_loss = [0.0307, 0.0264, 0.0201, 0.0133, 0.0107, 0.0097, 0.0097,
            0.0072, 0.0075, 0.0067, 0.0062, 0.0062, 0.0059, 0.0063,
            0.0061, 0.0059, 0.0058, 0.0060, 0.0059, 0.0061, 0.0060, 0.0060]
train_loss = [0.0346, 0.0287, 0.0225, 0.0150, 0.0109, 0.0080, 0.0058,
              0.0045, 0.0033, 0.0025, 0.0019, 0.0016, 0.0013, 0.0010,
              0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0004, 0.0003, 0.0003]

fig3, ax3 = plt.subplots(figsize=(12, 5.5))
ax3r = ax3.twinx()

l1, = ax3.plot(epochs, val_auc, color=GREEN, linewidth=2.5,
               marker="o", markersize=5, label="Val AUC", zorder=4)
l2, = ax3r.plot(epochs, train_loss, color=RED, linewidth=2,
                linestyle="--", marker="s", markersize=4,
                alpha=0.75, label="Train Loss", zorder=4)
l3, = ax3r.plot(epochs, val_loss, color=GOLD, linewidth=2,
                linestyle=":", marker="^", markersize=4,
                alpha=0.85, label="Val Loss", zorder=4)

# Best AUC marker
best_ep = val_auc.index(max(val_auc)) + 1
ax3.scatter([best_ep], [max(val_auc)], s=120, color=GREEN,
            zorder=5, edgecolor=WHITE, linewidth=2)
ax3.annotate(f"Best AUC = {max(val_auc):.4f}\n(epoch {best_ep})",
             xy=(best_ep, max(val_auc)),
             xytext=(best_ep + 1.5, max(val_auc) - 0.04),
             fontsize=11.5, color=GREEN, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

# Phase separator
ax3.axvspan(0.5, 3.5, alpha=0.06, color=TEAL, zorder=1)
ax3.axvspan(3.5, 22.5, alpha=0.04, color=GREEN, zorder=1)
ax3.axvline(3.5, color=TEAL, linewidth=1.8, linestyle=":", alpha=0.7)
ax3.text(1.9, 0.635, "Phase 1\n(frozen)", ha="center",
         fontsize=11, color=TEAL, style="italic")
ax3.text(6.5, 0.635, "Phase 2 — backbones unfrozen", ha="left",
         fontsize=11, color=GREEN, style="italic")

ax3.set_xlabel("Epoch", fontsize=13, color=NAVY)
ax3.set_ylabel("Val AUC", fontsize=13, color=GREEN)
ax3r.set_ylabel("Loss (ASL)", fontsize=13, color=RED)
ax3.set_xlim(0.5, 22.5)
ax3.set_ylim(0.60, 1.01)
ax3r.set_ylim(0, 0.038)
ax3.set_title("Multimodal Fusion — Training Dynamics (22 epochs, early stopping)",
              fontsize=15, fontweight="bold", color=NAVY, pad=12)
ax3.tick_params(labelsize=12)
ax3r.tick_params(labelsize=12)
ax3.yaxis.label.set_color(GREEN)
ax3r.yaxis.label.set_color(RED)
ax3.spines["left"].set_color(GREEN)
ax3r.spines["right"].set_color(RED)

lines = [l1, l2, l3]
ax3.legend(lines, [l.get_label() for l in lines],
           fontsize=12, loc="center right", framealpha=0.9)

fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, "fig3_training_curve.png"),
             dpi=200, bbox_inches="tight")
print("Saved fig3_training_curve.png")

print(f"\nAll figures saved to: {OUT_DIR}")
