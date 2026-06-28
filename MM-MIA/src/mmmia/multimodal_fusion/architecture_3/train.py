"""train.py — Entraînement PyTorch Lightning du Q-Former label-aligné.

Stack : CXR-BERT (texte, gelé) + ViT (image, gelé) + Q-Former 14 requêtes
label-alignées + Asymmetric Loss. Encodeurs gelés par défaut ; `--unfreeze_at N`
les dégèle à partir de l'epoch N.

Usage :
    IMAGE_DIR=~/data/Png python train.py --image_dir $IMAGE_DIR
    python train.py --image_dir /content/Png --mode 14 --text_feature_mode deep
"""

import os
import sys
import argparse

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss

ROOT = os.path.dirname(os.path.abspath(__file__))
MM_ROOT = os.path.dirname(ROOT)                # src/mmmia/multimodal_fusion
REPO_ROOT = os.path.abspath(os.path.join(ROOT, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)      # model.py (local)
sys.path.insert(0, MM_ROOT)   # common/ (partagé)

from model import FusionQFormer, build_cxr_tokenizer            # noqa: E402
from common.data import (FusionDataset, fusion_collate, load_paired_df,  # noqa: E402
                         resolve_label_cols, build_groups, grouped_train_val_test)
from common.transforms import TRAIN_TF, VAL_TF                   # noqa: E402
from common.losses import AsymmetricLoss                        # noqa: E402

DEFAULT_CSV = os.path.join(REPO_ROOT, "data", "shared", "dataset_labeled_major.csv")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# ======================================================================
#  LIGHTNING MODULE
# ======================================================================

class LitFusionQFormer(pl.LightningModule):
    def __init__(self, n_labels, pad_id, n_layers=3, text_feature_mode="last",
                 norm="post", radio_only=False, attn_pooled_head=False,
                 text_dropout=0.0, lr=2e-4, epochs=30, unfreeze_at=None,
                 cggm=False, cggm_lambda=0.1, cggm_detach=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = FusionQFormer(
            n_labels=n_labels, n_layers=n_layers, pad_id=pad_id,
            text_feature_mode=text_feature_mode,
            norm=norm, radio_only=radio_only, attn_pooled_head=attn_pooled_head,
            text_dropout=text_dropout, cggm=cggm,
            freeze_text=True, freeze_image=True,
        )
        self.loss_fn = AsymmetricLoss()
        self.val_probs, self.val_labels = [], []
        self.history = {"train": [], "val": [], "cggm": []}
        self._train_losses, self._val_losses = [], []

        # ── État CGGM (modulation de direction ; encodeurs gelés → pas de
        #    modulation de magnitude). coeff_i = (Σ Δε − Δε_i)/Σ Δε, mis à jour
        #    par epoch ; warm-up tant qu'on n'a pas deux epochs pour un diff.
        self.cggm = cggm
        self.cggm_lambda = cggm_lambda
        self.cggm_detach = cggm_detach
        self._coeff_txt, self._coeff_img = 0.0, 0.0
        self._cggm_ready = False
        self._prev_acc = None
        self._uni_acc_txt, self._uni_acc_img, self._uni_n = 0.0, 0.0, 0
        self._head_gn = {"txt": [], "img": [], "fus": []}

    def forward(self, ids, pixel_values):
        return self.model(ids, pixel_values)

    def on_train_epoch_start(self):
        # current_epoch est 0-indexé. Dégel optionnel des encodeurs.
        if self.hparams.unfreeze_at is not None \
                and self.current_epoch + 1 == self.hparams.unfreeze_at:
            self.model.unfreeze_encoders()
            print(f"  → Epoch {self.current_epoch + 1} : encodeurs dégelés")
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Epoch {self.current_epoch + 1}: {n:,} trainable params")

    def training_step(self, batch, _):
        ids, px, lab = batch
        if self.cggm:
            logits, logit_txt, logit_img = self.model.forward_cggm(
                ids, px, detach=self.cggm_detach)
            # L_task (fusion) + supervision unimodale (entraîne les têtes ; le
            # backbone aussi si --no-cggm_detach) + terme de direction l_gm.
            loss = (self.loss_fn(logits, lab)
                    + self.loss_fn(logit_txt, lab)
                    + self.loss_fn(logit_img, lab))
            if self._cggm_ready:
                loss = loss + self.cggm_lambda * self._l_gm()
            with torch.no_grad():
                self._accum_uni_acc(logit_txt, logit_img, lab)
        else:
            loss = self.loss_fn(self(ids, px), lab)
        self._train_losses.append(loss.detach())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ── CGGM : direction (l_gm) + accuracy unimodale → coeff ────────────────
    def _l_gm(self):
        """l_gm = (Σ|c_i| − Σ c_i·cos(w_F, w_i)) / M  (M=2). Minimiser tire la
        direction de la tête de fusion vers les têtes unimodales, pondéré par les
        coeff (modalité qui stagne → c_i grand → forte traction)."""
        wF = self.model.qformer.W.flatten()
        wT = self.model.head_txt.W.flatten()
        wI = self.model.head_img.W.flatten()
        cos = torch.nn.functional.cosine_similarity
        cT, cI = self._coeff_txt, self._coeff_img
        sim = cT * cos(wF, wT, dim=0) + cI * cos(wF, wI, dim=0)
        return (abs(cT) + abs(cI) - sim) / 2.0

    @torch.no_grad()
    def _accum_uni_acc(self, logit_txt, logit_img, lab):
        pt = (logit_txt.sigmoid() > 0.5).float()
        pi = (logit_img.sigmoid() > 0.5).float()
        n = lab.size(0)
        self._uni_acc_txt += (pt == lab).float().mean().item() * n
        self._uni_acc_img += (pi == lab).float().mean().item() * n
        self._uni_n += n

    @torch.no_grad()
    def _cos_now(self, which):
        wF = self.model.qformer.W.flatten()
        head = self.model.head_txt if which == "txt" else self.model.head_img
        return torch.nn.functional.cosine_similarity(wF, head.W.flatten(), dim=0).item()

    def on_after_backward(self):
        if not self.cggm:
            return
        gn = lambda p: p.grad.norm().item() if p.grad is not None else 0.0
        self._head_gn["fus"].append(gn(self.model.qformer.W))
        self._head_gn["txt"].append(gn(self.model.head_txt.W))
        self._head_gn["img"].append(gn(self.model.head_img.W))

    def on_train_epoch_end(self):
        self.history["train"].append(torch.stack(self._train_losses).mean().item())
        self._train_losses.clear()
        if self.cggm and self._uni_n > 0:
            self._update_cggm_coeff()

    def _update_cggm_coeff(self):
        """Fin d'epoch : Δaccuracy unimodale → nouveaux coeff (pour l'epoch
        suivante) + enregistrement des stats pour la figure 3-panneaux."""
        import numpy as _np
        acc_t = self._uni_acc_txt / self._uni_n
        acc_i = self._uni_acc_img / self._uni_n
        if self._prev_acc is not None:
            dT = acc_t - self._prev_acc[0]
            dI = acc_i - self._prev_acc[1]
            s = dT + dI + 1e-8                       # ← l'unique epsilon (anti /0)
            clamp = lambda x: float(min(max(x, 0.0), 2.0))   # anti sign-flip
            self._coeff_txt = clamp((s - dT) / s)
            self._coeff_img = clamp((s - dI) / s)
            self._cggm_ready = True
        self._prev_acc = (acc_t, acc_i)
        mean = lambda xs: float(_np.mean(xs)) if xs else 0.0
        self.history["cggm"].append({
            "epoch": self.current_epoch + 1,
            "acc_txt": acc_t, "acc_img": acc_i,
            "coeff_txt": self._coeff_txt, "coeff_img": self._coeff_img,
            "gnorm_txt": mean(self._head_gn["txt"]),
            "gnorm_img": mean(self._head_gn["img"]),
            "gnorm_fus": mean(self._head_gn["fus"]),
            "cos_txt": self._cos_now("txt"), "cos_img": self._cos_now("img"),
        })
        self._uni_acc_txt = self._uni_acc_img = 0.0
        self._uni_n = 0
        self._head_gn = {"txt": [], "img": [], "fus": []}

    def validation_step(self, batch, _):
        ids, px, lab = batch
        logits = self(ids, px)
        loss = self.loss_fn(logits, lab)
        self._val_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=True)
        self.val_probs.append(logits.sigmoid().cpu())
        self.val_labels.append(lab.cpu())

    def on_validation_epoch_end(self):
        vl = float("nan")
        if self._val_losses:
            vl = torch.stack(self._val_losses).mean().item()
            self.history["val"].append(vl)
            self._val_losses.clear()
        auc = float("nan")
        if self.val_probs:
            probs = torch.cat(self.val_probs).numpy()
            labs = torch.cat(self.val_labels).numpy()
            valid = [i for i in range(labs.shape[1]) if len(np.unique(labs[:, i])) > 1]
            if valid:
                auc = roc_auc_score(labs[:, valid], probs[:, valid], average="macro")
                self.log("val_auc", auc, prog_bar=True)
        self.val_probs.clear()
        self.val_labels.clear()
        # Ligne lisible dans le .out batch (la barre de progression Lightning,
        # écrite en \r sur TTY, ne s'affiche pas dans un fichier redirigé).
        tl = self.history["train"][-1] if self.history["train"] else float("nan")
        print(f"  Epoch {self.current_epoch + 1:2d} | train_loss={tl:.4f} | "
              f"val_loss={vl:.4f} | val_auc={auc:.4f}", flush=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.hparams.epochs)
        return [opt], [sched]


# ======================================================================
#  MAIN
# ======================================================================

def main(args):
    pl.seed_everything(42, workers=True)

    # ── Données ───────────────────────────────────────────────────────
    df = load_paired_df(args.csv)
    label_cols = resolve_label_cols(df, args.mode)
    tok, pad_id = build_cxr_tokenizer(max_len=args.max_len)

    print(f"CSV     : {args.csv}")
    print(f"Mode    : {args.mode} | {len(df)} images | {len(label_cols)} labels")
    print(f"Labels  : {label_cols}")

    train_idx, val_idx, test_idx = grouped_train_val_test(
        build_groups(df), train=0.70, val=0.15)

    # --no_findings : l'ENTRAÎNEMENT reste multimodal (indication+findings) ;
    # val/test (donc sélection de checkpoint ET inférence) n'ont que l'indication.
    eval_text_cols = ["indication"] if args.no_findings else None
    if args.no_findings:
        print("--no_findings : train=indication+findings | val/test=indication seul")
    train_ds = FusionDataset(df, label_cols, tok, args.image_dir, TRAIN_TF)
    eval_full = FusionDataset(df, label_cols, tok, args.image_dir, VAL_TF,
                              text_cols=eval_text_cols)
    train_set = Subset(train_ds, train_idx)
    val_set = Subset(eval_full, val_idx)
    test_set = Subset(eval_full, test_idx)
    print(f"Train   : {len(train_set)} | Val : {len(val_set)} | Test : {len(test_set)} "
          "(split groupé sans fuite)")

    dl = lambda ds, sh: DataLoader(ds, args.batch_size, shuffle=sh,
                                   collate_fn=fusion_collate, num_workers=args.num_workers)
    train_loader, val_loader, test_loader = dl(train_set, True), dl(val_set, False), dl(test_set, False)

    # ── Modèle : entraînement, OU rechargement pour éval-seule ────────
    if args.eval_ckpt:
        # Recharge un checkpoint (hparams restaurés) et saute l'entraînement →
        # AUC test sur le meilleur modèle déjà sauvegardé (ex. après un scancel).
        lit = LitFusionQFormer.load_from_checkpoint(args.eval_ckpt)
        print(f"Éval seule : {args.eval_ckpt} (pas d'entraînement)")
    else:
        lit = LitFusionQFormer(
            n_labels=len(label_cols), pad_id=pad_id, n_layers=args.n_layers,
            text_feature_mode=args.text_feature_mode,
            norm=args.norm, radio_only=args.radio_only,
            attn_pooled_head=args.attn_pooled_head,
            text_dropout=args.text_dropout, lr=args.lr,
            epochs=args.epochs, unfreeze_at=args.unfreeze_at,
            cggm=args.cggm, cggm_lambda=args.cggm_lambda,
            cggm_detach=args.cggm_detach,
        )

        # Tag de variante → checkpoints distincts (n'écrase pas les runs post-norm).
        tag = f"qformer_{args.mode}_{args.text_feature_mode}_{args.norm}"
        if args.radio_only:
            tag += "_radio"
        if args.attn_pooled_head:
            tag += "_apool"
        if args.text_dropout:
            tag += f"_td{args.text_dropout:g}"
        if args.cggm:
            tag += "_cggm"
        if args.no_findings:
            tag += "_nofind"
        print(f"Variante : norm={args.norm} | radio_only={args.radio_only} | "
              f"attn_pooled_head={args.attn_pooled_head} | "
              f"text_dropout={args.text_dropout} → ckpt '{tag}'")

        ckpt_cb = pl.callbacks.ModelCheckpoint(
            dirpath=CKPT_DIR, filename=tag,
            monitor="val_auc", mode="max", save_top_k=1)
        early_cb = pl.callbacks.EarlyStopping("val_auc", patience=args.patience, mode="max")

        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            callbacks=[ckpt_cb, early_cb],
            num_sanity_val_steps=0,
            logger=False,
            enable_progress_bar=False,   # batch SLURM : la barre \r pollue le .out
            overfit_batches=args.overfit_batches,
        )
        trainer.fit(lit, train_loader, val_loader)

        if args.overfit_batches:   # sanity check : pas d'évaluation test
            print("=> Sanity overfit terminé.")
            return

        # Recharge le meilleur checkpoint (val_auc) avant l'éval test.
        if ckpt_cb.best_model_path:
            best = torch.load(ckpt_cb.best_model_path, weights_only=False)
            lit.load_state_dict(best["state_dict"])

    # ── Évaluation sur le TEST tenu à l'écart ─────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit.to(device).eval()

    @torch.no_grad()
    def collect(loader):
        ps, ls = [], []
        for ids, px, lab in loader:
            ps.append(lit(ids.to(device), px.to(device)).sigmoid().cpu().numpy())
            ls.append(lab.numpy())
        return np.vstack(ps), np.vstack(ls)

    val_probs, val_labels = collect(val_loader)
    best_t = max(np.arange(0.1, 0.6, 0.05),
                 key=lambda t: f1_score(val_labels, (val_probs > t).astype(int),
                                        average="macro", zero_division=0))

    probs, labels_arr = collect(test_loader)
    preds = (probs > best_t).astype(int)
    valid = [i for i in range(labels_arr.shape[1]) if len(np.unique(labels_arr[:, i])) > 1]

    print(f"\n{'='*52}")
    print(f"  Q-Former [{args.mode}|{args.text_feature_mode}] — TEST — seuil={best_t:.2f}")
    print(f"{'='*52}")
    print(f"  F1 macro  : {f1_score(labels_arr, preds, average='macro', zero_division=0):.4f}")
    print(f"  F1 micro  : {f1_score(labels_arr, preds, average='micro', zero_division=0):.4f}")
    print(f"  AUC macro : {roc_auc_score(labels_arr[:, valid], probs[:, valid], average='macro'):.4f}")
    print(f"  AUC micro : {roc_auc_score(labels_arr[:, valid], probs[:, valid], average='micro'):.4f}")
    print(f"  Hamming   : {hamming_loss(labels_arr, preds):.4f}")

    auc_per = roc_auc_score(labels_arr[:, valid], probs[:, valid], average=None)
    names = [label_cols[i] for i in valid]
    print("\n-- AUC par label --")
    for name, auc in sorted(zip(names, auc_per), key=lambda x: x[1], reverse=True):
        print(f"  {name:22s} {auc:.4f}  {'#' * int(auc * 20)}")
    print(f"\n=> Checkpoint : {ckpt_cb.best_model_path}")

    # ── CGGM : historique JSON + figure 3-panneaux (acc / magnitude / direction)
    if not args.eval_ckpt and args.cggm and lit.history["cggm"]:
        import json
        res_dir = os.path.join(ROOT, "interpretability", "results", "cggm")
        os.makedirs(res_dir, exist_ok=True)
        hist_path = os.path.join(res_dir, f"{tag}.json")
        with open(hist_path, "w") as f:
            json.dump({k: lit.history[k] for k in ("cggm", "train", "val")}, f, indent=2)
        print(f"=> Historique CGGM : {hist_path}")
        try:
            from interpretability.cggm_curves import plot_cggm_history
            png = os.path.join(res_dir, f"{tag}.png")
            plot_cggm_history(lit.history["cggm"], png)
            print(f"=> Figure CGGM    : {png}")
        except Exception as e:                       # matplotlib absent en batch, p.ex.
            print(f"  (figure CGGM non générée : {e})")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV,
                   help="CSV image-level pairé (défaut data/shared/dataset_labeled_major.csv)")
    p.add_argument("--image_dir", required=True, help="Répertoire des PNG (ex. /content/Png)")
    p.add_argument("--eval_ckpt", default=None,
                   help="Recharge ce checkpoint et calcule l'AUC test SANS entraîner")
    p.add_argument("--mode", type=int, default=14, choices=[5, 14, 21],
                   help="Jeu de labels (défaut 14 NIH)")
    p.add_argument("--n_layers", type=int, default=3, help="Nombre de blocs Q-Former (2-4)")
    p.add_argument("--text_feature_mode", default="last", choices=["last", "deep"],
                   help="'last' (défaut) ou 'deep' (branchement BERT couche↔bloc)")
    p.add_argument("--norm", default="post", choices=["post", "pre"],
                   help="'post' (défaut, historique) ou 'pre' (anti-collapse des requêtes)")
    p.add_argument("--radio_only", action="store_true",
                   help="Image-seul : aucune entrée texte → force la voie image à localiser")
    p.add_argument("--attn_pooled_head", action="store_true",
                   help="logit_j = w_j·Σα_j·V(z) : met la cross-attention dans la loss")
    p.add_argument("--text_dropout", type=float, default=0.0,
                   help="Modality dropout texte : proba de masquer tout le texte d'un échantillon")
    p.add_argument("--cggm", action="store_true",
                   help="CGGM : modulation de DIRECTION sur θ_F (encodeurs gelés → pas de magnitude)")
    p.add_argument("--cggm_lambda", type=float, default=0.1,
                   help="Poids du terme de direction l_gm (défaut 0.1)")
    p.add_argument("--cggm_detach", action=argparse.BooleanOptionalAction, default=True,
                   help="Détache les reps unimodales (têtes-seules, défaut). "
                        "--no-cggm_detach : supervision unimodale propagée au backbone")
    p.add_argument("--no_findings", action="store_true",
                   help="Train=indication+findings ; val/test/inférence=indication seul")
    p.add_argument("--unfreeze_at", type=int, default=None,
                   help="Epoch (1-indexée) où dégeler les encodeurs ; défaut : jamais")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--overfit_batches", type=float, default=0.0,
                   help="Sanity check : surapprend sur N batches (ex. 2) puis s'arrête")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
