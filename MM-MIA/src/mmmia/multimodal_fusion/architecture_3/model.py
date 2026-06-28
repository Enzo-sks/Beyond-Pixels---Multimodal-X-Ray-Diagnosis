"""model.py — Q-Former label-aligné (famille 3) : fusion image + texte.

Architecture (cf. docs/architectures_fusion_multimodale_v3.md §Famille 3) :

    indication+findings --CXR-BERT (gelé)--> T (B, L, 768) + txt_pad (B, L)
    image PNG           --ViT      (gelé)--> Z (B, 197, 768)
    14 requêtes apprenables --N blocs Q-Former--> H (B, 14, 768)
    H --tête diagonale label-alignée--> 14 logits

Une requête par pathologie (alignement positionnel). La lecture en sortie est
DIAGONALE : logit_j = w_j · H_j + b_j, donc le logit de la pathologie j ne lit
que la requête j (aucun pooling des requêtes). L'ordre des labels est figé en
amont (dataset.py importe LABELS_14 de text_classification — source de vérité).

Chaque bloc enchaîne, dans cet ordre, trois sous-couches résiduelles+normalisées :
  1. Self-attention des requêtes sur [Q; T]  (seules les requêtes sont mises à
     jour — le texte n'est jamais en position de query).
  2. Cross-attention des requêtes vers l'image Z.
  3. Feed-forward sur les requêtes.

Branchement texte (text_feature_mode) :
  - "last" : tous les blocs lisent la dernière couche de CXR-BERT (baseline).
  - "deep" : le bloc l lit la l-ième couche cachée (style BLIP-2 hiérarchique) —
     coarse→fine sur les N dernières couches. Coût nul (BERT calcule déjà tout).
"""

import os
import sys
import contextlib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ViTModel

# Réutilisation de text_classification (convention sys.path du repo) : on injecte
# le paquet avant l'import, car model.py peut être importé avant dataset.py.
_TEXT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "..", "text_classification")
if _TEXT_ROOT not in sys.path:
    sys.path.insert(0, _TEXT_ROOT)

# Constante centralisée du projet (text_classification) — ne pas réinventer.
from models.cxr_bert_classifier import CXR_BERT_NAME  # noqa: E402

VIT_NAME = "codewithdark/vit-chest-xray"


def build_cxr_tokenizer(max_len: int = 256):
    """(backend_tokenizer, pad_id) de CXR-BERT, tronqué à max_len.

    Le tokenizer est apparié à l'encodeur texte CXR-BERT : il vit donc avec le
    modèle (cf. cxr_bert.py). On expose le backend `tokenizers.Tokenizer`
    (interface `encode_batch`) attendu par common.data.FusionDataset.
    """
    hf_tok = AutoTokenizer.from_pretrained(CXR_BERT_NAME, trust_remote_code=True)
    pad_id = hf_tok.pad_token_id
    tok = hf_tok.backend_tokenizer
    tok.enable_truncation(max_length=max_len)
    return tok, pad_id


class DiagHead(nn.Module):
    """Tête diagonale label-alignée : logit_j = w_j·q_j + b_j (une query par label).

    Utilisée pour les classifieurs unimodaux CGGM (texte-seul / image-seul). On
    garde la même forme que la tête de fusion (W : n_labels×d) pour que la
    modulation de DIRECTION puisse comparer leurs directions de poids :
    cos(w_fusion, w_modalité) ≈ alignement des gradients (cf. CGGM §direction).
    """

    def __init__(self, n_labels, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, q):                       # q : (B, n_labels, d)
        return (q * self.W.unsqueeze(0)).sum(-1) + self.b


class JointQueryBlock(nn.Module):
    """Un bloc Q-Former : SA([Q;T]) → CA(Q→Z) → FFN, chacune résiduelle + LN."""

    def __init__(self, d=768, n_heads=12, d_ff=3072, dropout=0.1, norm="post"):
        super().__init__()
        assert norm in ("post", "pre")
        self.norm = norm
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, T, Z, txt_pad=None, return_attn=False, use_image=True):
        # `T is None`   → voie image-seule (pas de texte dans la SA).
        # `use_image=F` → voie texte-seule (on saute la cross-attention image).
        # Ces deux drapeaux servent aux passes unimodales CGGM (encode_queries).
        B, n = q.size(0), q.size(1)

        # key_padding_mask pour la SA sur [Q; T] : True = ignoré (inverse HuggingFace).
        # Les n queries sont toujours visibles. None si pas de texte (radio_only).
        kpm = None
        if T is not None and txt_pad is not None:
            kpm = torch.cat(
                [torch.zeros(B, n, dtype=torch.bool, device=q.device), txt_pad], dim=1
            )

        ca_w = None
        if self.norm == "pre":
            # Pré-norm : le résidu q traverse HORS du LN → l'identité des requêtes
            # persiste en profondeur.
            qn = self.ln1(q)
            ctx = qn if T is None else torch.cat([qn, T], dim=1)          # (B, n[+L], d)
            h, sa_w = self.self_attn(qn, ctx, ctx, key_padding_mask=kpm,
                                     need_weights=return_attn)
            q = q + h
            if use_image:
                h, ca_w = self.cross_attn(self.ln2(q), Z, Z, need_weights=return_attn)
                q = q + h
            q = q + self.ffn(self.ln3(q))
        else:
            # Post-norm (LN après le résidu).
            ctx = q if T is None else torch.cat([q, T], dim=1)            # (B, n[+L], d)
            h, sa_w = self.self_attn(q, ctx, ctx, key_padding_mask=kpm,
                                     need_weights=return_attn)
            q = self.ln1(q + h)
            if use_image:
                h, ca_w = self.cross_attn(q, Z, Z, need_weights=return_attn)
                q = self.ln2(q + h)
            q = self.ln3(q + self.ffn(q))

        return q, sa_w, ca_w


class QFormerHead(nn.Module):
    """N blocs JointQueryBlock + tête label-alignée (1 query par pathologie)."""

    def __init__(self, n_query=14, d=768, n_layers=3, n_labels=14,
                 n_heads=12, d_ff=3072, dropout=0.1, norm="post",
                 attn_pooled_head=False):
        super().__init__()
        assert n_query == n_labels, "label-aligné : une query par pathologie"
        self.norm = norm
        self.attn_pooled_head = attn_pooled_head
        self.queries = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.blocks = nn.ModuleList(
            [JointQueryBlock(d, n_heads, d_ff, dropout, norm=norm) for _ in range(n_layers)]
        )
        # Pré-norm : la sortie du dernier bloc n'est pas normalisée → LN final
        # avant la tête (échelle stable). Inutile en post-norm (déjà normalisé).
        self.final_ln = nn.LayerNorm(d) if norm == "pre" else None
        # Tête attention-pooled : une cross-attention FINALE requête→image produit
        # v_j = Σ_p α_{j,p} V(z_p) ; le logit lit v_j (pas le contenu H_j). Ainsi
        # une carte α_j fausse → v_j faux → loss pénalise : l'attention EST dans le
        # chemin du gradient (la carte est causalement liée à la prédiction).
        self.pool_attn = (nn.MultiheadAttention(d, n_heads, dropout=dropout,
                                                batch_first=True)
                          if attn_pooled_head else None)
        self.W = nn.Parameter(torch.randn(n_labels, d) * 0.02)   # (14, d)
        self.b = nn.Parameter(torch.zeros(n_labels))             # (14,)

    def _run_blocks(self, T, Z, txt_pad=None, return_attn=False, use_image=True):
        """Enchaîne les N blocs et renvoie (q, sa_all, ca_all) avant la tête."""
        B = Z.size(0)
        is_per_layer = isinstance(T, (list, tuple))
        if is_per_layer:
            assert len(T) == len(self.blocks), \
                "branchement profond : un tenseur texte par bloc Q-Former"

        q = self.queries.expand(B, -1, -1)                    # (B, 14, d)
        sa_all, ca_all = [], []
        for i, blk in enumerate(self.blocks):
            T_i = T[i] if is_per_layer else T                 # None si radio_only
            q, sa_w, ca_w = blk(q, T_i, Z, txt_pad, return_attn, use_image=use_image)
            sa_all.append(sa_w)
            ca_all.append(ca_w)
        if self.final_ln is not None:
            q = self.final_ln(q)
        return q, sa_all, ca_all

    def encode_queries(self, T, Z, txt_pad=None, drop_text=False, drop_image=False):
        """Reps unimodales CGGM (sans tête) : texte-seul (`drop_image`) ou
        image-seul (`drop_text`). Réutilise T/Z déjà encodés (encodeurs gelés)."""
        T_eff = None if drop_text else T
        pad_eff = None if drop_text else txt_pad
        q, _, _ = self._run_blocks(T_eff, Z, pad_eff, use_image=not drop_image)
        return q

    def forward(self, T, Z, txt_pad=None, return_attn=False):
        """T : tenseur (B, L, d), liste de N tenseurs (deep), ou None (radio_only).
        Z : (B, Nv, d). Le batch B vient de Z (toujours présent)."""
        q, sa_all, ca_all = self._run_blocks(T, Z, txt_pad, return_attn=return_attn,
                                             use_image=True)

        if self.attn_pooled_head:
            # v_j = Σ_p α_{j,p} V(z_p) : le logit ne lit QUE l'image attendue.
            v, pool_w = self.pool_attn(q, Z, Z, need_weights=return_attn)  # (B,14,d),(B,14,Nv)
            ca_all.append(pool_w)            # carte décisive → aux["ca"][-1]
            read = v
        else:
            read = q                         # tête diagonale historique (lit H_j)
        # logit_j = w_j · read_j + b_j
        logits = (read * self.W.unsqueeze(0)).sum(-1) + self.b   # (B, 14)
        return logits, {"sa": sa_all, "ca": ca_all, "queries": q}


class FusionQFormer(nn.Module):
    """Modèle complet : CXR-BERT (texte) + ViT (image) + Q-Former label-aligné.

    Les deux encodeurs sont gelés par défaut ; le gel est paramétrable et
    indépendant (freeze_text / freeze_image).
    """

    def __init__(
        self,
        n_labels=14,
        d=768,
        n_layers=3,
        n_heads=12,
        d_ff=3072,
        dropout=0.1,
        pad_id=0,
        text_feature_mode="last",       # "last" | "deep"
        norm="post",                    # "post" (historique) | "pre" (anti-collapse)
        radio_only=False,               # image-seul : aucune entrée texte (force la voie image)
        attn_pooled_head=False,         # logit = w_j·Σα_j·V(z) (l'attention dans la loss)
        text_dropout=0.0,               # masquage de modalité texte (proba/échantillon)
        cggm=False,                     # têtes unimodales pour la modulation CGGM (direction)
        freeze_text=True,
        freeze_image=True,
        text_name=CXR_BERT_NAME,
        image_name=VIT_NAME,
    ):
        super().__init__()
        assert text_feature_mode in ("last", "deep")
        assert norm in ("post", "pre")
        self.pad_id = pad_id
        self.text_feature_mode = text_feature_mode
        self.radio_only = radio_only
        self.text_dropout = text_dropout

        # ── Encodeurs ────────────────────────────────────────────────────
        self.text_encoder = AutoModel.from_pretrained(text_name, trust_remote_code=True)
        # add_pooling_layer=False : on n'utilise que last_hidden_state (197 tokens),
        # jamais pooler_output ; évite d'initialiser un pooler aléatoire inutilisé
        # (le checkpoint codewithdark/vit-chest-xray n'en a pas, c'est un ViTForImageClassification).
        self.image_encoder = ViTModel.from_pretrained(image_name, add_pooling_layer=False)

        d_text = self.text_encoder.config.hidden_size
        d_image = self.image_encoder.config.hidden_size
        assert d_text == d == d_image, (
            f"dims attendues égales à d={d} (texte={d_text}, image={d_image}). "
            "Avec un encodeur image non-768, ajouter une projection."
        )

        # Mapping bloc Q-Former → couche cachée BERT pour le mode "deep" :
        # les N dernières couches en ordre croissant (coarse→fine). hidden_states
        # a une longueur num_hidden_layers+1 (index 0 = embeddings).
        n_hidden = self.text_encoder.config.num_hidden_layers
        if text_feature_mode == "deep":
            assert n_layers <= n_hidden, (
                f"mode deep : n_layers={n_layers} > couches BERT={n_hidden}"
            )
            self.layer_map = list(range(n_hidden - n_layers + 1, n_hidden + 1))
        else:
            self.layer_map = None

        # ── Q-Former ─────────────────────────────────────────────────────
        self.qformer = QFormerHead(
            n_query=n_labels, d=d, n_layers=n_layers, n_labels=n_labels,
            n_heads=n_heads, d_ff=d_ff, dropout=dropout, norm=norm,
            attn_pooled_head=attn_pooled_head,
        )

        # ── Têtes unimodales CGGM (créées seulement si cggm) ─────────────────
        # Elles lisent les requêtes des passes texte-seule / image-seule. Servent
        # à (1) mesurer Δaccuracy par modalité → coeff, (2) fournir une direction
        # de poids comparée à la tête de fusion (terme de direction l_gm).
        self.cggm = cggm
        if cggm:
            assert not radio_only, "CGGM nécessite la voie texte (incompatible radio_only)"
            self.head_txt = DiagHead(n_labels, d)
            self.head_img = DiagHead(n_labels, d)

        if freeze_text:
            self.set_text_trainable(False)
        if freeze_image:
            self.set_image_trainable(False)

    # ── Gel / dégel indépendant ──────────────────────────────────────────
    def set_text_trainable(self, flag: bool):
        for p in self.text_encoder.parameters():
            p.requires_grad = flag

    def set_image_trainable(self, flag: bool):
        for p in self.image_encoder.parameters():
            p.requires_grad = flag

    def freeze_encoders(self):
        self.set_text_trainable(False)
        self.set_image_trainable(False)

    def unfreeze_encoders(self):
        self.set_text_trainable(True)
        self.set_image_trainable(True)

    # ── Encodage (encodeurs gelés, exécutés une seule fois) ───────────────
    def _encode(self, ids, pixel_values):
        """Renvoie (T, Z, txt_pad). T : (B,L,d) | liste (deep) | None (radio_only)."""
        Z = self.image_encoder(pixel_values=pixel_values).last_hidden_state  # (B,197,d)

        if self.radio_only:
            # Image-seul : aucune entrée texte (T=None → SA sur les queries seules).
            return None, Z, None

        mask = (ids != self.pad_id).long()                   # 1 = vrai token
        txt_pad = (ids == self.pad_id)                       # True = ignoré (MHA)

        # Modality dropout (ModDrop, Neverova et al. 2016) : à l'entraînement,
        # avec proba text_dropout, masque TOUT le texte d'un échantillon → la
        # voie image doit suffire. Casse l'unimodal bias (« lire le rapport »).
        # Pas de NaN : chaque requête voit toujours les n requêtes dans la SA.
        if self.training and self.text_dropout > 0:
            drop = torch.rand(ids.size(0), device=ids.device) < self.text_dropout
            txt_pad = txt_pad | drop.unsqueeze(1)            # (B,1) broadcast → (B,L)

        need_hidden = self.text_feature_mode == "deep"
        out = self.text_encoder(
            input_ids=ids, attention_mask=mask, output_hidden_states=need_hidden
        )
        T = ([out.hidden_states[i] for i in self.layer_map] if need_hidden
             else out.last_hidden_state)                     # liste (deep) ou (B,L,d)
        return T, Z, txt_pad

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, ids, pixel_values, return_attn=False):
        T, Z, txt_pad = self._encode(ids, pixel_values)
        logits, aux = self.qformer(T, Z, txt_pad, return_attn=return_attn)
        if return_attn:
            return logits, aux
        return logits

    def forward_cggm(self, ids, pixel_values, detach=True):
        """Forward CGGM : logits fusion + logits unimodaux (texte-seul, image-seul).

        Les encodeurs tournent UNE fois ; le Q-Former tourne 3× (fusion avec
        graphe + 2 passes unimodales). Si `detach`, les reps unimodales sont
        calculées sous `no_grad` → seules les têtes unimodales s'entraînent
        (mesure pure ; backbone tiré par L_task + λ·l_gm). Sinon elles
        rétro-propagent dans le Q-Former (supervision unimodale auxiliaire).
        """
        T, Z, txt_pad = self._encode(ids, pixel_values)
        logits_fus, _ = self.qformer(T, Z, txt_pad)
        ctx = torch.no_grad() if detach else contextlib.nullcontext()
        with ctx:
            q_txt = self.qformer.encode_queries(T, Z, txt_pad, drop_image=True)
            q_img = self.qformer.encode_queries(T, Z, txt_pad, drop_text=True)
        logit_txt = self.head_txt(q_txt)
        logit_img = self.head_img(q_img)
        return logits_fus, logit_txt, logit_img
