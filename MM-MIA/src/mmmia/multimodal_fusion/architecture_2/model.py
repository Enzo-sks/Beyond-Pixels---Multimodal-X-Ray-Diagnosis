"""model.py — Projector MLP + LoRA sur Gemma 3 1B pour classification multi-label CXR.

Architecture (inspirée LLaVA / MedGemma) :
  pixel_values (B, 3, 224, 224)
      │
      ▼
  ViT encoder  (B, 197, 768)   ← gelé
      │
      ▼
  Projector MLP (768 → llm_dim) (B, 197, 1152)   ← entraîné
      │
  prepend aux tokens texte
      │
      ▼
  Gemma 3 1B  (B, 197+L, 1152)   ← gelé sauf LoRA
      │
      ▼
  [CLS-like] dernier token causal → head Linear(1152 → 21)   ← entraîné

Pourquoi Gemma 3 1B (et pas MedGemma 4B) ?
  - MedGemma 4B nécessite ~8-16 Go de mémoire et un GPU récent ; sur GPU
    Pascal (P100, sm_60), certaines opérations (notamment les convolutions
    cuDNN du ViT externe sous AMP) provoquent des erreurs backend
    ("GET was unable to find an engine to execute this computation") et la
    charge mémoire globale (modèle + dataset + activations) sature la RAM
    disponible du job Slurm.
  - google/gemma-3-1b-pt est texte seul (pas de vision_tower intégré à
    décharger), ~1B params, hidden_size=1152 — beaucoup plus léger à
    charger et à entraîner, tout en gardant la même architecture Gemma3
    (donc compatible PEFT/LoRA de la même façon).
  - Le ViT externe (codewithdark/vit-chest-xray) reste utilisé via le
    Projector, comme avec MedGemma.

Accès HuggingFace :
  google/gemma-3-1b-pt est public (pas de licence à accepter), contrairement
  à MedGemma. Un token HF simple suffit si besoin pour le rate-limit.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


# ── Constante modèle ─────────────────────────────────────────────────────────
GEMMA_NAME = "google/gemma-3-1b-pt"
GEMMA_DIM  = 1152   # hidden_size de Gemma 3 1B (confirmé : embed_tokens et
                     # q_proj/k_proj/v_proj sont tous en 1152)


# ── Projector MLP (2 couches, style LLaVA) ───────────────────────────────────

class Projector(nn.Module):
    """
    Projette les tokens ViT (image_dim) vers l'espace d'embedding du LLM (llm_dim).
    Architecture : Linear → GELU → Dropout → Linear (2 couches)

    Args:
        image_dim : dimension de sortie du ViT (768 pour ViT-B/16)
        llm_dim   : dimension d'embedding du LLM (2560 pour MedGemma 4B)
        dropout   : taux de dropout (défaut 0.1)
    """

    def __init__(self, image_dim: int = 768, llm_dim: int = GEMMA_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tokens : (B, N, image_dim)  — N=197 pour ViT-B/16
        Returns:
            (B, N, llm_dim)
        """
        return self.net(image_tokens)


# ── Modèle complet ────────────────────────────────────────────────────────────

class ProjectorLLM(nn.Module):
    """
    Pipeline complet :
      1. ViT extrait les tokens image              (B, 197, 768)
      2. Projector projette vers l'espace LLM      (B, 197, llm_dim=2560)
      3. Les tokens projetés sont prepend aux embeddings texte
      4. MedGemma 4B (+ LoRA) traite la séquence mixte
      5. Le dernier token causal → head → logits   (B, n_labels)

    Paramètres entraînables :
      - Projector  (~10M avec llm_dim=2560)
      - LoRA dans MedGemma (q_proj, v_proj de chaque layer) (~8-12M selon rank)
      - Head de classification (~54k)

    Args:
        vit        : ViTModel HuggingFace (expose .last_hidden_state, .config.hidden_size)
        gemma      : AutoModel MedGemma (déjà wrappé avec LoRA via get_peft_model)
        n_labels   : nombre de labels (défaut 21)
        image_dim  : dimension ViT (défaut 768)
        llm_dim    : dimension LLM (défaut 2560 pour MedGemma 4B)
        dropout    : taux dropout (défaut 0.1)
    """

    def __init__(
        self,
        vit,
        gemma,
        n_labels: int   = 21,
        image_dim: int  = 768,
        llm_dim: int    = GEMMA_DIM,
        dropout: float  = 0.1,
    ):
        super().__init__()
        self.vit       = vit
        self.gemma     = gemma
        self.projector = Projector(image_dim, llm_dim, dropout)
        self.head      = nn.Linear(llm_dim, n_labels)

        # Embedding table de MedGemma pour encoder les tokens texte.
        # AutoModel expose le modèle de base via .model (compatible PEFT).
        base = gemma.model if hasattr(gemma, "model") else gemma
        self.embed_tokens = base.embed_tokens

    def forward(
        self,
        input_ids:    torch.Tensor,   # (B, L)  tokens texte
        pixel_values: torch.Tensor,   # (B, 3, 224, 224)
    ) -> torch.Tensor:                # (B, n_labels)

        # ── 1. Encoder image (ViT gelé) ──────────────────────────────────
        # IMPORTANT: le ViT tourne sur CPU, pas GPU.
        # Sur ce GPU Pascal (P100, sm_60), la conv2d de patch_embeddings
        # échoue avec "GET was unable to find an engine to execute this
        # computation" — bug cuDNN/Pascal qui persiste même avec
        # cudnn.benchmark=False, cudnn.enabled=False, et en forçant float32.
        # Le ViT est gelé (torch.no_grad, pas de gradient à calculer), donc
        # l'exécuter sur CPU n'a qu'un coût de copie mémoire + calcul CPU,
        # pas de gradient perdu. Le aller-retour CPU<->GPU est limité à un
        # forward par batch (tenseur d'entrée + tenseur de sortie).
        with torch.no_grad():
            vit_device = next(self.vit.parameters()).device
            image_tokens = self.vit(
                pixel_values=pixel_values.to(vit_device, dtype=torch.float32)
            ).last_hidden_state
            image_tokens = image_tokens.to(pixel_values.device)
            # (B, 197, 768)

        # ── 2. Projector ─────────────────────────────────────────────────
        image_proj = self.projector(image_tokens)   # (B, 197, llm_dim)

        # ── 3. Embeddings texte via table MedGemma ───────────────────────
        text_embeds = self.embed_tokens(input_ids)  # (B, L, llm_dim)

        # ── 4. Concaténation [image_proj | text_embeds] ──────────────────
        inputs_embeds = torch.cat([image_proj, text_embeds], dim=1)
        # (B, 197+L, llm_dim)

        # ── 5. Passage dans MedGemma (+ LoRA) ────────────────────────────
        outputs = self.gemma(inputs_embeds=inputs_embeds)
        hidden  = outputs.last_hidden_state   # (B, 197+L, llm_dim)

        # ── 6. Pooling : dernier token causal ────────────────────────────
        cls_repr = hidden[:, -1, :]           # (B, llm_dim)

        # ── 7. Head de classification ────────────────────────────────────
        return self.head(cls_repr)            # (B, n_labels)


# ── Factory : construit le modèle complet avec LoRA ──────────────────────────

def build_model(
    vit,
    gemma_name: str     = GEMMA_NAME,
    n_labels: int       = 21,
    lora_rank: int      = 16,
    lora_alpha: int     = 32,
    lora_dropout: float = 0.05,
    dropout: float      = 0.1,
    device: str         = "cpu",
) -> ProjectorLLM:
    """
    Construit et retourne un ProjectorLLM avec Gemma 3 1B prêt à l'entraînement.

    - ViT          : gelé (passé en argument, déjà chargé), forcé sur CPU
                     (contournement bug cuDNN sur GPU Pascal)
    - Gemma 3 1B   : chargé depuis HuggingFace via AutoModelForCausalLM,
                     wrappé avec LoRA
    - Projector + head : entraînables

    Args:
        vit         : ViTModel déjà chargé
        gemma_name  : identifiant HuggingFace (défaut: google/gemma-3-1b-pt)
        n_labels    : nombre de labels de sortie
        lora_rank   : rang LoRA (défaut 16)
        lora_alpha  : alpha LoRA — scaling = alpha/rank (défaut 32)
        lora_dropout: dropout interne LoRA (défaut 0.05)
        dropout     : dropout du Projector (défaut 0.1)
        device      : "cuda" ou "cpu" (s'applique à Gemma/Projector/head ;
                      le ViT est toujours forcé sur CPU, voir ci-dessous)

    Returns:
        ProjectorLLM configuré et déplacé sur device
    """
    # ── ViT gelé ─────────────────────────────────────────────────────────
    for p in vit.parameters():
        p.requires_grad = False
    # IMPORTANT: ViT forcé sur CPU, indépendamment de `device`, pour
    # contourner un bug cuDNN sur GPU Pascal (P100) où la conv2d de
    # patch_embeddings échoue ("GET was unable to find an engine to execute
    # this computation"). Voir ProjectorLLM.forward() pour le aller-retour
    # CPU<->GPU correspondant. Le ViT étant gelé, ceci n'affecte pas
    # l'entraînement, seulement la vitesse du forward (acceptable: ViT-B/16
    # est petit, ~86M params, et tourne sur peu d'images par batch).
    vit = vit.to("cpu")

    # ── Gemma 3 1B via AutoModelForCausalLM ────────────────────────────────
    # Gemma 3 1B (contrairement à MedGemma 4B) est un checkpoint TEXTE SEUL :
    # pas de vision_tower à dépiler, AutoModelForCausalLM suffit directement.
    # Structure (même schéma que MedGemma, sans le niveau multimodal) :
    #   Gemma3ForCausalLM (= AutoModelForCausalLM)
    #     ├── model      (Gemma3TextModel) ← celui qu'on garde, a embed_tokens
    #     │     └── embed_tokens
    #     └── lm_head    (tête de génération, non utilisée ici)
    # Gemma3ForCausalLM.forward() ne retourne que logits/past_key_values
    # (pas de last_hidden_state) ; Gemma3TextModel.forward() retourne bien
    # last_hidden_state, ce dont ProjectorLLM.forward() a besoin.
    print(f"Chargement de {gemma_name}...")
    gemma_full = AutoModelForCausalLM.from_pretrained(
        gemma_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,   # évite de dupliquer le state_dict en RAM CPU pendant le chargement
    )
    medgemma_base = gemma_full.model
    del gemma_full   # libère lm_head (économie VRAM/RAM, minime mais gratuit)

    llm_dim = medgemma_base.config.hidden_size
    print(f"  hidden_size détecté : {llm_dim}")

    # ── LoRA sur MedGemma ─────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj"],   # Q et V de chaque bloc
        bias="none",
    )
    medgemma_lora = get_peft_model(medgemma_base, lora_cfg)

    # Geler tout sauf LoRA (PEFT le fait automatiquement, on double-vérifie)
    for name, p in medgemma_lora.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False

    medgemma_lora = medgemma_lora.to(device)

    # ── Assemblage ────────────────────────────────────────────────────────
    model = ProjectorLLM(
        vit=vit,
        gemma=medgemma_lora,
        n_labels=n_labels,
        image_dim=vit.config.hidden_size,
        llm_dim=llm_dim,
        dropout=dropout,
    ).to(device)

    # IMPORTANT: le .to(device) ci-dessus déplace TOUS les sous-modules,
    # y compris model.vit qu'on avait explicitement placé sur CPU plus haut
    # (contournement bug cuDNN P100). On le re-force donc sur CPU ici, en
    # dernier, pour que ça reste effectif après l'assemblage complet.
    model.vit = model.vit.to("cpu")

    # ── Résumé des paramètres ─────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux    : {total:,}")
    print(f"Paramètres entraîn.  : {trainable:,}  ({100*trainable/total:.1f}%)")

    return model