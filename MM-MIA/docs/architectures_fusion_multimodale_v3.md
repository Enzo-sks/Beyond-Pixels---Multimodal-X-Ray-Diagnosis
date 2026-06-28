# Les 4 familles d'architectures de fusion multimodale pour la classification médicale image + texte

Ce document recense les **quatre familles d'architectures** utilisées pour combiner une image médicale et un texte clinique afin de produire une classification (multi-label, diagnostic, etc.). Chaque famille est illustrée par des publications peer-reviewed de référence avec leurs repos.

---

## Vue d'ensemble

| Famille | Idée centrale | Référence principale | Lieu de publication | Responsable |
|---|---|---|---|---|
| **1. Cross-attention bidirectionnelle** | Deux encodeurs séparés qui s'enrichissent mutuellement via cross-attention | TransCheX c.f lr truc du prof, LXMERT (Tan & Bansal, EMNLP 2019), VILBERT (Lu et al., NeurIPS 2019) | MONAI / NVIDIA | |
| **2. Single-stream transformer** | djouhoud| Tokens image et texte concaténés dans une séquence unique | MMBT (Kiela et al., 2019) + Jacenków et al. (2022) + CaMCheX (Sloan et al., 2025) | ISBI 2022 / ML4H 2025 | |
| **3. Latent query distillation** | N queries apprenables qui distillent les deux modalités via cross-attention | Khader et al. (2023) + BLIP-2 (Li et al., 2023) | Radiology (RSNA) / ICML 2023 | Hugo |
| **4. Projector + LLM** | Tokens visuels projetés et prepended à l'entrée d'un LLM | LLaVA-Med (Li et al., 2023) et DeepStack (meng et al.) | NeurIPS 2023 (Spotlight) | |

---

## Famille 1 — Cross-attention bidirectionnelle

###Commentaire

Ca a pas l'air facile facile à implementer par, par contre les modèles ne sont pas figé donc faut TOUT rentrainer. A voir cu que c'est l'archiecture à recopier ce mettre à deux dessus c'est pas mal.

### Principe

Deux encodeurs séparés (un ViT pour l'image, un BERT pour le texte) traitent chaque modalité indépendamment. À certaines couches, des blocs de **cross-attention bidirectionnelle** sont insérés : le texte fait une attention où Q vient du texte et K,V viennent de l'image, et en parallèle l'image fait une attention où Q vient de l'image et K,V viennent du texte. Les deux flux gardent leurs représentations séparées jusqu'à la fin, mais s'enrichissent mutuellement à chaque couche de fusion.

La classification se fait généralement sur le `[CLS]` du flux texte, ou par concaténation des `[CLS]` des deux flux.

### Référence

LXMERT (Tan & Bansal, EMNLP 2019), VILBERT (Lu et al., NeurIPS 2019)

- Repo : [`Project-MONAI/research-contributions/TransChex`](https://github.com/Project-MONAI/research-contributions/tree/main/TransChex)
- Mécanisme central : `BertMixedLayer` contenant les deux cross-attentions parallèles
- VILBERT : https://arxiv.org/abs/1908.02265
- LXMERT : https://arxiv.org/abs/1908.07490
- Intégré au framework MONAI

### Avantages et limites

- ✅ Sépare proprement les modalités jusqu'à la fusion explicite
- ✅ Permet d'inspecter les cartes d'attention cross-modal (interprétabilité)
- ❌ Plus de paramètres trainables (deux encodeurs + couches mixtes)
- ❌ Sensible aux déséquilibres : si une modalité domine, l'autre peut être ignorée

---

## Famille 2 — Single-stream transformer

### Commentaire

Implé relativement facile,le dernier article montre une amelioration de FOU avec la multimodalité (à prendre avec des pincettes vuque c'est facile de se perdre,et c'est possible qu'il est mis les reponses en entrées, l'article à aucune citation). C'est ce qui a le plus de chance de marcher selon moi.
GROS avantages, ça permet aussi de diviser les images en "patch" et du coup de travailler avec les dicom plus facilement, à voir : Cf : https://arxiv.org/abs/2103.10504 : UNETR. 

### Principe

L'image est encodée en tokens patches (par un CNN ou un ViT), puis ces tokens sont **concaténés directement à la séquence de tokens texte** du tokenizer BERT. Un transformer unique traite la séquence mixte avec self-attention complète — il n'y a pas de mécanisme de fusion dédié, c'est la self-attention qui mélange les modalités implicitement.

Concrètement, la séquence d'entrée ressemble à : `[CLS_img] [patch_1] ... [patch_N] [SEP] [CLS_txt] [tok_1] ... [tok_M]`, avec des segment embeddings pour distinguer les modalités. La classification se fait sur un token `[CLS]` final.

### Références

**Architecture de base** : Kiela et al., *"Supervised Multimodal Bitransformers for Classifying Images and Text"* (Facebook AI Research, 2019)

- Repo : [`facebookresearch/mmbt`](https://github.com/facebookresearch/mmbt)
- Disponible aussi dans HuggingFace Transformers

**Adaptation médicale** : Jacenków et al., *"Indication as Prior Knowledge for Multimodal Disease Classification in Chest Radiographs with Transformers"* — ISBI 2022

- Démontre proprement le gain apporté par le champ `indication` seul (sans le rapport complet)
- Gain : +3.4 points d'AUROC sur MIMIC-CXR par rapport au classifieur image-seule
- Architecture : ResNet-152 + BERT, fusion par concaténation des tokens visuels et textuels
- *Note* : à moderniser avec RadDINO ou CXR-BERT pour un usage en 2025

**Version récente** : Sloan, Simpson, Mirmehdi, *"CaMCheX: Clinically-aligned Multi-modal Chest X-ray Classification"* — ML4H 2025 (Machine Learning for Health Symposium, PMLR vol. 297)

- Repo : [`phillipSloan/CaMCheX`](https://github.com/phillipSloan/CaMCheX)
- Single-stream transformer modernisé : encodeur ConvNeXt (Noisy Student training), encodeur texte BioBERT, tête de classification ML-Decoder
- Multi-vue (frontal + latéral) avec encodeurs spécialisés par projection, gestion explicite des modalités manquantes
- L'ablation montre +16.5 mAP / +4.4 AUROC quand on ajoute le champ `indication` à l'image seule
- *Note* : workshop paper peer-reviewed, mais auteurs sans track record établi en CXR — à citer comme confirmation récente, pas comme SOTA établi

### Avantages et limites

- ✅ Simple architecturalement : un seul transformer à entraîner
- ✅ Self-attention complète entre toutes les modalités à chaque couche
- ✅ Bien adapté aux petits datasets
- ❌ Pas de séparation explicite des modalités
- ❌ La longueur de séquence augmente (tokens image + tokens texte), donc complexité quadratique de l'attention

---

## Famille 3 — Latent query distillation (Perceiver / Q-Former)

###Commentaire

L'implé va surment être un petit enfer. par contre les derniers papiers bien cités qui font comme nous du multmodal texte images utilisent souvant cette architecture. En plus c'est assez prometeur pour avoir de l'explicapibilité si on sacrifie un peu de êrf en alignant les query avec les labels. (On associe Normal à une query distillé, Pneumonia à une autre etc...)

### Principe

Un petit nombre de **query embeddings apprenables** (typiquement 32 à 64) sont introduits. Ces queries font de la **cross-attention sur les features image et sur les features texte** : elles "interrogent" les deux modalités pour en extraire l'information pertinente à la tâche.

Les queries passent par plusieurs couches alternant self-attention (entre queries) et cross-attention (queries vers features image, puis queries vers features texte). À la sortie, on a N queries enrichies qui résument l'information multimodale. La classification se fait sur ces N vecteurs (pooling ou MLP).

**L'avantage clé** : la sortie a une taille fixe (N queries), indépendamment de la résolution de l'image ou de la longueur du texte. C'est un bottleneck contrôlé.

### Références

**Khader et al., *"Multimodal Deep Learning for Integrating Chest Radiographs and Clinical Parameters: A Case for Transformers"*** — Radiology (revue de la RSNA, facteur d'impact ~30), vol. 309(1):e230806, 2023

- Repo : [`FirasGit/lsmt`](https://github.com/FirasGit/lsmt)
- Architecture basée sur **Perceiver IO** (Jaegle et al., DeepMind, ICLR 2022)
- 64 latents apprenables, 9 couches alternant self-attention et cross-attention
- ViT-small patch32 384 comme backbone image
- Validation sur 45 016 patients (Aachen) + 53 150 patients (MIMIC-IV)
- Diagnostic multi-label de 25 conditions

**BLIP-2 / Q-Former** — Li, Li, Savarese, Hoi, *"BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"* — ICML 2023 (Salesforce Research)

- Repo : [`salesforce/LAVIS`](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
- Le **Q-Former** : transformer léger de 188M paramètres, initialisé depuis BERT-base
- 32 query embeddings apprenables de dimension 768 — bottleneck face aux features image (257×1024 pour un ViT-L/14)
- Pré-entraînement en deux étapes : (1) representation learning avec encodeur image gelé, trois objectifs joints (contrastif ITC, matching ITM, génération ITG) contrôlés par des masques d'attention ; (2) generative learning où les queries deviennent des soft prompts pour un LLM gelé
- Conçu pour la génération (VQA, captioning) mais les queries peuvent servir de features pour la classification
- *Note* : variante générative du paradigme latents — pertinente surtout si un pretraining large est disponible

### Avantages et limites

- ✅ Très peu de paramètres trainables (les query embeddings pèsent quelques milliers de params ; 188M pour le Q-Former entier)
- ✅ Bottleneck contrôlé : sortie de taille fixe N
- ✅ Modality-agnostic : peut accepter image, texte, vitaux, tabulaire dans la même architecture
- ✅ Publications dans Radiology et ICML = références top tier
- ❌ Nécessite généralement un pretraining pour que les latents apprennent à distiller utilement
- ❌ Risque de perte d'information si N est trop petit pour la tâche

---

## Famille 4 — Projector + LLM

###Commentaire

-Implé super simple a priori, c'est juste un encoder decoder, je pense que la seule façon que ça marche bien c'est que le llm utilisé soit TRES gros (typiquement gemma par exemple ça irais bien, mais faut voir ce qu'on peut faire tourner), on peut utiliser des llm plus gros vu que ici le lllm n'est jamais entrainé seulement utilisé. La seule partie entainé c'est la projection.
-par contre si on ceut faire plus poussé, (pas dis que ça marche mieux mais bon) on peut regarder deepstack qui au lieux de juste encodé les token image puis les faires process par le llm, on les distille petit à petit dans le context du llm, cependant ça demande de réentrainer le llm, donc pas sur que ça passe.

### Principe

Un encodeur image (typiquement un ViT pré-entraîné en contrastif type CLIP) produit des tokens visuels. Un **projector** — généralement un MLP à 2 couches — projette ces tokens depuis l'espace de l'encodeur image vers l'espace d'embedding du LLM.

Les tokens visuels projetés sont ensuite **prepended à la séquence de tokens texte** en entrée du LLM. Le LLM traite la séquence mixte via sa self-attention native et produit du texte en sortie (ou des logits via une head de classification ajoutée).

Pendant l'entraînement, on **gèle typiquement l'encodeur image et le LLM**, et on n'entraîne que le projector (et parfois une LoRA sur le LLM). C'est très peu de paramètres trainables compte tenu de la taille du modèle complet.

### Référence

**LLaVA-Med** — Li, Wong, Zhang, Usuyama et al., *"LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day"* — NeurIPS 2023 Datasets and Benchmarks Track (Spotlight)

- Repo : [`microsoft/LLaVA-Med`](https://github.com/microsoft/LLaVA-Med)
- 954 citations à ce jour
- Auteurs : Microsoft Research
- Encodeur image : CLIP-ViT pré-entraîné
- LLM : Vicuna (base LLaMA)
- Projector : MLP 2 couches
- Pretraining curriculum : (1) alignement vocabulaire biomédical sur 600k paires figure-caption PubMed, (2) instruction-tuning conversationnel sur 60k instructions générées par GPT-4
- Entraînement en 15 heures sur 8 A100s
- Surpasse les SOTA supervisés sur trois benchmarks de VQA biomédical

**Évolution récente** : MedGemma (Google DeepMind, 2025) suit le même paradigme avec Gemma 3 comme LLM et MedSigLIP comme encodeur image.

### Avantages et limites

- ✅ Bénéficie du pretraining massif d'un LLM généraliste
- ✅ Très peu de paramètres trainables (projector + éventuelle LoRA)
- ✅ Capacités émergentes : génération de rapport, VQA, dialogue, zero-shot sur nouvelles tâches
- ❌ Pour la classification multi-label pure, moins précis qu'un classifieur dédié
- ❌ Coût d'inférence élevé (forward pass d'un LLM 7B+ pour chaque prédiction)
- ❌ Risque d'hallucination en mode génératif
- ❌ Nécessite un pretraining massif (fait par Microsoft, Google, etc.)

---

## Comment choisir ?

Le choix dépend de trois facteurs :

**1. Taille du dataset downstream**
- Petit (< 10k samples) : famille 2 (single-stream) ou famille 3 (latents) si pretraining disponible
- Moyen (10-100k) : famille 1 (cross-attention) ou famille 3
- Grand (> 100k) : toutes les familles fonctionnent

**2. Objectif**
- Classification multi-label pure : familles 1, 2, 3
- Génération de rapport, VQA, dialogue : famille 4
- Mélange des deux : famille 4 avec head de classification ajoutée

**3. Ressources de calcul**
- Limitées : famille 2 (la plus simple) ou famille 3 (la plus économe en paramètres trainables)
- Abondantes avec pretraining déjà fait par un acteur tiers : famille 4

---

## Synthèse comparative

| Critère | Famille 1<br>Cross-attention | Famille 2<br>Single-stream | Famille 3<br>Latent queries | Famille 4<br>Projector + LLM |
|---|---|---|---|---|
| Complexité d'implémentation | Élevée | Faible | Moyenne | Faible (si LLM gelé) |
| Paramètres trainables | ~50M | ~110M (BERT entier) | ~5-25M (latents) / 188M (Q-Former) | <1M (projector seul) |
| Adapté petits datasets | Moyen | Bon | Très bon | Excellent (zero-shot) |
| Capacité générative | Non | Non | Oui (Q-Former) | Oui |
| Pretraining nécessaire | Modéré | Faible | Important | Massif |
| Longueur de séquence | Stable | Augmente | Fixe (N latents) | Augmente |
| Niveau de la publication de référence | NVIDIA / MONAI | ISBI / ML4H | Radiology / ICML (top tier) | NeurIPS (top tier) |

---

## Références bibliographiques complètes

1. **TransCheX** — Hatamizadeh, A., Sengupta, A. et al. *Multimodal Transformer for Joint Chest X-Ray and Radiology Report Classification*. MONAI Research Contributions, 2022.
   Repo : https://github.com/Project-MONAI/research-contributions/tree/main/TransChex

2. **MMBT** — Kiela, D., Bhooshan, S., Firooz, H., Perez, E., Testuggine, D. *Supervised Multimodal Bitransformers for Classifying Images and Text*. arXiv:1909.02950, Facebook AI Research, 2019.
   Repo : https://github.com/facebookresearch/mmbt

3. **MMBT médical** — Jacenków, G., O'Neil, A. Q., Tsaftaris, S. A. *Indication as Prior Knowledge for Multimodal Disease Classification in Chest Radiographs with Transformers*. IEEE ISBI 2022. arXiv:2202.06076.

4. **CaMCheX** — Sloan, P., Simpson, E., Mirmehdi, M. *Clinically-aligned Multi-modal Chest X-ray Classification*. ML4H 2025, PMLR vol. 297. arXiv:2511.09581.
   Repo : https://github.com/phillipSloan/CaMCheX

**ce qui sont bien pr la partie 2 :**

-LXMERT https://arxiv.org/abs/1908.07490

-VILBERT : https://arxiv.org/abs/1908.02265


5. **Khader / LSMT** — Khader, F., Müller-Franzes, G., Wang, T. et al. *Multimodal Deep Learning for Integrating Chest Radiographs and Clinical Parameters: A Case for Transformers*. Radiology, 309(1):e230806, 2023. DOI : 10.1148/radiol.230806
   Repo : https://github.com/FirasGit/lsmt

6. **BLIP-2 / Q-Former** — Li, J., Li, D., Savarese, S., Hoi, S. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*. ICML 2023, Salesforce Research. arXiv:2301.12597.
   Repo : https://github.com/salesforce/LAVIS/tree/main/projects/blip2

7. **LLaVA-Med** — Li, C., Wong, C., Zhang, S., Usuyama, N. et al. *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*. NeurIPS 2023 Datasets and Benchmarks Track (Spotlight). arXiv:2306.00890.
   Repo : https://github.com/microsoft/LLaVA-Med
