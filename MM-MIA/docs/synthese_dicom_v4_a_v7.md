# Synthèse des Travaux : Modélisation 3D DICOM (De la v4 à la v7)

Ce document retrace l'évolution complète du pipeline de classification multi-labels sur les volumes DICOM 3D (dossier `src/mmmia/dicom`). Il explique les défis rencontrés, les stratégies de prétraitement des données, et les différentes itérations de l'algorithme d'apprentissage profond pour passer d'un modèle basique à un système optimisé.

---

## 1. La Première Stratégie de Preprocessing
**Fichier associé :** `src/mmmia/dicom/notebooks/Preprocess_DICOM_to_Volumes_3D.ipynb`

### Objectif
Convertir des séries d'images DICOM brutes en volumes 3D exploitables par un réseau de neurones (format NumPy `.npy`).

### Fonctionnement
- **Extraction :** Lecture des pixels depuis les fichiers DICOM.
- **Normalisation :** Mise à l'échelle des valeurs de pixels entre 0 et 1.
- **Redimensionnement :** Standardisation de la taille de chaque volume à **128 × 128 × 64** (Largeur × Hauteur × Profondeur) via interpolation.

> **Les limites de cette version :**
> La résolution de 128x128 était trop faible pour discerner des détails fins comme les nodules. De plus, aucun fenêtrage spécifique (Windowing) n'était appliqué, ce qui donnait des images où les tissus mous masquaient souvent les structures pulmonaires (qui apparaissent noires par défaut).

---

## 2. Les Itérations d'Entraînement sur le Premier Preprocessing

Avec ces volumes 128x128x64, nous avons itéré sur le script d'entraînement (`train.py`) pour tenter d'extraire des signaux.

### Version 4 (v4) : Optimisation des Bases
- **Architecture :** ResNet18-3D (`r3d_18` de PyTorch torchvision).
- **Amélioration majeure :** Mise en place d'un split de données **stratifié multi-critères**. Pour garantir une répartition équitable, le dataset a été divisé (70% Train, 15% Val, 15% Test) en prenant en compte la présence de la classe "Normal" ET le nombre total de pathologies par patient.
- **Résultat :** AUC-ROC autour de **53.2%**. Le modèle apprenait à peine mieux que le hasard, la perte (loss) stagnait rapidement.

### Version 5 (v5) : Approche Progressive
- **Stratégie :** Introduction du "Progressive Unfreezing" (Dégel progressif). Au lieu d'entraîner tout le réseau d'un coup, on figeait les poids de base pour n'entraîner que la couche de classification (la tête), puis on débloquait les couches profondes (`layer4`, puis `layer3`) au fil des époques.
- **Résultat :** Le modèle overfittait sur les classes majoritaires et générait énormément de "Faux Positifs". L'AUC-ROC baissait à **52.3%**.

### Version 6 (v6) : Stabilisation et Gestion du Déséquilibre
Pour forcer le modèle à apprendre malgré le fort déséquilibre (beaucoup de patients sains, peu de cas pathologiques par classe spécifique), nous avons sorti l'artillerie lourde :
- **Poids pré-entraînés :** Utilisation stricte des poids Kinetics-400 pour initialiser le ResNet18-3D.
- **Asymmetric Loss (ASL) :** Remplacement de la perte classique par une fonction de perte asymétrique (`gamma_neg=4`, `gamma_pos=1`). Cette technique punit très sévèrement le modèle lorsqu'il prédit une maladie à tort (réduction des Faux Positifs), tout en étant plus tolérant sur les vrais positifs.
- **Nettoyage des Labels :** Suppression des 4 classes ultra-rares (moins de 10 exemples sur 6000 : Pneumothorax, Edema, Pleural_Thickening, Consolidation) qui empêchaient la convergence.
- **Résultat :** **Victoire !** L'AUC-ROC est montée à **59.2%**. Le modèle a commencé à réellement discriminer. Cependant, l'analyse de la matrice de confusion a prouvé que le modèle restait "aveugle" aux détails à cause de la qualité des images 128x128.

---

## 3. Le Nouveau Preprocessing Avancé (La Révolution)
**Fichier associé :** `src/mmmia/dicom/notebooks/Advanced_Preprocess_DICOM_to_Volumes_3D.ipynb`

Pour briser le plafond de verre des 59% d'AUC, nous avons entièrement revu la façon dont les images sont préparées avant d'être données au réseau.

### Les Améliorations Clés
1. **Conversion en Hounsfield Units (HU) et Lung Windowing (Fenêtrage Pulmonaire) :** 
   Les images médicales DICOM stockent les données brutes des capteurs. Pour les transformer en densités physiques (Unités Hounsfield, où l'eau = 0 et l'air = -1000), la première formule appliquée est :
   $$HU = (PixelValue \times RescaleSlope) + RescaleIntercept$$
   
   Ensuite, pour faire ressortir spécifiquement le parenchyme pulmonaire, un filtre radiologique ("Windowing") est appliqué (Width: 1500, Center: -600 HU). Les bornes de visibilité sont calculées ainsi :
   $$Min = Center - \frac{Width}{2} = -600 - \frac{1500}{2} = -1350$$
   $$Max = Center + \frac{Width}{2} = -600 + \frac{1500}{2} = +150$$
   
   Toutes les valeurs $HU$ en dehors de ces bornes sont coupées (clippées). La normalisation finale, indispensable pour les réseaux de neurones (valeurs entre 0 et 1), est donnée par la formule :
   $$ValeurNormalisee = \frac{HU_{clippe} - Min}{Max - Min}$$
   **Résultat :** Les os et les tissus mous disparaissent, les vaisseaux et le parenchyme pulmonaire deviennent nets et contrastés.
2. **Haute Résolution (224x224) :** Passage à des volumes de **224 × 224 × 64**. 224x224 est la taille native d'entraînement du ResNet18, ce qui évite au réseau de devoir réadapter ses filtres internes.
3. **Padding Adaptatif :** Au lieu d'écraser et de déformer les images rectangulaires pour en faire des carrés, le nouveau script ajoute des bandes noires (padding) pour préserver les proportions réelles de l'anatomie du patient.

---

## 4. La Dernière Version : L'Entraînement v7 (En détail)

La version 7 de notre pipeline de Machine Learning (`src/mmmia/dicom/dicom_gpu_telecom/`) est l'aboutissement de toutes nos itérations. Elle capitalise sur le nouveau preprocessing et consolide le code en un véritable pipeline MLOps de production.

### A. Refonte de la Pipeline de Données (`dataset.py` & `prepare_data.py`)
1. **Ingestion des nouveaux volumes :** Le Dataset PyTorch est désormais calibré pour ingérer les nouveaux tenseurs haute qualité au format `(3, 64, 224, 224)` (Canaux, Profondeur, Hauteur, Largeur).
2. **Data Augmentation 3D native (MONAI) :** Pour lutter contre le surapprentissage sur notre dataset de 2557 images, nous avons intégré la librairie **MONAI** (Medical Open Network for AI). Pendant l'entraînement, les volumes subissent des transformations dynamiques à la volée :
   - `RandFlip` : Inversions miroirs horizontales aléatoires.
   - `RandRotate90` : Rotations aléatoires de 90°.
   - `RandZoom` : Zooms spatiaux modérés.
   - `RandGaussianNoise` : Ajout de bruit gaussien pour rendre le modèle robuste aux variations des scanners réels.
3. **Split Dynamique et Sécurisé :** Le script `prepare_data.py` gère toujours le split stratifié multi-critères (70/15/15), mais le script principal du cluster supprime et recrée obligatoirement ces splits à chaque lancement pour s'assurer qu'aucun ancien patient (dont le fichier `.npy` n'existe plus) ne soit utilisé par erreur.

### B. Architecture Modulaire et Entraînement (`model.py` & `train.py`)
1. **Modèle de base :** Nous utilisons un **ResNet-18 3D** (`r3d_18`) pré-entraîné sur l'immense dataset vidéo Kinetics-400. Les réseaux vidéo 3D sont structurellement identiques aux réseaux médicaux 3D (temps = profondeur), ce qui permet un excellent transfert d'apprentissage.
2. **Progressive Unfreezing en 3 Phases :**
   - **Phase 1 (Epochs 1-5) :** Tout le réseau est gelé, seule la dernière couche (Classifieur linéaire pour les 17 classes) est entraînée avec un Learning Rate (LR) élevé (1e-3). Cela permet à la tête d'apprendre rapidement le nouveau format des labels.
   - **Phase 2 (Epochs 6-10) :** Dégel du `layer4` (le dernier bloc convolutif du ResNet). Le LR est drastiquement réduit (1e-5) pour affiner les caractéristiques sémantiques de haut niveau sans détruire les poids pré-entraînés.
   - **Phase 3 (Epochs 11+) :** Dégel du `layer3`. Le LR est divisé par deux (5e-6) pour un fine-tuning chirurgical en profondeur.
3. **Asymmetric Loss (ASL) :** Plutôt qu'un classique Binary Cross-Entropy (BCE), la fonction de perte ASL est utilisée avec `gamma_neg=4` et `gamma_pos=1`. Cela permet de "réduire au silence" les millions de "vrais négatifs" faciles (patients sains) pour obliger le réseau à se concentrer sur les quelques cas positifs (maladies).

### C. Robustesse de l'Infrastucture (Scripts d'automatisation)
Travailler sur un cluster distant avec des quotas stricts a nécessité l'ingénierie d'outils annexes :
- **Mapping des Labels (`map_labels.py`) :** Un script qui fusionne l'ancienne table de labels avec la nouvelle, assurant que seules les métadonnées valides sont conservées.
- **Filtre de Sécurité (`filter_csv_cluster.py`) :** La plus grande difficulté a été la disparition d'images causée par les quotas Google Drive. Ce script vérifie physiquement (via `os.path.exists`) la présence de chaque volume `.npy` sur les disques de Télécom Paris. S'il manque un fichier, la ligne est purgée du CSV *avant* de lancer PyTorch.
- **Logs Intelligents (`job.sh` & `train.py`) :** Le script de soumission SLURM injecte des diagnostics de santé au lancement : Nom du GPU alloué (ex: Tesla P100), quantité de VRAM, confirmation d'accès CUDA, et résumé détaillé du batch size et de l'architecture.

### D. Évaluation Dynamique Multi-seuils (`eval.py`)
Dans un problème médical multi-label déséquilibré, fixer un seuil arbitraire de `0.5` pour décider si un patient est malade ou non est une erreur.
La v7 intègre un algorithme de recherche de seuil :
1. Pour chaque classe (parmi les 17), l'algorithme teste des dizaines de seuils (de 0.1 à 0.9).
2. Il retient le seuil qui maximise le **F1-Score** (équilibre entre précision et rappel). Par exemple, le seuil de détection d'une fibrose peut être `0.15` alors que celui d'une masse sera `0.5`.
3. Ces seuils optimaux sont sauvegardés dans `optimal_thresholds.json` et utilisés pour générer une **Matrice de Confusion (Heatmap)** extrêmement précise par pathologie.

### Conclusion Globale
Le dossier `src/mmmia/dicom` abrite aujourd'hui un pipeline MLOps industriel. Du pré-traitement avancé à l'entraînement stratifié, jusqu'au déploiement et filtrage automatisé sur cluster, ce système est prêt à extraire la quintessence des volumes DICOM 3D.
