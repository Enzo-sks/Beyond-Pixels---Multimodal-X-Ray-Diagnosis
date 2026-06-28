# Analyse des Résultats v6

## 1. Évolution des Métriques (v4 → v5 → v6)

| Métrique | v4 | v5 | **v6** | Tendance |
|---|---|---|---|---|
| **AUC-ROC** | 53.2% | 52.3% | **59.2%** | 📈 **+6 pts** (le modèle commence enfin à discriminer !) |
| Accuracy | 55.2% | 0.78% | 0.26% | ⚠️ Voir explication ci-dessous |
| Precision | 21.3% | 26.7% | **29.8%** | 📈 Amélioration progressive |
| Recall | 38.6% | 87.6% | **78.1%** | Le modèle détecte bien les malades |
| F1-Score | 27.5% | 36.9% | **37.4%** | 📈 Meilleur score des 3 versions |

> [!IMPORTANT]
> **Pourquoi l'Accuracy est à 0.26% ?**
> L'`accuracy_score` de sklearn en multi-label exige un **exact match** : pour qu'un patient soit compté comme "correct", il faut que les 21 labels soient **tous** parfaitement prédits. Avec 21 classes, c'est quasi impossible. **Cette métrique n'est PAS pertinente pour du multi-label.** L'AUC-ROC et le F1 sont les vraies métriques à regarder.

## 2. La bonne nouvelle : l'AUC-ROC décolle

L'AUC passe de ~52% (hasard) à **59.2%**. C'est la première fois que le modèle montre une vraie capacité à distinguer les cas positifs des négatifs. Les poids pré-entraînés Kinetics + le dégel progressif + l'AsymmetricLoss améliorée fonctionnent.

## 3. Analyse de la Confusion Matrix (Heatmap)

![Confusion Heatmap](file:///C:/Users/ahmed/Desktop/artishow/artihow%20_final/MM-MIA/src/mmmia/dicom/dicom_gpu_telecom/results/confusion_heatmap.png)

### Classes où le modèle fonctionne le mieux :
| Classe | Recall | Precision | F1 | Constat |
|---|---|---|---|---|
| **Normal** | 100% | 55.2% | 71.1% | ✅ Détecte tous les patients sains |
| **Calcinosis** | 98.3% | 15.7% | 27.1% | Détecte bien, mais beaucoup de faux positifs |
| **Opacity** | 90.0% | 12.9% | 22.6% | Idem |

### Classes où il échoue complètement (Recall = 0%) :
| Classe | Support (nb vrais cas) | Constat |
|---|---|---|
| Pneumothorax | 1 seul cas | Impossible d'apprendre avec 1 exemple |
| Edema | 6 cas | Trop rare |
| Emphysema | 18 cas | Le fenêtrage actuel ne montre pas les poumons |
| Pleural_Thickening | 4 cas | Trop rare |

### Le problème principal : **trop de Faux Positifs (FP)**
La colonne "False Pos" de la heatmap est rouge vif partout. Le modèle prédit "malade" beaucoup trop souvent :
- Fibrosis : 4 TP mais **325 FP** → il crie "Fibrose !" pour 84% des patients
- Mass : 3 TP mais **258 FP**
- Nodule : 15 TP mais **282 FP**

## 4. Diagnostic Final

Le modèle a fait des progrès réels (AUC en hausse), mais il reste fondamentalement limité par **la qualité des images d'entrée** :
1. Le fenêtrage actuel (tissus mous, pas poumons) rend les pathologies pulmonaires invisibles
2. La résolution 128×128 écrase les détails fins
3. Certaines classes ont trop peu d'exemples (1 à 6 cas)

## 5. Prochaines Étapes (par ordre d'impact)

1. 🥇 **Relancer le preprocessing avec le nouveau notebook Colab** (`Advanced_Preprocess_DICOM_to_Volumes_3D.ipynb`) : fenêtrage Lung Window + résolution 224×224. C'est le changement qui aura le plus d'impact.
2. 🥈 **Supprimer les classes avec moins de 10 exemples** (Pneumothorax, Edema, Pleural_Thickening, Consolidation) du CSV de labels, car le modèle ne peut physiquement pas les apprendre.
3. 🥉 **Augmenter le dataset** avec de la data augmentation 3D plus agressive (rotations, elastic deformation) via MONAI transforms.
