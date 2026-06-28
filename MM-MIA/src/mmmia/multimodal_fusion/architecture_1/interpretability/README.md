# Interpretability Analysis Results

## 1. Attention Heatmaps (Image Pooling & Text→Image Attention)

### Overall Conclusion
The attention maps show some variation between pathologies, indicating that the model does not always focus on exactly the same image regions. However, several high-attention areas appear near image borders or outside clinically relevant anatomy. Therefore, the attention mechanism provides only limited evidence of pathology-specific localization and should be interpreted with caution.

### Pneumothorax
The strongest attention appears near the left image border and partially outside the lung field. Although the model predicts Pneumothorax with moderate confidence, the highlighted region is not anatomically consistent with the expected location of a pneumothorax, raising concerns about border-related artifacts.

### Cardiomegaly
Attention is distributed across several regions, including lower thoracic areas. Some attention is located closer to relevant anatomy than in the Pneumothorax case, but the cardiac silhouette is not clearly emphasized as the dominant region.

### Effusion
The attention pattern differs noticeably from the frontal-view cases and follows the lateral projection. Some highlighted regions overlap with lower thoracic areas where pleural effusions are commonly observed, although the localization remains diffuse.

### Normal
Multiple hotspots are present despite the absence of pathology. The highlighted regions do not correspond to a specific anatomical structure and include border areas, suggesting that attention is not exclusively focused on disease-related features.

---

## 2. Grad-CAM on Individual Images

### Overall Conclusion
Grad-CAM produces more pathology-dependent patterns than raw attention maps and is generally a more direct indicator of which image features influence the prediction. However, the localization remains weak and inconsistent across samples, with some maps showing little signal and others highlighting regions that are not clearly related to the pathology.

### Pneumothorax
The Grad-CAM map is nearly uniform with very little localized activation. This suggests that the prediction is not strongly driven by a specific image region for this sample.

### Cardiomegaly
A small localized hotspot is visible, indicating that a limited region contributes to the prediction. However, the highlighted area is not clearly centered on the enlarged cardiac silhouette, making the explanation difficult to interpret clinically.

### Effusion
The heatmap shows widespread activation across the image rather than a single focused region. The resulting pattern appears noisy and lacks a clear anatomical target, reducing confidence in the localization.

### Normal
Several localized hotspots appear near image borders and peripheral regions. Since no pathology is present, these activations may reflect general image characteristics rather than disease-specific findings.

---

## 3. Averaged Norm-Corrected Grad-CAM (15 Samples per Pathology)

### Overall Conclusion
After averaging Grad-CAM maps from multiple positive cases, no pathology exhibits a strong and reproducible hotspot. The averaged signals remain weak and diffuse, suggesting that different samples do not consistently activate the same image region. This indicates limited evidence for a stable pathology-specific localization strategy.

### Pneumothorax
The averaged map contains only weak activations scattered across the grid. No dominant region emerges across patients, indicating low agreement in localization.

### Cardiomegaly
The averaged signal is very weak and broadly distributed. No clear concentration appears around the heart region, suggesting that the model does not consistently rely on the same cardiac area across cases.

### Effusion
A slightly stronger pattern is visible near the lower portion of the grid, which could be loosely consistent with pleural fluid accumulation. However, the overall signal remains weak and lacks a sharply defined hotspot.

### Normal
The strongest averaged activations occur near image borders and lower edge regions. These patterns are more suggestive of image-position effects than of meaningful anatomical localization.

---

# Final Takeaway

Across all three analyses, the model shows **some sensitivity to image content**, since the highlighted regions are not completely identical across pathologies. However, the explanations do not consistently align with expected disease locations, and several activations occur near image borders or background regions. Overall, the results suggest that while the image branch contributes to the predictions, the model does not appear to rely on a clear and reproducible anatomical region for each pathology. This may indicate reliance on distributed visual cues, dataset-specific patterns, or information provided by the text modality in the multimodal fusion process.


## Cross-Modal Interaction Analysis


Blanking the report text caused a substantial decrease in prediction probabilities for pathology classes when the model was evaluated end-to-end:

| Pathology | Real Image + Real Text | Real Image + Blank Text |
|------------|------------|------------|
| Pneumothorax | 0.634 | 0.152 |
| Cardiomegaly | 0.755 | 0.248 |
| Effusion | 0.790 | 0.235 |
| Normal | 0.980 | 0.892 |

These results indicate that the model is not relying solely on visual information for pathology predictions. The textual input contributes significantly to the final output.

To understand this behavior, a representation-swap experiment was performed. When the image representation obtained from the **real-text forward pass** was combined with a **blank-text representation**, most of the original prediction score was recovered:

| Pathology | P(real, real) | P(blankText, realImage) |
|------------|------------|------------|
| Pneumothorax | 0.634 | 0.558 |
| Cardiomegaly | 0.755 | 0.752 |
| Effusion | 0.790 | 0.752 |

This suggests that the image representation itself has been modified through cross-attention with the report text. In other words, the fused image features are no longer purely visual representations but already contain information originating from the text modality.

In contrast, when the **real-text representation** was paired with the **image representation obtained from the blank-text forward pass**, prediction scores remained low:

| Pathology | P(realText, blankImage) |
|------------|------------|
| Pneumothorax | 0.170 |
| Cardiomegaly | 0.258 |
| Effusion | 0.233 |

This indicates that the performance drop is primarily caused by changes in the image representation after removing textual information, rather than by the text representation alone.

### Conclusion

The results suggest that a substantial portion of the diagnostic signal is transferred into the image representation through cross-attention. Consequently, the apparent contribution of the image branch may partly reflect information injected from the report text rather than independent visual evidence. This highlights a potential risk of **text leakage through fusion mechanisms** and suggests that high multimodal performance should not automatically be interpreted as successful integration of complementary visual and textual information.


## Cross-Modal Mediation Analysis — Full Test Set, AUC-Based (n=968)

### Method

For every test sample, the model is run twice through the shared backbone — once with the real radiology report, once with blank text — producing two representation pairs: `(text_R, image_R)` and `(text_B, image_B)`. These are then recombined and passed through the shared classification head to isolate each branch's contribution:

| Condition | Combination | Interpretation |
|---|---|---|
| `real_real` | `(text_R, image_R)` | Full model, reproduces reported test AUC |
| `blankT+realI` | `(text_B, image_R)` | Image representation alone (but shaped by cross-attention with the real report) |
| `realT+blankI` | `(text_R, image_B)` | Text representation alone (paired with a content-free image) |

AUC is computed per label over the full 968-sample test set (exact split reproduced from `train.py`, `random_state=42`).

### Per-Label Results

| Pathology | n_pos | AUC real | AUC blankT+realI | AUC realT+blankI | Δ (real − blankT+realI) |
|---|---|---|---|---|---|
| Atelectasis | 93 | 0.9825 | 0.9858 | 0.6512 | −0.0033 |
| Cardiomegaly | 82 | 0.9817 | 0.9844 | 0.9286 | −0.0027 |
| Effusion | 41 | 0.9758 | 0.9723 | 0.8736 | 0.0035 |
| Pneumonia | 22 | 0.9681 | 0.9435 | 0.6647 | 0.0246 |
| Pneumothorax | 6 | 0.9977 | 0.9971 | 0.9182 | 0.0007 |
| Edema | 25 | 0.9506 | 0.9703 | 0.8780 | −0.0197 |
| Emphysema | 32 | 0.9494 | 0.9576 | 0.8571 | −0.0082 |
| Fibrosis | 6 | 0.9463 | 0.9539 | 0.8557 | −0.0076 |
| Infiltration | 19 | 0.9685 | 0.9612 | 0.7677 | 0.0073 |
| Mass | 5 | 0.9915 | 0.9668 | 0.8546 | 0.0247 |
| Nodule | 28 | 0.9894 | 0.9938 | 0.6169 | −0.0044 |
| Hernia | 12 | 0.9997 | 0.9994 | 0.7494 | 0.0003 |
| Fracture | 27 | 0.9831 | 0.9959 | 0.7204 | −0.0128 |
| Pleural_Thickening | 13 | 0.9998 | 0.9989 | 0.7729 | 0.0010 |
| Opacity | 132 | 0.9996 | 0.9988 | 0.6720 | 0.0008 |
| Consolidation | 7 | 0.9915 | 0.9917 | 0.8910 | −0.0001 |
| Granuloma | 105 | 0.9999 | 0.9997 | 0.5967 | 0.0002 |
| Calcinosis | 129 | 0.9954 | 0.9930 | 0.6368 | 0.0024 |
| Scoliosis | 24 | 0.9916 | 0.9802 | 0.7138 | 0.0113 |
| Atherosclerosis | 33 | 1.0000 | 1.0000 | 0.7466 | 0.0000 |
| Normal | 553 | 0.9876 | 0.9853 | 0.6951 | 0.0022 |

### Summary

| Metric | Mean AUC |
|---|---|
| `real_real` (full model) | **0.9833** |
| `blankT+realI` (image branch alone) | **0.9824** |
| `realT+blankI` (text branch alone) | **0.7648** |
| Δ (real − blankT+realI) | **0.0010** |

### Interpretation

- Removing the text entirely — while keeping the image representation as it was produced during a forward pass that *did* see the real report — costs essentially **zero AUC** (Δ ≈ 0.001, and negative i.e. slightly *better* for several labels: Atelectasis, Cardiomegaly, Edema, Emphysema, Fibrosis, Nodule, Fracture).
- The text branch alone (paired with a content-free image) is much weaker and more variable (mean AUC 0.7648, range 0.60–0.93) — still well above chance, so reports do carry real signal, but clearly secondary.
- Together this indicates the dominant discriminative signal lives in the **image representation slot**, but only because cross-attention with the real report contaminated it during the forward pass — not because the visual features are independently that informative.
