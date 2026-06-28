# 1. Learned Image Pooling Attention

## What is it?

After the ViT processes the image, it produces a set of image patch embeddings:

\[
I_1, I_2, ..., I_N
\]

where each patch contains information about a small image region.

Before classification, the model needs a **single image representation**.

Instead of simply averaging all patches, the model learns attention weights:

\[
\alpha_1, \alpha_2, ..., \alpha_N
\]

and computes:

\[
I_{\text{pooled}}
=
\sum_i \alpha_i I_i
\]

The heatmap visualizes these learned weights \(\alpha_i\).

## Example

Suppose the model assigns the following weights:

| Patch | Weight |
|---------|---------|
| Left lung apex | 0.40 |
| Heart region | 0.10 |
| Right lung | 0.05 |
| Background | 0.01 |

Then the pooled image representation is dominated by the left lung apex.

The heatmap would therefore highlight that region.

## Important Limitation

This does **not** necessarily mean:

> "This region caused the Pneumothorax prediction."

It only means:

> "This region was important for building the image representation."

The same pooling attention pattern could potentially contribute to multiple pathology labels.

---

# 2. Text[CLS] → Image Cross-Attention

## What is it?

The text encoder (BERT) produces a special token:

\[
\text{CLS}
\]

which summarizes the textual information.

During cross-attention, the CLS token acts as a **query**:

\[
Q = \text{CLS}
\]

while image patches act as **keys** and **values**:

\[
K,V = I_1, I_2, ..., I_N
\]

The attention score for each image patch is:

\[
\text{Attention}(\text{CLS}, I_i)
\]

## What does it mean?

The map answers:

> "Which image patches does the text representation look at?"

or

> "Which image regions are most relevant to the text features?"

## Example

Imagine the text branch contains information strongly related to:

- Pleural effusion
- Fluid accumulation

Then the CLS token might attend more strongly to:

- Lower lung zones
- Costophrenic angles

than to other image regions.

## Intuition

### Image Pooling Attention

Answers:

> "What parts of the image are important overall?"

### Text[CLS] → Image Attention

Answers:

> "Given what the text branch knows, where does it look in the image?"