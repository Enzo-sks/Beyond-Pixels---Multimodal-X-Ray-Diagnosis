# Project Tracking

This document is used to record sessions, tasks, and observations. Simply fill in the corresponding sections.

## General Information

!!!! When it said "Activities Completed before session" : it is what has been done between the seesion before and this one

- **Project Name: Beyond Pixel**
- **Project Lead: AHMED SAID Djouhoud**
- **Start Date: 11/02**
- **Expected End Date: No fixed end date, continuous improvement approach**
- **Member 1 : enzo**
- **Member 2 : Djouhoud**
- **Member 3 : Hugo**
- **Member 3 : Aziz**
- **Member 4 : Elias**

---

## Objectives Checklist

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Objectives</title>
</head>
<body>

  <h3>Short-term Objectives (quick wins)</h3>
  <ul>
    <li><label><input type="checkbox"> Understand the project goal and the</label></li> 
    <li><label><input type="checkbox">learn and understand how all  libraries work </label></li>
    <li><label><input type="checkbox"> Set up the working environment (Git structure, folders, notebooks) </label></li>
    <li><label><input type="checkbox"> Split learning topics: PyTorch, MONAI, scikit-learn, NumPy, pandas </label></li>
    <li><label><input type="checkbox">Read the documentation provided by the supervisor </label></li>

  </ul>

  <h3>Achievable Objectives (reasonable commitment)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined after the learning phase  </label></li>
    <li><label><input type="checkbox">Build a clean dataset (match images with labels, handle NaN, normalize) </label></li>
    <li><label><input type="checkbox">Implement a first baseline model for image classification </label></li>
    <li><label><input type="checkbox">Build or Fine-tune a pretrained model </label></li>
    <li><label><input type="checkbox">Organize the code into reusable modules / scripts </label></li>
    <li><label><input type="checkbox">Document the current pipeline (preprocessing → model → evaluation) </label></li>
    <li><label><input type="checkbox"> Work with DICOM images using MONAI in a more advanced way</label></li>
    <li><label><input type="checkbox"> Explore multimodal models (image + text)</label></li>
  </ul>

  <h3>Advanced Objectives (long-term / complex)</h3>
  <ul>
    <li><label><input type="checkbox">to be defined </label></li>
    <li><label><input type="checkbox">Perform advanced hyperparameter optimization (Optuna, Ray Tune) </label></li>
    <li><label><input type="checkbox">Compare multiple architectures and write an in-depth analysis </label></li>
    <li><label><input type="checkbox">to be defined </label></li>
  </ul>

</body>
</html>

---

## Session Tracking

Sessions are listed by date.

### Session 1 [20/02/26]

**Session Objectives:**

Complete the planning and set objectives for each member of the team, starting with the short term

**Activities Completed:**

- Planning
- Division of tasks

**Decisions / Results:**

- The current schedule is not final; After the training phase it may change if necessary.

**Next Steps for the Following Session:**

- Learning about Machine Learning and Deep Learning Libraries
- Setting up a planning schedule that should be effective. The lack of knowledge but especially the fact of not knowing to what extent we will be capable of doing things (a problem also mentioned by our supervisor who advised us to do things little by little even if we don't go all the way) pushed us to create a very progressive planning with ultimately few details on the pure practice of what we will do. In theory and ideas, the planning is however very precise.

**Decisions / Results:**

- Planning.md

**Next Steps for the Following Session:**

- Review all the documentation provided by our supervisor

### Session 2 [23/02/26]

**Session Objectives:**

- Start reviewing the documentation

**Activities Completed:**

In this session, each person focused on where they wanted to start. Overall, everything will need to be completed:

- Enzo: Started watching the 24-hour video on PyTorch, a deep learning library.
- Djouhoud : started to learn about the MONAI project
- Aziz: Started watching videos about scikit-learn library
- Hugo: A lot about Numpy and panda library. Started learning about basic ML, and application with scikit-learn, through https://inria.github.io/scikit-learn-mooc/index.html. 

**Decisions / Results:**

- Enzo: Video not finished but well underway
- Aziz: understood the basics of linear models, SVMs, and SGD.
- Hugo : Understood the basic of linear models. I now am able to manipulate panda and numpy a bit better. Started through scikit, to learn how to analyze a basic (here 2d) datasets.

**Next Steps for the Following Session:**

- Enzo: Continue the video and also review certain points from the presented book: "https://scikit-learn.org/stable/" on machine learning
- Aziz: continue reading more about the scikit-learn library
- Hugo: continue with basic Ml and scikit.

## General Remarks

-

---

### Session 3 [13/03/26]

**Activities Completed before session:**

- Enzo: Already done before (vacation): Video DL finisehd. Write a first DL model with a dataset already existed.
- Hugo: Built an end-to-end text classification pipeline in `Text classification/first_phases.ipynb`
- Djouhoud: Djouhoud: use of the DICOM dataset API, start the treatment of the DICOM images 
- Aziz: Continued reading the DL book, completed a HF course on CV.
**Session Objectives:**

- Enzo: Start the traitement of the data
- Hugo: Implement a BERT-based text classifier (fine-tuning on the same dataset) to compare against this TF-IDF baseline

**Activities Completed:**

In this session, each person focused on where they wanted to start. Overall, everything will need to be completed:

- Enzo: (focus on png images only): create our custom dataset, sorting images into test/train folders based on labels. Start normalizing the images
-Aziz: Created a dataset where he linked the images to their respective XML files. He also added in the dataset hugo labels which scores the patient on 12 different diseases. `image_preprocess/merged_df_meta.csv`


**Decisions / Results:**

- Enzo: organization of the dataset
- djouhoud : focus on the dicom images 


**Next Steps for the Following Session:**

- Building a first model to train

## General Remarks
-Hugo label classification method is not great and it has room for improvement: in fact, it assigned NaN values to more than 2000 rows.
- enzo : it takes too much time to run the model on all the data, for the next session, we have to find a solution 

---

### Session 4 [16/03/26]

**Activities Completed before session:** -> meaning between the two session



**Session Objectives:**

-meeting

**Activities Completed:**


**Decisions / Results:**
-We decided to continue on our program basicly.
-enzo:  need to implement fully the pipeline he beggined
-aziz:  try to solve overfittting problem





### Session 5 [27/03/26]

**Activities Completed before session:**


**Session Objectives:**


-Discuss the work each one did so far
Enzo - Complete the model from start to finish and strive to be efficient when working on these models.

**Activities Completed:**

-Aziz: Built the first version of a computer vision model for multi-label classification: Fine-tuned a DenseNet-121 on the png dataset (7470 images). Achieved a score of 0.78 AUC
The model can be found in `image classification (png)\cv_model_01.ipynb` with the documentation in `image classification (png)\readme.md`

Enzo - The model we created works, but it’s not very efficient, so we’re going to switch to a more optimized model: we’ll take a look at PyTorch 
Enzo - Transfer it as a python file so it can more easily be used (can be found in image classification (png)\own created model) and also work on the structure itseld (OO)
-Hugo : I started building my own bert model to understand better the transformer architectiure. For that i used d2l.ia very good step by step guide to build an attention mecanism and then a (quite modern ! transformer), i implemented (without always understanding everything of course but still) multiheadedattention, Rope, add and rms norme.

**Decisions / Results:**
-Hugo : I need to clean my repo

**Next Steps for the Following Session:**

Aziz: 

-Work on improving the CV model through changing the hyper-parameters;

-Fine-tuning an open source model already trained on medical images like the HF model: `codewithdark/vit-chest-xray`

- enzo : ask to the supervisor wether it is a good idea to continu developping our own model or focus on fine-tuning
## General Remarks

### Session 6 [03/04/26]

**Activities Completed before session:**
-hugo : continue and finish with the bert classifier
- enzo : did some research on how improving our own model

**Session Objectives:**

Djouhoud:
- Create a comprehensive mapping system linking images to their diagnostic labels.
- Reorganize the `projet_dicom/` module for better modularity and maintainability.
- Convert medical DICOM images to standard RGB PNG format for model training.
- Fix alignment and synchronization issues between images and their labels.
Enzo:
- Developing and improving our own model in an effort to create something that runs in a reasonable amount of time and works reasonably well. For now, the focus is primarily on simply optimizing the model to achieve satisfactory results.
Hugo:
-engineer the prtrain, find a bigger cxray datasets to pretrain 

**Activities Completed:**

- enzo : the model has been improved, more efficient with more layers and a more optimized augmentation
-hugo : Following once again the very good chap 15 of d2l.ai, i built a pretrain functiun on a portion of MIMICXR datasets (29k)
-hugo : Fine tuned the model on open-i

-Aziz: Worked on improving the vit model using MultilabelStratifiedShuffleSplit, Stronger data augmentation, using asymmetric loss, and a richer classification head. Achieved a score of AUC: 75%


-djouhoud :
- Applied Hounsfield unit scaling and windowing technique to preserve diagnostic information in chest X-rays
- Windowing parameters properly calibrated for pulmonary imaging (window center: 40 HU, window width: 400 HU)
- Batch converted part of  DICOM files in `fragment_data_set/` in the drive  to high-resolution PNG images
- Created logical folder structure: raw inputs → processed data → analysis → outputs
- Separated `fragment_data_set/` (medical DICOM files), `NLMCXR_reports/` (annotations), and output directories
- Implemented clear separation of concerns for preprocessing, analysis, and model training pipelines
- Added comprehensive documentation in README files
- Built `png_label_mapping.csv` establishing bidirectional relationships: image_id ↔ png_filename ↔ 21 pathology labels
- Integrated XML metadata from NLMCXR dataset containing clinical annotations
- Mapped 21 distinct pathologies: Atelectasis, Cardiomegaly, Effusion, Pneumonia, Pneumothorax, Edema, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Hernia, Fracture, Pleural_Thickening, Opacity, Consolidation, Granuloma, Calcinosis, Scoliosis, Atherosclerosis, Normal
- Corrected filename mismatches between DICOM originals and PNG conversions
- Fixed label assignment errors from XML parsing
- Verified image-label correspondence across entire dataset
- Implemented validation checks to prevent future misalignments

**Decisions / Results:**
-Very very Satisfying result on the bert classifier, i obtain an auc score of .94 which is not so far from the .97 that we aim for. Although my model overfitt because i need to train all the layer during fine-tunning as my model is quite small (5m param/300mfor cxr-bert)

-Aziz: The model is no longer overfitting thanks to the new loss function. I got similar results as the previous model.


**Next Steps for the Following Session:**

- enzo : the model now needs to be optimize in a deeper way 
- Hugo : fine-tune CXR-bert (microsoft) and compare with my model 

---


### Session 6 [10/04/26]


**Activities Completed before session:**
Enzo: Implement torch.lightning to the codes so that it is easier to use the GPUs. exam revision
hugo : exam revision.
djouhoud: improve the dicom  model

**Session Objectives:**
Enzo : nothing due to the exams
hugo : nothing due to the exams.


**Activities Completed:**
-Aziz: Worked on preparing the presentation for the intermediate evaluation.
-hugo : exam period, nothing substantial. On the side, started reading about how text and image can be combined (early vs late fusion, joint embeddings) to prepare the multimodal phase.

Elias :
- understand how hugo's test model works
- help djouhoud with dicom - understand what he's doing
- learn how to use telecom's GPU

**Decisions / Results:**

**Next Steps for the Following Session:**

### [9/04/2026]


**Session Objectives:**


- understand how hugo's test model works         (same as yesterday)
- help djouhoud with dicom - understand what he's doing (same as yesterday)


- design dicom API
- train multimodal head
**Activities Completed:**


- nothing, I try to catch up and understand everyone's problem
- I tried to upload dicom dataset on telecom server


**Decisions / Results:**


- we didn't need an API for the dicom dataset because I discovered that we could upload the whole dataset on telecom server so it can train with the gpu.
I got my IP address banned because of too much API I requested on google drive to upload all the dicom
I think we will be forced to download on hardware the dataset so that we can upload it without any API.


**Next Steps for the Following Session:**


- find a solution on how to upload 100GB dataset
- I want to work more on models and less on data
- watch 24h video on pytorch
---


### [10/04/2026]


**Session Objectives:**
- understand Aziz and Hugo model to build the multimodal head to use png and text

**Activities Completed:**
- no

**Decisions / Results:**
- I need to learn more about ML and PyTorch to understand

**Next Steps for the Following Session:**

- Learn more about PyTorch

### Session 7 [15/04/26]


**Activities Completed before session:**
Enzo: Try to implement finetuning (better) in our own created model (juste to see, btw we will switch to the model developped with Aziz) to compare and se the differences with a model not so good, exam revision
hugo : exam revision.

**Session Objectives:**
Enzo : nothing due to the exams
hugo : nothing due to the exams.
djouhoud: nothing due to the exams

**Activities Completed:**
-hugo : still exam period. Kept surveying the multimodal fusion literature on the side, building a first mental map of the main families of approaches (concatenation, cross-attention, single-stream transformers) before diving deeper next sessions.



**Decisions / Results:**

**Next Steps for the Following Session:**
hugo : start a serious, structured bibliography on multimodal fusion methods.

### Session 8 [05/05/26]

**Activities Completed before session:** 
Enzo : Watched the video about multimodel implementation : 
Stanford CS224N NLP with Deep Learning | 2023 | Lecture 16 - Multimodal Deep Learning, Douwe Kiela
Start reading the article mentionned in the video : Learning Transferable Visual Models From Natural Language Supervision

Aziz: Used also the learning material of Stanford CS224N to learn about multimodal implemebtatiob.
Read several blog posts about techniques we can use of the multimodal implementation
Djouhoud: Downloaded the dataset to the Telecom GPU server and verified the total number of DICOM files.
hugo : finished the exams. Started a structured bibliography on multimodal fusion, reading the foundational papers (CLIP for contrastive image-text alignment, ViLBERT/LXMERT for dual-stream cross-attention).


**Session Objectives:**
 - Put averything together and for the planning for the next steps
Djouhoud: Preprocessed DICOM files using the multi-window windowing method on the Telecom GPU server.
hugo : compare the main fusion paradigms and figure out which one fits our image+report setup best.
**Activities Completed:**
Enzo : discuss about what I've learned with the others and focus on the planning for the next sessions
Aziz: Read online blogs about multi-model implementation "Building a Multimodal Classifier in PyTorch: A Step-by-Step Guide".
hugo : went deeper into the fusion taxonomy — early fusion (concat of features), late fusion (decision-level), and intermediate fusion via cross-attention. Took notes on the trade-offs (parameter count, where the two modalities actually interact, alignment requirements). Identified cross-attention as the most promising direction for our case.

**Decisions / Results:**
We decided to work one more week on the Deecom but give up if it still doesn't work, for the Fusion Phase,we shared what we know and organize what people are going to do 
Djouhoud: Encountered disk quota limitation during DICOM preprocessing. The output directory exceeded available storage on the Telecom GPU server.
**Next Steps for the Following Session:**
Enzo: try to think about a simple API
Aziz: Help Elias in the multi-model implementation.
Djouhoud: Explore alternative DICOM processing technique using 3D volumetric representation.
### Session 8 [13/05/26]

**Activities Completed before session:** 


Enzo : Made a simple API and tried if it works with enzo's model : https://test-api-2-production.up.railway.app/docs   . The Api take in consideration the model according to the dictionnary containing all the values each neuron has after training the model. To use the API with another model, we only need to change the `model_full.pth`. The API use `Dockers`, to containes things and the deployement has been mage using `Railway`

Aziz: 
- Read papers on multimodal fusion via bidirectional cross-attention:
  - **LXMERT** (Tan & Bansal, EMNLP 2019): two separate encoders (text + image) that mutually enrich each other via cross-attention, with a final fusion layer
  - **ViLBERT** (Lu et al., NeurIPS 2019): dual-stream architecture with cross-attention between visual and text tokens, a precursor to bimodal approaches
  - Explored **MONAI (NVIDIA)** as a potential framework for medical data handling and multimodal training

**Session Objectives:**
Enzo: work with the others with the multimodal goals.




**Activities Completed:**
djouhoud: -  fine-tuning of a DenseNet-121 model using a preprocessed dataset (windowing applied and reduced image size).
- Training was performed for half of the data set and 5 epochs to test whether the model can run and converge successfully.
- start to preprocess the dataset for the volume methode 
enzo : discuss about how we should do the multimodal implementation. Read about that

Aziz: 
- Implemented `multimodal_fusion/` module combining the custom text encoder (BERT MLM, d=256) and ViT (`codewithdark/vit-chest-xray`, d=768) via bidirectional cross-attention projected to d=512
hugo : continued the bibliography and synthesized everything into a written document `docs/architectures_fusion_multimodale_v3.md`, an extensive review of the fusion methods (single-stream vs dual-stream, cross-attention variants, Q-Former / BLIP-2 style querying, contrastive pretraining). The goal is to give the whole team a shared reference to choose our architecture from.
Aziz: -Transfered his code from the main branch to the new refactored branch created by hugo before the final merge.

**Decisions / Results:**
Djouhoud: The windowing method is not suitable for Google Colab, as it significantly increases training time even when using only half of the dataset, and the resulting metrics are not satisfactory.
hugo : the document converges on two candidates for my part: bidirectional cross-attention (close to what Aziz started) and a Q-Former style module that learns a small set of query tokens to extract image information conditioned on text.
Aziz: Satisfied with the initial implementation and excited to keep working on finishing the code.

**Next Steps for the Following Session:**
enzo : continue to think of ways to improve our multimodal implementation.
djouhoud : focusing in the volume methode
hugo : clean up and restructure the repo so the multimodal work has a proper home, then start my own fusion implementation.
Aziz: Start brainstorm ideas for the poster design+keep working on the multi-modal implementation.

### Session 9 [27/05/26]

**Activities Completed before session:** 
enzo : improve the model : The code trains a multimodal model that combines a chest X‑ray processed by EfficientNet‑B0 with TF‑IDF text features from clinical reports. The two embeddings are concatenated and passed through a fusion MLP that predicts 14 thoracic pathologies. Training happens in two stages: first the fusion head is learned while the image branch is frozen, then the entire network is fine‑tuned ----> see the branch enzo_multimodal for now.
hugo : started the big refactor of the repo (the codebase had grown into a pile of loose notebooks and folders, hard to reuse across the team).

**Session Objectives:**
enzo : use the .pth of the other's model to check if everything works with the model done.
hugo : turn the scattered scripts/notebooks into a clean, installable project.

**Activities Completed:**
hugo : did a full refactor towards a monorepo structure. Created the `mmmia` package under `src/mmmia/` (text encoder, image encoder, fusion, pretraining utilities), moved the figures into `docs/`, and integrated the existing `multimodal_fusion` work into the package instead of having it live as standalone notebooks. Cleaned up imports so every member can `import mmmia` instead of copy-pasting code between notebooks.

Aziz: Worked on writing the code for the bidirectionnal multi modal approach. In particular, chose to write python scripts instead of jupyter notebooks to have production level code. Started running the code on Colab GPU.


**Decisions / Results:**
hugo : the project is now a proper Python package; this is the new shared base for everyone (branch `refactor/clean-architecture` then merged). Still need to regenerate a complete lockfile so installs are reproducible.



**Next Steps for the Following Session:**
hugo : finish stabilising the install (requirements lock) and start my own fusion implementation. Also investigate the suspiciously high AUC scores we are getting.

Aziz: Finish the multi-modal implementation and get the results.

### Session 10 [10/06/26]

**Activities Completed before session:** 
enzo : use the .pth of the other's model to check if everything works with the model done. It works, not so good, AUC very low.
hugo : finalised the refactor — regenerated a complete `requirements-lock.txt` with install instructions so the project is reproducible for everyone.

**Session Objectives:**
elias : again learning about single stream transformer in order to implement one multimodal head.
enzo: create the flyer for the presentation
hugo : investigate the suspiciously high ~0.96–0.99 AUC of the fusion model (possible data leak), then start my own fusion architecture.



**Activities Completed:**

enzo : the canva of the Flyer has been created and some information has already been written.
hugo : started digging into the potential data leak between MIMIC-CXR (the CXR set used to pretrain the encoders) and Open-i (our eval set). The almost-perfect AUC (several classes at 0.99–1.00, mean ~0.98–0.96) is too good to be true for chest X-ray multi-label classification, and a leak between the pretraining and evaluation distributions would exactly explain it. Began checking for overlap (same patients / studies leaking across the pretrain→finetune split) and how the train/test split is built. In parallel, started my own part of the multimodal fusion: a first implementation of a Q-Former architecture (a small set of learnable query tokens that attend over the image features to extract a compact, text-conditioned visual representation), as an alternative to the bidirectional cross-attention fusion.

elias : Implemented the initial Single Stream Transformer architecture 
Cleaned up the tree structure 
Performed minor codebase optimizations

djouhoud: setup the telecom's volume  for the dicom volume preprocess. 
Solve the git access problem 

Aziz: Built the `multimodal_fusion/` module that combines a custom BERT text encoder and a Vision Transformer (ViT) image encoder into a single multi-label classifier for 21 chest pathologies, using bidirectional cross-attention as the fusion mechanism.
**`BidirectionalCrossAttention`**

Both modalities are projected to a common space (d=512) before attention:

- Text tokens `(B, L, 256)` → Linear projection → `(B, L, 512)`
- Image tokens `(B, 197, 768)` → Linear projection → `(B, 197, 512)`
- Two cross-attention directions computed **in parallel**:
  - Text → Image (`t2i`): text queries attend over image keys/values
  - Image → Text (`i2t`): image queries attend over text keys/values
- Each branch has a residual connection, FFN (512 → 1024 → 512), and LayerNorm

**`MultimodalFusion`**

| Component | Details |
|-----------|---------|
| Text encoder | Custom BERT (d=256, 6 layers, 8 heads, RoPE+RMSNorm+SiLU) — 4.3M params |
| Image encoder | `codewithdark/vit-chest-xray` ViT-B/16 (d=768, 197 tokens) — 86M params |
| Cross-attention | BidirectionalCrossAttention, 8 heads, d_model=512 |
| Text pooling | CLS token (index 0) of fused text sequence |
| Image pooling | Learned soft-attention over all 197 fused image tokens |
| Classification head | Linear(1024 → 512) + LayerNorm + GELU + Dropout + Linear(512 → 21) |
| **Total parameters** | **95,931,925** |


### Results

**Training curve (key epochs):**

| Epoch | Phase | Val AUC |
|-------|-------|---------|
| 1 | Frozen | 0.6247 |
| 3 | Frozen | 0.9206 |
| 4 | Unfrozen | 0.9596 |
| 8 | Unfrozen | 0.9879 ★ |
| 15 | Unfrozen | 0.9881 ★ best |
| 22 | Unfrozen | early stop |

**Per-class AUC on test set (968 samples):**

| Pathology | AUC |
|-----------|-----|
| Atelectasis | 0.9875 |
| Cardiomegaly | 0.9887 |
| Effusion | 0.9731 |
| Pneumonia | 0.9670 |
| Pneumothorax | 0.9969 |
| Edema | 0.9922 |
| Emphysema | 0.9604 |
| Fibrosis | 0.9550 |
| Infiltration | 0.9839 |
| Mass | 0.9946 |
| Nodule | 0.9918 |
| Hernia | 0.9997 |
| Fracture | 0.9922 |
| Pleural_Thickening | 1.0000 |
| Opacity | 0.9999 |
| Consolidation | 0.9936 |
| Granuloma | 0.9999 |
| Calcinosis | 0.9974 |
| Scoliosis | 0.9950 |
| Atherosclerosis | 1.0000 |
| Normal | 0.9920 |
| **Mean AUC** | **0.9886** |

**Best val AUC: 0.9881 — Test AUC: 0.9886**

Checkpoint saved at: `multimodal_fusion/checkpoints/multimodal_fusion.pt`
More documentation can be found on `multimodal_fusion/checkpoints/README.md`


Aziz: Worked on the poster design using Canvas, created figures that represent the results of his multi modal approach which can be found on the folder `figures`

**Decisions / Results:**
hugo : strong suspicion that the ~0.96+ AUC is at least partly inflated by a leak between MIMIC-CXR and Open-i — needs to be confirmed before we trust/communicate these numbers. The Q-Former branch is at an early prototype stage.

Aziz: Happy with the results of the multi-modal bidirectional cross attention approach. 

**Next Steps for the Following Session:**
elias : implementing this single stream transformer (see branch1 before merging) 
enzo : complete the flyer and finalise the API with the new model Aziz made.
hugo : confirm (or rule out) the MIMIC-CXR / Open-i leak and re-evaluate the fusion model on a clean split; continue the Q-Former implementation.
Aziz: Continue working on the poster

### Session 11 [15/06/26]

**Activities Completed before session:** 

**Session Objectives:**

**Activities Completed:**
-Aziz: Worked on the poster along the other teammates, Made sure it was submitted before tonight's deadline.
-We also changed the date of our meeting with the professor from this friday to Tuesday 23rd.
- Enzo: finished the poster with the consideration of Nikolai, and sent it. 
-djouhoud : learn about the implementation 
**Decisions / Results:**
-Poster sent at time and reviewed by the professor 

**Next Steps for the Following Session:**
-Aziz: Prepare a presentation for the next meeting with the professor
-Enzo: Prepare a presentation for the next meeting with the professor

### Session 12 [22/06/26]

**Activities Completed before session:** 
- djouhoud: Studied the architecture and prepared access to the GPU cluster.

**Session Objectives:**
- djouhoud: Deploy and train a 3D model on the DICOM volumes dataset for multi-label classification (21 pathologies) using the P100 GPUs on the Telecom Paris cluster.

Enzo : try to implement a model using lmm to compare with the other model 

Aziz: 
- Assess whether the model focuses on clinically relevant image regions.
- Evaluate the consistency of localization across different pathologies and patients.
- Investigate the contribution of the text modality to the fused image representation and final predictions.

**Activities Completed:**
Enzo : I implemented a multimodal model inspired by LLaVA, designed for multi‑label chest X‑ray classification with 21 output labels. The system combines a frozen ViT encoder, a trainable two‑layer projector MLP, and the Gemma 3 1B language model equipped with LoRA adapters. The ViT produces 197 image tokens of dimension 768, which are then projected to the 2048‑dimensional embedding space of Gemma. These projected image tokens are prepended to the text embeddings before being processed by Gemma. The LLM remains frozen except for LoRA applied to the q_proj and v_proj modules (rank 16, alpha 32, dropout 0.05). After Gemma processes the mixed sequence, the final causal token is extracted and passed through a linear classification head mapping 2048 to 21 logits. Only the projector, the LoRA parameters, and the classification head are trainable; the ViT and the rest of Gemma remain frozen.

Aziz: 
- Analyzed attention heatmaps for Pneumothorax, Cardiomegaly, Effusion, and Normal cases.
- Evaluated Grad-CAM explanations on individual samples.
- Performed text/image ablation experiments using real and blank reports.
- Measured cosine similarity between fused image representations obtained with real versus blank text inputs.
- Compared multimodal, image-only, and text-only prediction probabilities.

- djouhoud: 
  - Configured the virtual environment (`requirements.txt`) and the SLURM job script (`job.sh`) on Telecom's `gpu-gw` cluster.
  - Implemented and tested several iterations of the 3D training pipeline:
    - **v4**: Implemented a multi-criteria stratified split, but the model suffered from severe under-prediction.
    - **v5**: Tested the MONAI architecture (3D-ResNet18) from scratch with dynamic thresholds. Failed to converge due to the lack of pre-trained weights.
    - **v6**: Reverted to torchvision's `r3d_18` with pre-trained Kinetics-400 weights. Added a 3-phase progressive unfreezing schedule, AUC-based model selection, and Asymmetric Loss. Global AUC-ROC increased to 59.2%.
  - Set up comprehensive monitoring tools (`eval.py`): saved training history, generated per-pathology confusion matrices, and plotted confusion heatmaps.
  - Diagnosed a fundamental issue in the initial DICOM preprocessing (soft tissue windowing instead of lungs, and resolution too low at 128x128).
  - Created a new Colab preprocessing pipeline to extract 224x224x64 volumes with a proper "Lung Window", along with metadata extraction (age, sex).
  - Implemented **v7**: Integrated 3D MONAI transforms (aggressive data augmentation) and removed 4 extremely rare classes to stabilize training.

**Decisions / Results:**
- djouhoud: The transfer learning strategy (Kinetics) and progressive unfreezing were validated. The initial performance bottlenecks mainly stemmed from incorrect DICOM windowing. The new 224x224 dataset is ready to be used on the cluster.

Aziz:
- Attention maps showed some variation across pathologies but frequently highlighted image borders or non-anatomical regions.
- Grad-CAM explanations exhibited weak and inconsistent localization, with limited alignment to expected pathology locations.
- Cross-attention significantly altered the fused image representation (cosine similarity: 0.208 for Pneumothorax, 0.098 for Cardiomegaly, 0.053 for Effusion), indicating strong sensitivity to textual input.

**Next Steps for the Following Session:**

-Aziz:

-Work on preparing the presentation for tomorrow meeting with the professor

### Session 13 [23/06/26]

**Activities Completed before session:** 

**Session Objectives:**
Aziz:
-Finish the cross-modal ablation experiments  
-Prepare the pptx presentation  

Enzo: 
-Deploy the API locally and online  
-Create a web interface to access the model  
-Present the deployment workflow to the supervisor  

Elias  : 
- Connect text and image backbone models
- Establish dataset loading and job automation routines for training the SST network
- Set up and write the dedicated pipeline training loop

**Activities Completed:**
Aziz: 
-Worked on the cross modal ablation where I calculated the cosine similarity between the image representation with real medical report vs blank text.  
-Also calculated the probabilities of measuring an anomaly given the real/blank image representation associated with the real/blank medical report.  
-Prepared the powerpoint presentation where I talked about the image classification models I built along with the multimodal bidirectional cross attention architecture with interpretability part.  
-Coordinated with other members to finish the presentation on time and assisted them with a canva template I prepared.

Enzo: 
-Set up the full inference API using FastAPI, first running locally through Docker to validate the environment.  
-Deployed the API on Railway, connected to GitHub for CI/CD and to HuggingFace Hub to store the heavy `.pt` model weights.  
-Created a simple HTML/JS website hosted on GitHub Pages allowing users to query the API easily.  
-Tested end‑to‑end communication between the website and the API (CORS, JSON payloads, error handling).  
-Presented the deployment pipeline and user workflow to the supervisor.

Elias :

- Retrieved text and PNG base architectures to prepare feature fusion ("récupération des modèles text et png")

- Created and configured the SLURM/bash training configuration script ("ajout job pour train la tête")

- Integrated the specific SST architecture skeleton ("ajout architecture SST")

- Loaded and leveraged pretrained BERT and ViT (Vision Transformer) weights into the models ("utilisation des Bert et Vit pretrained")

- Consolidated and structured the training data source directory ("ajout dataset pour SST")

- Coded and implemented the full python execution module for the model's training phase ("écriture du train afin de train le SST")

**Decisions / Results:**

Aziz:  
-Results show that the image representation combined with real medical report carry the highest signal used by the ML model. The cross attention particularly (t2i) allowed the image to carry the necessary text signal needed to get excellent results. (98.9 AUC)  
-Presentation went great and we are proud of what we did so far !

Enzo:
-The API is fully functional both locally and online.  
-The GitHub Pages website successfully interacts with the Railway API.  
-HuggingFace hosting solves the large‑file issue.  
-The deployment pipeline is stable and ready for public use.

Elias :

Utilizing industry-standard pretrained models (BERT and ViT) will significantly save computational time and boost initial diagnostic/multimodal performance

Separating the dataset injection logic from the network parameters ensures cleaner, modular execution

**Next Steps for the Following Session:**

Aziz:  
-Compare my multi modal with Enzo and Elias. For now, mine outperforms Hugo's model (98.9 AUC > 97 AUC.)

djouhoud:  
-Launch the v7 training on the cluster with the new 224x224x64 volumes. Analyze the new confusion matrix, and initiate the transition toward a multimodal model integrating metadata (Age/Sex) with the 3D volume.

Enzo:  
-Prepare cluster access to train the model on Telecom’s P100 GPUs.  
-Add QR codes and example images to help users test the API easily.

Elias :

Monitor training behavior, evaluate initial loss curves, and perform validation checks
Debug and optimize computational/GPU bottlenecks in the training run if any arise

---

### Session 14 [24/06/26]

**Activities Completed before session:**
- djouhoud: Prepared the 224x224x64 volumes with the advanced preprocessing (HU clipping, lung windowing) to fix the issues encountered in v6.


**Session Objectives:**
- djouhoud: Launch the v7 training on the cluster with the new 224x224x64 volumes, analyze the results, and finalize a robust MLOps pipeline.

Enzo:  
-Find a viable training strategy for the multimodal model on limited GPU resources.

Elias : 
- Adjust data paths for image loading in the training pipeline
- Optimize and debug the cluster cluster job execution settings (job.sh)

**Activities Completed:**
- djouhoud: 
  - Deployed the v7 pipeline on the Telecom cluster using the new high-resolution 224x224x64 volumes with Lung Windowing and adaptive padding.
  - Integrated native 3D Data Augmentation via MONAI (RandFlip, RandRotate90, RandZoom, RandGaussianNoise) in the PyTorch dataset.
  - Implemented the 3-phase progressive unfreezing schedule on the ResNet-18 3D (Kinetics-400 pre-trained) with Asymmetric Loss (gamma_neg=4, gamma_pos=1).
  - Developed cluster robustness scripts: `filter_csv_cluster.py` to automatically purge missing files, and `map_labels.py`.
  - Refined `eval.py` to calculate dynamic optimal thresholds per pathology based on maximizing the F1-Score, and generation of precise confusion heatmaps.

- Aziz:
 - Completed the cross-modal mediation analysis on the test dataset on every pathology out of the 21 labels.
 - Started working on the report in OverLeaf

Enzo:  
-Connected to the Telecom Paris GPU cluster and configured the environment for training.  
-Adapted the multimodal architecture to use **Gemma 4B**, a lighter LLM compatible with P100 GPUs.  
-Launched training runs and validated that the model trains correctly despite reduced LLM capacity.  
-Created QR codes and a Drive folder with example medical images to allow users to test the API easily.  
-Improved the GitHub Pages website by integrating these resources.

Elias :
- Modified train.py to target the specific Png directory for image retrieval ("modif du train.py afin qu'il cherche les png dans le dossier Png")
- Refactored the core execution shell script ("changement du job.sh")
- Tuned resource constraints by modifying the allocated runtime in the job script ("chgt temps alloué dans le job")
- Iterated on iterative job setup adjustments and configuration fixes ("chgt job" and "job")

**Decisions / Results:**
- djouhoud: 
  - The v7 is a fully realized industrial MLOps pipeline.
  - Despite the Lung Windowing and high resolution, the AUC actually dropped from 59% to 50%.
  - The multi-threshold dynamic evaluation significantly balances precision and recall compared to a naive 0.5 threshold. 
  - The dataset pipeline is now totally secure against missing files, enabling stable remote training.

- Aziz:
  - Results confirm the hypothesis I got on 23/06: image representation influenced by the medical report carry the most important signal and information used by the model to make its predictions.

Enzo:  
-The Gemma 4B model trains successfully on P100 GPUs.  
-The API is now easier to use thanks to QR codes and example datasets.  
-The deployment ecosystem (API + website + examples) is ready for demonstration.

Elias :

- Updated data-loading paths to guarantee the training loop successfully locates the PNG image inputs
- Extended or optimized the job script's allocated time to prevent timeout errors during long training epochs on the cluster

**Next Steps for the Following Session:**
- djouhoud: Initiate the transition toward a multimodal model integrating metadata (Age/Sex) with the 3D volume.
-launch the train with the windowing methode on the telecom gpu 

- Aziz: 
 - work with other members to prepare tomorrow's presentation
 - continue the report with the help of other teammates

Enzo:  
-Run full training on the cluster and evaluate performance.  
-Start preparing the final presentation video.

Elias :
- Launch and monitor the updated training job to ensure it runs to completion without path errors or time failures
- Evaluate the first metrics generated by the Single Stream Transformer loop

---

### Session 15 [25/06/26]

**Session Objectives:**
- djouhoud: Combine v1 and v2 datasets, fix data loader issues, and relaunch the 3D model training on the cluster.

Enzo:  
-Train the multimodal model on the cluster and prepare final presentation materials.
Aziz:
-Prepare the presentation video
-Complete the written report
-Help other teammates in case they need help
Hugo:
-Build a label-aligned multimodal fusion (chest X-ray image + radiology report) and test the core hypothesis: can a fully multimodal training improve a more *unimodal* inference (image + indication only, no findings) ?

**Activities Completed:**
- djouhoud:
  - Created a combined dataset (v1 + v2) for 3D training (`dataset_labeled_volumes_3d_combined.csv`).
  - Modified `dataset.py` to handle both dataset versions and dynamically resize all volumes to `64x224x224`.
  - Fixed MONAI transforms tensor output compatibility in the dataloader.
  - Updated the SLURM `job.sh` and successfully relaunched the training on the Telecom cluster, generating new evaluation metrics and heatmaps.
  - Wrote new notebooks for resumable Colab preprocessing and PNG-to-NPY conversion for the windowing methode

Elias : 
- merged my branch into the main
- start training the sst

Enzo:  
-Trained the Gemma‑based multimodal model on the P100 GPUs.  
-Achieved an AUC of **0.97**, confirming good performance but still below the previous stronger model.  
-Created and edited a **presentation video** for the final demo.  
-Improved the GitHub Pages website UI and integrated the video and documentation.  
-Wrote the full section of the global report describing the model, training process, results, and deployment pipeline.

Aziz:
-Worked on the **presentation video** for the demo with other teammates.
-Converted ViT jupyter notebooks to python scripts.
-Proposed the idea of the written report and wrote my part on the image classification models, multimodal bidirectional cross-attention, and interpretability part.
-Tested my multimodal on Enzo's API.

Hugo: (catching up the tracking since my text-classification work — multimodal fusion line, `architecture_3`)
- Built `architecture_3`: a label-aligned **Q-Former** fusion — frozen CXR-BERT (text) + frozen ViT (image), 14 learnable queries (one per pathology), diagonal label-aligned head (`logit_j = w_j·H_j`), Asymmetric Loss. Factored a shared `common/` (paired dataset, grouped train/val/test split without leakage, losses, transforms).
- Interpretability: per-block query-collapse diagnostics, label-aligned cross-attention maps, a pre-norm anti-collapse variant, an **attention-pooled head** (puts the cross-attention in the gradient path so the maps are causally tied to the prediction), and a "deep" text branching (BLIP-2 style, BERT layer ↔ Q-Former block).
- Implemented **CGGM** (NeurIPS 2024, *Classifier-guided Gradient Modulation*) adapted to our **frozen-encoder** regime: only the **direction** modulation applies (magnitude has no encoder gradient to rescale since CXR-BERT/ViT are frozen). Composite loss `L_task + L_txt + L_img + λ·l_gm` with per-epoch coefficients derived from unimodal accuracy, auxiliary unimodal heads computed on text-only / image-only Q-Former passes (encoders run once, reused), and a 3-panel CGGM figure (accuracy / gradient magnitude / direction).
- Added a `--no_findings` flag: train multimodal on indication+findings, but **validate/test/infer on image + indication only** — so checkpoint selection matches the deployment scenario. All gated behind `--cggm` (checkpoint-compatible when off).

**Decisions / Results:**
- djouhoud: The combined dataset (v1+v2) is now functional, and the model handles heterogeneous input sizes dynamically during training. Achieved an AUC of 60%.

Enzo:  
-The Gemma 4B multimodal model reaches 0.97 AUC, validating the architecture but confirming that the earlier larger model performs better.  
-The final user-facing ecosystem (API + website + examples + video) is complete and ready for the final presentation.

Hugo:
- Ran CGGM + `--no_findings` (deep mode) on the cluster (P100): the **image + indication** inference reaches **AUC macro 0.70 / micro 0.76**. Visual pathologies rank well (Pneumothorax 0.875, Effusion 0.870, Cardiomegaly 0.81); low-image-signal ones collapse (Mass / Fibrosis ≈ 0.42) — the model leans on the image, the short indication does not recover the lost findings.
- Verified `codewithdark/vit-chest-xray` is a ViT fine-tuned on **CheXpert with a 5-class head** → its native head can't be reused for our 14 labels.
- Conclusion: the honest baseline is **internal** (`--radio_only`, same frozen pipeline), not the external image-only ViT (0.77 macro). The image-only path is already > 0.70, so beating it (and confirming CGGM's effect vs a no-CGGM ablation) needs dedicated runs — deferred to a separate session/branch.

Aziz: 
-Happy & proud with the results we have and excited for tomorrow's presentation.

**Next Steps for the Following Session:**
- djouhoud: Analyze the updated results from the cluster and continue towards multimodal integration (Age/Sex + 3D volume).

Enzo:  
-Finalize the presentation and rehearse the final pitch.

Hugo:
- Run the `--radio_only` internal image-only baseline and the no-CGGM ablation to isolate CGGM's contribution; explore architecture improvements (e.g. reusing a stronger pretrained image classifier, stronger modality balancing) to try to beat the image-only baseline — in a separate branch.

Aziz:
- Reharse the final pitch, print the written report, and have fun at tomorrow's presentation.
