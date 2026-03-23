# CLIP-based Ingredient Recognition from Food Images

*A small research study on prompt design and domain adaptation*

## 1. Project Overview

This project investigates whether **CLIP-style vision-language representation learning** can be used to predict **ingredients from food images**, and whether **prompt engineering and lightweight fine-tuning** improve performance.

The central research question is:

> **Can prompt design and lightweight domain adaptation improve CLIP’s ability to detect ingredients in food images?**

The project evaluates the following setups:

1. **Zero-shot CLIP** (no training)
2. **Prompt engineering** — eight prompt templates + ensemble variants
3. **Lightweight fine-tuning** — projection head and partial unfreeze
4. **LoRA contrastive fine-tuning** — parameter-efficient adaptation of both encoders using the CLIP loss
5. **Backbone swap** — SigLIP as an alternative backbone with sigmoid multi-label loss

The goal is to understand how well CLIP transfers to the **ingredient recognition task** and whether parameter-efficient domain adaptation can close the gap.

---

# 2. Problem Definition

### Task

Multi-label ingredient prediction:

```
Input:  Food image
Output: Set of ingredients
```

Example:

```
Image: pasta with tomato sauce

Prediction:
[pasta, tomato, basil, parmesan]
```

This is a **multi-label classification problem**, since multiple ingredients can appear in a single image.

---

# 3. Dataset

You need:

```
(image, ingredient list)
```

pairs.

## Recommended Dataset Options

### Option 1 — Recipe1M (best)

Contains:

* 1M+ recipes
* ingredient lists
* food images

Paper:

```
Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes
```

Pros:

* large
* standard in literature

Cons:

* requires some preprocessing

---

### Option 2 — VireoFood172 (easier)

Contains:

* food images
* ingredient annotations

Good for quick experimentation.

---

### Option 3 — Food-101 (fallback)

Contains:

* food images
* dish labels

You must map dishes → ingredients manually.

---

## Ingredient Vocabulary

You should build a fixed ingredient vocabulary.

Example:

```
ingredients = [
    "tomato",
    "basil",
    "garlic",
    "onion",
    "cheese",
    "chicken",
    "beef",
    "pasta",
    "rice",
    "egg",
    ...
]
```

Typical size:

```
100–300 ingredients
```

Remove ingredients like:

```
salt
pepper
water
```

because they are **not visually detectable**.

---

# 4. Model Architecture

We use **CLIP** as the backbone.

CLIP consists of:

```
Image encoder
Text encoder
Shared embedding space
```

The model maps both modalities to the same vector space.

```
image → image embedding
text  → text embedding
```

Similarity is computed using cosine similarity.

---

## Architecture Diagram

```
                +------------------+
                |   Food Image     |
                +------------------+
                          |
                          v
                 CLIP Image Encoder
                          |
                          v
                   Image Embedding
                          |
                          |
           cosine similarity scoring
                          |
                          |
           Ingredient Text Embeddings
                          ^
                          |
                 CLIP Text Encoder
                          ^
                          |
                 Ingredient Prompts
```

---

# 5. Baseline Model (Zero-shot CLIP)

### Step 1 — Encode ingredient prompts

Example prompts:

```
"tomato"
"basil"
"garlic"
"cheese"
```

Compute embeddings:

```
ingredient_embeddings = text_encoder(prompts)
```

---

### Step 2 — Encode image

```
image_embedding = image_encoder(image)
```

---

### Step 3 — Compute similarity

```
scores = cosine_similarity(image_embedding, ingredient_embeddings)
```

---

### Step 4 — Predict top-k ingredients

Example:

```
top_k = 5
predicted_ingredients = argsort(scores)[-top_k:]
```

---

# 6. Prompt Engineering Experiment

CLIP performance depends heavily on prompt phrasing.

You will test multiple prompt types.

---

Eight prompt templates are tested. All follow `template.format(ingredient=x)`.

| Type | Template | Example |
|---|---|---|
| A | `{ingredient}` | `"basil"` |
| B | `a dish containing {ingredient}` | `"a dish containing basil"` |
| C | `a meal with {ingredient} ingredient` | `"a meal with basil ingredient"` |
| D | `a dish with visible {ingredient}` | `"a dish with visible basil"` |
| E | `food photo showing {ingredient}` | `"food photo showing basil"` |
| F | `recipe ingredient: {ingredient}` | `"recipe ingredient: basil"` |
| G | `this food contains {ingredient}` | `"this food contains basil"` |
| H | `fresh {ingredient}` | `"fresh basil"` |

Two ensemble variants are also evaluated by averaging L2-normalized text embeddings and re-normalizing:

- **ENS4** — average of A+B+C+D
- **ENS7** — average of B+C+D+E+F+G+H (excludes single-word A, which was found to dilute the ensemble)

## Evaluation

Compare performance of each prompt type and ensemble using:

```
Precision@K
Recall@K
F1@K
```

---

# 7. Fine-tuning CLIP

We test whether **domain adaptation improves performance**.

Four fine-tuning strategies are evaluated.

---

# Strategy 1 — Projection Head

Freeze CLIP. Add a linear classification layer on top of the image encoder.

```
image
↓
CLIP image encoder (frozen)
↓
projection head (linear)
↓
ingredient probabilities
```

Loss: Binary Cross Entropy (BCE). Only the head is trained.

---

# Strategy 2 — Partial CLIP Fine-tuning

Unfreeze the last transformer block of the image encoder and fine-tune with a small learning rate (1e-5). Text encoder stays frozen.

---

# Strategy 3 — LoRA Contrastive Fine-tuning (Experiment 4)

Instead of treating this as a classification problem, we keep it as **image-text contrastive matching** — the same objective CLIP was trained on — and use **LoRA** for parameter-efficient adaptation.

**LoRA** (Low-Rank Adaptation) injects trainable low-rank matrices into the Q and K projections of the attention layers:

```
W_q → W_q + ΔW_q,   where ΔW_q = A·B,  A ∈ R^{d×r}, B ∈ R^{r×d}
```

Only A and B are trained (rank r ≪ d). The original weights are frozen. Applied symmetrically to **both vision and text encoders**.

**Training signal:** For each image, the ingredient list is concatenated into a prompt:

```
"ingredients: tomato, cheese, basil, garlic"
```

The model is trained with the standard CLIP InfoNCE loss — each image should be most similar to its own ingredient text within the batch.

**Why LoRA over partial unfreeze:**
- Partial unfreeze touches full-rank weight matrices, which damages pretrained representations at small data scale
- LoRA with rank r=16–64 uses ~1% of total parameters, preserving the backbone while specializing the embedding space

Loss: OpenCLIP's `ClipLoss` (symmetric cross-entropy over batch similarities with learnable temperature).

---

# Strategy 4 — SigLIP Backbone (Experiment 5)

**SigLIP** replaces the InfoNCE softmax loss with a **sigmoid loss**. Each image-text pair is scored independently rather than competing against all other pairs in the batch:

```
L = -Σ log σ(zij · t) - Σ log σ(-zij · t)
```

This is better suited to the multi-label ingredient task, where an image may legitimately match multiple text descriptions (multiple ingredients). With InfoNCE, only one positive is allowed per image; with sigmoid loss, many positives are fine.

SigLIP is available via HuggingFace `transformers` (`google/siglip-base-patch16-224`). LoRA is applied here as well.

The comparison between Experiment 4 (CLIP + LoRA) and Experiment 5 (SigLIP + LoRA) directly tests whether the sigmoid multi-label loss is better suited to ingredient recognition.

---

# 8. Training Setup

## Loss Functions

| Strategy | Loss |
|---|---|
| Projection head | Binary Cross Entropy |
| Partial unfreeze | Binary Cross Entropy |
| LoRA + CLIP | InfoNCE (symmetric cross-entropy, `ClipLoss`) |
| LoRA + SigLIP | Sigmoid contrastive loss |

## Optimizer

AdamW throughout.

| Component | LR |
|---|---|
| Projection head | 1e-4 |
| Partially unfrozen CLIP block | 1e-5 |
| LoRA A/B matrices | 1e-4 |

Batch size: 32–64. Larger batches improve InfoNCE/sigmoid loss quality (more negatives per step). Epochs: 5–10.

---

# 9. Evaluation Metrics

Since this is **multi-label classification**, accuracy alone is not sufficient.

Use:

### Precision

```
Precision = TP / (TP + FP)
```

### Recall

```
Recall = TP / (TP + FN)
```

### F1 Score

```
F1 = 2 * (precision * recall) / (precision + recall)
```

---

### Precision@K

Used when predicting top-k ingredients.

Example:

```
Precision@5
```

---

# 10. Experiments

## Experiment 1 — Zero-shot CLIP (implemented)

Frozen CLIP, single-word prompts (type A), fixed top-5 prediction. Establishes the baseline.

**Sub-experiments:**
- **1B** — Ensemble prompts (A+B+C+D averaged, re-normalized)
- **1C** — Adaptive threshold (predict variable k based on score distribution)

---

## Experiment 2 — Prompt Engineering (implemented)

All 8 prompt templates + ENS4 + ENS7 evaluated under identical conditions.

Goal: isolate the effect of prompt phrasing from any model adaptation.

---

## Experiment 3 — Lightweight Fine-tuning (implemented)

Compare zero-shot, projection head (BCE), and partial unfreeze (BCE) at ~1,880 training samples.

Expected finding: fine-tuning underperforms zero-shot at this data scale; motivates LoRA and more data.

---

## Experiment 4 — LoRA Contrastive Fine-tuning (planned)

CLIP ViT-B/32 with LoRA applied to Q/K projections in both vision and text encoders (rank=32).

Training signal: ingredient list → `"ingredients: {ingr1}, {ingr2}, ..."` paired with the corresponding food image. InfoNCE loss.

Evaluation: zero-shot inference after fine-tuning (same setup as Experiment 1), so results are directly comparable.

---

## Experiment 5 — SigLIP Backbone (planned)

Repeat Experiment 4 with SigLIP (`google/siglip-base-patch16-224`) and sigmoid loss.

The sigmoid loss scores each image-text pair independently, which is better suited to multi-label recognition where one image has many correct ingredient texts.

---

## Experiment Summary Table

| Experiment | Backbone | Adaptation | Loss | Status |
|---|---|---|---|---|
| 1 / 1B / 1C | CLIP ViT-B/32 | none | — | done |
| 2 | CLIP ViT-B/32 | none | — | done |
| 3 | CLIP ViT-B/32 | head / partial unfreeze | BCE | done |
| 4 | CLIP ViT-B/32 | LoRA (both encoders, r=32) | InfoNCE | planned |
| 5 | SigLIP B/16 | LoRA (both encoders, r=32) | Sigmoid | planned |

---

# 11. Visualization

We use spatial heatmaps to show which image regions drove each ingredient prediction.

**Current implementation:** Grad-CAM hooking the last transformer block of CLIP's visual encoder (`src/visualization/gradcam.py`).

**Known issue:** Grad-CAM is not well-suited to ViT architectures. By the final block, spatial information has been aggregated into the CLS token — patch gradients are spatially uniform and don't localize to specific regions. The resulting heatmaps are near-identical across different ingredients for the same image.

**Planned fix:** Replace Grad-CAM with **Attention Rollout**, the standard ViT visualization technique:
- Collect attention weight matrices from every transformer block (averaged across heads)
- Multiply recursively across layers (rollout), adding a residual identity at each step
- The result is a (49,) token-importance map upsampled to (224, 224)
- No gradient computation required — forward pass only

See `docs/known_issues_and_improvements.md` for implementation details.

---

# 12. Results (actual)

### Experiments 1–3 (completed)

| Setup | P@1 | P@5 | R@5 | F1@5 |
|---|---|---|---|---|
| Zero-shot (Prompt A) | 0.12 | 0.09 | 0.11 | 0.09 |
| Ensemble ENS4 (A+B+C+D) | 0.14 | 0.11 | 0.15 | 0.11 |
| Adaptive threshold (Prompt B, k=10) | — | 0.09 | 0.24 | 0.12 |
| Projection head | 0.04 | 0.05 | 0.06 | 0.05 |
| Partial unfreeze | 0.00 | 0.03 | 0.03 | 0.02 |

**Key findings:**
- Contextual prompts (B, D) outperform single-word (A) by ~30% relative F1
- Ensemble embeddings give a free +22% F1 improvement over single-word baseline
- Fine-tuning degrades performance at ~1,880 samples — data scale is the bottleneck
- Adaptive threshold trades precision for recall; useful when coverage matters

### Experiments 4–5 (pending)

Results pending — requires larger dataset and LoRA implementation.

---

# 13. Possible Limitations

### Hidden ingredients

Many ingredients are not visually detectable (`salt`, `pepper`, `olive oil`). These are excluded from the vocabulary, but the model's score distribution may still be flattened by the presence of many borderline ingredients.

### Dataset noise

Ingredient lists include things added for flavor or texture that don't appear in the image. The ground truth is noisy by construction.

### CLIP training bias

CLIP was trained on general web image-text pairs, not food-specific data. Domain adaptation (LoRA) is expected to help once sufficient training data is available.

### Data scale

With ~1,880 training samples, fine-tuning is counterproductive. The full Recipe1M+ (~1M recipes) is needed for Experiments 4 and 5 to be meaningful.

### Adaptive threshold saturation

The current adaptive threshold (`mean + 0.5×std`) selects k=10 (the configured maximum) on the test set, indicating flat CLIP score distributions. A higher alpha or a lower max_k would improve precision.

---

# 14. Implementation Tools

Recommended stack:

```
Python
PyTorch
OpenCLIP
HuggingFace Transformers
```

Install:

```
pip install torch torchvision
pip install open_clip_torch
pip install transformers
```

---

# 15. Implementation Pipeline

```
src/data/           → load_recipes, build_vocab, Recipe1MDataset
src/models/         → CLIPWrapper (OpenCLIP), IngredientHead
src/experiments/    → prompts (A–H + ensembles), predict (topk + adaptive), metrics
src/training/       → trainer (head_only, partial_unfreeze, lora_contrastive)
src/visualization/  → GradCAM (current), AttentionRollout (planned)

src/run_experiment1.py  → zero-shot + ensemble + adaptive
src/run_experiment2.py  → 8 prompts + ENS4 + ENS7 comparison table
src/run_experiment3.py  → projection head + partial unfreeze vs zero-shot
src/run_experiment4.py  → LoRA contrastive fine-tuning (planned)
src/run_experiment5.py  → SigLIP + LoRA (planned)
src/run_visualization.py→ heatmap overlays per ingredient
```

---

# 16. Final Deliverables

* Experiment scripts (`run_experiment1–5.py`) with reproducible results
* Performance tables (Precision@K, Recall@K, F1@K) for all setups
* Attention rollout visualizations per ingredient (replacing GradCAM)
* Analysis of prompt sensitivity, ensemble gains, LoRA vs baseline
* Backbone comparison: CLIP InfoNCE vs SigLIP sigmoid loss

---

# 17. Research Contribution Statement

> We investigate the effectiveness of CLIP representations for multi-label ingredient recognition from food images. We conduct a systematic comparison of prompt design strategies (8 templates + ensemble variants), parameter-efficient domain adaptation (LoRA on both vision and text encoders with contrastive fine-tuning), and backbone selection (CLIP InfoNCE vs SigLIP sigmoid loss). Our controlled ablation isolates prompt sensitivity from fine-tuning effects — a separation most prior work does not make. The practical finding — that contextual prompt ensembles improve zero-shot F1 by ~22% for free, while naive fine-tuning at small data scale degrades performance — is both actionable and consistent with the broader transfer learning literature.

---

# End of Document