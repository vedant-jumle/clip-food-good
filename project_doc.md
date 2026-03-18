# CLIP-based Ingredient Recognition from Food Images

*A small research study on prompt design and domain adaptation*

## 1. Project Overview

This project investigates whether **CLIP-style vision-language representation learning** can be used to predict **ingredients from food images**, and whether **prompt engineering and lightweight fine-tuning** improve performance.

The central research question is:

> **Can prompt design and lightweight domain adaptation improve CLIP’s ability to detect ingredients in food images?**

The project evaluates three main setups:

1. **Zero-shot CLIP** (no training)
2. **Prompt engineering**
3. **Lightweight fine-tuning**

The goal is to understand how well CLIP transfers to the **ingredient recognition task**.

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

## Prompt Type A — Single word

```
"basil"
"tomato"
"garlic"
```

---

## Prompt Type B — Ingredient phrases

```
"a dish containing basil"
"a dish containing tomato"
"a dish containing garlic"
```

---

## Prompt Type C — Cooking context

```
"food with basil garnish"
"a meal with tomato ingredient"
"a dish cooked with garlic"
```

---

## Prompt Type D — Visual context

```
"a dish topped with basil leaves"
"a tomato-based food dish"
"a dish with visible garlic pieces"
```

---

## Evaluation

Compare performance of each prompt type using:

```
Precision@K
Recall@K
F1-score
```

---

# 7. Fine-tuning CLIP

We test whether **domain adaptation improves performance**.

Two fine-tuning strategies are recommended.

---

# Strategy 1 — Projection Head (Recommended)

Freeze CLIP.

Add a small MLP layer.

Architecture:

```
image
↓
CLIP image encoder (frozen)
↓
projection head
↓
ingredient prediction
```

Example PyTorch head:

```python
class IngredientHead(nn.Module):
    def __init__(self, input_dim, num_ingredients):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_ingredients)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
```

Loss:

```
Binary Cross Entropy (BCE)
```

---

# Strategy 2 — Partial CLIP Fine-tuning

Unfreeze only the **last layers of the image encoder**.

Keep:

```
text encoder frozen
```

Train with:

```
small learning rate (1e-5)
```

This avoids destroying CLIP's pretrained representation.

---

# 8. Training Setup

## Loss Function

For multi-label classification:

```
Binary Cross Entropy Loss
```

```
L = BCE(predicted_ingredients, ground_truth_labels)
```

---

## Optimization

Recommended optimizer:

```
AdamW
```

Typical settings:

```
learning rate = 1e-4 (head)
learning rate = 1e-5 (CLIP layers)
batch size = 32
epochs = 5–10
```

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

You should perform at least **three experiments**.

---

## Experiment 1 — Zero-shot CLIP

Goal:

```
Evaluate baseline performance
```

Setup:

```
frozen CLIP
single-word prompts
```

---

## Experiment 2 — Prompt Engineering

Goal:

```
Measure prompt sensitivity
```

Compare:

```
single word prompts
ingredient phrases
visual context prompts
```

---

## Experiment 3 — Domain Fine-tuning

Compare:

```
zero-shot CLIP
CLIP + projection head
partial CLIP fine-tuning
```

---

## Optional Experiment — Visible vs Invisible Ingredients

Split ingredients into:

```
visible ingredients
hidden ingredients
```

Example:

Visible:

```
basil
cheese
tomato
egg
```

Hidden:

```
salt
pepper
olive oil
```

Evaluate whether CLIP detects visible ingredients better.

---

# 11. Visualization (Highly Recommended)

Use **Grad-CAM** or **attention maps**.

Example:

```
input image
↓
highlight region responsible for ingredient prediction
```

Example output:

```
image → basil prediction → highlight green leaves
```

This improves interpretability.

---

# 12. Expected Results

Typical observations:

* CLIP performs reasonably well zero-shot
* prompts significantly affect results
* domain fine-tuning improves performance slightly
* visible ingredients are easier to detect

---

# 13. Possible Limitations

Important to discuss:

### Hidden ingredients

Many ingredients cannot be seen.

Example:

```
salt
pepper
olive oil
```

### Dataset noise

Ingredient lists may include irrelevant items.

### CLIP training bias

CLIP was not trained specifically for food analysis.

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

# 15. Minimal Implementation Pipeline

```
1. Load dataset
2. Build ingredient vocabulary
3. Load CLIP model
4. Encode ingredient prompts
5. Encode images
6. Compute similarity
7. Predict top-k ingredients
8. Evaluate metrics
9. Run prompt experiments
10. Fine-tune projection head
```

---

# 16. Final Deliverables

Your project should include:

* trained models
* experiment results
* performance tables
* visualizations
* analysis of results

---

# 17. Example Research Contribution Statement

Example statement for report:

> We investigate the effectiveness of CLIP representations for ingredient recognition from food images. Specifically, we evaluate the impact of prompt design and lightweight domain adaptation on ingredient prediction performance. Our experiments show that prompt phrasing significantly influences prediction accuracy, and that modest domain adaptation improves performance on visually observable ingredients.

---

# End of Document