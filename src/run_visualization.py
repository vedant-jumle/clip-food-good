from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm.auto import tqdm

from data.recipe1m import load_recipes
from data.vocab import build_vocab
from data.dataset import Recipe1MDataset, default_transform
from models.clip_wrapper import CLIPWrapper
from visualization.gradcam import GradCAM

# --- Config ---
DET_INGRS  = "data/recipe1m/det_ingrs.json"
LAYER1     = "data/recipe1m/layer1.json"
LAYER2     = "data/recipe1m/layer2.json"
IMAGE_ROOT = "data/recipe1m/0"
OUTPUT_DIR = "outputs"
N_SAMPLES  = 5
TOP_K      = 3
VOCAB_SIZE = 200

NON_VISUAL = {
    "salt", "pepper", "water", "oil", "sugar", "flour", "butter",
    "baking powder", "baking soda", "vanilla extract", "olive oil",
    "vegetable oil", "black pepper", "kosher salt", "sea salt",
    "garlic powder", "onion powder", "paprika", "cumin", "vinegar",
}


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CLIP-normalized tensor back to a displayable RGB image."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB image using a jet colormap."""
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.5 * image_rgb + 0.5 * colored).astype(np.uint8)
    return blended


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading recipes...")
    recipes = load_recipes(
        det_ingrs_path=DET_INGRS,
        layer1_path=LAYER1,
        layer2_path=LAYER2,
        image_root=IMAGE_ROOT,
        partition="test",
        require_images=True,
    )
    if not recipes:
        print("No recipes found. Check data paths and image availability.")
        return
    print(f"  {len(recipes)} test recipes loaded.")

    print("Building vocab...")
    vocab = build_vocab(recipes, top_n=VOCAB_SIZE, min_freq=2, exclude=NON_VISUAL)
    print(f"  Vocab size: {len(vocab)}")

    print("Loading CLIP...")
    clip = CLIPWrapper()
    gradcam = GradCAM(clip)

    print("Pre-encoding ingredient text embeddings (prompt type A)...")
    text_embs = clip.encode_text(vocab)  # (V, D)

    transform = default_transform()
    samples = recipes[:N_SAMPLES]

    for recipe in tqdm(samples, desc="Visualizing recipes"):
        recipe_id = recipe["id"]
        image_path = recipe["image_path"]

        print(f"\nProcessing recipe {recipe_id}: {recipe['title']}")
        print(f"  Ground truth ingredients: {recipe['ingredients'][:8]}")

        # load and preprocess image
        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = transform(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

        # predict top-k ingredients via cosine similarity
        with torch.no_grad():
            image_emb = clip.encode_image(image_tensor)           # (1, D)
            scores = (image_emb @ text_embs.T).squeeze(0)         # (V,)
        top_indices = scores.topk(TOP_K).indices.tolist()
        top_ingredients = [vocab[i] for i in top_indices]
        print(f"  Top-{TOP_K} predictions: {top_ingredients}")

        # generate grad-cam heatmaps
        image_rgb = denormalize(image_tensor.squeeze(0))
        fig, axes = plt.subplots(1, TOP_K + 1, figsize=(4 * (TOP_K + 1), 4))
        fig.suptitle(f"{recipe['title']}", fontsize=12)

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for col, ingredient in enumerate(top_ingredients, start=1):
            heatmap = gradcam(image_tensor, ingredient)
            overlay = overlay_heatmap(image_rgb, heatmap)
            axes[col].imshow(overlay)
            axes[col].set_title(f"{ingredient}")
            axes[col].axis("off")

        out_path = os.path.join(OUTPUT_DIR, f"gradcam_{recipe_id}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  Saved → {out_path}")

    gradcam.remove_hooks()
    print(f"\nDone. {len(samples)} visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
