from __future__ import annotations

import json
import os
import sys
from typing import Dict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import Recipe1MDataset
from src.data.recipe1m import load_recipes
from src.data.vocab import build_vocab
from src.models.clip_wrapper import CLIPWrapper
from src.models.ingredient_head import IngredientHead
from src.training.trainer import train

# Data paths
DET_INGRS = "data/recipe1m/det_ingrs.json"
LAYER1 = "data/recipe1m/layer1.json"
LAYER2 = "data/recipe1m/layer2.json"
IMAGE_ROOT = os.environ.get("RECIPE1M_IMAGE_ROOT", "data/recipe1m/0")
NUM_WORKERS = int(os.environ.get("RECIPE1M_NUM_WORKERS", "0"))

# Non-visual ingredients
NON_VISUAL = {
    "salt", "pepper", "water", "oil", "sugar", "flour", "butter",
    "baking powder", "baking soda", "vanilla extract", "olive oil",
    "vegetable oil", "black pepper", "kosher salt", "sea salt",
    "garlic powder", "onion powder", "paprika", "cumin", "vinegar",
}


# Try importing Person A module
try:
    from src.experiments.prompts import make_prompts
    from src.experiments.predict import compute_scores, predict_topk
    from src.experiments.metrics import precision_at_k, recall_at_k, f1_at_k
except ImportError:
    print(
    "Warning: Person A modules not found. Using temporary fallback stubs; "
    "evaluation metrics will be placeholders.")

    def make_prompts(vocab, prompt_type):
        return [f"{v}" for v in vocab]

    def compute_scores(image_emb, text_emb):
        return image_emb @ text_emb.T

    def predict_topk(scores, k=5):
        return scores.topk(k, dim=-1).indices

    def precision_at_k(preds, labels, k):
        return 0.0

    def recall_at_k(preds, labels, k):
        return 0.0

    def f1_at_k(preds, labels, k):
        return 0.0


def evaluate_zero_shot(
    clip: CLIPWrapper,
    dataloader: DataLoader,
    vocab: list[str],
    prompt_type: str = "A",
    device: str = "cuda",
) -> Dict[str, float]:
    prompts = make_prompts(vocab, prompt_type)
    text_emb = clip.encode_text(prompts)  # (V, D), normalized

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Zero-shot eval", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"]

        image_emb = clip.encode_image(images)   # (N, D), normalized
        scores = compute_scores(image_emb, text_emb)  # (N, V)
        preds = predict_topk(scores, k=5)  # (N, 5)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        "P@1": precision_at_k(preds[:, :1], labels, k=1),
        "P@5": precision_at_k(preds, labels, k=5),
        "R@5": recall_at_k(preds, labels, k=5),
        "F1@5": f1_at_k(preds, labels, k=5),
    }


@torch.no_grad()
def evaluate_head_model(
    clip: CLIPWrapper,
    head: IngredientHead,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    head.eval()
    clip.model.eval()

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating head", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"]

        image_emb = clip.encode_image(images)   # normalized embeddings
        probs = head(image_emb)                 # (N, V)
        preds = predict_topk(probs, k=5)        # top-k ingredient indices

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        "P@1": precision_at_k(preds[:, :1], labels, k=1),
        "P@5": precision_at_k(preds, labels, k=5),
        "R@5": recall_at_k(preds, labels, k=5),
        "F1@5": f1_at_k(preds, labels, k=5),
    }
def check_data_paths() -> None:
    required_paths = [DET_INGRS, LAYER1, LAYER2, IMAGE_ROOT]
    missing = [path for path in required_paths if not Path(path).exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Recipe1M data files/directories:\n"
            + "\n".join(f" - {path}" for path in missing)
        )

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_data_paths()
    print("Loading Recipe1M splits...")
    train_recipes = load_recipes(
        DET_INGRS,
        LAYER1,
        LAYER2,
        IMAGE_ROOT,
        partition="train",
        require_images=True,
    )
    test_recipes = load_recipes(
        DET_INGRS,
        LAYER1,
        LAYER2,
        IMAGE_ROOT,
        partition="test",
        require_images=True,
    )

    print(f"Train recipes: {len(train_recipes)}")
    print(f"Test recipes:  {len(test_recipes)}")

    print("Building vocabulary...")
    vocab = build_vocab(
        train_recipes,
        top_n=200,
        min_freq=2,
        exclude=NON_VISUAL,
    )
    print(f"Vocab size: {len(vocab)}")

    print("Building datasets and dataloaders...")
    train_dataset = Recipe1MDataset(train_recipes, vocab)
    test_dataset = Recipe1MDataset(test_recipes, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Loading CLIP...")
    clip = CLIPWrapper(device=device)

    # Setup A: Zero-shot
    print("\nRunning zero-shot baseline...")
    zero_shot_metrics = evaluate_zero_shot(
        clip=clip,
        dataloader=test_loader,
        vocab=vocab,
        prompt_type="A",
        device=device,
    )

    # Setup B: Projection head
    print("\nTraining projection head...")
    head_projection = IngredientHead(
        input_dim=clip.embedding_dim,
        num_ingredients=len(vocab),
    )

    train(
        clip=clip,
        head=head_projection,
        dataloader=train_loader,
        strategy="head_only",
        epochs=5,
        lr_head=1e-4,
        device=device,
    )

    print("Evaluating projection head...")
    projection_metrics = evaluate_head_model(
        clip=clip,
        head=head_projection,
        dataloader=test_loader,
        device=device,
    )

    # Setup C: Partial unfreeze
    print("\nTraining partial unfreeze model...")
    clip_partial = CLIPWrapper(device=device)
    head_partial = IngredientHead(
        input_dim=clip_partial.embedding_dim,
        num_ingredients=len(vocab),
    )

    train(
        clip=clip_partial,
        head=head_partial,
        dataloader=train_loader,
        strategy="partial_unfreeze",
        epochs=5,
        lr_head=1e-4,
        lr_clip=1e-5,
        device=device,
    )

    print("Evaluating partial unfreeze model...")
    partial_metrics = evaluate_head_model(
        clip=clip_partial,
        head=head_partial,
        dataloader=test_loader,
        device=device,
    )

    # Final output table
    print("\n=== Experiment 3: Fine-tuning ===")
    print(f"{'Setup':<18} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'F1@5':>6}")
    print("-" * 46)
    print(
        f"{'Zero-shot':<18} "
        f"{zero_shot_metrics['P@1']:.3f} "
        f"{zero_shot_metrics['P@5']:.3f} "
        f"{zero_shot_metrics['R@5']:.3f} "
        f"{zero_shot_metrics['F1@5']:.3f}"
    )
    print(
        f"{'Projection head':<18} "
        f"{projection_metrics['P@1']:.3f} "
        f"{projection_metrics['P@5']:.3f} "
        f"{projection_metrics['R@5']:.3f} "
        f"{projection_metrics['F1@5']:.3f}"
    )
    print(
        f"{'Partial unfreeze':<18} "
        f"{partial_metrics['P@1']:.3f} "
        f"{partial_metrics['P@5']:.3f} "
        f"{partial_metrics['R@5']:.3f} "
        f"{partial_metrics['F1@5']:.3f}"
    )

    # Save results to JSON
    results = {
        "vocab_size": len(vocab),
        "n_train": len(train_recipes),
        "n_test": len(test_recipes),
        "results": [
            {"setup": "Zero-shot", **{k: round(v, 4) for k, v in zero_shot_metrics.items()}},
            {"setup": "Projection head", **{k: round(v, 4) for k, v in projection_metrics.items()}},
            {"setup": "Partial unfreeze", **{k: round(v, 4) for k, v in partial_metrics.items()}},
        ],
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/experiment3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/experiment3_results.json")


if __name__ == "__main__":
    main()