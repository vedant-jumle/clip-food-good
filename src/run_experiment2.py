from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import Recipe1MDataset
from src.data.recipe1m import load_recipes
from src.data.vocab import build_vocab
from src.experiments.metrics import f1_at_k, precision_at_k, recall_at_k
from src.experiments.predict import compute_scores, predict_topk
from src.experiments.prompts import make_prompts
from src.models.clip_wrapper import CLIPWrapper


DET_INGRS = "data/recipe1m/det_ingrs.json"
LAYER1 = "data/recipe1m/layer1.json"
LAYER2 = "data/recipe1m/layer2.json"
IMAGE_ROOT = "data/recipe1m/0"

NON_VISUAL = {
    "salt",
    "pepper",
    "water",
    "oil",
    "sugar",
    "flour",
    "butter",
    "baking powder",
    "baking soda",
    "vanilla extract",
    "olive oil",
    "vegetable oil",
    "black pepper",
    "kosher salt",
    "sea salt",
    "garlic powder",
    "onion powder",
    "paprika",
    "cumin",
    "vinegar",
}

PROMPT_TYPES = ["A", "B", "C", "D"]
BATCH_SIZE = 32
TOP_K = 5


def build_test_loader() -> tuple[list[dict], list[str], DataLoader]:
    recipes = load_recipes(
        DET_INGRS,
        LAYER1,
        LAYER2,
        IMAGE_ROOT,
        partition="test",
        require_images=True,
    )
    vocab = build_vocab(recipes, top_n=200, min_freq=2, exclude=NON_VISUAL)

    if not recipes:
        raise RuntimeError("No test recipes were loaded.")
    if not vocab:
        raise RuntimeError("Vocabulary is empty after filtering.")

    dataset = Recipe1MDataset(recipes=recipes, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return recipes, vocab, loader


def evaluate_prompt_type(
    loader: DataLoader,
    clip: CLIPWrapper,
    vocab: list[str],
    prompt_type: str,
) -> tuple[float, float, float, int]:
    prompts = make_prompts(vocab, prompt_type)
    text_embeddings = clip.encode_text(prompts)

    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        image_embeddings = clip.encode_image(batch["image"])
        scores = compute_scores(image_embeddings, text_embeddings)
        predictions = predict_topk(scores, k=TOP_K)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["labels"].cpu())

    preds = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return (
        precision_at_k(preds, labels, k=5),
        recall_at_k(preds, labels, k=5),
        f1_at_k(preds, labels, k=5),
        labels.size(0),
    )


def main() -> None:
    recipes, vocab, loader = build_test_loader()
    clip = CLIPWrapper()

    print("=== Experiment 2: Prompt Engineering ===")
    print(f"Vocab size: {len(vocab)}")
    print()
    print("Prompt  | P@5   | R@5   | F1@5")
    print("--------|-------|-------|------")

    n_evaluated = None
    for prompt_type in PROMPT_TYPES:
        precision, recall, f1, n = evaluate_prompt_type(loader, clip, vocab, prompt_type)
        if n_evaluated is None:
            n_evaluated = n
        print(f"{prompt_type:<7} | {precision:.2f}  | {recall:.2f}  | {f1:.2f}")

    print()
    print(f"Recipes evaluated: {n_evaluated}")


if __name__ == "__main__":
    main()
