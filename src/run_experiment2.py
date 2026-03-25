from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset import Recipe1MDataset
from src.data.recipe1m import expand_recipes, load_recipes
from src.data.vocab import build_vocab
import torch.nn.functional as F

from src.experiments.metrics import f1_at_k, precision_at_k, recall_at_k
from src.experiments.predict import compute_scores, predict_topk
from src.experiments.prompts import make_prompts
from src.models.clip_wrapper import CLIPWrapper


DET_INGRS = "data/recipe1m/det_ingrs.json"
LAYER1 = "data/recipe1m/layer1.json"
LAYER2 = "data/recipe1m/layer2.json"
IMAGE_ROOT = os.environ.get("RECIPE1M_IMAGE_ROOT", "data/recipe1m/0")
NUM_WORKERS = int(os.environ.get("RECIPE1M_NUM_WORKERS", "0"))

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

PROMPT_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "ENS4", "ENS7"]
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
    samples = expand_recipes(recipes)

    if not recipes:
        raise RuntimeError("No test recipes were loaded.")
    if not vocab:
        raise RuntimeError("Vocabulary is empty after filtering.")

    dataset = Recipe1MDataset(recipes=samples, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return recipes, vocab, loader


def make_ensemble_embeddings(vocab: list[str], clip: CLIPWrapper, types: list[str]) -> torch.Tensor:
    embeddings = []
    for pt in types:
        prompts = make_prompts(vocab, pt)
        embeddings.append(clip.encode_text(prompts))  # (V, D)
    ensemble = torch.stack(embeddings).mean(dim=0)    # (V, D)
    return F.normalize(ensemble, dim=-1)


# ENS4 = original A+B+C+D, ENS7 = B+C+D+E+F+G+H (no single-word A, no "fresh X")
ENSEMBLE_GROUPS = {
    "ENS4": ["A", "B", "C", "D"],
    "ENS7": ["B", "C", "D", "E", "F", "G", "H"],
}

PROMPT_LABELS = {
    "A": "A  (single word)",
    "B": "B  (dish containing X)",
    "C": "C  (meal with X ingr.)",
    "D": "D  (visible X)",
    "E": "E  (food photo of X)",
    "F": "F  (recipe ingr: X)",
    "G": "G  (this food contains X)",
    "H": "H  (fresh X)",
    "ENS4": "ENS4 (A+B+C+D)",
    "ENS7": "ENS7 (B-H, no A)",
}


def evaluate_prompt_type(
    loader: DataLoader,
    clip: CLIPWrapper,
    vocab: list[str],
    prompt_type: str,
) -> tuple[float, float, float, int]:
    if prompt_type in ENSEMBLE_GROUPS:
        text_embeddings = make_ensemble_embeddings(vocab, clip, ENSEMBLE_GROUPS[prompt_type])
    else:
        prompts = make_prompts(vocab, prompt_type)
        text_embeddings = clip.encode_text(prompts)

    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc=f"Prompt {prompt_type}", leave=False):
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

    tqdm.write("=== Experiment 2: Prompt Engineering ===")
    tqdm.write(f"Vocab size: {len(vocab)}")
    tqdm.write("")
    tqdm.write(f"{'Prompt':<26} | P@5   | R@5   | F1@5")
    tqdm.write(f"{'-'*26}-|-------|-------|------")

    n_evaluated = None
    prompt_results = []
    for prompt_type in tqdm(PROMPT_TYPES, desc="Prompt types"):
        precision, recall, f1, n = evaluate_prompt_type(loader, clip, vocab, prompt_type)
        if n_evaluated is None:
            n_evaluated = n
        label = PROMPT_LABELS.get(prompt_type, prompt_type)
        tqdm.write(f"{label:<26} | {precision:.2f}  | {recall:.2f}  | {f1:.2f}")
        prompt_results.append({
            "prompt": prompt_type,
            "label": label,
            "P@5": round(precision, 4),
            "R@5": round(recall, 4),
            "F1@5": round(f1, 4),
        })

    tqdm.write("")
    tqdm.write(f"Recipes evaluated: {n_evaluated}")

    # Save results to JSON
    results = {
        "vocab_size": len(vocab),
        "n_evaluated": n_evaluated,
        "results": prompt_results,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/experiment2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    tqdm.write("Results saved to outputs/experiment2_results.json")


if __name__ == "__main__":
    main()
