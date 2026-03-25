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
from src.data.recipe1m import load_recipes
from src.data.vocab import build_vocab
import torch.nn.functional as F

from src.experiments.metrics import f1_at_k, precision_at_k, recall_at_k
from src.experiments.predict import compute_scores, predict_adaptive, predict_topk
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
    vocab = build_vocab(recipes, top_n=500, min_freq=2, exclude=NON_VISUAL)

    if not recipes:
        raise RuntimeError("No test recipes were loaded.")
    if not vocab:
        raise RuntimeError("Vocabulary is empty after filtering.")

    dataset = Recipe1MDataset(recipes=recipes, vocab=vocab)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return recipes, vocab, loader


def make_ensemble_embeddings(vocab: list[str], clip: CLIPWrapper, types: list[str] = ["A", "B", "C", "D"]) -> torch.Tensor:
    embeddings = []
    for pt in types:
        prompts = make_prompts(vocab, pt)
        embeddings.append(clip.encode_text(prompts))  # (V, D)
    ensemble = torch.stack(embeddings).mean(dim=0)    # (V, D)
    return F.normalize(ensemble, dim=-1)


def evaluate_zero_shot(
    loader: DataLoader,
    clip: CLIPWrapper,
    vocab: list[str],
    text_embeddings: torch.Tensor,
    adaptive: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image_embeddings = clip.encode_image(batch["image"])
        scores = compute_scores(image_embeddings, text_embeddings)
        if adaptive:
            predictions = predict_adaptive(scores)
        else:
            predictions = predict_topk(scores, k=TOP_K)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["labels"].cpu())

    return torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0)


def main() -> None:
    recipes, vocab, loader = build_test_loader()
    clip = CLIPWrapper()

    # --- Baseline: Prompt A ---
    text_emb_a = clip.encode_text(make_prompts(vocab, "A"))
    preds, labels = evaluate_zero_shot(loader, clip, vocab, text_emb_a)

    print("=== Experiment 1: Zero-shot CLIP (Prompt A) ===")
    print(f"Recipes evaluated: {labels.size(0)}")
    print(f"Vocab size: {len(vocab)}")
    print()
    print(f"Precision@1: {precision_at_k(preds[:, :1], labels, k=1):.2f}")
    print(f"Precision@3: {precision_at_k(preds[:, :3], labels, k=3):.2f}")
    print(f"Precision@5: {precision_at_k(preds, labels, k=5):.2f}")
    print(f"Recall@5:    {recall_at_k(preds, labels, k=5):.2f}")
    print(f"F1@5:        {f1_at_k(preds, labels, k=5):.2f}")

    # --- Ensemble prompts (A+B+C+D averaged) ---
    print()
    print("=== Experiment 1B: Ensemble Prompts (A+B+C+D) ===")
    ensemble_emb = make_ensemble_embeddings(vocab, clip)
    preds_ens, labels_ens = evaluate_zero_shot(loader, clip, vocab, ensemble_emb)
    print(f"Precision@1: {precision_at_k(preds_ens[:, :1], labels_ens, k=1):.2f}")
    print(f"Precision@3: {precision_at_k(preds_ens[:, :3], labels_ens, k=3):.2f}")
    print(f"Precision@5: {precision_at_k(preds_ens, labels_ens, k=5):.2f}")
    print(f"Recall@5:    {recall_at_k(preds_ens, labels_ens, k=5):.2f}")
    print(f"F1@5:        {f1_at_k(preds_ens, labels_ens, k=5):.2f}")

    # --- Adaptive threshold (Prompt B, best single prompt) ---
    print()
    print("=== Experiment 1C: Adaptive Threshold (Prompt B) ===")
    text_emb_b = clip.encode_text(make_prompts(vocab, "B"))
    preds_ada, labels_ada = evaluate_zero_shot(loader, clip, vocab, text_emb_b, adaptive=True)
    k_ada = preds_ada.size(1)
    p_ada = precision_at_k(preds_ada, labels_ada, k=k_ada)
    r_ada = recall_at_k(preds_ada, labels_ada, k=k_ada)
    f1_ada = f1_at_k(preds_ada, labels_ada, k=k_ada)
    print(f"Adaptive k selected: {k_ada}")
    print(f"Precision@k: {p_ada:.2f}")
    print(f"Recall@k:    {r_ada:.2f}")
    print(f"F1@k:        {f1_ada:.2f}")

    # Save results to JSON
    results = {
        "vocab_size": len(vocab),
        "n_evaluated": int(labels.size(0)),
        "zero_shot": {
            "prompt": "A",
            "P@1": round(precision_at_k(preds[:, :1], labels, k=1), 4),
            "P@3": round(precision_at_k(preds[:, :3], labels, k=3), 4),
            "P@5": round(precision_at_k(preds, labels, k=5), 4),
            "R@5": round(recall_at_k(preds, labels, k=5), 4),
            "F1@5": round(f1_at_k(preds, labels, k=5), 4),
        },
        "ensemble": {
            "prompts": ["A", "B", "C", "D"],
            "P@1": round(precision_at_k(preds_ens[:, :1], labels_ens, k=1), 4),
            "P@3": round(precision_at_k(preds_ens[:, :3], labels_ens, k=3), 4),
            "P@5": round(precision_at_k(preds_ens, labels_ens, k=5), 4),
            "R@5": round(recall_at_k(preds_ens, labels_ens, k=5), 4),
            "F1@5": round(f1_at_k(preds_ens, labels_ens, k=5), 4),
        },
        "adaptive": {
            "prompt": "B",
            "k_selected": k_ada,
            "P@k": round(p_ada, 4),
            "R@k": round(r_ada, 4),
            "F1@k": round(f1_ada, 4),
        },
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/experiment1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to outputs/experiment1_results.json")


if __name__ == "__main__":
    main()
