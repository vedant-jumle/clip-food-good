from __future__ import annotations

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
from src.experiments.metrics import f1_at_k, precision_at_k, recall_at_k
from src.experiments.predict import compute_scores, predict_topk
from src.experiments.prompts import make_prompts
from src.models.clip_wrapper import CLIPWrapper
from src.training.trainer import train_contrastive


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

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4
LORA_RANK = 32
LORA_ALPHA = 1.0
TOP_K = 5


def build_loaders(device: str) -> tuple[list[str], DataLoader, DataLoader]:
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
    vocab = build_vocab(train_recipes, top_n=200, min_freq=2, exclude=NON_VISUAL)

    if not train_recipes:
        raise RuntimeError("No training recipes were loaded.")
    if not test_recipes:
        raise RuntimeError("No test recipes were loaded.")
    if not vocab:
        raise RuntimeError("Vocabulary is empty after filtering.")

    train_dataset = Recipe1MDataset(recipes=train_recipes, vocab=vocab)
    test_dataset = Recipe1MDataset(recipes=test_recipes, vocab=vocab)

    pin_memory = device.startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    return vocab, train_loader, test_loader


def evaluate(clip: CLIPWrapper, loader: DataLoader, vocab: list[str]) -> dict[str, float]:
    clip.model.eval()

    # Prompt B was best in Experiment 2
    text_embeddings = clip.encode_text(make_prompts(vocab, "B"))

    all_predictions: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image_embeddings = clip.encode_image(batch["image"])
        scores = compute_scores(image_embeddings, text_embeddings)
        predictions = predict_topk(scores, k=TOP_K)

        all_predictions.append(predictions.cpu())
        all_labels.append(batch["labels"].cpu())

    preds = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        "P@1": precision_at_k(preds[:, :1], labels, k=1),
        "P@5": precision_at_k(preds, labels, k=5),
        "R@5": recall_at_k(preds, labels, k=5),
        "F1@5": f1_at_k(preds, labels, k=5),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab, train_loader, test_loader = build_loaders(device=device)
    clip = CLIPWrapper(device=device)

    print("=== Experiment 4: LoRA Contrastive Fine-tuning ===")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")
    print(f"Vocab size:    {len(vocab)}")
    print(f"LoRA rank:     {LORA_RANK}")
    print(f"LoRA alpha:    {LORA_ALPHA}")
    print()

    # Before fine-tuning
    print("Evaluating zero-shot baseline (before fine-tuning)...")
    before = evaluate(clip, test_loader, vocab)
    print(
        f"Baseline -> P@1: {before['P@1']:.3f}, "
        f"P@5: {before['P@5']:.3f}, "
        f"R@5: {before['R@5']:.3f}, "
        f"F1@5: {before['F1@5']:.3f}"
    )
    print()

    # LoRA contrastive training
    print("Training with LoRA contrastive loss...")
    losses = train_contrastive(
        clip=clip,
        dataloader=train_loader,
        vocab=vocab,
        epochs=EPOCHS,
        lr=LR,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        device=device,
    )
    print(f"Final epoch loss: {losses[-1]:.4f}")
    print()

    # After fine-tuning
    print("Evaluating after fine-tuning...")
    clip.model.eval()
    after = evaluate(clip, test_loader, vocab)

    print("\n=== Results ===")
    print(f"{'Setup':<22} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'F1@5':>6}")
    print("-" * 52)
    print(
        f"{'Zero-shot (before)':<22} "
        f"{before['P@1']:.3f}  {before['P@5']:.3f}  {before['R@5']:.3f}  {before['F1@5']:.3f}"
    )
    print(
        f"{'LoRA fine-tuned':<22} "
        f"{after['P@1']:.3f}  {after['P@5']:.3f}  {after['R@5']:.3f}  {after['F1@5']:.3f}"
    )

    os.makedirs("outputs", exist_ok=True)
    results = {
        "experiment": "experiment4_lora_contrastive",
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset),
        "vocab_size": len(vocab),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS,
        "lr": LR,
        "losses": [round(x, 6) for x in losses],
        "before": {k: round(v, 4) for k, v in before.items()},
        "after": {k: round(v, 4) for k, v in after.items()},
        "eval_prompt": "B",
        "top_k": TOP_K,
    }

    with open("outputs/experiment4_results.json", "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2)

    print("\nResults saved to outputs/experiment4_results.json")


if __name__ == "__main__":
    main()