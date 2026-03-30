"""Evaluate a saved exp5 visual encoder checkpoint locally.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint outputs/checkpoints/exp5_best.pt \
        --layer2 data/recipe1m/layer2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.recipe1m import expand_recipes, load_recipes
from src.data.vocab import build_vocab
from src.data.dataset import Recipe1MDataset
from src.experiments.metrics import f1_at_k, precision_at_k, recall_at_k
from src.experiments.predict import compute_scores, predict_topk
from src.experiments.prompts import make_prompts, prompt_templates
from src.models.clip_wrapper import CLIPWrapper

NON_VISUAL = {
    "salt", "pepper", "water", "oil", "sugar", "flour", "butter",
    "baking powder", "baking soda", "vanilla extract", "olive oil",
    "vegetable oil", "black pepper", "kosher salt", "sea salt",
    "garlic powder", "onion powder", "paprika", "cumin", "vinegar",
}

DET_INGRS  = "data/recipe1m/det_ingrs.json"
LAYER1     = "data/recipe1m/layer1.json"
TOP_K      = 5


def evaluate(clip, loader, vocab, prompt_type):
    text_embeddings = clip.encode_text(make_prompts(vocab, prompt_type))
    all_predictions, all_labels = [], []

    for batch in tqdm(loader, desc=f"Prompt {prompt_type}", leave=False):
        image_embeddings = clip.encode_image(batch["image"])
        scores = compute_scores(image_embeddings, text_embeddings)
        predictions = predict_topk(scores, k=TOP_K)
        all_predictions.append(predictions.cpu())
        all_labels.append(batch["labels"].cpu())

    preds  = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels,      dim=0)

    return {
        "P@1":  precision_at_k(preds[:, :1], labels, k=1),
        "P@5":  precision_at_k(preds,        labels, k=5),
        "R@5":  recall_at_k(preds,           labels, k=5),
        "F1@5": f1_at_k(preds,               labels, k=5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/checkpoints/exp5_best.pt")
    parser.add_argument("--layer2",     default=os.environ.get("RECIPE1M_LAYER2", "data/recipe1m/layer2.json"))
    parser.add_argument("--image-root", default=os.environ.get("RECIPE1M_IMAGE_ROOT", "data/recipe1m/0"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size",  type=int, default=128)
    parser.add_argument("--output",      default="outputs/eval_checkpoint_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load data
    test_recipes = load_recipes(
        DET_INGRS, LAYER1, args.layer2, args.image_root,
        partition="test", require_images=True,
    )
    train_recipes = load_recipes(
        DET_INGRS, LAYER1, args.layer2, args.image_root,
        partition="train", require_images=True,
    )
    vocab = build_vocab(train_recipes, top_n=200, min_freq=2, exclude=NON_VISUAL)
    test_samples = expand_recipes(test_recipes)
    dataset = Recipe1MDataset(recipes=test_samples, vocab=vocab)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Test samples: {len(dataset)}, Vocab: {len(vocab)}")

    # Load model
    clip = CLIPWrapper(device=device)
    state = torch.load(args.checkpoint, map_location=device)
    # Checkpoint is full model state_dict — load directly into clip.model
    clip.model.load_state_dict(state)
    print("Checkpoint loaded.\n")

    # Eval all prompts
    print(f"{'Prompt':<8} {'P@1':>6} {'P@5':>6} {'R@5':>6} {'F1@5':>6}")
    print("-" * 38)
    results = []
    for prompt_type in prompt_templates:
        m = evaluate(clip, loader, vocab, prompt_type)
        print(f"{prompt_type:<8} {m['P@1']:.3f}  {m['P@5']:.3f}  {m['R@5']:.3f}  {m['F1@5']:.3f}")
        results.append({"prompt": prompt_type, **{k: round(v, 4) for k, v in m.items()}})

    output = {
        "checkpoint": args.checkpoint,
        "test_samples": len(dataset),
        "vocab_size": len(vocab),
        "top_k": TOP_K,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
