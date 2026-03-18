from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

CLIP_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def default_transform() -> transforms.Compose:
    """Return a CLIP-compatible image preprocessing pipeline."""
    return transforms.Compose(
        [
            transforms.Resize(
                CLIP_IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(CLIP_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


class Recipe1MDataset(Dataset[dict[str, Any]]):
    """PyTorch dataset serving image tensors and multi-hot ingredient labels."""

    def __init__(
        self,
        recipes: Sequence[dict[str, Any]],
        vocab: Sequence[str],
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.recipes = list(recipes)
        self.vocab = list(vocab)
        self.transform = transform if transform is not None else default_transform()
        self.vocab_to_idx = {ingredient: idx for idx, ingredient in enumerate(self.vocab)}

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.recipes:
            raise IndexError("Recipe1MDataset is empty.")

        total = len(self.recipes)
        start_index = index % total

        for offset in range(total):
            current_index = (start_index + offset) % total
            recipe = self.recipes[current_index]
            image_path = Path(recipe["image_path"])

            try:
                image = self._load_image(image_path)
            except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
                recipe_id = recipe.get("id", "<unknown>")
                LOGGER.warning(
                    "Skipping unreadable image for recipe %s at %s: %s",
                    recipe_id,
                    image_path,
                    exc,
                )
                continue

            labels = self._encode_ingredients(recipe.get("ingredients", ()))
            return {
                "image": self.transform(image),
                "labels": labels,
                "id": recipe.get("id", ""),
                "image_path": str(image_path),
            }

        raise RuntimeError("No readable images were found in the provided recipes.")

    def _encode_ingredients(self, ingredients: Sequence[str]) -> torch.Tensor:
        labels = torch.zeros(len(self.vocab), dtype=torch.float32)
        for ingredient in set(ingredients):
            vocab_index = self.vocab_to_idx.get(ingredient)
            if vocab_index is not None:
                labels[vocab_index] = 1.0
        return labels

    @staticmethod
    def _load_image(image_path: Path) -> Image.Image:
        with Image.open(image_path) as image:
            return image.convert("RGB")


def _first_recipe1m_image(images_root: Path) -> Path | None:
    for image_path in images_root.rglob("*.jpg"):
        return image_path
    return None


def _dummy_recipes(image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": "dummy-1",
            "title": "Dummy Recipe 1",
            "ingredients": ["tomato", "garlic"],
            "image_path": str(image_path),
            "partition": "train",
        },
        {
            "id": "dummy-2",
            "title": "Dummy Recipe 2",
            "ingredients": ["cheese"],
            "image_path": str(image_path),
            "partition": "train",
        },
    ]


def smoke_test() -> None:
    logging.basicConfig(level=logging.INFO)

    images_root = Path("data/recipe1M")
    image_path = _first_recipe1m_image(images_root)
    if image_path is None:
        raise FileNotFoundError(f"No .jpg files found under {images_root}")

    recipes = _dummy_recipes(image_path)
    vocab = ["tomato", "garlic", "cheese", "chicken"]

    dataset = Recipe1MDataset(recipes=recipes, vocab=vocab)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    print(f"dataset_size={len(dataset)}")
    print(f"image_shape={tuple(batch['image'].shape)}")
    print(f"labels_shape={tuple(batch['labels'].shape)}")
    print(f"ids={list(batch['id'])}")


if __name__ == "__main__":
    smoke_test()
