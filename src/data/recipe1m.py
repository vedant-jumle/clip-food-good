from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def norm_ing(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid_ing(text: str) -> bool:
    if not text:
        return False
    if text.startswith("makes about"):
        return False
    if not re.search(r"[a-zA-Z]", text):
        return False
    return True


def extract_ings(det_entry: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    ingredients = det_entry.get("ingredients", [])
    valid_flags = det_entry.get("valid", [True] * len(ingredients))

    for ing, is_valid in zip(ingredients, valid_flags):
        if not is_valid:
            continue
        if isinstance(ing, dict):
            text = ing.get("text", "")
        elif isinstance(ing, str):
            text = ing
        else:
            continue
        text = norm_ing(text)
        if is_valid_ing(text):
            result.append(text)

    return result


def index_det_ingrs(det_ingrs_path: str | Path) -> Dict[str, List[str]]:
    data = read_json(det_ingrs_path)
    index: Dict[str, List[str]] = {}
    for entry in data:
        recipe_id = str(entry["id"])
        index[recipe_id] = extract_ings(entry)
    return index


def index_layer1(layer1_path: str | Path) -> Dict[str, Any]:
    data = read_json(layer1_path)
    index: Dict[str, Any] = {}
    for entry in data:
        recipe_id = str(entry["id"])
        index[recipe_id] = {
            "title": entry.get("title", "").strip(),
            "partition": entry.get("partition"),
        }
    return index


def index_layer2(layer2_path: str | Path) -> Dict[str, List[str]]:
    data = read_json(layer2_path)
    index: Dict[str, List[str]] = {}
    for entry in data:
        recipe_id = str(entry["id"])
        image_ids = []
        for img in entry.get("images", []):
            img_id = img.get("id", "") if isinstance(img, dict) else str(img)
            stem = img_id.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
            if stem:
                image_ids.append(stem)
        index[recipe_id] = image_ids
    return index


def build_image_index(image_root: str | Path) -> Dict[str, str]:
    """Scan image_root once and return a dict mapping image stem -> full path."""
    index: Dict[str, str] = {}
    for p in Path(image_root).rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[p.stem] = str(p)
    return index


def load_recipes(
    det_ingrs_path: str | Path,
    layer1_path: str | Path,
    layer2_path: str | Path,
    image_root: str | Path,
    partition: Optional[str] = None,
    require_images: bool = True,
) -> List[Dict[str, Any]]:
    det_index = index_det_ingrs(det_ingrs_path)
    layer_index = index_layer1(layer1_path)
    layer2_index = index_layer2(layer2_path)
    image_index = build_image_index(image_root)

    recipes: List[Dict[str, Any]] = []

    shared_ids = det_index.keys() & layer_index.keys()

    for recipe_id in shared_ids:
        meta = layer_index[recipe_id]

        if partition is not None and meta["partition"] != partition:
            continue

        ingredients = det_index[recipe_id]
        if not ingredients:
            continue

        image_paths = [
            image_index[img_id]
            for img_id in layer2_index.get(recipe_id, [])
            if img_id in image_index
        ]

        if require_images and not image_paths:
            continue

        recipes.append(
            {
                "id": recipe_id,
                "title": meta["title"],
                "partition": meta["partition"],
                "ingredients": ingredients,
                "image_paths": image_paths,
            }
        )

    return recipes


def expand_recipes(recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each recipe into one entry per available image.

    Each returned entry is a copy of the recipe dict with a single
    ``"image_path"`` key (str) instead of ``"image_paths"`` (list),
    so it is compatible with Recipe1MDataset without any changes.
    """
    samples: List[Dict[str, Any]] = []
    for recipe in recipes:
        for path in recipe["image_paths"]:
            samples.append({**recipe, "image_path": path})
    return samples
