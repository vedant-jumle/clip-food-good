from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

def read_json(path: str | Path) -> Any:
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
    
    if not re.search(r"[a-zA-z]", text):
        return False
    
    return True

def extract_ings(det_entry: Dict[str, Any]) -> List[str]:
    result: List[str] = []
    
    for ing in det_entry.get("ingredients", []):
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

def index_layer1(layer1_path: str | Path) -> Dict[str, List[str]]:
    data = read_json(layer1_path)
    index: Dict[str, List[str]] = {}
    
    for entry in data:
        recipe_id = str(entry["id"])
        index[recipe_id] = {
            "title": entry.get("title", "").strip(),
            "partition": entry.get("partition"),
        }
        
    return index

def recipe_id_to_image_path(image_root: str | Path, recipe_id: str) -> Optional[str]:
    
    image_root = Path(image_root)
    recipe_id = str(recipe_id)

    if len(recipe_id) < 3:
        return None

    candidate = image_root / recipe_id[0] / recipe_id[1] / recipe_id[2] / f"{recipe_id}.jpg"
    if candidate.exists():
        return str(candidate)

    jpeg_candidate = image_root / recipe_id[0] / recipe_id[1] / recipe_id[2] / f"{recipe_id}.jpeg"
    if jpeg_candidate.exists():
        return str(jpeg_candidate)

    png_candidate = image_root / recipe_id[0] / recipe_id[1] / recipe_id[2] / f"{recipe_id}.png"
    if png_candidate.exists():
        return str(png_candidate)

    return None

def candidate_img_paths(image_root: Path, image_id: str) -> List[Path]:
    image_id = str(image_id)
    candidates: List[Path] = []

    if len(image_id) >= 3:
        base = image_root / image_id[0] / image_id[1] / image_id[2]
        candidates.append(base / f"{image_id}.jpg")
        candidates.append(base / f"{image_id}.jpeg")
        candidates.append(base / f"{image_id}.png")

    candidates.append(image_root / f"{image_id}.jpg")
    candidates.append(image_root / f"{image_id}.jpeg")
    candidates.append(image_root / f"{image_id}.png")

    return candidates

def resolve_first_img_path(images: List[Any], image_root: str | Path) -> Optional[str]:
    image_root = Path(image_root)
    
    for img in images:
        if isinstance(img, dict):
            image_id = img.get("id")
        elif isinstance(img, str):
            image_id = img
        else:
            continue

        if not image_id:
            continue

        for candidate in candidate_img_paths(image_root, str(image_id)):
            if candidate.exists():
                return str(candidate)

    return None

def load_recipes(
    det_ingrs_path: str | Path,
    layer1_path: str | Path,
    image_root: str | Path,
    partition: Optional[str] = None,
    require_images: bool = True,
) -> List[Dict[str, Any]]:
    det_index = index_det_ingrs(det_ingrs_path)
    layer_index = index_layer1(layer1_path)

    recipes: List[Dict[str, Any]] = []

    shared_ids = det_index.keys() & layer_index.keys()

    for recipe_id in shared_ids:
        meta = layer_index[recipe_id]

        if partition is not None and meta["partition"] != partition:
            continue

        ingredients = det_index[recipe_id]
        if not ingredients:
            continue

        image_path = recipe_id_to_image_path(image_root, recipe_id)

        if require_images and image_path is None:
            continue

        recipes.append(
            {
                "id": recipe_id,
                "title": meta["title"],
                "partition": meta["partition"],
                "ingredients": ingredients,
                "image_paths": image_path,
            }
        )

    return recipes
