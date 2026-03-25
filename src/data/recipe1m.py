from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pqdm.processes import pqdm as pqdm_proc
from pqdm.threads import pqdm as pqdm_threads


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


def _cache_path(image_root: str | Path, cache_dir: Path) -> Path:
    key = hashlib.md5(str(Path(image_root).resolve()).encode()).hexdigest()[:8]
    return cache_dir / f"recipes_cache_{key}.pkl"


def _scan_subdir(subdir: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for p in subdir.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[p.stem] = str(p)
    return index


def build_image_index(image_root: str | Path) -> Dict[str, str]:
    """Scan image_root once and return a dict mapping image stem -> full path."""
    root = Path(image_root)
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return _scan_subdir(root)
    results = pqdm_proc(subdirs, _scan_subdir, n_jobs=len(subdirs), desc="Scanning image dirs")
    index: Dict[str, str] = {}
    for partial in results:
        index.update(partial)
    return index


def load_recipes(
    det_ingrs_path: str | Path,
    layer1_path: str | Path,
    layer2_path: str | Path,
    image_root: str | Path,
    partition: Optional[str] = None,
    require_images: bool = True,
) -> List[Dict[str, Any]]:
    cache_dir = Path(image_root).parent / ".cache"
    cache_file = _cache_path(image_root, cache_dir)

    if cache_file.exists():
        print(f"Loading recipes from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            all_recipes: List[Dict[str, Any]] = pickle.load(f)
    else:
        print("Building recipe index (first run, this may take a while)...")

        def _load(args):
            name, fn, path = args
            print(f"  Loading {name}...")
            return fn(path)

        tasks = [
            ("det_ingrs", index_det_ingrs, det_ingrs_path),
            ("layer1",    index_layer1,    layer1_path),
            ("layer2",    index_layer2,    layer2_path),
        ]
        results = pqdm_threads(tasks, _load, n_jobs=3, desc="Loading JSON files")
        det_index, layer_index, layer2_index = results

        image_index = build_image_index(image_root)

        common_ids = list(det_index.keys() & layer_index.keys())

        def _build_entry(recipe_id):
            meta = layer_index[recipe_id]
            ingredients = det_index[recipe_id]
            if not ingredients:
                return None
            image_paths = [
                image_index[img_id]
                for img_id in layer2_index.get(recipe_id, [])
                if img_id in image_index
            ]
            return {
                "id": recipe_id,
                "title": meta["title"],
                "partition": meta["partition"],
                "ingredients": ingredients,
                "image_paths": image_paths,
            }

        entries = pqdm_threads(common_ids, _build_entry, n_jobs=4, desc="Building recipe entries")
        all_recipes = [e for e in entries if e is not None]

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(all_recipes, f)
        print(f"Cached {len(all_recipes):,} recipes to {cache_file}")

    # Filter by partition and require_images after loading from cache
    recipes = []
    for recipe in all_recipes:
        if partition is not None and recipe["partition"] != partition:
            continue
        if require_images and not recipe["image_paths"]:
            continue
        recipes.append(recipe)

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
