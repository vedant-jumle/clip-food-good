"""Build the recipe index cache.

Run this once before any experiment to pre-build the pickle cache:

    python scripts/build_cache.py

Env vars:
    RECIPE1M_IMAGE_ROOT  path to image directory (default: data/recipe1m/0)
    RECIPE1M_LAYER2      path to layer2 JSON     (default: data/recipe1m/layer2+.json)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.recipe1m import load_recipes

DET_INGRS  = "data/recipe1m/det_ingrs.json"
LAYER1     = "data/recipe1m/layer1.json"
LAYER2     = os.environ.get("RECIPE1M_LAYER2",     "data/recipe1m/layer2+.json")
IMAGE_ROOT = os.environ.get("RECIPE1M_IMAGE_ROOT", "data/recipe1m/0")

if __name__ == "__main__":
    print(f"IMAGE_ROOT : {IMAGE_ROOT}")
    print(f"LAYER2     : {LAYER2}")
    recipes = load_recipes(
        DET_INGRS,
        LAYER1,
        LAYER2,
        IMAGE_ROOT,
        partition=None,
        require_images=False,
    )
    total_images = sum(len(r["image_paths"]) for r in recipes)
    with_images = sum(1 for r in recipes if r["image_paths"])
    print(f"Done. {len(recipes):,} recipes indexed.")
    print(f"      {with_images:,} recipes with images, {total_images:,} total images.")
