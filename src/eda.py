from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.recipe1m import (
    build_image_index,
    index_det_ingrs,
    index_layer1,
    index_layer2,
)

DET_INGRS = os.environ.get("RECIPE1M_DET_INGRS", "data/recipe1m/det_ingrs.json")
LAYER1    = os.environ.get("RECIPE1M_LAYER1",    "data/recipe1m/layer1.json")
LAYER2    = os.environ.get("RECIPE1M_LAYER2",    "data/recipe1m/layer2+.json")
IMAGE_ROOT = os.environ.get("RECIPE1M_IMAGE_ROOT", "data/recipe1m/0")

NON_VISUAL = {
    "salt", "pepper", "water", "oil", "sugar", "flour", "butter",
    "baking powder", "baking soda", "vanilla extract", "olive oil",
    "vegetable oil", "black pepper", "kosher salt", "sea salt",
    "garlic powder", "onion powder", "paprika", "cumin", "vinegar",
}

SEP  = "=" * 60
SEP2 = "-" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main() -> None:
    print("Loading JSON indexes (this may take a moment)...")
    det_index    = index_det_ingrs(DET_INGRS)
    layer_index  = index_layer1(LAYER1)
    layer2_index = index_layer2(LAYER2)

    print(f"Building image index from: {IMAGE_ROOT}")
    image_index = build_image_index(IMAGE_ROOT)

    # ------------------------------------------------------------------ #
    # 1. JSON-level counts (before image filter)
    # ------------------------------------------------------------------ #
    section("1. JSON-level counts (before image filter)")

    n_det    = len(det_index)
    n_layer1 = len(layer_index)
    n_layer2 = len(layer2_index)
    shared   = det_index.keys() & layer_index.keys()

    print(f"  det_ingrs entries  : {n_det:>10,}")
    print(f"  layer1 entries     : {n_layer1:>10,}")
    print(f"  layer2 entries     : {n_layer2:>10,}")
    print(f"  shared (det∩layer1): {len(shared):>10,}")

    # ------------------------------------------------------------------ #
    # 2. Image index
    # ------------------------------------------------------------------ #
    section("2. Downloaded images")
    print(f"  Total image files  : {len(image_index):>10,}")

    # ------------------------------------------------------------------ #
    # 3. Build full recipe list with image resolution
    # ------------------------------------------------------------------ #
    section("3. Recipe statistics (with images)")

    partitions = {"train": [], "val": [], "test": [], "unknown": []}
    ing_counts: list[int] = []
    all_ingredients: Counter = Counter()
    visual_ingredients: Counter = Counter()
    recipes_no_image = 0
    recipes_no_ingredients = 0

    for recipe_id in tqdm(shared, desc="Processing recipes"):
        meta = layer_index[recipe_id]
        partition = meta.get("partition") or "unknown"

        ingredients = det_index[recipe_id]
        if not ingredients:
            recipes_no_ingredients += 1
            continue

        # Resolve image
        image_path = None
        for img_id in layer2_index.get(recipe_id, []):
            if img_id in image_index:
                image_path = image_index[img_id]
                break

        if image_path is None:
            recipes_no_image += 1
            continue

        ing_counts.append(len(ingredients))
        for ing in ingredients:
            all_ingredients[ing] += 1
            if ing not in NON_VISUAL:
                visual_ingredients[ing] += 1

        partitions[partition if partition in partitions else "unknown"].append(recipe_id)

    total_with_images = sum(len(v) for v in partitions.values())

    print(f"\n  Recipes with images (usable):")
    print(f"    Total            : {total_with_images:>10,}")
    for part, ids in partitions.items():
        print(f"    {part:<8}         : {len(ids):>10,}")

    print(f"\n  Dropped (no image) : {recipes_no_image:>10,}")
    print(f"  Dropped (no ingrs) : {recipes_no_ingredients:>10,}")

    # ------------------------------------------------------------------ #
    # 4. Ingredient statistics
    # ------------------------------------------------------------------ #
    section("4. Ingredient statistics")

    if ing_counts:
        avg   = sum(ing_counts) / len(ing_counts)
        mn    = min(ing_counts)
        mx    = max(ing_counts)
        sorted_counts = sorted(ing_counts)
        median = sorted_counts[len(sorted_counts) // 2]
        p25   = sorted_counts[len(sorted_counts) // 4]
        p75   = sorted_counts[3 * len(sorted_counts) // 4]

        print(f"  Ingredients per recipe:")
        print(f"    Mean             : {avg:>10.2f}")
        print(f"    Median           : {median:>10}")
        print(f"    Min / Max        : {mn:>5} / {mx}")
        print(f"    25th / 75th pct  : {p25:>5} / {p75}")

    print(f"\n  Unique ingredients (raw)   : {len(all_ingredients):>8,}")
    print(f"  Unique ingredients (visual): {len(visual_ingredients):>8,}")

    # Frequency buckets
    freq_buckets = {">= 1000": 0, "100-999": 0, "10-99": 0, "2-9": 0, "1": 0}
    for freq in all_ingredients.values():
        if freq >= 1000:   freq_buckets[">= 1000"] += 1
        elif freq >= 100:  freq_buckets["100-999"] += 1
        elif freq >= 10:   freq_buckets["10-99"]   += 1
        elif freq >= 2:    freq_buckets["2-9"]      += 1
        else:              freq_buckets["1"]        += 1

    print(f"\n  Ingredient frequency distribution:")
    for bucket, count in freq_buckets.items():
        print(f"    {bucket:<12}: {count:>8,} ingredients")

    # ------------------------------------------------------------------ #
    # 5. Top ingredients
    # ------------------------------------------------------------------ #
    section("5. Top 30 ingredients (all)")
    print(f"  {'Rank':<5} {'Ingredient':<30} {'Count':>8}")
    print(f"  {SEP2}")
    for rank, (ing, cnt) in enumerate(all_ingredients.most_common(30), 1):
        print(f"  {rank:<5} {ing:<30} {cnt:>8,}")

    section("6. Top 30 visual ingredients (excluding non-visual)")
    print(f"  {'Rank':<5} {'Ingredient':<30} {'Count':>8}")
    print(f"  {SEP2}")
    for rank, (ing, cnt) in enumerate(visual_ingredients.most_common(30), 1):
        print(f"  {rank:<5} {ing:<30} {cnt:>8,}")

    # ------------------------------------------------------------------ #
    # 7. Vocab coverage
    # ------------------------------------------------------------------ #
    section("7. Vocab coverage at different top-N sizes")
    print(f"  {'top_n':<8} {'vocab size':<12} {'% recipes covered (>=1 match)'}")
    print(f"  {SEP2}")

    for top_n in [50, 100, 200, 500, 1000]:
        vocab = {ing for ing, _ in visual_ingredients.most_common(top_n)}
        covered = sum(
            1 for ids in partitions.values()
            for recipe_id in ids
            if any(ing in vocab for ing in det_index[recipe_id])
        )
        pct = 100 * covered / total_with_images if total_with_images else 0
        print(f"  {top_n:<8} {len(vocab):<12} {pct:.1f}%")

    # ------------------------------------------------------------------ #
    # 8. Images per recipe distribution
    # ------------------------------------------------------------------ #
    section("8. Images available per recipe (downloaded)")
    imgs_per_recipe_counts = [
        len([
            img_id for img_id in layer2_index.get(rid, [])
            if img_id in image_index
        ])
        for part_ids in partitions.values()
        for rid in part_ids
    ]
    total_samples = sum(imgs_per_recipe_counts)
    sorted_ipr = sorted(imgs_per_recipe_counts)
    ipr_mean   = sum(sorted_ipr) / len(sorted_ipr) if sorted_ipr else 0
    ipr_median = sorted_ipr[len(sorted_ipr) // 2]
    ipr_p25    = sorted_ipr[len(sorted_ipr) // 4]
    ipr_p75    = sorted_ipr[3 * len(sorted_ipr) // 4]
    ipr_min    = sorted_ipr[0]
    ipr_max    = sorted_ipr[-1]

    print(f"  Mean images/recipe   : {ipr_mean:>8.2f}")
    print(f"  Median               : {ipr_median:>8}")
    print(f"  Min / Max            : {ipr_min:>4} / {ipr_max}")
    print(f"  25th / 75th pct      : {ipr_p25:>4} / {ipr_p75}")
    print(f"  Total samples (flat) : {total_samples:>8,}  (recipes × images)")

    img_per_recipe = Counter(imgs_per_recipe_counts)
    print(f"\n  Distribution (top 20 buckets):")
    for n_imgs in sorted(img_per_recipe)[:20]:
        print(f"  {n_imgs:>4} image(s): {img_per_recipe[n_imgs]:>8,} recipes")

    print(f"\n{SEP}\n  Done.\n{SEP}\n")


if __name__ == "__main__":
    main()
