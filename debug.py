# from recipe1m import load_recipes

# recipes = load_recipes(
#     det_ingrs_path="data/recipe1M/det_ingrs.json",
#     layer1_path="data/recipe1M/layer1.json",
#     image_root="data/recipe1M/0",
#     partition="train",
#     require_images=True,
# )

# print("Loaded recipes:", len(recipes))
# print(recipes[0])

import json

with open("data/recipe1M/layer1.json", "r", encoding="utf-8") as f:
    layer = json.load(f)

print(layer[0].keys())