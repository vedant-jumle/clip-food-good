from recipe1m import load_recipes
from vocab import build_vocab

recipes = load_recipes(
    det_ingrs_path="../data/recipe1m/det_ingrs.json",
    layer1_path="../data/recipe1m/layer1.json",
    image_root="../data/recipe1m/0",
    partition="train",
    require_images=False,
)

print("Loaded recipes:", len(recipes))
print(recipes[0])

vocab = build_vocab(
    recipes,
    top_n=5000,
    min_freq=2,
    specials=["<pad>", "<unk>"],
)

print("Vocab size:", len(vocab))
print(vocab[:20])

# import json

# with open("data/recipe1M/layer1.json", "r", encoding="utf-8") as f:
#     layer = json.load(f)

# print(layer[0].keys())