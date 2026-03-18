from recipe1m import load_recipes
from vocab import build_vocab

DET_INGRS = "data/recipe1M/det_ingrs.json"
LAYER1 = "data/recipe1M/layer1.json"
IMAGE_ROOT = "data/recipe1M/0"

recipes = load_recipes(
    det_ingrs_path=DET_INGRS,
    layer1_path=LAYER1,
    image_root=IMAGE_ROOT,
    partition="train",      # or None
    require_images=True,
)

print("Loaded recipes:", len(recipes))

if recipes:
    print("\nFirst recipe:")
    print(recipes[0])

    print("\nFields:")
    print("id:", recipes[0]["id"])
    print("title:", recipes[0]["title"])
    print("partition:", recipes[0]["partition"])
    print("num ingredients:", len(recipes[0]["ingredients"]))
    print("num images:", len(recipes[0]["image_paths"]))

vocab = build_vocab(
    recipes,
    top_n=50,
    min_freq=1,
    specials=["<pad>", "<unk>"],
)

print("\nVocab size:", len(vocab))
print("First 20 vocab items:")
print(list(vocab.items())[:20])