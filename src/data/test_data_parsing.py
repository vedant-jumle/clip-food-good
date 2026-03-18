import sys
from recipe1m import load_recipes, index_det_ingrs, index_layer1, index_layer2, build_image_index
from vocab import build_vocab

DET_INGRS = "data/recipe1m/det_ingrs.json"
LAYER1 = "data/recipe1m/layer1.json"
LAYER2 = "data/recipe1m/layer2.json"
IMAGE_ROOT = "data/recipe1m/0"

print("=== Step 1: Parsing det_ingrs.json ...")
det_index = index_det_ingrs(DET_INGRS)
print(f"  det_ingrs entries: {len(det_index)}")

print("=== Step 2: Parsing layer1.json ...")
layer_index = index_layer1(LAYER1)
print(f"  layer1 entries: {len(layer_index)}")

print("=== Step 3: Parsing layer2.json ...")
layer2_index = index_layer2(LAYER2)
print(f"  layer2 entries: {len(layer2_index)}")

print("=== Step 4: Building image index ...")
image_index = build_image_index(IMAGE_ROOT)
print(f"  images on disk: {len(image_index)}")

shared = det_index.keys() & layer_index.keys()
print(f"\n  shared recipe IDs (det & layer1): {len(shared)}")
in_layer2 = sum(1 for rid in shared if rid in layer2_index)
print(f"  of those, in layer2: {in_layer2}")
with_images = sum(
    1 for rid in shared
    if any(img_id in image_index for img_id in layer2_index.get(rid, []))
)
print(f"  of those, with at least one image on disk: {with_images}")

print("\n=== Step 5: Loading recipes (partition=train, require_images=True) ...")
recipes = load_recipes(
    det_ingrs_path=DET_INGRS,
    layer1_path=LAYER1,
    layer2_path=LAYER2,
    image_root=IMAGE_ROOT,
    partition="train",
    require_images=True,
)
print(f"  Loaded recipes: {len(recipes)}")

if not recipes:
    print("\nNo recipes loaded. Check that images are present in IMAGE_ROOT.")
    sys.exit(1)

print("\n=== First recipe:")
r = recipes[0]
print(f"  id:          {r['id']}")
print(f"  title:       {r['title']}")
print(f"  partition:   {r['partition']}")
print(f"  ingredients: {r['ingredients']}")
print(f"  image_path:  {r['image_path']}")

print("\n=== Step 6: Building vocab (top_n=50, min_freq=1) ...")
vocab = build_vocab(recipes, top_n=50, min_freq=1)
print(f"  Vocab size: {len(vocab)}")
print(f"  Top 20: {vocab[:20]}")