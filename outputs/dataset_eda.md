# Dataset EDA — Recipe1M+ (Shard 0)

Run: 2026-03-25 | Image root: `data/recipe1m/0` | 424,304 images on disk

---

## 1. JSON-level counts

| | Count |
|---|---|
| det_ingrs entries | 1,029,720 |
| layer1 entries | 1,029,720 |
| layer2 entries | 402,760 |
| shared (det∩layer1) | 1,029,720 |

---

## 2. Downloaded images

| | Count |
|---|---|
| Total image files on disk | 424,304 |

---

## 3. Usable recipes (with images)

| Partition | Count |
|---|---|
| **Total** | **18,496** |
| train | 12,801 |
| val | 2,755 |
| test | 2,940 |
| Dropped (no image) | 1,010,201 |
| Dropped (no ingredients) | 1,023 |

> 98% of recipes have no images in shard 0. Each additional shard adds ~18k usable recipes.

---

## 4. Ingredient statistics

| Metric | Value |
|---|---|
| Mean ingredients/recipe | 8.83 |
| Median | 8 |
| Min / Max | 1 / 62 |
| 25th / 75th percentile | 6 / 11 |
| Unique ingredients (raw) | 6,148 |
| Unique ingredients (visual) | 6,128 |

### Frequency distribution

| Bucket | # ingredients |
|---|---|
| >= 1000 | 22 |
| 100–999 | 228 |
| 10–99 | 1,257 |
| 2–9 | 2,573 |
| 1 (hapax) | 2,068 |

---

## 5. Top 30 ingredients (all)

| Rank | Ingredient | Count |
|---|---|---|
| 1 | salt | 7,007 |
| 2 | butter | 5,260 |
| 3 | sugar | 3,349 |
| 4 | eggs | 3,318 |
| 5 | water | 2,781 |
| 6 | all-purpose flour | 2,717 |
| 7 | olive oil | 2,717 |
| 8 | milk | 2,563 |
| 9 | garlic cloves | 2,145 |
| 10 | white sugar | 1,843 |
| 11 | vanilla extract | 1,808 |
| 12 | egg | 1,777 |
| 13 | baking powder | 1,771 |
| 14 | onion | 1,751 |
| 15 | flour | 1,736 |
| 16 | brown sugar | 1,694 |
| 17 | vegetable oil | 1,509 |
| 18 | baking soda | 1,507 |
| 19 | onions | 1,359 |
| 20 | salt and pepper | 1,225 |
| 21 | parmesan cheese | 1,089 |
| 22 | pepper | 1,060 |
| 23 | lemon juice | 984 |
| 24 | garlic powder | 973 |
| 25 | ground cinnamon | 906 |
| 26 | sour cream | 899 |
| 27 | vanilla | 893 |
| 28 | cinnamon | 876 |
| 29 | cream cheese | 869 |
| 30 | unsalted butter | 867 |

---

## 6. Top 30 visual ingredients (non-visual excluded)

| Rank | Ingredient | Count |
|---|---|---|
| 1 | eggs | 3,318 |
| 2 | all-purpose flour | 2,717 |
| 3 | milk | 2,563 |
| 4 | garlic cloves | 2,145 |
| 5 | white sugar | 1,843 |
| 6 | egg | 1,777 |
| 7 | onion | 1,751 |
| 8 | brown sugar | 1,694 |
| 9 | onions | 1,359 |
| 10 | salt and pepper | 1,225 |
| 11 | parmesan cheese | 1,089 |
| 12 | lemon juice | 984 |
| 13 | ground cinnamon | 906 |
| 14 | sour cream | 899 |
| 15 | vanilla | 893 |
| 16 | cinnamon | 876 |
| 17 | cream cheese | 869 |
| 18 | unsalted butter | 867 |
| 19 | soy sauce | 846 |
| 20 | honey | 819 |
| 21 | ground black pepper | 807 |
| 22 | garlic clove | 803 |
| 23 | garlic | 760 |
| 24 | green onions | 729 |
| 25 | carrots | 694 |
| 26 | tomatoes | 689 |
| 27 | mayonnaise | 643 |
| 28 | granulated sugar | 636 |
| 29 | celery | 596 |
| 30 | cornstarch | 576 |

---

## 7. Vocab coverage

| top_n | % recipes with ≥1 match |
|---|---|
| 50 | 86.0% |
| 100 | 91.4% |
| 200 | 95.4% |
| 500 | 98.0% |
| 1000 | 99.2% |

> top_n=200 (current default) covers 95.4% of recipes.

---

## 8. Images per recipe (matched to downloaded shard)

| Metric | Value |
|---|---|
| Mean images/recipe | 1.20 |
| Median | 1 |
| Min / Max | 1 / 53 |
| 25th / 75th percentile | 1 / 1 |
| **Total samples after expand_recipes()** | **22,281** |

### Distribution

| Images | Recipes |
|---|---|
| 1 | 16,824 |
| 2 | 1,058 |
| 3 | 264 |
| 4 | 120 |
| 5 | 73 |
| 6 | 34 |
| 7 | 32 |
| 8 | 21 |
| 9 | 17 |
| 10 | 10 |
| 11+ | 50 |

> Most recipes have only 1 matched image in shard 0 (median=1). The 424k images on disk mostly belong to recipes whose metadata is in other shards.
