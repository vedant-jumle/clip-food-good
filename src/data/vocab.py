from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional, Sequence


def build_vocab(
    recipes: Sequence[dict],
    top_n: Optional[int] = None,
    min_freq: int = 1,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    exclude_set = set(exclude or [])
    counter = Counter()

    for recipe in recipes:
        for ing in recipe.get("ingredients", []):
            if ing and ing not in exclude_set:
                counter[ing] += 1

    items = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if top_n is not None:
        items = items[:top_n]

    return [token for token, _ in items]
