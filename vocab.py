from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional


def build_vocab(
    recipes: List[dict],
    top_n: Optional[int] = None,
    min_freq: int = 1,
    exclude: Optional[Iterable[str]] = None,
    specials: Optional[List[str]] = None,
) -> Dict[str, int]:
    exclude = set(exclude or [])
    counter = Counter()

    for recipe in recipes:
        for ing in recipe.get("ingredients", []):
            if ing not in exclude:
                counter[ing] += 1

    items = [(token, freq) for token, freq in counter.items() if freq >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    if top_n is not None:
        items = items[:top_n]

    vocab_tokens = list(specials or []) + [token for token, _ in items]
    return {token: idx for idx, token in enumerate(vocab_tokens)}