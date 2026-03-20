from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

class IngredientHead(nn.Module):
    """
    Prediction head for mulitple label ingredient prediction.
    Maps CLIP image embeddings to ingredient probs
    """
    def __init__(self, input_dim: int, num_ingredients: int):
        """
        input_dim: dimension of the CLIP img embedds
        num_ingredients: size of the ingredient vocab
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_ingredients)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: img embeddings of shape (N, input_dim)

        Returns:
        Ingredient probs of shape (N, num_ingredients),
        with values in [0,1]
        """
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs