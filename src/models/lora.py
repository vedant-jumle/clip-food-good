from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

# Lora implementation for fine tuning clip
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 32, alpha: float = 1.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.A = nn.Parameter(torch.randn(self.in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, self.out_features))

    @property
    def weight(self) -> torch.Tensor:
        delta = (self.A @ self.B).transpose(0, 1)
        return self.linear.weight + delta * self.scale

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)

# Iterate through transformer blocks in clip model
def _iter_transformer_blocks(model) -> Iterable[nn.Module]:
    if hasattr(model, "visual") and hasattr(model.visual, "transformer"):
        yield from model.visual.transformer.resblocks
    if hasattr(model, "transformer"):
        yield from model.transformer.resblocks

# Freeze Clip and use lora for each attention output
def apply_lora_to_clip(model, rank: int = 32, alpha: float = 1.0) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in _iter_transformer_blocks(model):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "out_proj"):
            continue
        if isinstance(attn.out_proj, LoRALinear):
            continue
        attn.out_proj = LoRALinear(attn.out_proj, rank=rank, alpha=alpha)

# Return lora parameters
def get_lora_parameters(model) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
