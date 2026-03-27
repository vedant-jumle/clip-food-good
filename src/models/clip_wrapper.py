from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
import open_clip


class CLIPWrapper:
    """Thin wrapper around OpenCLIP exposing normalized image and text encoders."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of preprocessed images.

        Args:
            images: (N, 3, 224, 224) tensor, already CLIP-preprocessed.

        Returns:
            L2-normalized embeddings of shape (N, D).
        """
        images = images.to(self.device)
        base = self.model.module if hasattr(self.model, "module") else self.model
        embeddings = base.encode_image(images)
        return F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of text strings.

        Args:
            texts: list of N strings.

        Returns:
            L2-normalized embeddings of shape (N, D).
        """
        tokens = self.tokenizer(texts).to(self.device)
        base = self.model.module if hasattr(self.model, "module") else self.model
        embeddings = base.encode_text(tokens)
        return F.normalize(embeddings, dim=-1)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality (512 for ViT-B-32)."""
        base = self.model.module if hasattr(self.model, "module") else self.model
        return base.visual.output_dim
