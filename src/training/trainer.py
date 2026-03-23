from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.clip_wrapper import CLIPWrapper
from src.models.ingredient_head import IngredientHead

def train(
    clip: CLIPWrapper,
    head: IngredientHead,
    dataloader: DataLoader,
    strategy: str,
    epochs: int = 5,
    lr_head: float = 1e-4,
    lr_clip: float = 1e-5,
    device: str = "cuda",
) -> List[float]:
    clip.model = clip.model.to(device)
    head = head.to(device)

    if strategy not in {"head_only", "partial_unfreeze"}:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected 'head_only' or 'partial_unfreeze'"
        )
    
    head.train()

    if strategy == "head_only":
        for p in clip.model.parameters():
            p.requires_grad_(False)

        clip.model.eval()

        optimizer = AdamW(head.parameters(), lr=lr_head)

    else:
        for p in clip.model.parameters():
            p.requires_grad_(False)

        for p in clip.model.visual.transformer.resblocks[-1].parameters():
            p.requires_grad_(True)

        clip.model.train()

        optimizer = AdamW(
            [
                {"params": head.parameters(), "lr": lr_head},
                {
                    "params": clip.model.visual.transformer.resblocks[-1].parameters(),
                    "lr": lr_clip,
                },
            ]
        ) 

    epoch_losses: List[float] = []
    epoch_bar = tqdm(range(epochs), desc=f"Training [{strategy}]")
    for epoch in epoch_bar:
        head.train()

        if strategy == "head_only":
            clip.model.eval()
        else:
            clip.model.train()

        running_loss = 0.0
        num_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if strategy == "head_only":
                image_emb = clip.encode_image(images)
            else:
                image_emb = clip.model.encode_image(images)
                image_emb = F.normalize(image_emb, dim=-1)

            preds = head(image_emb)
            loss = F.binary_cross_entropy(preds, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(epoch_loss)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")

    return epoch_losses