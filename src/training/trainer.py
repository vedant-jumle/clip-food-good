from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from open_clip.loss import ClipLoss

from src.models.clip_wrapper import CLIPWrapper
from src.models.ingredient_head import IngredientHead
from src.models.lora import apply_lora_to_clip, get_lora_parameters

##### Helper functions #####

def build_texts(labels: torch.Tensor, vocab: list[str]) -> list[str]:
    texts: list[str] = []
    for row in labels:
        indices = row.nonzero(as_tuple=True)[0].tolist()
        ingredients = [vocab[i] for i in indices]
        texts.append("ingredients: " + ", ".join(ingredients) if ingredients else "food")
    return texts



##### Main functions #####

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

    clip.model.eval()
    return epoch_losses

def train_contrastive(
    clip: CLIPWrapper,
    dataloader: DataLoader,
    vocab: list[str],
    epochs: int = 5,
    lr: float = 1e-4,
    rank: int = 32,
    alpha: float = 1.0,
    device: str = "cuda",
) -> list[float]:
    apply_lora_to_clip(clip.model, rank=rank, alpha=alpha)
    clip.model.to(device)
    lora_params = list(get_lora_parameters(clip.model))

    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found!")

    optimizer = AdamW(lora_params, lr=lr)
    criterion = ClipLoss()

    epoch_losses: list[float] = []

    clip.model.train()

    epoch_bar = tqdm(range(epochs), desc="Contrastive training", leave=True)
    for epoch in epoch_bar:
        running_loss = 0.0
        num_batches = 0

        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in batch_bar:
            images = batch["image"].to(device)
            labels = batch["labels"]  # CPU; dataloader never puts labels on GPU

            texts = build_texts(labels, vocab)

            optimizer.zero_grad()

            image_features = clip.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            text_tokens = clip.tokenizer(texts).to(device)
            text_features = clip.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            loss = criterion(image_features, text_features, clip.model.logit_scale)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1

            batch_bar.set_postfix(loss=f"{loss_value:.4f}")

        avg_loss = running_loss / max(1, num_batches)
        epoch_losses.append(avg_loss)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}")

    return epoch_losses
