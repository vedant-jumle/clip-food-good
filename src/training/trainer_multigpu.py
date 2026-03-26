from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from open_clip.loss import ClipLoss

from src.models.clip_wrapper import CLIPWrapper
from src.models.lora import apply_lora_to_clip, get_lora_parameters
from src.training.trainer import build_texts


def train_contrastive_multigpu(
    clip: CLIPWrapper,
    dataloader: DataLoader,
    vocab: list[str],
    epochs: int = 5,
    lr: float = 1e-4,
    rank: int = 32,
    alpha: float = 1.0,
    device: str = "cuda",
    patience: int = 5,
    min_delta: float = 0.001,
) -> list[float]:
    apply_lora_to_clip(clip.model, rank=rank, alpha=alpha)
    clip.model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        clip.model = torch.nn.DataParallel(clip.model)

    base_model = clip.model.module if hasattr(clip.model, "module") else clip.model
    lora_params = list(get_lora_parameters(base_model))

    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found!")

    optimizer = AdamW(lora_params, lr=lr)
    criterion = ClipLoss()

    epoch_losses: list[float] = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    base_model.train()

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

            image_features = base_model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)

            text_tokens = clip.tokenizer(texts).to(device)
            text_features = base_model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)

            loss = criterion(image_features, text_features, base_model.logit_scale)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            num_batches += 1

            batch_bar.set_postfix(loss=f"{loss_value:.4f}")

        avg_loss = running_loss / max(1, num_batches)
        epoch_losses.append(avg_loss)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", no_improve=epochs_without_improvement)

        if epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    base_model.eval()
    return epoch_losses
