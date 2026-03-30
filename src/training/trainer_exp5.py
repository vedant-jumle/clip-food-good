from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.clip_wrapper import CLIPWrapper
from src.experiments.prompts import make_prompts


def asymmetric_ingredient_loss(
    sims: torch.Tensor,
    labels: torch.Tensor,
    neg_weights: torch.Tensor,
) -> torch.Tensor:
    """Asymmetric multi-label BCE loss.

    False positives on rare ingredients are penalized heavily (neg_weights[i] is
    large for rare ingredients). False negatives are treated softly (weight = 1.0).

    Args:
        sims:        (B, V) raw logit similarities
        labels:      (B, V) multi-hot float labels
        neg_weights: (V,)   per-ingredient negative penalty (inverse frequency)

    Returns:
        scalar loss
    """
    # pos_weight in BCEWithLogitsLoss scales the positive term only
    # We want asymmetry on the negative side, so we compute manually:
    # loss = -[ labels * log(sigma(s)) + neg_weights * (1 - labels) * log(1 - sigma(s)) ]
    sig = torch.sigmoid(sims)
    pos_loss = labels * torch.log(sig + 1e-8)
    neg_loss = neg_weights.unsqueeze(0) * (1.0 - labels) * torch.log(1.0 - sig + 1e-8)
    return -(pos_loss + neg_loss).mean()


def train_exp5(
    clip: CLIPWrapper,
    dataloader: DataLoader,
    vocab: list[str],
    vocab_freqs: list[int],
    epochs: int = 10,
    lr: float = 1e-5,
    device: str = "cuda",
    patience: int = 5,
    min_delta: float = 0.001,
    prompt_type: str = "B",
    checkpoint_path: str = "outputs/checkpoints/exp5_best.pt",
) -> list[float]:
    """Fine-tune CLIP image encoder with asymmetric frequency-weighted ingredient loss.

    Text embeddings for all vocab ingredients are used as fixed classifier weights.
    The image encoder is fine-tuned end-to-end. Rare ingredients get a higher
    false-positive penalty than common ones.

    Args:
        clip:        CLIPWrapper (image encoder will be fine-tuned)
        dataloader:  training DataLoader
        vocab:       list of ingredient strings (length V)
        vocab_freqs: ingredient frequencies matching vocab order (length V)
        epochs:      max training epochs
        lr:          learning rate (lower than LoRA since full encoder is updated)
        device:      cuda or cpu
        patience:    early stopping patience
        min_delta:   minimum loss improvement to reset patience
        prompt_type: prompt template to use for ingredient text embeddings
    """
    # Unfreeze image encoder, freeze text encoder
    clip.model.to(device)
    for p in clip.model.parameters():
        p.requires_grad_(False)
    for p in clip.model.visual.parameters():
        p.requires_grad_(True)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        clip.model = torch.nn.DataParallel(clip.model)

    base_model = clip.model.module if hasattr(clip.model, "module") else clip.model

    optimizer = AdamW(
        [p for p in base_model.visual.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Build per-ingredient negative weights: max_freq / freq[i]
    # Rare ingredients get higher penalty for false positives
    freqs = torch.tensor(vocab_freqs, dtype=torch.float32)
    neg_weights = (freqs.max() / freqs).to(device)
    # Clamp to avoid extreme values for very rare ingredients
    neg_weights = neg_weights.clamp(max=10.0)

    # Pre-compute text embeddings once — these are fixed classifier weights
    print(f"Pre-computing text embeddings for {len(vocab)} ingredients...")
    with torch.no_grad():
        text_embs = clip.encode_text(make_prompts(vocab, prompt_type))  # (V, D)
    text_embs = text_embs.to(device)

    logit_scale = base_model.logit_scale.exp()

    epoch_losses: list[float] = []
    best_loss = float("inf")
    epochs_without_improvement = 0

    base_model.visual.train()

    epoch_bar = tqdm(range(epochs), desc="Exp5 training", leave=True)
    for epoch in epoch_bar:
        running_loss = 0.0
        num_batches = 0

        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for batch in batch_bar:
            images = batch["image"].to(device)
            labels = batch["labels"].float().to(device)  # (B, V)

            optimizer.zero_grad()

            # Encode images
            image_embs = base_model.encode_image(images)
            image_embs = F.normalize(image_embs, dim=-1)  # (B, D)

            # Similarity against all ingredient text embeddings
            sims = image_embs @ text_embs.T * logit_scale  # (B, V)

            loss = asymmetric_ingredient_loss(sims, labels, neg_weights)
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
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(base_model.state_dict(), checkpoint_path)
            tqdm.write(f"Checkpoint saved (loss={best_loss:.4f}) -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", no_improve=epochs_without_improvement)

        if epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    base_model.visual.eval()
    return epoch_losses
