import torch


def compute_scores(image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    return image_embeddings @ text_embeddings.T      

def predict_topk(scores: torch.Tensor, k: int = 5) -> torch.Tensor:
       if k <= 0:
           raise ValueError(f"k must be positive, got {k}")
       
       if scores.ndim != 2:
           raise ValueError(f"scores must have shape (N, V), got {tuple(scores.shape)}")
       
       vocab_size = scores.size(1)
       if k > vocab_size:
           raise ValueError(f"k={k} cannot be greater than vocab size V={vocab_size}")
       
       return torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices


def predict_adaptive(
    scores: torch.Tensor,
    alpha: float = 0.5,
    min_k: int = 1,
    max_k: int = 10,
) -> torch.Tensor:
    """Predict top-k ingredients where k is chosen adaptively per batch.

    For each image, counts ingredients whose score exceeds mean + alpha * std.
    Uses the median count across the batch as k, clamped to [min_k, max_k].

    Args:
        scores: (N, V) similarity scores.
        alpha: threshold sensitivity — higher means fewer predictions.
        min_k: minimum number of predictions.
        max_k: maximum number of predictions.

    Returns:
        (N, k) indices of top-k ingredients.
    """
    thresholds = scores.mean(dim=1, keepdim=True) + alpha * scores.std(dim=1, keepdim=True)
    above = (scores > thresholds).sum(dim=1).float()  # (N,)
    k = int(above.median().clamp(min_k, max_k).item())
    return torch.topk(scores, k=k, dim=1, largest=True, sorted=True).indices
