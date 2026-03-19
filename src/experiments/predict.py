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

