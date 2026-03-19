import torch

# here are all the helper functions
def validate_inputs(preds: torch.Tensor, labels: torch.Tensor, k: int) -> None:
    if preds.ndim != 2:
        raise ValueError(f"preds must have shape (N, k), got {tuple(preds.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N, V), got {tuple(labels.shape)}")
    if preds.size(0) != labels.size(0):
        raise ValueError(
            f"preds and labels must have same batch size, got {preds.size(0)} and {labels.size(0)}"
        )
    if preds.size(1) != k:
        raise ValueError(
            f"preds second dimension must equal k={k}, got {preds.size(1)}"
        )
        
def true_positives(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    gathered = torch.gather(labels, dim=1, index=preds.long())
    return gathered.sum(dim=1)


# here are where we have our actual functions!!
def precision_at_k(preds: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    validate_inputs(preds, labels, k)
    
    valid_mask = labels.sum(dim=1) > 0
    if valid_mask.sum().item() == 0:
        return 0.0
    
    preds_valid = preds[valid_mask]
    labels_valid = labels[valid_mask]
    
    tp = true_positives(preds_valid, labels_valid).float()
    precision = tp / float(k)
    
    return precision.mean().item()
    
def recall_at_k(preds: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    validate_inputs(preds, labels, k)
    
    valid_mask = labels.sum(dim=1) > 0
    if valid_mask.sum().item() == 0:
        return 0.0
    
    preds_valid = preds[valid_mask]
    labels_valid = labels[valid_mask]
    
    tp = true_positives(preds_valid, labels_valid).float()
    gt_counts = labels_valid.sum(dim=1).float()
    recall = tp / gt_counts
    
    return recall.mean().item()
    
def f1_at_k(preds: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    validate_inputs(preds, labels, k)
    
    valid_mask = labels.sum(dim=1) > 0
    if valid_mask.sum().item() == 0:
        return 0.0
    
    preds_valid = preds[valid_mask]
    labels_valid = labels[valid_mask]
    
    tp = true_positives(preds_valid, labels_valid).float()
    gt_counts = labels_valid.sum(dim=1).float()
    precision = tp / float(k)
    recall = tp / gt_counts

    denom = precision + recall
    f1 = torch.where(
        denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom)
    )
    
    return f1.mean().item()

