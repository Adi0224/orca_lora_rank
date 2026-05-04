import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def stabilize_features(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Add tiny jitter to prevent geomloss failures on degenerate pooled clouds."""
    noise = torch.randn_like(feats) * eps
    return feats + noise


def compute_otdd(
    target_feats: torch.Tensor,
    target_labels: torch.Tensor,
    source_feats: torch.Tensor,
    source_labels: torch.Tensor,
    maxsamples_per_class: int = 64,
    use_exact: bool = False,
):
    """
    Compute OTDD between target and source feature sets.
    Uses geomloss Sinkhorn divergence for differentiable transport.

    Args:
        target_feats: (N, D) features from target (hard) set through adapter
        target_labels: (N,) class labels for target
        source_feats: (M, D) features from source (easy) set, detached
        source_labels: (M,) class labels for source
        maxsamples_per_class: max samples per class bucket
        use_exact: if True, use exact OT (slower); else Sinkhorn approx

    Returns:
        Differentiable scalar OTDD loss
    """
    from geomloss import SamplesLoss

    # Use Sinkhorn divergence (differentiable, fast)
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, scaling=0.8)

    # Compute per-class transport and aggregate
    unique_labels = torch.unique(source_labels)
    total_loss = torch.tensor(0.0, device=target_feats.device, requires_grad=True)
    n_pairs = 0

    for label in unique_labels:
        src_mask = source_labels == label
        tgt_mask = target_labels == label

        if src_mask.sum() < 2 or tgt_mask.sum() < 2:
            continue

        src_cls = source_feats[src_mask][:maxsamples_per_class]
        tgt_cls = target_feats[tgt_mask][:maxsamples_per_class]

        # Stabilize
        src_cls = stabilize_features(src_cls)
        tgt_cls = stabilize_features(tgt_cls)

        dist = loss_fn(tgt_cls, src_cls)
        total_loss = total_loss + dist
        n_pairs += 1

    if n_pairs > 0:
        total_loss = total_loss / n_pairs

    return total_loss
