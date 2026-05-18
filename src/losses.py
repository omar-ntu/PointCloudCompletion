from __future__ import annotations

import torch


def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer Distance for batches of point clouds.

    a: [B, Na, 3]
    b: [B, Nb, 3]
    """
    d = torch.cdist(a, b, p=2)
    return d.min(dim=2).values.mean(dim=1).mean() + d.min(dim=1).values.mean(dim=1).mean()


@torch.no_grad()
def fscore(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    d = torch.cdist(a, b, p=2)
    precision = (d.min(dim=2).values < threshold).float().mean(dim=1)
    recall = (d.min(dim=1).values < threshold).float().mean(dim=1)
    return (2 * precision * recall / (precision + recall + 1e-8)).mean()
