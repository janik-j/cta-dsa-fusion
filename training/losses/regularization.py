"""Regularization losses (Gaussian/geometry-space)."""

from __future__ import annotations

from typing import Any

import torch

from training.losses.base import RegularizationLoss


class PriorAnchoringLoss(RegularizationLoss):
    """Pull Gaussians toward the prior surface (outpainting-friendly)."""

    def __init__(
        self,
        weight: float = 0.01,
        start_iter: int = 0,
        anchor_radius: float = 0.5,
        k: int = 8,
        sample_size: int = 2048,
        **kwargs,
    ):
        super().__init__(weight=weight, start_iter=start_iter, **kwargs)
        self.anchor_radius = float(anchor_radius)
        self.k = int(k)
        self.sample_size = int(sample_size)

    def _compute(
        self, gaussians, prior_xyz: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        xyz = gaussians.get_xyz

        if prior_xyz is None or prior_xyz.numel() == 0:
            return torch.tensor(0.0, device=xyz.device), {"n_anchored": 0}

        n_points = int(xyz.shape[0])
        if n_points < 1:
            return torch.tensor(0.0, device=xyz.device), {"n_anchored": 0}

        sample_size = min(max(1, self.sample_size), n_points)
        idx = torch.randperm(n_points, device=xyz.device)[:sample_size]
        sampled_xyz = xyz[idx]

        prior_xyz_device = prior_xyz.to(xyz.device)
        dists = torch.cdist(sampled_xyz, prior_xyz_device, p=2)
        min_dist = dists.min(dim=1).values

        anchor_mask = min_dist <= self.anchor_radius
        n_anchored = int(anchor_mask.sum().item())
        if n_anchored == 0:
            return torch.tensor(0.0, device=xyz.device), {"n_anchored": 0}

        k = min(self.k, int(prior_xyz.shape[0]))
        knn_idx = torch.topk(dists[anchor_mask], k=k, dim=1, largest=False).indices
        prior_neighbors = prior_xyz_device[knn_idx]
        prior_mean = prior_neighbors.mean(dim=1)

        diff = sampled_xyz[anchor_mask] - prior_mean
        loss = (diff * diff).sum(dim=1).mean()

        info = {
            "n_anchored": n_anchored,
            "n_sampled": sample_size,
            "anchor_ratio": n_anchored / float(sample_size),
        }
        return loss, info


__all__ = [
    "PriorAnchoringLoss",
]
