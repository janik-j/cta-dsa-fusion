"""Rendering (image-space) losses."""

from __future__ import annotations

from typing import Any

import torch

from training.losses.base import RenderingLoss
from training.losses.loss_utils import ssim


class L1Loss(RenderingLoss):
    """L1 (Mean Absolute Error) loss for pixel-wise reconstruction."""

    def _compute(
        self,
        image: torch.Tensor,
        gt_image: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.mean(torch.abs(image - gt_image)), {}


class DSSIMLoss(RenderingLoss):
    """DSSIM (Structural Dissimilarity) loss: 1 - SSIM."""

    def __init__(self, weight: float = 0.2, window_size: int = 11, **kwargs):
        super().__init__(weight=weight, **kwargs)
        self.window_size = int(window_size)

    def _compute(
        self,
        image: torch.Tensor,
        gt_image: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ssim_val = ssim(image, gt_image, window_size=self.window_size)
        if not torch.isfinite(ssim_val):
            ssim_val = torch.tensor(0.0, device=image.device)
        dssim = 1.0 - ssim_val
        return dssim, {"ssim": ssim_val}


__all__ = ["DSSIMLoss", "L1Loss"]
