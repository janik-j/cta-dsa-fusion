"""
Image utilities for this framework.

Small, dependency-light metrics and normalization helpers.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def psnr(img1: torch.Tensor | np.ndarray, img2: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Compute PSNR between two batched images. Returns tensor."""
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between two images. Returns scalar."""
    mse = ((pred - target) ** 2).mean()
    mse_v = float(mse.item() if torch.is_tensor(mse) else mse)
    return -10 * math.log10(mse_v) if mse_v != 0.0 else float("inf")


def data_norm(data: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Normalize data to [0, 1] range."""
    if torch.is_tensor(data):
        denom = data.max() - data.min()
        if float(denom.detach()) == 0.0:
            return torch.zeros_like(data)
        return (data - data.min()) / denom

    denom = data.max() - data.min()
    if float(denom) == 0.0:
        return np.zeros_like(data)
    return (data - data.min()) / denom


def get_ssim_2d(pred: np.ndarray, target: np.ndarray, data_range: float = 1) -> float:
    """Compute SSIM of two 2D arrays."""
    from skimage.metrics import structural_similarity

    return structural_similarity(pred, target, data_range=data_range)


class Averager:
    """Running average accumulator."""

    def __init__(self) -> None:
        self.sum: float = 0
        self.count: int = 0

    def add(self, value: float) -> None:
        self.sum += value
        self.count += 1

    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0
