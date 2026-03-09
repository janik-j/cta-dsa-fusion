"""
Loss primitives used by training losses.

Kept separate from the loss classes to:
- keep `rendering.py`/`manager.py` readable
- avoid mixing low-level tensor ops with scheduling/orchestration
"""

from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(network_output - gt).mean()


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def _as_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        return x
    if x.dim() == 3:  # CHW
        return x.unsqueeze(0)
    if x.dim() == 2:  # HW
        return x.unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Expected 2D/3D/4D tensor, got shape={tuple(x.shape)}")


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int, channel: int) -> torch.Tensor:
    # CuDNN's v8 conv path may require nvrtc at runtime; disable it here to keep
    # this small fixed-kernel SSIM computation portable in plain `uv run` shells.
    with torch.backends.cudnn.flags(enabled=False):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    return ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    img1 = _as_nchw(img1)
    img2 = _as_nchw(img2)

    channel = int(img1.shape[1])
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=(1, 2, 3))


__all__ = [
    "l1_loss",
    "ssim",
]
