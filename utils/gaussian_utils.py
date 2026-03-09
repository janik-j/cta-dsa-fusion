"""
Gaussian utilities for this framework.

Math helpers used by the Gaussian model and related training/eval code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Inverse of sigmoid function."""
    return torch.log(x / (1 - x))


def inverse_sigmoid_clamp(x: torch.Tensor) -> torch.Tensor:
    """Inverse sigmoid with clamping for numerical stability."""
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1 - x))


def exp_decay(initial_lr: float, final_lr: float, begin_step: int, end_step: int) -> callable:
    """Exponential decay learning rate schedule."""

    def helper(step: int) -> float:
        t = np.clip((step - begin_step) / (end_step - begin_step), 0, 1)
        return initial_lr * (final_lr / initial_lr) ** t

    return helper


def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    """Extract lower diagonal of 3x3 matrix batch."""
    uncertainty = torch.zeros((L.shape[0], 6), dtype=L.dtype, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    """Extract symmetric matrix representation."""
    return strip_lowerdiag(sym)


def build_rotation(quat: torch.Tensor) -> torch.Tensor:
    """Build rotation matrix from quaternion (w, x, y, z)."""
    q = quat / torch.norm(quat, dim=1, keepdim=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.size(0), 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _build_scaling_rotation_core(s: torch.Tensor, r: torch.Tensor, invert: bool = False) -> torch.Tensor:
    """Core function to build scaling-rotation matrix."""
    R = build_rotation(r)
    scale = 1 / s if invert else s
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    L[:, 0, 0] = scale[:, 0]
    L[:, 1, 1] = scale[:, 1]
    L[:, 2, 2] = scale[:, 2]
    return R @ L


def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build scaling-rotation matrix."""
    return _build_scaling_rotation_core(s, r, invert=False)


def get_scale_bound(dataset) -> NDArray | None:
    """Compute scale bound from dataset parameters."""
    if dataset.use_scale_bound and dataset.scale_min is not None and dataset.scale_max is not None:
        return np.asarray([dataset.scale_min, dataset.scale_max])
    return None
