"""Image utilities for clinical DSA pipeline.

Handles loading DiffDRR registration .pt files and zoom transformations.
"""

from __future__ import annotations

import pathlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Cross-platform .pt loading (handles WindowsPath in pickled dicts)
# ---------------------------------------------------------------------------


class _WindowsPathFixUnpickler(pickle.Unpickler):
    """Convert WindowsPath → PosixPath when loading on Linux."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PosixPath
        return super().find_class(module, name)


class _PickleModule:
    Unpickler = _WindowsPathFixUnpickler


def load_pt_fixed(pt_path: Path) -> Any:
    """Load a ``.pt`` file with WindowsPath → PosixPath compatibility fix."""
    import torch

    return torch.load(
        pt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_PickleModule,
    )


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def zoom_centered(img: np.ndarray, zoom_factor: float, *, order: int = 1) -> np.ndarray:
    """Zoom into image centre by *zoom_factor* (>1 zooms in), preserving shape."""
    if zoom_factor == 1.0:
        return img

    h, w = img.shape[:2]
    zh, zw = int(h / zoom_factor), int(w / zoom_factor)
    y0, x0 = (h - zh) // 2, (w - zw) // 2

    is_rgb = img.ndim == 3
    if is_rgb:
        cropped = img[y0 : y0 + zh, x0 : x0 + zw, :]
        zoomed = zoom(cropped, (h / zh, w / zw, 1), order=order)
    else:
        cropped = img[y0 : y0 + zh, x0 : x0 + zw]
        zoomed = zoom(cropped, (h / zh, w / zw), order=order)
    return zoomed.astype(img.dtype)


