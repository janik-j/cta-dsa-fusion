"""
System utilities for this framework.

Focused helpers for filesystem, reproducibility seeding, and training-state resume.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("PyTorch is required for training and evaluation utilities.") from exc
    return torch


def mkdir_p(folder_path: str | Path) -> None:
    """Create directory if it doesn't exist (like mkdir -p)."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def searchForMaxIteration(folder: str | Path) -> int:
    """Find the maximum iteration number in a checkpoint folder."""
    folder_p = Path(folder)
    if not folder_p.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {folder_p}")
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder_p)]
    if not saved_iters:
        raise ValueError(f"No iterations found under: {folder_p}")
    return max(saved_iters)


def seed_everything(seed: int) -> int:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value (int).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def _checkpoint_dir(model_path: str | Path, iteration: int) -> Path:
    return Path(model_path) / "point_cloud" / f"iteration_{int(iteration)}"


def capture_rng_state() -> dict[str, Any]:
    torch = _require_torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    state["cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch = _require_torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        cuda_state = state.get("cuda")
        if cuda_state is None:
            raise ValueError("Checkpoint missing CUDA RNG state but CUDA is available.")
        torch.cuda.set_rng_state_all(cuda_state)


def save_training_state(
    *,
    model_path: str | Path,
    iteration: int,
    gaussians,
) -> Path:
    torch = _require_torch()
    ckpt_dir = _checkpoint_dir(model_path, iteration)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / "training_state.pth"

    if getattr(gaussians, "optimizer", None) is None:
        raise ValueError("Cannot save training state: gaussians.optimizer is not initialized.")

    state = {
        "iteration": int(iteration),
        "optimizer": gaussians.optimizer.state_dict(),
        "rng": capture_rng_state(),
        "densification": {
            "xyz_gradient_accum": getattr(gaussians, "xyz_gradient_accum", None),
            "avgopacity_accum": getattr(gaussians, "avgopacity_accum", None),
            "denom": getattr(gaussians, "denom", None),
            "max_radii2D": getattr(gaussians, "max_radii2D", None),
        },
    }

    torch.save(state, out_path)
    return out_path


def load_training_state(*, model_path: str | Path, iteration: int) -> dict[str, Any]:
    torch = _require_torch()
    ckpt_dir = _checkpoint_dir(model_path, iteration)
    path = ckpt_dir / "training_state.pth"
    if not path.exists():
        raise FileNotFoundError(f"Missing training state for resume: {path}")
    # PyTorch >=2.6 defaults `weights_only=True`, which breaks loading our
    # full training-state checkpoint (optimizer + RNG state with Python/NumPy objects).
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only arg.
        return torch.load(path, map_location="cpu")


def apply_training_state(*, gaussians, state: dict[str, Any]) -> None:
    torch = _require_torch()
    expected_iter = int(state.get("iteration", -1))
    if expected_iter < 0:
        raise ValueError("Invalid training state: missing `iteration`.")

    if getattr(gaussians, "optimizer", None) is None:
        raise ValueError("Cannot load training state: gaussians.optimizer is not initialized.")
    gaussians.optimizer.load_state_dict(state["optimizer"])
    device = gaussians.get_xyz.device
    for st in gaussians.optimizer.state.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.to(device=device)

    dens = state.get("densification", {}) or {}
    for name in ("xyz_gradient_accum", "avgopacity_accum", "denom", "max_radii2D"):
        value = dens.get(name)
        if value is not None:
            setattr(gaussians, name, value.to(device=device))

    rng_state = state.get("rng")
    if not isinstance(rng_state, dict):
        raise ValueError("Invalid training state: missing `rng` dict.")
    restore_rng_state(rng_state)
