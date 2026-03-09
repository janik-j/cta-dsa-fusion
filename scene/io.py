"""I/O helpers for dataset loading (projections, transforms, PLY)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from plyfile import PlyData


@dataclass(frozen=True)
class ProjectionPaths:
    projections_dir: Path
    mask_run_file: Path
    fill_run_file: Path


def resolve_projection_paths(datapath: str | Path) -> ProjectionPaths:
    root = Path(datapath)
    projections_dir = root / "projections"
    if not projections_dir.exists():
        raise FileNotFoundError(f"Projections directory not found at {projections_dir}")
    mask_run_file = projections_dir / "mask_run.nii.gz"
    fill_run_file = projections_dir / "fill_run.nii.gz"
    if not mask_run_file.exists():
        raise FileNotFoundError(f"Missing projection file: {mask_run_file}")
    if not fill_run_file.exists():
        raise FileNotFoundError(f"Missing projection file: {fill_run_file}")
    return ProjectionPaths(
        projections_dir=projections_dir,
        mask_run_file=mask_run_file,
        fill_run_file=fill_run_file,
    )


def load_transforms_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"transforms.json not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"transforms.json must be a JSON object: {p}")
    return data


def load_mask_fill(paths: ProjectionPaths) -> tuple[np.ndarray, np.ndarray]:
    mask_run = sitk.GetArrayFromImage(sitk.ReadImage(str(paths.mask_run_file)))
    fill_run = sitk.GetArrayFromImage(sitk.ReadImage(str(paths.fill_run_file)))
    return mask_run, fill_run


def read_ply_vertices(ply_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read xyz/opacity/normals from a PLY (vertex element)."""
    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    points = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)

    if "opacity" in v.data.dtype.names:
        opacities = np.asarray(v["opacity"], dtype=np.float32).reshape(-1, 1)
    else:
        opacities = np.ones((points.shape[0], 1), dtype=np.float32) * 0.1

    normals = None
    if all(k in v.data.dtype.names for k in ("nx", "ny", "nz")):
        normals = np.stack([np.asarray(v["nx"]), np.asarray(v["ny"]), np.asarray(v["nz"])], axis=1).astype(np.float32)

    return points, opacities, normals


def subsample_deterministic(
    points: np.ndarray,
    opacities: np.ndarray,
    normals: np.ndarray | None,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Deterministically subsample to at most max_points (stable, no RNG)."""
    if max_points is None:
        return points, opacities, normals
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, opacities, normals

    step = float(points.shape[0]) / float(max_points)
    idx = (np.arange(max_points) * step).astype(np.int64)
    idx = np.clip(idx, 0, points.shape[0] - 1)
    return points[idx], opacities[idx], (normals[idx] if normals is not None else None)
