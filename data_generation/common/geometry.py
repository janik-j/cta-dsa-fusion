"""Shared geometry helpers for dataset generation (coords + PLY I/O)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
def voxel_to_world_coords(
    pts_voxel_xyz: np.ndarray,
    *,
    volume_origin_xyz: np.ndarray,
    volume_spacing_xyz: np.ndarray,
) -> np.ndarray:
    """
    Convert voxel-index coordinates to world (mm) coordinates.

    With voxel-center convention:
        world = origin + (i + 0.5) * spacing
    """
    pts = np.asarray(pts_voxel_xyz, dtype=np.float32)
    origin = np.asarray(volume_origin_xyz, dtype=np.float32).reshape(1, 3)
    spacing = np.asarray(volume_spacing_xyz, dtype=np.float32).reshape(1, 3)
    return (origin + (pts + 0.5) * spacing).astype(np.float32, copy=False)


def write_ply_xyz_opacity(path: Path, pts_xyz: np.ndarray, opacity: float | np.ndarray = 0.1) -> None:
    """Write XYZ point cloud with per-vertex opacity to a PLY file (ASCII)."""
    from plyfile import PlyData, PlyElement

    pts = np.asarray(pts_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts_xyz must have shape [N,3], got {pts.shape}")

    n = int(pts.shape[0])
    if np.isscalar(opacity):
        op = np.full((n,), float(opacity), dtype=np.float32)
    else:
        op = np.asarray(opacity, dtype=np.float32).reshape(-1)
        if int(op.shape[0]) != n:
            raise ValueError(f"opacity must have shape [N], got {op.shape} for N={n}")

    verts = np.empty((n,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    verts["x"] = pts[:, 0]
    verts["y"] = pts[:, 1]
    verts["z"] = pts[:, 2]
    verts["opacity"] = op

    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=True).write(str(path))


def load_ply_xyz_opacity(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ point cloud and opacity from PLY file."""
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if "opacity" in v.data.dtype.names:
        opacities = np.asarray(v["opacity"], dtype=np.float32)
    else:
        opacities = np.ones((len(pts),), dtype=np.float32)
    return pts, opacities
