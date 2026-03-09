"""Residual (M2) point-cloud generation for clinical datasets.

Generates two PLY files analogous to the TopBrain residual pipeline:

* ``fdk_residual.ply`` — thresholded backprojection volume.
* ``intersected_fdk_residual.ply`` — additionally gated by multi-view
  intersection support mask.

For clinical data there is no "complete vs incomplete" ground truth.
Instead an *error map* per view is derived by removing the projected
CTA prior from the exported DSA MIP (already segmentation-masked).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation

from data_generation.common.geometry import voxel_to_world_coords, write_ply_xyz_opacity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_residual_plys(
    *,
    dataset_dir: Path,
    prior_pts_world_xyz: np.ndarray,
    max_points: int = 30_000,
    voxel_size_mm: float = 2.0,
    prior_proj_dilate_px: int = 3,
    residual_intensity_threshold: float = 0.01,
    residual_support_threshold: float = 0.01,
    residual_mask_dilate_px: int = 0,
    volume_threshold_rel: float = 0.02,
) -> dict[str, Any]:
    """Generate residual PLYs for a clinical dataset directory.

    Reads ``projections/DSA.nii.gz`` and ``transforms.json`` from
    *dataset_dir*, computes per-view residuals, backprojects into a
    coarse 3-D volume, and writes the PLY files.

    Returns:
        Dict written into ``transforms.json["residuals"]`` with
        provenance metadata.
    """
    dataset_dir = Path(dataset_dir)
    transforms_path = dataset_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing transforms.json: {transforms_path}")

    with open(transforms_path, encoding="utf-8") as f:
        transforms = json.load(f)
    frames = list(transforms.get("frames") or [])
    n_views = int(transforms.get("N_views") or 0)
    if n_views <= 0:
        raise ValueError("transforms.json missing/invalid N_views")
    if len(frames) < n_views:
        raise ValueError(f"frames length {len(frames)} < N_views {n_views}")

    dsa_path = dataset_dir / "projections" / "DSA.nii.gz"
    if not dsa_path.exists():
        raise FileNotFoundError(f"Missing DSA.nii.gz: {dsa_path}")
    dsa_stack = sitk.GetArrayFromImage(sitk.ReadImage(str(dsa_path))).astype(np.float32)
    if dsa_stack.ndim != 3 or int(dsa_stack.shape[0]) < n_views:
        raise ValueError(f"DSA stack must be [N_views,H,W], got {dsa_stack.shape}")
    dsa_stack = dsa_stack[:n_views]

    residual_stack = _compute_residual_stack(
        dsa_stack,
        frames[:n_views],
        prior_pts_world_xyz,
        prior_proj_dilate_px=int(prior_proj_dilate_px),
        residual_intensity_threshold=float(residual_intensity_threshold),
    )

    vol_origin = np.asarray(
        transforms.get("volume_origin")
        or (-np.asarray(transforms["volume_phy"]) / 2).tolist(),
        dtype=np.float32,
    )
    vol_phy = np.asarray(transforms["volume_phy"], dtype=np.float32)

    vol, intersect = _backproject_and_intersect(
        residual_stack,
        frames[:n_views],
        volume_origin_xyz=vol_origin,
        volume_phy_xyz=vol_phy,
        voxel_size_mm=float(voxel_size_mm),
        residual_support_threshold=float(residual_support_threshold),
        residual_mask_dilate_px=int(residual_mask_dilate_px),
    )

    residuals_dir = dataset_dir / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)

    fdk_path = residuals_dir / "fdk_residual.ply"
    inter_path = residuals_dir / "intersected_fdk_residual.ply"

    n_fdk = _write_volume_ply(
        fdk_path, vol,
        volume_origin_xyz=vol_origin,
        voxel_size_mm=float(voxel_size_mm),
        threshold_rel=float(volume_threshold_rel),
        max_points=int(max_points),
    )
    n_inter = _write_volume_ply(
        inter_path, vol,
        volume_origin_xyz=vol_origin,
        voxel_size_mm=float(voxel_size_mm),
        threshold_rel=float(volume_threshold_rel),
        max_points=int(max_points),
        intersection_mask=intersect,
    )

    # Persist provenance
    residuals_meta: dict[str, Any] = {
        "enabled": True,
        "method": "dsa_minus_projected_prior",
        "voxel_size_mm": float(voxel_size_mm),
        "prior_proj_dilate_px": int(prior_proj_dilate_px),
        "residual_intensity_threshold": float(residual_intensity_threshold),
        "residual_support_threshold": float(residual_support_threshold),
        "residual_mask_dilate_px": int(residual_mask_dilate_px),
        "volume_threshold_rel": float(volume_threshold_rel),
        "files": {
            "fdk_residual": {
                "path": str(fdk_path.relative_to(dataset_dir)),
                "n_points": int(n_fdk),
            },
            "intersected_fdk_residual": {
                "path": str(inter_path.relative_to(dataset_dir)),
                "n_points": int(n_inter),
            },
        },
    }
    transforms["residuals"] = residuals_meta
    (dataset_dir / "transforms.json").write_text(
        json.dumps(transforms, indent=2) + "\n", encoding="utf-8"
    )
    return residuals_meta


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _project_points_to_mask(
    pts_world_xyz: np.ndarray,
    frame: dict[str, Any],
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    h, w = int(frame["height"]), int(frame["width"])
    FovX, FovY = float(frame["FovX"]), float(frame["FovY"])
    fx = float(w) / (2.0 * float(np.tan(FovX / 2.0)))
    fy = float(h) / (2.0 * float(np.tan(FovY / 2.0)))
    cx, cy = float(w) / 2.0, float(h) / 2.0

    R = np.asarray(frame["R"], dtype=np.float32).reshape(3, 3)
    T = np.asarray(frame["T"], dtype=np.float32).reshape(3)

    pts = np.asarray(pts_world_xyz, dtype=np.float32)
    cam = pts @ R.T + T.reshape(1, 3)
    z = cam[:, 2]
    valid = z > 1e-3
    if not np.any(valid):
        return np.zeros((h, w), dtype=bool)

    cam, z = cam[valid], z[valid]
    px = (cam[:, 0] / z) * fx + cx
    py = (cam[:, 1] / z) * fy + cy
    u = np.round(px).astype(np.int32)
    v = np.round(py).astype(np.int32)
    ib = (u >= 0) & (u < w) & (v >= 0) & (v < h)

    mask = np.zeros((h, w), dtype=bool)
    if np.any(ib):
        mask[v[ib], u[ib]] = True
    if int(dilate_px) > 0:
        mask = binary_dilation(mask, iterations=int(dilate_px))
    return mask.astype(bool, copy=False)


def _compute_residual_stack(
    dsa_stack: np.ndarray,
    frames: list[dict[str, Any]],
    prior_pts: np.ndarray,
    *,
    prior_proj_dilate_px: int,
    residual_intensity_threshold: float,
) -> np.ndarray:
    n_views = int(dsa_stack.shape[0])
    residual = np.zeros_like(dsa_stack, dtype=np.float32)
    for i in range(n_views):
        prior_mask = _project_points_to_mask(
            prior_pts, frames[i], dilate_px=prior_proj_dilate_px
        )
        view_res = np.asarray(dsa_stack[i], dtype=np.float32).copy()
        view_res[prior_mask] = 0.0
        if residual_intensity_threshold > 0:
            view_res[view_res < residual_intensity_threshold] = 0.0
        residual[i] = view_res
    return residual


def _backproject_and_intersect(
    residual_stack: np.ndarray,
    frames: list[dict[str, Any]],
    *,
    volume_origin_xyz: np.ndarray,
    volume_phy_xyz: np.ndarray,
    voxel_size_mm: float,
    residual_support_threshold: float,
    residual_mask_dilate_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    vol_origin = np.asarray(volume_origin_xyz, dtype=np.float32).reshape(3)
    vol_phy = np.asarray(volume_phy_xyz, dtype=np.float32).reshape(3)
    v = float(voxel_size_mm)

    grid_size = np.maximum(1, np.ceil(vol_phy / v).astype(np.int32))
    nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

    x_coords = vol_origin[0] + (np.arange(nx, dtype=np.float32) + 0.5) * v
    y_coords = vol_origin[1] + (np.arange(ny, dtype=np.float32) + 0.5) * v
    z_coords = vol_origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * v

    vol = np.zeros((nx, ny, nz), dtype=np.float32)
    intersect = np.ones((nx, ny, nz), dtype=bool)

    chunk_z = 16
    n_views = int(residual_stack.shape[0])
    for vi in range(n_views):
        frame = frames[vi]
        img = np.asarray(residual_stack[vi], dtype=np.float32)
        h, w = img.shape
        R = np.asarray(frame["R"], dtype=np.float32).reshape(3, 3)
        T = np.asarray(frame["T"], dtype=np.float32).reshape(3)
        FovX, FovY = float(frame["FovX"]), float(frame["FovY"])
        fx = float(w) / (2.0 * float(np.tan(FovX / 2.0)))
        fy = float(h) / (2.0 * float(np.tan(FovY / 2.0)))
        cx, cy = float(w) / 2.0, float(h) / 2.0

        support = img > float(residual_support_threshold)
        if int(residual_mask_dilate_px) > 0:
            support = binary_dilation(support, iterations=int(residual_mask_dilate_px))

        for zs in range(0, nz, chunk_z):
            ze = min(zs + chunk_z, nz)
            xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords[zs:ze], indexing="ij")
            cam_x = xx * R[0, 0] + yy * R[0, 1] + zz * R[0, 2] + T[0]
            cam_y = xx * R[1, 0] + yy * R[1, 1] + zz * R[1, 2] + T[1]
            cam_z = xx * R[2, 0] + yy * R[2, 1] + zz * R[2, 2] + T[2]

            in_front = cam_z > 1e-3
            u = fx * cam_x / (cam_z + 1e-8) + cx
            vi_ = fy * cam_y / (cam_z + 1e-8) + cy
            u_int = np.round(u).astype(np.int32)
            v_int = np.round(vi_).astype(np.int32)
            ib = in_front & (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)

            sample = np.zeros_like(cam_z, dtype=np.float32)
            sup_vol = np.zeros_like(ib, dtype=bool)
            if np.any(ib):
                sample[ib] = img[v_int[ib], u_int[ib]]
                sup_vol[ib] = support[v_int[ib], u_int[ib]]

            vol[:, :, zs:ze] += sample
            intersect[:, :, zs:ze] &= sup_vol

    return vol, intersect


def _write_volume_ply(
    out_path: Path,
    vol_xyz: np.ndarray,
    *,
    volume_origin_xyz: np.ndarray,
    voxel_size_mm: float,
    threshold_rel: float,
    max_points: int,
    intersection_mask: np.ndarray | None = None,
) -> int:
    if vol_xyz.size == 0:
        return 0
    vmax = float(np.max(vol_xyz))
    if vmax <= 0:
        return 0

    mask = vol_xyz > vmax * float(threshold_rel)
    if intersection_mask is not None:
        mask &= intersection_mask.astype(bool)
    if not np.any(mask):
        return 0

    coords = np.argwhere(mask)
    intensities = vol_xyz[mask].astype(np.float32, copy=False)

    if int(max_points) > 0 and coords.shape[0] > int(max_points):
        idx = np.argpartition(intensities, -int(max_points))[-int(max_points):]
        idx = idx[np.argsort(-intensities[idx], kind="stable")]
        coords = coords[idx]
        intensities = intensities[idx]

    origin = np.asarray(volume_origin_xyz, dtype=np.float32)
    spacing = np.array([float(voxel_size_mm)] * 3, dtype=np.float32)
    pts_world = voxel_to_world_coords(
        coords.astype(np.float32), volume_origin_xyz=origin, volume_spacing_xyz=spacing
    )
    opacity = np.clip(intensities / vmax, 0.1, 1.0)
    write_ply_xyz_opacity(out_path, pts_world, opacity)
    return int(pts_world.shape[0])
