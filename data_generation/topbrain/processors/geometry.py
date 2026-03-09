"""
Geometry utilities: PLY point clouds, mesh export, coordinate conversion.
Consolidates mesh/PLY generation from generate_gt_from_cta.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from data_generation.common.geometry import voxel_to_world_coords, write_ply_xyz_opacity

# ============================================================================
# Shared Utilities
# ============================================================================

def _subsample_indices(n: int, k: int) -> np.ndarray:
    """Deterministically subsample k indices from [0..n)."""
    if k <= 0 or n <= k:
        return np.arange(n, dtype=np.int64)
    # Use linspace so we span the full range [0, n-1] (inclusive).
    # A simple fixed-step sampler can miss the tail where sparse structures may live.
    return np.linspace(0, n - 1, num=k, dtype=np.int64)


def _get_volume_params(transforms: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract volume parameters from transforms dict.

    Returns:
        Tuple of (resolution, physical_size, origin) as numpy arrays in [x, y, z] order.
    """
    vol_res = np.array(transforms["volume_resolution"], dtype=np.float32)
    vol_phy = np.array(transforms["volume_phy"], dtype=np.float32)
    default_origin = -vol_phy / 2
    vol_origin = np.array(transforms.get("volume_origin", default_origin.tolist()), dtype=np.float32)
    return vol_res, vol_phy, vol_origin


def _resample_to_target(volume: np.ndarray, target_shape_zyx: tuple[int, ...], order: int = 0) -> np.ndarray:
    """Resample volume to target shape if needed.

    Args:
        volume: Input volume [z, y, x]
        target_shape_zyx: Target shape as (z, y, x)
        order: Interpolation order (0=nearest, 1=linear)
    """
    if tuple(volume.shape) == target_shape_zyx:
        return volume
    zoom_factors = [target_shape_zyx[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=order)


def _voxel_to_physical(coords_zyx: np.ndarray, transforms: dict) -> np.ndarray:
    """Convert voxel coordinates [z, y, x] to physical coordinates [x, y, z].

    Args:
        coords_zyx: Voxel coordinates [N, 3] in (z, y, x) order
        transforms: Transforms dict with volume parameters

    Returns:
        Physical coordinates [N, 3] in (x, y, z) order
    """
    vol_res, vol_phy, vol_origin = _get_volume_params(transforms)
    spacing = vol_phy / vol_res
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float32)
    return voxel_to_world_coords(coords_xyz, volume_origin_xyz=vol_origin, volume_spacing_xyz=spacing)


def _create_vessel_mask(
    seg_volume: np.ndarray, excluded_labels: list[int] | None = None, include_labels: list[int] | None = None
) -> np.ndarray:
    """Create binary vessel mask from segmentation.

    Args:
        seg_volume: Segmentation volume (any axis order)
        excluded_labels: Labels to exclude (ignored if include_labels is set)
        include_labels: If set, include ONLY these labels

    Returns:
        Binary mask of vessels
    """
    if include_labels:
        return np.isin(seg_volume, include_labels) & (seg_volume > 0)
    if excluded_labels:
        return np.isin(seg_volume, excluded_labels, invert=True) & (seg_volume > 0)
    return seg_volume > 0


# ============================================================================
# PLY Point Cloud Generation
# ============================================================================


def create_ply_from_fdk_volume(
    fdk_volume: np.ndarray,
    transforms: dict,
    output_path: Path,
    threshold_rel: float = 0.02,
    max_points: int = 30000,
    intensity_weighted: bool = True,
    intersection_mask: np.ndarray | None = None,
) -> None:
    """Generate PLY point cloud from FDK reconstruction volume with intensity weighting.

    Args:
        fdk_volume: FDK reconstruction volume [z, y, x]
        transforms: Transforms dict with volume parameters
        output_path: Output PLY file path
        threshold_rel: Threshold as fraction of max intensity (default: 0.02)
        max_points: Maximum number of points to sample
        intensity_weighted: Weight sampling by intensity (default: True)
        intersection_mask: Optional binary mask for intersection gating [z, y, x] (default: None)
    """
    target_shape_zyx = tuple(int(v) for v in transforms["volume_resolution"][::-1])
    fdk_volume = _resample_to_target(fdk_volume, target_shape_zyx, order=1)

    if intersection_mask is not None:
        intersection_mask = _resample_to_target(intersection_mask.astype(np.float32), target_shape_zyx, order=0) > 0.5

    threshold = fdk_volume.max() * threshold_rel
    mask = fdk_volume > threshold

    if intersection_mask is not None:
        if intersection_mask.shape != mask.shape:
            print(f"  Warning: Mask shape mismatch {mask.shape} vs {intersection_mask.shape}, resampling...")
            intersection_mask = _resample_to_target(intersection_mask.astype(np.float32), mask.shape, order=0) > 0.5
        print(f"  Applying intersection mask: {mask.sum():,} -> ", end="")
        mask = mask & intersection_mask
        print(f"{mask.sum():,} voxels")

    if not mask.any():
        print("Warning: No voxels above threshold, skipping PLY generation")
        return

    coords = np.argwhere(mask)
    intensities = fdk_volume[mask]

    if int(max_points) > 0 and len(coords) > int(max_points):
        max_points = int(max_points)
        if intensity_weighted:
            indices = np.argpartition(intensities, -max_points)[-max_points:]
            indices = indices[np.argsort(-intensities[indices], kind="stable")]
        else:
            indices = _subsample_indices(len(coords), max_points)
        coords = coords[indices]
        intensities = intensities[indices]

    physical_coords = _voxel_to_physical(coords, transforms)
    opacity = np.clip(intensities / fdk_volume.max(), 0.1, 1.0)

    write_ply_xyz_opacity(output_path, physical_coords, opacity)
    print(f"Created FDK PLY: {len(physical_coords)} points at {output_path}")


def create_ply_from_segmentation(
    seg_volume: np.ndarray,
    transforms: dict,
    output_path: Path,
    excluded_labels: list[int] | None = None,
    max_points: int = 30000,
) -> None:
    """Generate PLY point cloud from segmentation in FDK-aligned coordinates.

    CRITICAL: Coordinates must match FDK reconstruction convention for proper alignment.
    The segmentation is resampled to the FDK grid before coordinate conversion.

    Args:
        seg_volume: Segmentation [z, y, x] from SimpleITK
        transforms: Transforms dict with volume parameters
        output_path: Output PLY file path
        excluded_labels: Labels to exclude (for incomplete vessel init)
        max_points: Maximum number of points to sample
    """
    target_shape_zyx = tuple(int(v) for v in transforms["volume_resolution"][::-1])
    seg_volume = _resample_to_target(seg_volume, target_shape_zyx, order=0)

    vessel_mask = _create_vessel_mask(seg_volume, excluded_labels=excluded_labels)

    if not vessel_mask.any():
        print("Warning: No vessels in mask, skipping PLY generation")
        return

    coords = np.argwhere(vessel_mask)

    # NOTE: CLI uses `--max-points 0` to mean "no cap" (use all vessel voxels).
    if int(max_points) > 0 and len(coords) > int(max_points):
        max_points = int(max_points)
        coords = coords[_subsample_indices(len(coords), max_points)]

    physical_coords = _voxel_to_physical(coords, transforms)
    write_ply_xyz_opacity(output_path, physical_coords, opacity=1.0)
    print(f"Generated PLY with {len(physical_coords)} points: {output_path}")


# ============================================================================
# Mesh Export
# ============================================================================


def export_mesh_obj(
    seg_volume: np.ndarray,
    transforms: dict,
    output_path: Path,
    excluded_labels: list[int] | None = None,
    threshold: float = 0.5,
    *,
    include_labels: list[int] | None = None,
) -> None:
    """Export vessel mesh as OBJ in voxel space.

    Args:
        seg_volume: Segmentation [z, y, x]
        transforms: Transforms dict
        output_path: Output OBJ path
        excluded_labels: Labels to exclude
        include_labels: If set, export ONLY these labels (overrides excluded_labels)
        threshold: Marching cubes threshold
    """
    import mcubes

    seg_xyz = seg_volume.transpose(2, 1, 0)
    vessel_mask = _create_vessel_mask(seg_xyz, excluded_labels, include_labels)

    if not vessel_mask.any():
        raise ValueError("No vessels for mesh export (check labels/exclusions)")

    target_shape = tuple(transforms["volume_resolution"])
    if vessel_mask.shape != target_shape:
        zoom_factors = [target_shape[i] / vessel_mask.shape[i] for i in range(3)]
        vessel_mask = zoom(vessel_mask.astype(np.float32), zoom_factors, order=0) > 0.5

    vertices, triangles = mcubes.marching_cubes(vessel_mask.astype(np.float32), threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mcubes.export_obj(vertices, triangles, str(output_path))
    print(f"Exported mesh with {len(vertices)} vertices: {output_path}")
