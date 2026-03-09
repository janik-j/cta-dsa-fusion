"""
Core CTA processing utilities shared by the TopBrain pipeline.
Consolidates: load/save, TIGRE ops, volume processing, geometry creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from scipy.spatial import cKDTree

if TYPE_CHECKING:  # pragma: no cover
    pass  # type: ignore

# ============================================================================
# CTA I/O Operations
# ============================================================================


def load_cta_and_segmentation(ct_path: str, seg_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load CT and vessel segmentation from NIfTI files.

    Returns:
        ct_volume: CT volume [z, y, x] in HU
        seg_volume: Segmentation volume [z, y, x] with integer labels
        metadata: Dict with spacing, origin, direction
    """
    ct_img = sitk.ReadImage(ct_path)
    seg_img = sitk.ReadImage(seg_path)

    ct_volume = sitk.GetArrayFromImage(ct_img)
    seg_volume = sitk.GetArrayFromImage(seg_img)

    metadata = {
        "spacing": ct_img.GetSpacing(),
        "origin": ct_img.GetOrigin(),
        "direction": ct_img.GetDirection(),
        "size": ct_img.GetSize(),
    }

    return ct_volume, seg_volume, metadata


# ============================================================================
# Per-case Volume Geometry
# ============================================================================


def apply_cta_volume_to_transforms(
    transforms: dict, metadata: dict, *, case_name: str | None = None, volume_resolution_xyz: list[int] | None = None
) -> dict:
    """Return a transforms dict whose *volume* fields match the CTA's physical extent.

    We keep the template's projection geometry and (optionally) its voxel resolution,
    but set `volume_phy` to the CTA physical size so mm units are consistent across cases.

    Args:
        transforms: Base transforms template (will not be mutated)
        metadata: Dict from `load_cta_and_segmentation()` (expects `spacing` and `size`)
        case_name: Optional case name to store in `obj_index`
        volume_resolution_xyz: Optional target volume resolution [X, Y, Z].
            If omitted, uses `transforms["volume_resolution"]`.
    """
    spacing_xyz = np.asarray(metadata["spacing"], dtype=np.float64)
    size_xyz = np.asarray(metadata["size"], dtype=np.float64)
    phy_xyz = size_xyz * spacing_xyz

    out = dict(transforms)
    if case_name is not None:
        out["obj_index"] = str(case_name)

    # Keep original CTA geometry as extra metadata (safe: schema allows extra keys).
    out["source_volume_resolution"] = [int(x) for x in size_xyz.tolist()]
    out["source_volume_spacing"] = [float(x) for x in spacing_xyz.tolist()]
    out["source_volume_phy"] = [float(x) for x in phy_xyz.tolist()]

    res_xyz = (
        np.asarray(volume_resolution_xyz, dtype=np.float64)
        if volume_resolution_xyz is not None
        else np.asarray(out["volume_resolution"], dtype=np.float64)
    )
    if res_xyz.shape != (3,):
        raise ValueError(f"volume_resolution_xyz must be length 3, got {res_xyz}")

    out["volume_resolution"] = [int(x) for x in res_xyz.tolist()]
    out["volume_phy"] = [float(x) for x in phy_xyz.tolist()]
    out["volume_spacing"] = [float(x) for x in (phy_xyz / res_xyz).tolist()]
    return out


# ============================================================================
# Vessel Thickness Filtering (Diameter Threshold)
# ============================================================================


_KIMIMARO_TEASAR_PARAMS: dict[str, float] = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_const": 300,
    "soma_invalidation_scale": 2,
}


def _full_connectivity_structure(ndim: int) -> np.ndarray:
    return generate_binary_structure(int(ndim), int(ndim))


def _connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components for a binary mask. Background is 0, components are 1..n."""
    labeled, n_components = label(binary_mask, structure=_full_connectivity_structure(binary_mask.ndim))
    return labeled.astype(np.int32, copy=False), int(n_components)


def filter_segmentation_by_diameter(
    seg_volume: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    min_diameter_mm: float,
) -> np.ndarray:
    """Filter a label volume by local vessel diameter.

    Keeps voxels where the estimated local diameter is at least `min_diameter_mm`
    and sets others to background (0).

    Definition (matches `scripts/analysis/vessel_thickness_analysis.py`):
    - Extract TEASAR centerlines per connected component via Kimimaro.
    - Convert Kimimaro radii (mm) to diameters with a half-voxel surface correction:
        diam_mm = max(2*(r - 0.5*min(spacing_zyx)), 0)
    - Assign each vessel voxel the diameter of its nearest centerline vertex within
      its connected component, then threshold by `diam_mm >= min_diameter_mm`.
    """
    thr = float(min_diameter_mm)
    if thr <= 0:
        return seg_volume

    vessel_mask_all = seg_volume > 0
    if not vessel_mask_all.any():
        return seg_volume

    try:
        import kimimaro
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Kimimaro is required for thickness-based priors (diameter filtering). "
            "Install `kimimaro` and `crackle-codec` in the active environment."
        ) from e

    spacing_xyz_f = np.asarray(spacing_xyz, dtype=np.float64).reshape(3)
    spacing_zyx = tuple(float(v) for v in spacing_xyz_f[::-1])
    surface_offset_mm = 0.5 * float(min(spacing_zyx))

    keep = np.zeros_like(vessel_mask_all, dtype=bool)

    label_ids = np.unique(seg_volume)
    label_ids = label_ids[label_ids > 0]
    spacing_arr = np.asarray(spacing_zyx, dtype=np.float32).reshape(1, 3)

    for lid in label_ids.tolist():
        vessel_mask = seg_volume == int(lid)
        if not vessel_mask.any():
            continue

        comp_labels, n_components = _connected_components(vessel_mask)
        if n_components == 0:
            continue

        skels = kimimaro.skeletonize(
            comp_labels.astype(np.uint32, copy=False),
            teasar_params=_KIMIMARO_TEASAR_PARAMS,
            anisotropy=spacing_zyx,
            dust_threshold=0,
            progress=False,
            fix_branching=True,
            fix_borders=True,
            parallel=1,
        )

        verts_all: list[np.ndarray] = []
        diam_all: list[np.ndarray] = []
        comp_all: list[np.ndarray] = []

        for obj_id, skel in skels.items():
            if skel is None or getattr(skel, "vertices", None) is None or len(skel.vertices) == 0:
                continue
            verts = np.asarray(skel.vertices, dtype=np.float32)
            radii = getattr(skel, "radii", getattr(skel, "radius", None))
            if radii is None:
                continue
            radii = np.asarray(radii, dtype=np.float32)
            if radii.ndim != 1 or len(radii) != len(verts):
                continue
            diam = np.maximum(2.0 * (radii - surface_offset_mm), 0.0).astype(np.float32, copy=False)
            verts_all.append(verts)
            diam_all.append(diam)
            comp_all.append(np.full((len(verts),), int(obj_id), dtype=np.int32))

        if not verts_all:
            continue

        cl_verts = np.concatenate(verts_all, axis=0)
        cl_diam = np.concatenate(diam_all, axis=0)
        cl_comp = np.concatenate(comp_all, axis=0)

        coords = np.argwhere(vessel_mask)
        voxel_comp = comp_labels[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int32, copy=False)
        voxel_mm = coords.astype(np.float32, copy=False) * spacing_arr

        for comp_id in np.unique(voxel_comp):
            if comp_id == 0:
                continue
            idx_vox = np.nonzero(voxel_comp == comp_id)[0]
            idx_cl = np.nonzero(cl_comp == comp_id)[0]
            if len(idx_cl) == 0:
                continue

            tree = cKDTree(cl_verts[idx_cl])
            _, nn = tree.query(voxel_mm[idx_vox], k=1)
            diam_assigned = cl_diam[idx_cl[np.asarray(nn, dtype=np.int64)]]
            keep_local = diam_assigned >= thr
            if not np.any(keep_local):
                continue
            keep_coords = coords[idx_vox[np.nonzero(keep_local)[0]]]
            keep[keep_coords[:, 0], keep_coords[:, 1], keep_coords[:, 2]] = True

    out = seg_volume.copy()
    out[~keep] = 0
    return out


# ============================================================================
# Volume Processing
# ============================================================================


def create_baseline_contrast_volumes(
    ct_volume: np.ndarray,
    seg_volume: np.ndarray,
    contrast_enhancement: float = 1200.0,
    excluded_labels: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create baseline and contrast-enhanced volumes.

    Args:
        ct_volume: CT volume in HU
        seg_volume: Vessel segmentation with integer labels
        contrast_enhancement: HU enhancement for vessels (default: 1200)
        excluded_labels: Labels to exclude from contrast volume

    Returns:
        baseline_volume: Copy of CT
        contrast_volume: CT + enhanced vessels
    """
    baseline_volume = ct_volume.copy()
    contrast_volume = ct_volume.copy()

    # Create vessel mask
    if excluded_labels is not None and len(excluded_labels) > 0:
        vessel_mask = np.isin(seg_volume, excluded_labels, invert=True) & (seg_volume > 0)
    else:
        vessel_mask = seg_volume > 0

    # Enhance vessels
    contrast_volume[vessel_mask] += contrast_enhancement

    return baseline_volume, contrast_volume


def resample_volume(volume: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    """Resample volume to target shape.

    Args:
        volume: Input volume [z, y, x]
        target_shape: Target shape (z, y, x)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    from scipy.ndimage import zoom

    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return zoom(volume, zoom_factors, order=order)


# ============================================================================
# TIGRE Operations
# ============================================================================


def create_tigre_geometry(transforms: dict) -> Any:
    """Create TIGRE geometry from transforms dict.

    Args:
        transforms: Dict with camera parameters

    Returns:
        TIGRE geometry object
    """
    try:
        import tigre  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("tigre is required for forward/back-projection utilities.") from e

    geo = tigre.geometry()

    # Detector parameters
    geo.nDetector = np.array(transforms["proj_resolution"])
    geo.dDetector = np.array(transforms["proj_spacing"])
    geo.sDetector = geo.nDetector * geo.dDetector

    # Volume parameters
    geo.nVoxel = np.array(transforms["volume_resolution"][::-1])  # [z, y, x]
    vol_phy = np.array(transforms["volume_phy"])
    geo.dVoxel = vol_phy[::-1] / geo.nVoxel  # [z, y, x] spacing
    geo.sVoxel = geo.nVoxel * geo.dVoxel

    # Source/detector distances
    geo.DSD = transforms["sid"]  # Source-to-detector
    geo.DSO = transforms["sad"]  # Source-to-axis (isocenter)

    # Offsets (centered)
    geo.offOrigin = np.array([0, 0, 0])
    geo.offDetector = np.array([0, 0])

    # Accuracy
    geo.accuracy = 0.5

    return geo


def forward_project(volume: np.ndarray, geometry: Any, angles: np.ndarray) -> np.ndarray:
    """TIGRE forward projection with flip convention.

    Args:
        volume: Volume to project [z, y, x]
        geometry: TIGRE geometry
        angles: Projection angles in radians [N_views]

    Returns:
        Projections [N_views, H, W]
    """
    try:
        import tigre  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("tigre is required for forward projection.") from e

    # TIGRE expects [z, y, x] order (already correct from SimpleITK)
    projections = tigre.Ax(volume, geometry, angles)

    # Flip horizontally to match FDK reconstruction convention
    projections = projections[:, ::-1, :]

    return projections


def fdk_reconstruct(projections: np.ndarray, geometry: Any, angles: np.ndarray) -> np.ndarray:
    """TIGRE FDK backprojection.

    Args:
        projections: Projections [N_views, H, W]
        geometry: TIGRE geometry
        angles: Projection angles in radians [N_views]

    Returns:
        Reconstructed volume [z, y, x]
    """
    try:
        import tigre  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("tigre is required for FDK reconstruction.") from e

    # Flip projections before reconstruction (reverse of forward)
    projections_flipped = projections[:, ::-1, :]

    volume = tigre.algorithms.fdk(projections_flipped, geometry, angles)

    return volume


# ============================================================================
# Residual Map Generation
# ============================================================================


def compute_residual_maps(
    baseline_complete: np.ndarray,
    contrast_complete: np.ndarray,
    baseline_incomplete: np.ndarray,
    contrast_incomplete: np.ndarray,
    geometry: Any,
    angles: np.ndarray,
    seg_volume: np.ndarray | None = None,
    metadata: dict | None = None,
) -> dict:
    """Compute residual maps (difference between complete and incomplete) with filtering.

    Args:
        baseline_complete: Complete baseline projections [N_views, H, W]
        contrast_complete: Complete contrast projections [N_views, H, W]
        baseline_incomplete: Incomplete baseline projections [N_views, H, W]
        contrast_incomplete: Incomplete contrast projections [N_views, H, W]
        geometry: TIGRE geometry
        angles: Projection angles in radians
        seg_volume: Complete segmentation (optional, for corridor gating)
        metadata: CT metadata (optional, for spacing)

    Returns:
        Dict with residual arrays
    """
    # Compute residuals (difference maps)
    dsa_residual = (contrast_complete - baseline_complete) - (contrast_incomplete - baseline_incomplete)

    # FDK reconstruction of residual
    print("  FDK reconstruction of residual volume...")
    residual_volume = fdk_reconstruct(dsa_residual, geometry, angles)

    # --- ADVANCED FILTERING ---
    print("  Filtering residual volume (Intersection & Corridor Gating)...")

    # 1. Thresholding (reduced from 0.25 to 0.05 for more residual points)
    threshold = residual_volume.max() * 0.05
    print(f"    Thresholding at {threshold:.2f}")
    residual_mask = (residual_volume > threshold).astype(np.uint8)

    # 2. Intersection Gating (AP ∩ LAT) if 2+ views
    if len(angles) >= 2:
        print("    Applying Intersection Gating...")
        # Project individual views
        mask_intersect = np.ones_like(residual_mask)

        # We can approximate by taking just the first and last view if available, or all.
        # But 'generate_residual_maps.py' logic was specific to 2 views or per-view backproj.
        # Let's do per-view intersection for robustness.
        for i in range(len(angles)):
            # Create single-view projection array (rest zeros)
            # This is expensive for many views, but fine for 2-5 views.
            single_proj = np.zeros_like(dsa_residual)
            single_proj[i] = dsa_residual[i]

            # Backproject single view
            vol_single = fdk_reconstruct(single_proj, geometry, angles)
            mask_single = (vol_single > threshold * 0.5).astype(np.uint8)
            mask_intersect *= mask_single

        residual_mask *= mask_intersect
        print(f"    Intersection complete. Voxels: {residual_mask.sum()}")

    # 3. CTA Corridor Gating (if segmentation provided)
    if seg_volume is not None and metadata is not None:
        try:
            print("    Applying CTA Corridor Gating...")
            # Create vessel mask from segmentation (assuming it's the unfiltered one)
            vessel_mask_complete = seg_volume > 0

            # Dilate corridor (8mm)
            dilation_radius_mm = 8.0
            # residual_volume is already on the TIGRE grid -> use its voxel spacing.
            # binary_dilation is isotropic in voxel units, so use the minimum voxel size
            # to avoid under-dilating in high-res axes.
            spacing_zyx = np.asarray(getattr(geometry, "dVoxel", [1.0, 1.0, 1.0]), dtype=np.float64)
            dilation_voxels = int(np.ceil(dilation_radius_mm / float(np.min(spacing_zyx))))

            # We need to resample the vessel mask to the TIGRE volume shape first?
            # residual_volume is [z, y, x] (TIGRE grid). seg_volume is [z, y, x] (Original CT grid).
            # They might differ if 'volume_resolution' != 'ct_size'.
            # 'resample_volume' can map seg to residual shape.

            target_shape = residual_volume.shape
            if seg_volume.shape != target_shape:
                # Resample segmentation (nearest neighbor for labels, but for mask linear is fine then thresh)
                # Actually for binary mask, order=0 is safest or 1 then >0.5
                vessel_mask_tigre = (
                    resample_volume(vessel_mask_complete.astype(np.float32), target_shape, order=0) > 0.5
                )
            else:
                vessel_mask_tigre = vessel_mask_complete

            corridor_mask = binary_dilation(vessel_mask_tigre, iterations=dilation_voxels).astype(np.uint8)

            residual_mask *= corridor_mask
            print(f"    Corridor gating complete. Voxels: {residual_mask.sum()}")

        except Exception as e:
            print(f"    Warning: Corridor gating failed: {e}")
            import traceback

            traceback.print_exc()

    # Fallback if empty
    if residual_mask.sum() == 0:
        print("    WARNING: Mask empty after filtering. Falling back to simple low threshold.")
        residual_mask = (residual_volume > residual_volume.max() * 0.1).astype(np.uint8)

    return {
        "dsa_residual": dsa_residual,
        "residual_volume": residual_volume,
        "residual_mask": residual_mask,
    }
