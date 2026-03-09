"""Vessel prior loading and visual hull filtering.

Loads CTA-derived vessel priors and optionally filters them using
calibrated 2-D segmentation masks (pointwise visual hull projection).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from data_generation.clinical.utils.geometry import ViewMaskGeometry
class PriorLoader:
    """Load and filter vessel prior point clouds."""

    def __init__(self, raw_root: Path, patient_id: str) -> None:
        self.raw_root = Path(raw_root)
        self.patient_id = str(patient_id)
        self.base_dir = self.raw_root / self.patient_id

    def load_vessel_prior(
        self,
        vessel_seg_path: Path,
        cta_path: Path,
        *,
        label: int = 2,
        max_points: int = 100_000,
        seed: int = 0,
        opacity: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load vessel prior from CTA segmentation in world coordinates.

        Uses SimpleITK to read the segmentation and CTA, extract the
        physical-space coordinates of labelled voxels, and subsample.

        Args:
            vessel_seg_path: Path to vessel segmentation (``.nrrd`` or ``.nii.gz``).
            cta_path: Path to CTA volume (used for affine/spacing reference).
            label: Segmentation label to extract (default 2).
            max_points: Subsample cap (0 = no cap).
            seed: RNG seed for subsampling.
            opacity: Uniform opacity assigned to every point.

        Returns:
            ``(points [N, 3], opacities [N])`` in world (mm) coordinates,
            centred on the volume origin.
        """
        import SimpleITK as sitk

        # Read the CTA to get physical metadata
        cta_img = sitk.ReadImage(str(cta_path))
        origin = np.array(cta_img.GetOrigin(), dtype=np.float32)  # [x, y, z]
        spacing = np.array(cta_img.GetSpacing(), dtype=np.float32)
        direction = np.array(cta_img.GetDirection(), dtype=np.float32).reshape(3, 3)
        size = np.array(cta_img.GetSize(), dtype=np.float32)  # [x, y, z]

        # Read segmentation
        seg_img = sitk.ReadImage(str(vessel_seg_path))
        seg_arr = sitk.GetArrayFromImage(seg_img)  # [Z, Y, X]

        # Find voxels matching the label
        coords_zyx = np.argwhere(seg_arr == int(label)).astype(np.float32)
        if coords_zyx.size == 0:
            raise RuntimeError(
                f"No voxels found for label={label} in {vessel_seg_path}"
            )

        # Convert from [Z, Y, X] voxel indices to [X, Y, Z] world coordinates
        # using the image affine: world = origin + direction @ (voxel * spacing)
        coords_xyz = coords_zyx[:, ::-1]  # [N, 3] in (X, Y, Z) order
        pts_world = origin + (direction @ (coords_xyz * spacing).T).T

        # Centre on volume midpoint (matches DiffDRR centre_volume=True)
        vol_extent = direction @ np.diag(spacing * size)
        vol_centre = origin + vol_extent.sum(axis=1) / 2.0
        pts_centred = (pts_world - vol_centre).astype(np.float32)

        # Subsample
        rng = np.random.default_rng(int(seed))
        if max_points > 0 and pts_centred.shape[0] > max_points:
            idx = rng.choice(pts_centred.shape[0], size=max_points, replace=False)
            pts_centred = pts_centred[idx]

        opacities = np.full(pts_centred.shape[0], float(opacity), dtype=np.float32)
        return pts_centred, opacities

    def filter_by_visual_hull(
        self,
        pts: np.ndarray,
        masks_with_geometry: list[ViewMaskGeometry],
        *,
        mask_dilate_px: int = 0,
    ) -> np.ndarray:
        """Filter points to those inside the visual hull.

        Projects each 3-D point into every calibrated 2-D mask and keeps
        only points visible in all views.

        Args:
            pts: ``[N, 3]`` world-space points.
            masks_with_geometry: Calibrated 2-D masks for each view.
            mask_dilate_px: Dilate 2-D masks before testing.

        Returns:
            Filtered points ``[M, 3]`` where ``M <= N``.
        """
        mask = _visual_hull_pointwise(
            pts, masks_with_geometry, mask_dilate_px=mask_dilate_px
        )
        return pts[mask]

    def build_view_geometry(
        self,
        mask_2d: np.ndarray,
        frame_info: dict,
        source_file: str = "",
    ) -> ViewMaskGeometry:
        """Construct :class:`ViewMaskGeometry` from a transforms.json frame."""
        w2c = np.array(frame_info["world2cam"], dtype=np.float32)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        cam_pos = (-R.T @ T).astype(np.float32)
        h, w = mask_2d.shape
        FovX = float(frame_info.get("FovX", 0.5))
        FovY = float(frame_info.get("FovY", 0.5))
        fx = w / (2.0 * np.tan(FovX / 2.0))
        fy = h / (2.0 * np.tan(FovY / 2.0))

        return ViewMaskGeometry(
            mask_2d=(mask_2d > 0.5).astype(bool),
            cam_pos=cam_pos,
            cam_forward=R[2, :].astype(np.float32),
            cam_right=R[0, :].astype(np.float32),
            cam_up=R[1, :].astype(np.float32),
            fx=float(fx),
            fy=float(fy),
            cx=w / 2.0,
            cy=h / 2.0,
            source_file=str(source_file),
        )


# ---------------------------------------------------------------------------
# Visual hull internals
# ---------------------------------------------------------------------------


def _visual_hull_pointwise(
    pts: np.ndarray,
    masks_with_geometry: list[ViewMaskGeometry],
    *,
    mask_dilate_px: int,
) -> np.ndarray:
    """Per-point projection voting (requires all views)."""
    pts = np.asarray(pts, dtype=np.float32)
    n = int(pts.shape[0])
    n_views = len(masks_with_geometry)
    votes = np.zeros(n, dtype=np.uint8)

    for view in masks_with_geometry:
        mask_2d = view.mask_2d
        if int(mask_dilate_px) > 0:
            mask_2d = binary_dilation(mask_2d.astype(bool), iterations=int(mask_dilate_px))
        mask_2d = mask_2d.astype(bool, copy=False)

        h, w = mask_2d.shape
        vec = pts - view.cam_pos[None, :]
        cam_x = np.sum(vec * view.cam_right[None, :], axis=1)
        cam_y = np.sum(vec * view.cam_up[None, :], axis=1)
        cam_z = np.sum(vec * view.cam_forward[None, :], axis=1)

        u = view.fx * cam_x / (cam_z + 1e-8) + view.cx
        v = view.fy * cam_y / (cam_z + 1e-8) + view.cy
        u_int = np.rint(u).astype(np.int32)
        v_int = np.rint(v).astype(np.int32)

        in_bounds = (cam_z > 0) & (u_int >= 0) & (u_int < w) & (v_int >= 0) & (v_int < h)
        hit = np.zeros(n, dtype=bool)
        if np.any(in_bounds):
            hit[in_bounds] = mask_2d[v_int[in_bounds], u_int[in_bounds]]
        votes += hit.astype(np.uint8)

    mask = votes >= n_views
    print(
        f"  VH(point): {int(mask.sum())}/{n} points kept "
        f"({100.0 * float(mask.sum()) / max(1, n):.1f}%)"
    )
    return mask


