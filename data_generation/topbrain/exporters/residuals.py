"""TopBrain residual generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_generation.common.transforms import load_transforms
from data_generation.topbrain.config import PriorConfig, ViewConfig
from data_generation.topbrain.exporters.priors import load_excluded_labels
from data_generation.topbrain.loaders import cases as topbrain_cases
from data_generation.topbrain.processors import cta as topbrain_cta
from data_generation.topbrain.processors import geometry as topbrain_geometry


def _residual_outputs_complete(residual_dir: Path) -> bool:
    required = [
        "fdk_residual.ply",
        "intersected_fdk_residual.ply",
    ]
    return all((residual_dir / name).exists() for name in required)


def generate_residual(
    *,
    ct_path: Path,
    seg_path: Path,
    view_config: ViewConfig,
    prior_config: PriorConfig,
    output_root: Path,
    contrast_enhancement: float,
    max_points: int,
) -> Path | None:
    """Generate residual PLYs for a (view, prior) combination."""
    case_name = topbrain_cases.extract_case_name(ct_path)
    output_dir = output_root / case_name
    residual_dir = output_dir / "residuals" / f"{view_config.name}__{prior_config.name}"

    if _residual_outputs_complete(residual_dir):
        return residual_dir

    prior_dir = output_dir / "priors" / prior_config.name
    excluded_labels = load_excluded_labels(prior_dir, prior_config)
    min_diam = prior_config.min_diameter_mm
    has_filter = bool(excluded_labels) or (min_diam is not None and float(min_diam) > 0)
    if not has_filter:
        return None

    ct_volume, seg_volume, metadata = topbrain_cta.load_cta_and_segmentation(str(ct_path), str(seg_path))
    seg_incomplete = seg_volume
    if excluded_labels:
        seg_incomplete = seg_incomplete.copy()
        seg_incomplete[np.isin(seg_incomplete, excluded_labels)] = 0
    if min_diam is not None and float(min_diam) > 0:
        seg_incomplete = topbrain_cta.filter_segmentation_by_diameter(
            seg_incomplete, spacing_xyz=metadata["spacing"], min_diameter_mm=float(min_diam)
        )

    view_dir = output_dir / "views" / view_config.name
    transforms = load_transforms(view_dir / "transforms.json")

    geo = topbrain_cta.create_tigre_geometry(transforms)
    target_shape = tuple(transforms["volume_resolution"][::-1])

    baseline_complete, contrast_complete = topbrain_cta.create_baseline_contrast_volumes(
        ct_volume, seg_volume, contrast_enhancement, excluded_labels=None
    )
    baseline_incomplete, contrast_incomplete = topbrain_cta.create_baseline_contrast_volumes(
        ct_volume, seg_incomplete, contrast_enhancement, excluded_labels=None
    )

    baseline_complete_tigre = topbrain_cta.resample_volume(baseline_complete, target_shape, order=1)
    contrast_complete_tigre = topbrain_cta.resample_volume(contrast_complete, target_shape, order=1)
    baseline_incomplete_tigre = topbrain_cta.resample_volume(baseline_incomplete, target_shape, order=1)
    contrast_incomplete_tigre = topbrain_cta.resample_volume(contrast_incomplete, target_shape, order=1)

    angles_rad = np.deg2rad(view_config.angles_deg)
    baseline_projs_complete = topbrain_cta.forward_project(baseline_complete_tigre, geo, angles_rad)
    contrast_projs_complete = topbrain_cta.forward_project(contrast_complete_tigre, geo, angles_rad)
    baseline_projs_incomplete = topbrain_cta.forward_project(baseline_incomplete_tigre, geo, angles_rad)
    contrast_projs_incomplete = topbrain_cta.forward_project(contrast_incomplete_tigre, geo, angles_rad)

    residual_dir.mkdir(parents=True, exist_ok=True)
    residual_data = topbrain_cta.compute_residual_maps(
        baseline_projs_complete,
        contrast_projs_complete,
        baseline_projs_incomplete,
        contrast_projs_incomplete,
        geo,
        angles_rad,
        seg_volume=seg_volume,
        metadata=metadata,
    )

    topbrain_geometry.create_ply_from_fdk_volume(
        residual_data["residual_volume"],
        transforms,
        residual_dir / "fdk_residual.ply",
        threshold_rel=0.02,
        max_points=max_points,
        intensity_weighted=True,
        intersection_mask=None,
    )
    topbrain_geometry.create_ply_from_fdk_volume(
        residual_data["residual_volume"],
        transforms,
        residual_dir / "intersected_fdk_residual.ply",
        threshold_rel=0.02,
        max_points=max_points,
        intensity_weighted=True,
        intersection_mask=residual_data.get("residual_mask"),
    )

    return residual_dir
