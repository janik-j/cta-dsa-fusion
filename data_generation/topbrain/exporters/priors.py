"""TopBrain prior generation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_generation.common.transforms import load_transforms
from data_generation.topbrain.config import PriorConfig
from data_generation.topbrain.loaders import cases as topbrain_cases
from data_generation.topbrain.processors import cta as topbrain_cta
from data_generation.topbrain.processors import geometry as topbrain_geometry

_DIAMETER_ESTIMATOR_ID = "kimimaro_teasar_nn_v1"


def load_excluded_labels(prior_dir: Path, prior_config: PriorConfig) -> list[int]:
    """Load excluded_labels from stored config or use defaults."""
    config_path = prior_dir / "prior_config.json"
    if config_path.exists():
        try:
            prior_info = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(prior_info, dict) and isinstance(prior_info.get("excluded_labels"), list):
                return [int(x) for x in prior_info["excluded_labels"] if isinstance(x, (int, float, str))]
        except Exception as e:
            import warnings

            warnings.warn(f"Failed to load prior config from {config_path}: {e}", stacklevel=2)
    return list(prior_config.excluded_labels)


def prior_config_matches(prior_dir: Path, prior_config: PriorConfig, excluded_labels: list[int]) -> bool:
    config_path = prior_dir / "prior_config.json"
    if not config_path.exists():
        return False
    try:
        info = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(info, dict):
        return False

    if str(info.get("name", "")).strip() != str(prior_config.name).strip():
        return False

    prev_excluded = info.get("excluded_labels", [])
    if not isinstance(prev_excluded, list):
        return False
    if sorted(int(x) for x in prev_excluded) != sorted(int(x) for x in excluded_labels):
        return False

    prev_min = info.get("min_diameter_mm", None)
    cur_min = prior_config.min_diameter_mm
    if prev_min is None and cur_min is not None:
        return False
    if prev_min is not None and cur_min is None:
        return False
    if prev_min is not None and cur_min is not None and abs(float(prev_min) - float(cur_min)) > 1e-6:
        return False

    return str(info.get("diameter_estimator", "")).strip() == _DIAMETER_ESTIMATOR_ID


def generate_prior(
    *,
    ct_path: Path,
    seg_path: Path,
    prior_config: PriorConfig,
    output_root: Path,
    transforms_template: Path,
    max_points: int,
) -> Path:
    """Generate prior PLY for a specific coverage level."""
    case_name = topbrain_cases.extract_case_name(ct_path)
    output_dir = output_root / case_name
    prior_dir = output_dir / "priors" / prior_config.name

    excluded_labels = load_excluded_labels(prior_dir, prior_config)
    artifacts_exist = (
        (prior_dir / "vessels.ply").exists()
        and (prior_dir / "mesh.obj").exists()
        and (prior_dir / "prior_config.json").exists()
    )
    if artifacts_exist and prior_config_matches(prior_dir, prior_config, excluded_labels):
        return prior_dir

    _ct_volume, seg_volume, metadata = topbrain_cta.load_cta_and_segmentation(str(ct_path), str(seg_path))
    prior_dir.mkdir(parents=True, exist_ok=True)

    base_transforms = load_transforms(transforms_template)
    base_transforms = topbrain_cta.apply_cta_volume_to_transforms(base_transforms, metadata, case_name=case_name)
    transforms = topbrain_cases.create_view_transforms(base_transforms, [-90, -45, 0])

    # Apply filtering (labels first, then diameter threshold).
    seg_filtered = seg_volume
    if excluded_labels:
        seg_filtered = seg_filtered.copy()
        seg_filtered[np.isin(seg_filtered, excluded_labels)] = 0
    if prior_config.min_diameter_mm is not None and float(prior_config.min_diameter_mm) > 0:
        seg_filtered = topbrain_cta.filter_segmentation_by_diameter(
            seg_filtered, spacing_xyz=metadata["spacing"], min_diameter_mm=float(prior_config.min_diameter_mm)
        )

    if not (prior_dir / "vessels.ply").exists():
        topbrain_geometry.create_ply_from_segmentation(
            seg_filtered,
            transforms,
            prior_dir / "vessels.ply",
            excluded_labels=None,
            max_points=max_points,
        )

    if not (prior_dir / "mesh.obj").exists():
        topbrain_geometry.export_mesh_obj(
            seg_filtered,
            transforms,
            prior_dir / "mesh.obj",
            excluded_labels=None,
        )

    if not (prior_dir / "mesh_missing.obj").exists():
        missing_mask = (seg_volume > 0) & (seg_filtered == 0)
        if missing_mask.any():
            missing_vol = missing_mask.astype(np.uint8)
            topbrain_geometry.export_mesh_obj(
                missing_vol,
                transforms,
                prior_dir / "mesh_missing.obj",
                include_labels=[1],
            )

    (prior_dir / "prior_config.json").write_text(
        json.dumps(
            {
                "name": prior_config.name,
                "excluded_labels": excluded_labels,
                "min_diameter_mm": prior_config.min_diameter_mm,
                "diameter_estimator": _DIAMETER_ESTIMATOR_ID,
                "diameter_comparison": ">=",
                "diameter_surface_offset": "0.5*min(spacing_zyx)",
                "description": prior_config.description,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return prior_dir
