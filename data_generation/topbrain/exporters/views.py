"""TopBrain view generation utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from data_generation.common.dataset_io import write_dataset
from data_generation.common.transforms import load_transforms, set_mesh_ref, write_transforms
from data_generation.topbrain.config import ViewConfig
from data_generation.topbrain.loaders import cases as topbrain_cases
from data_generation.topbrain.processors import cta as topbrain_cta


def create_view_transforms(base_transforms: dict, view_config: ViewConfig) -> dict:
    """Create transforms dict with primary and secondary angles."""
    transforms = dict(base_transforms)
    base_frame = (
        dict(transforms["frames"][0]) if transforms.get("frames") else {"PrimaryAngle": 0.0, "SecondaryAngle": 0.0}
    )

    frames = []
    for i, (primary, secondary) in enumerate(zip(view_config.angles_deg, view_config.secondary_angles_deg)):
        frame = dict(base_frame)
        frame["PrimaryAngle"] = float(np.deg2rad(primary))
        frame["SecondaryAngle"] = float(np.deg2rad(secondary))
        frame["file"] = f"frame_{i:04d}"
        frames.append(frame)

    transforms["frames"] = frames
    transforms["N_views"] = len(frames)
    return transforms


def ensure_view_mesh_ref(*, case_root: Path, view_dir: Path) -> None:
    src = case_root / "geometry" / "mesh_ref.obj"
    if not src.exists():
        raise FileNotFoundError(f"Missing base mesh_ref.obj at {src}. Run `generate_base()` before views.")
    dst = view_dir / "mesh_ref.obj"
    if dst.exists():
        return
    view_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def ensure_view_transforms_mesh_ref(*, view_dir: Path, mesh_ref_space: str) -> None:
    tf_path = view_dir / "transforms.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"Missing transforms.json in view dir: {tf_path}")
    transforms = load_transforms(tf_path)

    before = json.dumps(transforms, sort_keys=True)
    set_mesh_ref(transforms, rel_path="mesh_ref.obj", space=mesh_ref_space)
    after = json.dumps(transforms, sort_keys=True)
    if after != before:
        write_transforms(tf_path, transforms, validate=True)


def generate_view(
    *,
    ct_path: Path,
    seg_path: Path,
    view_config: ViewConfig,
    output_root: Path,
    transforms_template: Path,
    contrast_enhancement: float,
) -> Path:
    """Generate projections for a specific view configuration."""
    case_name = topbrain_cases.extract_case_name(ct_path)
    output_dir = output_root / case_name
    view_dir = output_dir / "views" / view_config.name

    if (view_dir / "projections" / "mask_run.nii.gz").exists():
        ensure_view_mesh_ref(case_root=output_dir, view_dir=view_dir)
        ensure_view_transforms_mesh_ref(view_dir=view_dir, mesh_ref_space="voxel")
        return view_dir

    ct_volume, seg_volume, metadata = topbrain_cta.load_cta_and_segmentation(str(ct_path), str(seg_path))

    base_transforms = load_transforms(transforms_template)
    base_transforms = topbrain_cta.apply_cta_volume_to_transforms(base_transforms, metadata, case_name=case_name)
    transforms = create_view_transforms(base_transforms, view_config)

    baseline_volume, contrast_volume = topbrain_cta.create_baseline_contrast_volumes(
        ct_volume, seg_volume, contrast_enhancement, excluded_labels=None
    )

    geo = topbrain_cta.create_tigre_geometry(transforms)
    target_shape = tuple(transforms["volume_resolution"][::-1])
    baseline_tigre = topbrain_cta.resample_volume(baseline_volume, target_shape, order=1)
    contrast_tigre = topbrain_cta.resample_volume(contrast_volume, target_shape, order=1)

    angles_rad = np.deg2rad(view_config.angles_deg)
    baseline_projs = topbrain_cta.forward_project(baseline_tigre, geo, angles_rad)
    contrast_projs = topbrain_cta.forward_project(contrast_tigre, geo, angles_rad)

    transforms["volume_resolution"] = [int(geo.nVoxel[2]), int(geo.nVoxel[1]), int(geo.nVoxel[0])]
    transforms["volume_phy"] = [
        float(geo.nVoxel[2] * geo.dVoxel[2]),
        float(geo.nVoxel[1] * geo.dVoxel[1]),
        float(geo.nVoxel[0] * geo.dVoxel[0]),
    ]
    transforms["pose_type"] = "angles_rad"
    transforms["dsa_convention"] = "fill_minus_mask"

    set_mesh_ref(transforms, rel_path="mesh_ref.obj", space="voxel")

    view_dir.mkdir(parents=True, exist_ok=True)
    write_dataset(
        output_dir=view_dir,
        arrays={"mask_run": baseline_projs, "fill_run": contrast_projs},
        transforms=transforms,
        validate=True,
    )
    ensure_view_mesh_ref(case_root=output_dir, view_dir=view_dir)

    (view_dir / "view_config.json").write_text(
        json.dumps(
            {
                "name": view_config.name,
                "angles_deg": view_config.angles_deg,
                "secondary_angles_deg": view_config.secondary_angles_deg,
                "description": view_config.description,
                "n_views": len(view_config.angles_deg),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return view_dir
