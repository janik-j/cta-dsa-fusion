"""Shared dataset I/O for projections + transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data_generation.common.transforms import write_transforms


def write_projection_arrays(*, output_dir: Path, arrays: dict[str, np.ndarray]) -> Path:
    proj_dir = output_dir / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        path = proj_dir / f"{name}.nii.gz"
        sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.float32)), str(path))
    return proj_dir


def write_dataset(
    *,
    output_dir: Path,
    arrays: dict[str, np.ndarray],
    transforms: dict,
    validate: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_projection_arrays(output_dir=output_dir, arrays=arrays)
    write_transforms(output_dir / "transforms.json", transforms, validate=validate)
