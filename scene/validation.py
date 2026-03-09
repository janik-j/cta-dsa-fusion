"""Dataset schema validation for transforms.json and file layout."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .conventions import DSA_CONVENTIONS, POSE_TYPES


class DatasetSchemaError(ValueError):
    pass


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _require_key(d: Mapping[str, Any], key: str, *, where: str) -> Any:
    if key not in d:
        raise DatasetSchemaError(f"{where}: missing required field '{key}'")
    return d[key]


def _require_list(value: Any, *, name: str, length: int, number: bool = True) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        raise DatasetSchemaError(f"{name}: expected list length {length}, got {type(value).__name__} len={getattr(value, '__len__', None)}")
    if number and not all(_is_number(v) for v in value):
        raise DatasetSchemaError(f"{name}: expected numeric list of length {length}")
    return value


def _validate_frame(frame: Mapping[str, Any], *, pose_type: str, idx: int) -> None:
    where = f"frames[{idx}]"
    _require_key(frame, "file", where=where)
    if not isinstance(frame["file"], str) or not frame["file"]:
        raise DatasetSchemaError(f"{where}.file must be a non-empty string")

    pt = str(pose_type).strip().lower()
    if pt == "angles_rad":
        _require_key(frame, "PrimaryAngle", where=where)
        if not _is_number(frame["PrimaryAngle"]):
            raise DatasetSchemaError(f"{where}.PrimaryAngle must be numeric")
        if "SecondaryAngle" in frame and not _is_number(frame["SecondaryAngle"]):
            raise DatasetSchemaError(f"{where}.SecondaryAngle must be numeric")
        return

    if pt == "world2cam_4x4":
        has_w2c = "world2cam" in frame
        has_rt = "R" in frame and "T" in frame
        if not (has_w2c or has_rt):
            raise DatasetSchemaError(f"{where}: require world2cam (4x4) or R/T for pose_type=world2cam_4x4")
        if has_w2c:
            w2c = np.asarray(frame["world2cam"], dtype=np.float32)
            if w2c.shape != (4, 4):
                raise DatasetSchemaError(f"{where}.world2cam must be 4x4")
        if has_rt:
            r = np.asarray(frame["R"], dtype=np.float32)
            t = np.asarray(frame["T"], dtype=np.float32)
            if r.shape != (3, 3):
                raise DatasetSchemaError(f"{where}.R must be 3x3")
            if t.shape not in {(3,), (3, 1)}:
                raise DatasetSchemaError(f"{where}.T must be length-3")
        return

    raise DatasetSchemaError(f"pose_type '{pose_type}' is not supported")


def validate_transforms(transforms: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(transforms, Mapping):
        raise DatasetSchemaError("transforms.json must be a JSON object")

    # Backwards compatibility: older datasets omit pose_type/dsa_convention.
    # - TopCoW legacy transforms store PrimaryAngle/SecondaryAngle (radians) per frame.
    # - DSA convention historically used fill-minus-mask subtraction.
    pose_type = str(transforms.get("pose_type", "angles_rad")).strip().lower()
    if pose_type not in POSE_TYPES:
        raise DatasetSchemaError(f"pose_type must be one of {sorted(POSE_TYPES)}, got {pose_type!r}")

    dsa_convention = str(transforms.get("dsa_convention", "fill_minus_mask")).strip().lower()
    if dsa_convention not in DSA_CONVENTIONS:
        raise DatasetSchemaError(
            f"dsa_convention must be one of {sorted(DSA_CONVENTIONS)}, got {dsa_convention!r}"
        )

    n_views = _require_key(transforms, "N_views", where="transforms")
    if not _is_number(n_views) or int(n_views) <= 0:
        raise DatasetSchemaError("N_views must be a positive integer")

    _require_list(_require_key(transforms, "proj_resolution", where="transforms"), name="proj_resolution", length=2)
    _require_list(_require_key(transforms, "proj_phy", where="transforms"), name="proj_phy", length=2)
    _require_list(_require_key(transforms, "proj_spacing", where="transforms"), name="proj_spacing", length=2)

    _require_list(_require_key(transforms, "volume_resolution", where="transforms"), name="volume_resolution", length=3)
    _require_list(_require_key(transforms, "volume_spacing", where="transforms"), name="volume_spacing", length=3)
    _require_list(_require_key(transforms, "volume_phy", where="transforms"), name="volume_phy", length=3)

    if "volume_origin" in transforms:
        _require_list(transforms["volume_origin"], name="volume_origin", length=3)

    frames = _require_key(transforms, "frames", where="transforms")
    if not isinstance(frames, list) or not frames:
        raise DatasetSchemaError("frames must be a non-empty list")
    if len(frames) < int(n_views):
        raise DatasetSchemaError(f"frames length ({len(frames)}) < N_views ({int(n_views)})")

    for idx, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise DatasetSchemaError(f"frames[{idx}] must be an object")
        _validate_frame(frame, pose_type=pose_type, idx=idx)

    out = dict(transforms)
    out.setdefault("pose_type", pose_type)
    out.setdefault("dsa_convention", dsa_convention)
    return out


def validate_dataset_layout(source_path: str | Path) -> dict[str, Path]:
    root = Path(source_path)
    transforms_path = root / "transforms.json"
    if not transforms_path.exists():
        raise DatasetSchemaError(f"Missing transforms.json at {transforms_path}")

    projections_dir = root / "projections"
    if not projections_dir.exists():
        raise DatasetSchemaError(f"Missing projections/ at {projections_dir}")

    mask_run = projections_dir / "mask_run.nii.gz"
    fill_run = projections_dir / "fill_run.nii.gz"
    if not mask_run.exists():
        raise DatasetSchemaError(f"Missing required file: {mask_run}")
    if not fill_run.exists():
        raise DatasetSchemaError(f"Missing required file: {fill_run}")

    return {"root": root, "transforms": transforms_path, "projections": projections_dir}
