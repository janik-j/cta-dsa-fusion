"""
VQR evaluation from a predicted volume (no 4DRGS Scene required).

This is used by baseline integrations (e.g., R2-Gaussian) to produce
`train/VQR_output/vqr_metrics.json` with the same schema/protocol as 4DRGS:
- fixed/adaptive thresholding
- marching-cubes mesh extraction
- optional voxel->world conversion when GT mesh_ref is world-space
- ICP alignment before CD/HD95 reporting
- optional ROI metrics if a prior PLY is provided (mesh_missing/mesh.obj)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.mesh_utils import (
    compute_metric,
    compute_metric_missing_region,
    compute_metric_missing_region_partitioned,
    icp_align_obj,
    mesh_extraction,
)


def _try_import_sitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception:
        return None
    return sitk


@dataclass(frozen=True)
class VQRVolumeEvalConfig:
    threshold_mode: str
    fixed_threshold: float
    adaptive_scale: float

    def __post_init__(self) -> None:
        mode = str(self.threshold_mode).strip().lower()
        if mode not in {"adaptive", "fixed", "otsu"}:
            mode = "fixed"
        object.__setattr__(self, "threshold_mode", mode)
        object.__setattr__(self, "fixed_threshold", float(self.fixed_threshold))
        object.__setattr__(self, "adaptive_scale", float(self.adaptive_scale))


def _read_transforms(source_path: Path) -> dict[str, Any]:
    json_path = source_path / "transforms.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid transforms.json (expected object): {json_path}")
    return data


def _resolve_mesh_ref(transforms: dict[str, Any], *, source_path: Path) -> tuple[Path, bool]:
    geo = transforms.get("geometry")
    if not isinstance(geo, dict):
        raise ValueError("transforms.json missing required `geometry` section (dataset contract)")

    mesh_ref = geo.get("mesh_ref")
    if not isinstance(mesh_ref, dict):
        raise ValueError("transforms.json missing required `geometry.mesh_ref` section (dataset contract)")

    rel_path = str(mesh_ref.get("path", "")).strip()
    if not rel_path:
        raise ValueError("transforms.json missing required `geometry.mesh_ref.path` (dataset contract)")

    mesh_ref_path = (source_path / rel_path).resolve()
    if not mesh_ref_path.exists():
        raise FileNotFoundError(f"mesh_ref.obj not found at path declared in transforms.json: {mesh_ref_path}")

    is_world = str(mesh_ref.get("space", "voxel")).strip().lower() == "world"
    return mesh_ref_path, is_world


def _resolve_roi_mesh_paths(*, ply_path: str | Path | None) -> tuple[str | None, str | None]:
    if not ply_path:
        return None, None
    prior_ply = Path(str(ply_path)).expanduser()
    if not prior_ply.is_absolute():
        prior_ply = prior_ply.resolve()
    if not prior_ply.exists():
        return None, None

    prior_dir = prior_ply.parent
    mesh_missing = prior_dir / "mesh_missing.obj"
    mesh_prior = prior_dir / "mesh.obj"
    return (str(mesh_missing) if mesh_missing.exists() else None, str(mesh_prior) if mesh_prior.exists() else None)


def _convert_obj_voxel_to_world(
    in_path: Path,
    out_path: Path,
    *,
    volume_origin: np.ndarray,
    volume_spacing: np.ndarray,
) -> None:
    origin = np.asarray(volume_origin, dtype=np.float64).reshape(1, 3)
    spacing = np.asarray(volume_spacing, dtype=np.float64).reshape(1, 3)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(in_path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64).reshape(1, 3)
                    w = origin + (v + 0.5) * spacing
                    fout.write(f"v {w[0, 0]:.8f} {w[0, 1]:.8f} {w[0, 2]:.8f}\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)
    os.replace(tmp_path, out_path)


def _roi_metrics(
    *,
    pred_mesh_path: str,
    mesh_missing_path: str | None,
    mesh_prior_path: str | None,
    spacing_for_metrics: np.ndarray,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {"missing_cd_mm": None, "missing_hd95_mm": None, "prior_cd_mm": None, "prior_hd95_mm": None}

    if mesh_missing_path and mesh_prior_path:
        cd_m, hd_m, cd_p, hd_p = compute_metric_missing_region_partitioned(
            mesh_missing_path,
            pred_mesh_path,
            mesh_prior_path,
            spacing_for_metrics,
        )
        return {
            "missing_cd_mm": float(cd_m),
            "missing_hd95_mm": float(hd_m),
            "prior_cd_mm": float(cd_p),
            "prior_hd95_mm": float(hd_p),
        }

    if mesh_missing_path:
        cd, hd = compute_metric_missing_region(mesh_missing_path, pred_mesh_path, spacing_for_metrics)
        out["missing_cd_mm"] = float(cd)
        out["missing_hd95_mm"] = float(hd)

    if mesh_prior_path:
        cd, hd = compute_metric(mesh_prior_path, pred_mesh_path, spacing_for_metrics)
        out["prior_cd_mm"] = float(cd)
        out["prior_hd95_mm"] = float(hd)

    return out


def _select_threshold(vol_pred: np.ndarray, cfg: VQRVolumeEvalConfig) -> tuple[float, str]:
    mode = str(cfg.threshold_mode).strip().lower()
    if mode == "fixed":
        return float(cfg.fixed_threshold), "fixed"
    if mode == "otsu":
        finite = np.asarray(vol_pred, dtype=np.float32)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            return float(cfg.fixed_threshold), "otsu"

        values = finite[finite > 0.0]
        if values.size < 64:
            values = finite
        vmin = float(values.min()) if values.size else 0.0
        vmax = float(values.max()) if values.size else 0.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
            return float(cfg.fixed_threshold), "otsu"

        try:
            from skimage.filters import threshold_otsu

            th = float(threshold_otsu(values.astype(np.float32, copy=False)))
            if not np.isfinite(th):
                return float(cfg.fixed_threshold), "otsu"
            return th, "otsu"
        except Exception:
            return float(cfg.fixed_threshold), "otsu"

    vol_max = float(vol_pred.max()) if vol_pred.size else 0.0
    if vol_max > 0.0:
        return float(vol_max * float(cfg.adaptive_scale)), "adaptive"

    # Fallback for empty volumes.
    return float(cfg.fixed_threshold), "adaptive"


def evaluate_vqr_from_volume(
    *,
    vol_pred: np.ndarray,
    dataset_source_path: str | Path,
    output_dir: str | Path,
    cfg: VQRVolumeEvalConfig,
    ply_path_for_roi: str | Path | None = None,
    icp_seed: int = 0,
    write_vol_nifti: bool = True,
) -> dict[str, Any]:
    """
    Evaluate a predicted voxel volume against the dataset's GT mesh_ref.

    Args:
        vol_pred: Predicted volume as a numpy array in (X, Y, Z) voxel index order.
        dataset_source_path: Dataset folder containing transforms.json + mesh_ref.
        output_dir: Where to write VQR artifacts (mesh_pred.obj, vqr_metrics.json, ...).
        cfg: Threshold mode/values.
        ply_path_for_roi: Optional prior PLY path used only to locate ROI meshes (mesh_missing.obj, mesh.obj).
        icp_seed: Seed used by ICP subsampling (for deterministic evaluation).
        write_vol_nifti: If True, write vol_pred.nii.gz when SimpleITK is available.

    Returns:
        Structured dict matching `evaluation/pipeline.py`'s VQR JSON schema.
    """
    source_path = Path(str(dataset_source_path)).expanduser().resolve()
    out_dir = Path(str(output_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vol_pred = np.asarray(vol_pred, dtype=np.float32)
    if vol_pred.ndim != 3:
        raise ValueError(f"Expected vol_pred to be 3D (X,Y,Z), got shape={vol_pred.shape}")

    sitk = _try_import_sitk()
    if write_vol_nifti and sitk is not None:
        # SimpleITK expects array in (Z, Y, X).
        sitk.WriteImage(sitk.GetImageFromArray(vol_pred.transpose(2, 1, 0)), str(out_dir / "vol_pred.nii.gz"))

    transforms = _read_transforms(source_path)
    mesh_ref_path, mesh_ref_is_world = _resolve_mesh_ref(transforms, source_path=source_path)

    volume_phy = np.asarray(transforms.get("volume_phy", [0, 0, 0]), dtype=np.float32).reshape(3)
    volume_res = np.asarray(transforms.get("volume_resolution", [1, 1, 1]), dtype=np.float32).reshape(3)
    volume_spacing = (volume_phy / np.maximum(volume_res, 1.0)).astype(np.float32)
    volume_origin = np.asarray(transforms.get("volume_origin", (-volume_phy / 2.0).tolist()), dtype=np.float32).reshape(3)

    # Metric helpers assume OBJ vertices are in voxel space and then scale by spacing.
    # If the GT mesh_ref is already in world/mm coordinates, do NOT apply an additional spacing scale.
    spacing_for_metrics = (
        np.ones((3,), dtype=np.float32)
        if mesh_ref_is_world
        else np.asarray(transforms.get("volume_spacing", volume_spacing.tolist()), dtype=np.float32).reshape(3)
    )

    thres, thres_mode = _select_threshold(vol_pred, cfg)

    mesh_pred = out_dir / "mesh_pred.obj"
    mesh_pred_icp = out_dir / "mesh_pred_icp.obj"

    mesh_extraction(vol_pred, float(thres), str(mesh_pred))

    if mesh_ref_is_world:
        _convert_obj_voxel_to_world(mesh_pred, mesh_pred, volume_origin=volume_origin, volume_spacing=volume_spacing)

    icp_info: dict[str, Any] | None = None
    try:
        icp_info = icp_align_obj(
            mesh_ref_path=str(mesh_ref_path),
            mesh_pred_path=str(mesh_pred),
            mesh_pred_aligned_path=str(mesh_pred_icp),
            max_iterations=75,
            sample_size=2000,
            seed=int(icp_seed),
            trim_fraction=0.8,
        )
        mesh_for_metrics = str(mesh_pred_icp) if icp_info.get("ok") else str(mesh_pred)
    except Exception:
        # Fall back to raw mesh.
        mesh_for_metrics = str(mesh_pred)

    cd_mm, hd95_mm = compute_metric(str(mesh_ref_path), mesh_for_metrics, spacing_for_metrics)

    mesh_missing, mesh_prior = _resolve_roi_mesh_paths(ply_path=ply_path_for_roi)
    roi = _roi_metrics(
        pred_mesh_path=mesh_for_metrics,
        mesh_missing_path=mesh_missing,
        mesh_prior_path=mesh_prior,
        spacing_for_metrics=spacing_for_metrics,
    )

    results: dict[str, Any] = {
        "metrics": {
            "global": {"cd_mm": float(cd_mm), "hd95_mm": float(hd95_mm)},
            "roi": dict(roi),
        },
        "threshold": float(thres),
        "threshold_mode": str(thres_mode),
        "meshes": {"pred": str(mesh_pred), "pred_icp": str(mesh_pred_icp)},
    }

    # Keep alignment diagnostics if available (ignored by collectors, useful for debugging).
    if icp_info is not None:
        results["icp"] = dict(icp_info)

    with open(out_dir / "vqr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
