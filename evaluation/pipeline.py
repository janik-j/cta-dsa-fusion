"""
Evaluation utilities for this framework.

Single responsibility:
- 2D rendering metrics
- VQR mesh extraction + 3D metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer_voxelizer import query
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.gaussian_utils import get_scale_bound
from utils.image_utils import Averager, data_norm, get_psnr, get_ssim_2d
from utils.mesh_utils import (
    compute_metric,
    compute_metric_missing_region,
    compute_metric_missing_region_partitioned,
    icp_align_obj,
    mesh_extraction,
)

_VERBOSE = os.environ.get("DEBUG_EVAL", "").lower() in ("1", "true", "yes")


# ============================================================================
# Lazy dependency helpers
# ============================================================================


def _require_sitk():
    try:
        import SimpleITK as sitk  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("SimpleITK is required to write NIfTI outputs.") from e
    return sitk


# ============================================================================
# VQR (from vqr.py)
# ============================================================================


@dataclass(frozen=True)
class VQREvalConfig:
    fixed_threshold: float

    @classmethod
    def from_dataset(cls, dataset) -> VQREvalConfig:
        return cls(
            fixed_threshold=float(getattr(dataset, "eval_threshold", 0.008)),
        )


def _convert_obj_voxel_to_world(
    in_path: str,
    out_path: str,
    *,
    volume_origin: np.ndarray,
    volume_spacing: np.ndarray,
) -> None:
    origin = np.asarray(volume_origin, dtype=np.float64).reshape(1, 3)
    spacing = np.asarray(volume_spacing, dtype=np.float64).reshape(1, 3)
    tmp_path = out_path + ".tmp"
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


def _read_transforms(dataset) -> dict[str, Any]:
    json_path = Path(dataset.source_path) / "transforms.json"
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid transforms.json (expected object): {json_path}")
    return data


def _resolve_mesh_ref_from_transforms(transforms: dict[str, Any], *, source_path: Path) -> tuple[str, bool]:
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
    return str(mesh_ref_path), is_world


def _resolve_roi_mesh_paths(dataset) -> tuple[str | None, str | None]:
    ply_path = str(getattr(dataset, "ply_path", "") or "").strip()
    if not ply_path:
        return None, None

    prior_ply = Path(ply_path)
    if not prior_ply.is_absolute():
        prior_ply = (Path(dataset.source_path) / prior_ply).resolve()
    if not prior_ply.exists():
        return None, None

    prior_dir = prior_ply.parent
    mesh_missing = prior_dir / "mesh_missing.obj"
    mesh_prior = prior_dir / "mesh.obj"
    return (str(mesh_missing) if mesh_missing.exists() else None, str(mesh_prior) if mesh_prior.exists() else None)


def _roi_metrics(
    *,
    pred_mesh_path: str,
    mesh_missing_path: str | None,
    mesh_prior_path: str | None,
    spacing_for_metrics: np.ndarray,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {"missing_cd_mm": None, "missing_hd95_mm": None, "prior_cd_mm": None, "prior_hd95_mm": None}

    if mesh_missing_path and mesh_prior_path:
        try:
            cd_m, hd_m, cd_p, hd_p = compute_metric_missing_region_partitioned(
                mesh_missing_path,
                pred_mesh_path,
                mesh_prior_path,
                spacing_for_metrics,
            )
            out = {
                "missing_cd_mm": float(cd_m),
                "missing_hd95_mm": float(hd_m),
                "prior_cd_mm": float(cd_p),
                "prior_hd95_mm": float(hd_p),
            }
        except Exception as e:
            import warnings
            warnings.warn(f"Partitioned ROI metric computation failed: {e}", stacklevel=2)
        return out

    if mesh_missing_path:
        try:
            cd, hd = compute_metric_missing_region(mesh_missing_path, pred_mesh_path, spacing_for_metrics)
            out["missing_cd_mm"] = float(cd)
            out["missing_hd95_mm"] = float(hd)
        except Exception as e:
            import warnings
            warnings.warn(f"Missing-region metric computation failed: {e}", stacklevel=2)

    if mesh_prior_path:
        try:
            # Prior-region metrics should be comparable to global CD/HD95 (bidirectional).
            # `compute_metric_missing_region` is intentionally one-directional (GT->pred) and is only appropriate
            # for the missing-region recall-style metric.
            cd, hd = compute_metric(mesh_prior_path, pred_mesh_path, spacing_for_metrics)
            out["prior_cd_mm"] = float(cd)
            out["prior_hd95_mm"] = float(hd)
        except Exception as e:
            import warnings
            warnings.warn(f"Prior-region metric computation failed: {e}", stacklevel=2)

    return out


def _eval_vqr(
    *,
    threshold: float,
    vol_pred: np.ndarray,
    output_path: Path,
    mesh_ref: str,
    volume_origin: np.ndarray,
    volume_spacing: np.ndarray,
    spacing_for_metrics: np.ndarray,
    use_world_mesh_ref: bool,
    mesh_missing: str | None,
    mesh_prior: str | None,
) -> dict[str, Any]:
    """
    Extract a single prediction mesh and compute metrics.

    Contract:
    - No post-processing that changes topology (raw-only evaluation).
    - One output mesh per run (chosen by `threshold_mode`).
    """
    mesh_pred = str(output_path / "mesh_pred.obj")

    mesh_extraction(vol_pred, threshold, mesh_pred)

    if use_world_mesh_ref:
        # Keep `mesh_pred.obj` in the same coordinate frame as the GT mesh_ref (world/mm).
        _convert_obj_voxel_to_world(
            mesh_pred,
            mesh_pred,
            volume_origin=volume_origin,
            volume_spacing=volume_spacing,
        )

    # Paper protocol: align reference/reconstruction with ICP before reporting CD/HD.
    # We always write the aligned mesh for auditability.
    mesh_pred_icp = str(output_path / "mesh_pred_icp.obj")
    icp_info: dict[str, Any] | None = None
    try:
        icp_info = icp_align_obj(
            mesh_ref_path=mesh_ref,
            mesh_pred_path=mesh_pred,
            mesh_pred_aligned_path=mesh_pred_icp,
            max_iterations=75,
            sample_size=2000,
            seed=0,
            trim_fraction=0.8,
        )
        mesh_for_metrics = mesh_pred_icp if icp_info.get("ok") else mesh_pred
    except Exception as e:
        import warnings

        warnings.warn(f"ICP alignment failed; falling back to raw mesh: {e}", stacklevel=2)
        mesh_for_metrics = mesh_pred

    global_metrics: dict[str, float | None] = {"cd_mm": None, "hd95_mm": None}
    try:
        cd, hd95 = compute_metric(mesh_ref, mesh_for_metrics, spacing_for_metrics)
        global_metrics = {"cd_mm": float(cd), "hd95_mm": float(hd95)}
    except Exception as e:
        import warnings
        warnings.warn(f"Global metric computation failed: {e}", stacklevel=2)

    roi_metrics = _roi_metrics(
        pred_mesh_path=mesh_for_metrics,
        mesh_missing_path=mesh_missing,
        mesh_prior_path=mesh_prior,
        spacing_for_metrics=spacing_for_metrics,
    )

    return {
        "threshold_mode": "fixed",
        "threshold": float(threshold),
        "meshes": {"pred": mesh_pred, "pred_icp": mesh_pred_icp},
        "icp": icp_info,
        "metrics": {"global": global_metrics, "roi": roi_metrics},
    }


def compute_opacity(scene, gaussians) -> torch.Tensor:
    """Query the opacity field."""
    field_output = gaussians._field(gaussians, scene.recon_args)
    return field_output["final_opacity"]


def query_volume(*, gaussians, recon_args: dict[str, Any], pipe, opacity_query: torch.Tensor) -> np.ndarray:
    query_pkg = query(
        gaussians,
        recon_args,
        recon_args["volume_resolution"],
        recon_args["volume_phy"],
        [0, 0, 0],
        pipe,
        opacity_precomp=opacity_query,
    )
    return query_pkg["vol"].detach().cpu().numpy()


def run_vqr(scene, gaussians, pipe, dataset, *, phase: str = "test") -> dict[str, Any]:
    cfg = VQREvalConfig.from_dataset(dataset)

    output_path = Path(dataset.model_path) / phase / "VQR_output"
    output_path.mkdir(parents=True, exist_ok=True)

    opacity_query = compute_opacity(scene, gaussians)
    vol_pred = query_volume(gaussians=gaussians, recon_args=scene.recon_args, pipe=pipe, opacity_query=opacity_query)

    sitk = _require_sitk()
    sitk.WriteImage(
        sitk.GetImageFromArray(vol_pred.transpose(2, 1, 0)),
        str(output_path / "vol_pred.nii.gz"),
    )

    transforms = _read_transforms(dataset)
    mesh_ref_path, use_world_mesh_ref = _resolve_mesh_ref_from_transforms(transforms, source_path=Path(dataset.source_path))

    mesh_missing, mesh_prior = _resolve_roi_mesh_paths(dataset)

    volume_origin = np.asarray(scene.recon_args["volume_origin"], dtype=np.float32)
    volume_spacing = np.asarray(scene.recon_args["volume_phy"], dtype=np.float32) / np.asarray(
        scene.recon_args["volume_resolution"], dtype=np.float32
    )
    # Metric helpers in `utils/mesh_utils.py` assume OBJ vertices are in voxel space and then scale by `spacing`.
    # If the GT mesh_ref is already in world/mm coordinates, do NOT apply an additional spacing scale.
    if use_world_mesh_ref:
        spacing_for_metrics = np.ones((3,), dtype=np.float32)
    else:
        spacing_for_metrics = np.asarray(scene.recon_args["volume_spacing"], dtype=np.float32)

    results = _eval_vqr(
        threshold=cfg.fixed_threshold,
        vol_pred=vol_pred,
        output_path=output_path,
        mesh_ref=mesh_ref_path,
        volume_origin=volume_origin,
        volume_spacing=volume_spacing,
        spacing_for_metrics=spacing_for_metrics,
        use_world_mesh_ref=use_world_mesh_ref,
        mesh_missing=mesh_missing,
        mesh_prior=mesh_prior,
    )

    # Persist a single structured schema (no duplicated flat keys).
    results_out: dict[str, Any] = {
        "metrics": {
            "global": dict(results["metrics"]["global"]),
            "roi": dict(results["metrics"]["roi"]),
        },
        "threshold": float(results["threshold"]),
        "threshold_mode": str(results["threshold_mode"]),
        "meshes": dict(results.get("meshes") or {}),
    }

    with open(output_path / "vqr_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)

    return results_out


# ============================================================================
# Render metrics
# ============================================================================


def _fmt(v: Any, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def render_set(scene: Scene, gaussians: GaussianModel, pipe, dataset, phase: str = "test") -> dict[str, Any]:
    """Render all views and compute 2D metrics (PSNR, SSIM)."""
    from gaussian_renderer_voxelizer import render

    all_views = scene.getAllCameras().copy()
    output_path = os.path.join(dataset.model_path, phase, "render_output")
    os.makedirs(output_path, exist_ok=True)

    render_images = []
    gt_images = []
    for view in tqdm(all_views, desc="Rendering", disable=not _VERBOSE):
        render_pkg = render(view, gaussians, scene.recon_args, pipe)
        render_images.append(render_pkg["render"][0].cpu().numpy())
        gt_images.append(view.original_image[0].cpu().numpy())
    render_images = np.stack(render_images)
    render_images = np.clip(render_images, 0, render_images.max())
    gt_images = np.stack(gt_images)
    gt_images = np.clip(gt_images, 0, gt_images.max())

    sitk = _require_sitk()
    sitk.WriteImage(sitk.GetImageFromArray(render_images), os.path.join(output_path, "proj.nii.gz"))

    train_indice = set(scene.get_indice()["train_indice"])
    metrics = {"train": {"psnr": Averager(), "ssim": Averager()}, "eval": {"psnr": Averager(), "ssim": Averager()}}
    counts = {"train": 0, "eval": 0}

    for idx in range(len(all_views)):
        rendering = data_norm(render_images[idx])
        gt = data_norm(gt_images[idx])
        split = "train" if idx in train_indice else "eval"
        metrics[split]["psnr"].add(get_psnr(rendering, gt))
        metrics[split]["ssim"].add(get_ssim_2d(rendering, gt))
        counts[split] += 1

    results = {k: {"psnr": np.round(v["psnr"].avg(), 4), "ssim": np.round(v["ssim"].avg(), 4)} for k, v in metrics.items()}

    # Common in 2/3-view experiments: all available views are used for training,
    # leaving an empty eval set. In that case, report eval metrics on the same
    # (train) views to avoid misleading zeros.
    if counts["eval"] == 0 and counts["train"] > 0:
        results["eval"] = dict(results["train"])

    metrics_dict = {
        "num_gaussians": int(gaussians._xyz.shape[0]),
        "train_psnr": float(results["train"]["psnr"]),
        "train_ssim": float(results["train"]["ssim"]),
        "eval_psnr": float(results["eval"]["psnr"]),
        "eval_ssim": float(results["eval"]["ssim"]),
        "train_views": int(counts["train"]),
        "eval_views": int(counts["eval"]),
    }
    with open(os.path.join(output_path, "render_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics_dict


def voxel_query_reconstruction(scene: Scene, gaussians: GaussianModel, pipe, dataset, phase: str = "test") -> dict[str, Any]:
    """Run VQR (raw mesh only) and return metrics."""
    return run_vqr(scene, gaussians, pipe, dataset, phase=phase)


def run_evaluation(dataset, pipeline, field_conf, args) -> dict[str, Any]:
    """Run evaluation and return metrics dict."""
    results: dict[str, Any] = {}

    with torch.no_grad():
        scale_bound = get_scale_bound(dataset)
        gaussians = GaussianModel(field_conf)
        scene = Scene(
            dataset,
            gaussians,
            scale_bound,
            load_iteration=args.iteration,
            shuffle=False,
        )

        phase = getattr(args, "phase", "test")
        if args.render_2d:
            results["render"] = render_set(scene, gaussians, pipeline, dataset, phase)
        if args.VQR:
            results["vqr"] = voxel_query_reconstruction(scene, gaussians, pipeline, dataset, phase)

    return results


def print_results_summary(results: dict[str, Any], model_path: str) -> None:
    """Print clean results summary."""
    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")

    if "render" in results:
        r = results["render"]
        print("\n2D Rendering:")
        print(f"  Gaussians: {r.get('num_gaussians', '-')}")
        print(f"  Train PSNR: {_fmt(r.get('train_psnr'))} dB  SSIM: {_fmt(r.get('train_ssim'))}")
        print(f"  Eval  PSNR: {_fmt(r.get('eval_psnr'))} dB  SSIM: {_fmt(r.get('eval_ssim'))}")

    if "vqr" in results:
        v = results["vqr"]
        print("\n3D Reconstruction (VQR):")
        global_m = (v.get("metrics") or {}).get("global") if isinstance(v, dict) else {}
        roi_m = (v.get("metrics") or {}).get("roi") if isinstance(v, dict) else {}
        if not isinstance(global_m, dict):
            global_m = {}
        if not isinstance(roi_m, dict):
            roi_m = {}

        print(f"  CD:   {_fmt(global_m.get('cd_mm', v.get('cd_mm')))} mm")
        print(f"  HD95: {_fmt(global_m.get('hd95_mm', v.get('hd95_mm')))} mm")
        # Always report ROI splits (may be '-' if meshes are unavailable).
        print(f"  CD (missing):   {_fmt(roi_m.get('missing_cd_mm', v.get('cd_missing_mm')))} mm")
        print(f"  HD95 (missing): {_fmt(roi_m.get('missing_hd95_mm', v.get('hd95_missing_mm')))} mm")
        print(f"  CD (prior):     {_fmt(roi_m.get('prior_cd_mm', v.get('cd_prior_mm')))} mm")
        print(f"  HD95 (prior):   {_fmt(roi_m.get('prior_hd95_mm', v.get('hd95_prior_mm')))} mm")

    print(f"{'=' * 60}\n")
