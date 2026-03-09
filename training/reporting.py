"""Training reporting and visualization helpers (I/O + plots + validation)."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.image_utils import data_norm, psnr


def _wandb_log(wandb_run, data: dict[str, Any], *, step: int) -> None:
    if not wandb_run:
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return
    wandb.log(data, step=step)


def _wandb_image(path: str, caption: str | None = None):
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb.Image(path, caption=caption) if caption else wandb.Image(path)


def resolve_vis_config(vis_config: dict | None, iterations: int, testing_iterations) -> dict:
    if vis_config is None:
        vis_config = {"save_dsa": True, "interval": 1000}

    if "vis_iterations" not in vis_config:
        interval = vis_config.get("interval")
        if interval:
            vis_config["vis_iterations"] = [0, *list(range(int(interval), iterations + 1, int(interval)))]
        else:
            vis_config["vis_iterations"] = [0, *list(testing_iterations)]

    return vis_config


def print_vis_config(vis_config: dict, model_path: str) -> None:
    vis_iterations = vis_config.get("vis_iterations", [])
    vis_enabled = bool(vis_config.get("save_dsa", False))

    if not vis_enabled or not vis_iterations:
        return

    print(f"\n{'=' * 80}")
    print("Visualization Configuration:")
    if vis_config.get("save_dsa", False):
        print("  - DSA images enabled")

    if "interval" in vis_config:
        print(f"  - Saving every {vis_config['interval']} iterations")
    elif len(vis_iterations) > 1:
        intervals = [vis_iterations[i + 1] - vis_iterations[i] for i in range(len(vis_iterations) - 1)]
        if len(set(intervals)) == 1:
            print(f"  - Saving every {intervals[0]} iterations")
        else:
            print(f"  - Saving at specific iterations: {vis_iterations}")
    else:
        print(f"  - Saving at iteration: {vis_iterations}")

    print(f"  - Output: {model_path}/")
    print(f"{'=' * 80}\n")


def log_point_cloud_init(gaussians, dataset) -> None:
    num_pts = gaussians.get_xyz.shape[0]
    print("\n[METRIC] Point Cloud Initialization:")
    print(f"  - Total Points: {num_pts}")

    actual_counts_logged = False
    is_m1 = getattr(gaussians, "is_m1", None)
    if torch.is_tensor(is_m1) and is_m1.numel() == int(num_pts):
        m1_actual = int(is_m1.sum().item())
        m2_actual = int(num_pts) - m1_actual
        print(f"  - M1 (Prior, actual): {m1_actual}")
        print(f"  - M2 (Residual/Scaffold, actual): {m2_actual}")
        actual_counts_logged = True

    if not actual_counts_logged:
        print(f"  - M1 (Prior, configured cap): {getattr(dataset, 'M1', 'n/a')}")
        print(f"  - M2 (Residual/Scaffold, configured cap): {getattr(dataset, 'M2', 'n/a')}")

    if hasattr(dataset, "residuals_path") and dataset.residuals_path:
        print(f"  - M2 Source: {dataset.residuals_path}")
    print(f"{'=' * 80}\n", flush=True)


def log_scalar_metrics(wandb_run, iteration, loss_dict, gaussians) -> None:
    if not wandb_run:
        return

    wandb_dict = {"iteration": iteration}

    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            wandb_dict[f"train_loss_{key}"] = value.item()
        elif isinstance(value, (int, float)):
            wandb_dict[f"train_loss_{key}"] = float(value)

    wandb_dict["total_points"] = gaussians.get_xyz.shape[0]

    avgopacity = gaussians.avgopacity_accum / gaussians.denom
    avgopacity[avgopacity.isnan()] = 0.0
    wandb_dict["avgopacity_min"] = avgopacity.min().item()
    wandb_dict["avgopacity_max"] = avgopacity.max().item()
    wandb_dict["avgopacity_mean"] = avgopacity.mean().item()
    wandb_dict["avgopacity_median"] = avgopacity.median().item()

    _wandb_log(wandb_run, wandb_dict, step=iteration)


def log_pruning_stats(wandb_run, iteration, prune_num_record) -> None:
    if not prune_num_record or not wandb_run:
        return

    wandb_dict = {f"prune_{key}": value for key, value in prune_num_record.items()}
    _wandb_log(wandb_run, wandb_dict, step=iteration)


def run_validation(scene, gaussians, render_func, pipe, iteration, wandb_run) -> None:
    torch.cuda.empty_cache()

    train_cams = scene.getTrainCameras()
    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {"name": "train", "cameras": [train_cams[idx % len(train_cams)] for idx in range(5, 30, 5)]},
    )

    for config in validation_configs:
        cameras = config["cameras"]
        if not cameras:
            continue

        psnr_total = 0.0
        for viewpoint in cameras:
            render_pkg = render_func(viewpoint, gaussians, scene.recon_args, pipe)
            rendered = render_pkg["render"]
            image = data_norm(torch.clamp(rendered, 0, float(rendered.max().item())))
            gt = viewpoint.original_image
            if torch.is_tensor(gt):
                gt = gt.to(render_pkg["render"].device)
            gt_image = data_norm(torch.clamp(gt, 0, float(gt.max().item())))
            psnr_total += psnr(image, gt_image).mean().double()

        psnr_avg = psnr_total / len(cameras)
        print(f"\n[ITER {iteration}] Evaluating {config['name']}: PSNR {psnr_avg}")

        if wandb_run:
            _wandb_log(wandb_run, {f"{config['name']}_psnr": psnr_avg}, step=iteration)

    torch.cuda.empty_cache()


def training_report(
    wandb_run,
    iteration,
    loss_dict,
    prune_num_record,
    testing_iterations,
    scene,
    gaussians,
    render_func,
    pipe,
    vis_config=None,
) -> None:
    if vis_config is None:
        vis_config = {"save_dsa": False, "vis_iterations": testing_iterations}

    log_scalar_metrics(wandb_run, iteration, loss_dict, gaussians)
    log_pruning_stats(wandb_run, iteration, prune_num_record)

    if iteration in testing_iterations:
        run_validation(scene, gaussians, render_func, pipe, iteration, wandb_run)

    vis_iterations = vis_config.get("vis_iterations", testing_iterations)
    interval = vis_config.get("interval", 0)
    do_vis = iteration in vis_iterations or (interval > 0 and iteration > 0 and iteration % interval == 0)

    if not do_vis:
        return

    train_cameras = scene.getTrainCameras()

    if vis_config.get("save_dsa", False) and train_cameras:
        _save_dsa_visualizations(
            train_cameras,
            vis_config,
            render_func,
            gaussians,
            scene.recon_args,
            pipe,
            scene.model_path,
            iteration,
            wandb_run,
        )


def _select_cameras_for_dsa(cams, vis_config):
    if not cams:
        return []

    max_views = vis_config.get("save_dsa_max_views")
    return cams[: int(max_views)] if max_views else cams


def _to_numpy_2d_image(img) -> np.ndarray:
    if torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    return np.asarray(arr, dtype=np.float32)


def _image_stats(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0, "p99": 0.0}

    arr = np.where(np.isfinite(arr), arr, 0.0)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
    }


def _normalize_image(img, *, percentile: float | None = None) -> tuple[np.ndarray, float, dict[str, float]]:
    arr = _to_numpy_2d_image(img)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    stats = _image_stats(arr)

    scale = float(stats["max"])
    if percentile is not None:
        pct = float(percentile)
        if 0.0 < pct < 100.0:
            scale = float(np.percentile(arr, pct)) if arr.size > 0 else 0.0
        else:
            print(
                f"[DSA_DEBUG][WARN] Invalid save_dsa_norm_percentile={percentile}; "
                "falling back to max normalization.",
                flush=True,
            )

    if scale > 0:
        arr = np.clip(arr / scale, 0.0, 1.0)
    return arr, float(scale), stats


def _build_camera_tag(viewpoint_cam) -> str:
    camera_name = getattr(viewpoint_cam, "image_name", None)
    cam_uid = getattr(viewpoint_cam, "uid", None)
    cam_angle = getattr(viewpoint_cam, "PrimaryAngle", None)

    tag_parts = []
    if camera_name is not None:
        tag_parts.append(str(camera_name))
    if cam_uid is not None:
        try:
            tag_parts.append(f"uid_{int(cam_uid):04d}")
        except Exception:
            tag_parts.append(f"uid_{cam_uid}")
    if cam_angle is not None:
        try:
            tag_parts.append(f"ang_{int(np.degrees(float(cam_angle))):+03d}")
        except Exception:
            tag_parts.append("ang_unknown")

    if not tag_parts:
        tag_parts.append(f"camera_{getattr(viewpoint_cam, 'timestamp', 0):.3f}")

    return "_".join(tag_parts)


def _ensure_dir(model_path: str, subdir: str) -> str:
    path = os.path.join(model_path, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _save_dsa_visualizations(
    train_cameras, vis_config, render_func, gaussians, recon_args, pipe, model_path, iteration, wandb_run
) -> None:
    sel = _select_cameras_for_dsa(train_cameras, vis_config)
    print(f"  -> Saving DSA for {len(sel)}/{len(train_cameras)} views...")

    norm_percentile = vis_config.get("save_dsa_norm_percentile")
    debug_stats = bool(vis_config.get("save_dsa_debug_stats", False))
    invert_synth = bool(vis_config.get("save_dsa_invert", False))
    if norm_percentile is not None:
        print(f"  -> DSA normalization percentile: {float(norm_percentile):.2f}")
    if debug_stats:
        print("  -> DSA debug stats: enabled")
    if invert_synth:
        print("  -> DSA synthesized image inversion: enabled")

    for viewpoint_cam in sel:
        try:
            render_pkg = render_func(viewpoint_cam, gaussians, recon_args, pipe)
            gt = viewpoint_cam.original_image
            if torch.is_tensor(gt):
                gt = gt.to(render_pkg["render"].device)
            save_synthesized_dsa_image(
                render_pkg["render"],
                viewpoint_cam,
                model_path,
                iteration,
                gt,
                wandb_run,
                norm_percentile=norm_percentile,
                debug_stats=debug_stats,
                invert_synth=invert_synth,
            )
        except Exception as e:
            print(
                f"[RenderError] Skipping DSA visualization: iter={iteration} "
                f"uid={getattr(viewpoint_cam, 'uid', None)} "
                f"err={type(e).__name__}: {e}",
                flush=True,
            )

    print(f"     Saved to: {model_path}/dsa/")


def save_synthesized_dsa_image(
    image,
    viewpoint_cam,
    model_path,
    iteration,
    gt_image=None,
    wandb_run=None,
    norm_percentile=None,
    debug_stats: bool = False,
    invert_synth: bool = False,
) -> str:
    synthed_dsa_dir = _ensure_dir(model_path, "dsa")
    image_np_raw, synth_scale, synth_stats = _normalize_image(image, percentile=norm_percentile)
    image_np = image_np_raw.copy()
    if invert_synth:
        image_np = 1.0 - image_np
    view_tag = _build_camera_tag(viewpoint_cam)
    filename = f"synthed_dsa_iter_{iteration:06d}_{view_tag}.png"
    filepath = os.path.join(synthed_dsa_dir, filename)

    gt_scale = None
    gt_stats = None
    if gt_image is not None:
        gt_image_np_raw, gt_scale, gt_stats = _normalize_image(gt_image, percentile=norm_percentile)
        gt_image_np = gt_image_np_raw.copy()
        if invert_synth:
            gt_image_np = 1.0 - gt_image_np
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.imshow(image_np, cmap="gray")
        ax1.set_title("Synthesized DSA")
        ax1.axis("off")
        ax2.imshow(gt_image_np, cmap="gray")
        ax2.set_title("Ground Truth DSA")
        ax2.axis("off")
        wandb_key = "synthesized_dsa_comparison"
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        ax1.imshow(image_np, cmap="gray")
        ax1.set_title(f"Synthesized DSA - Iteration {iteration}")
        ax1.axis("off")
        wandb_key = "synthesized_dsa"

    if debug_stats:
        debug_msg = (
            f"[DSA_DEBUG] iter={iteration} "
            f"uid={getattr(viewpoint_cam, 'uid', None)} "
            f"dicom_frame={getattr(viewpoint_cam, 'dicom_frame', None)} "
            f"synth_scale={synth_scale:.6g} "
            f"synth[min={synth_stats['min']:.6g}, max={synth_stats['max']:.6g}, "
            f"mean={synth_stats['mean']:.6g}, p95={synth_stats['p95']:.6g}, p99={synth_stats['p99']:.6g}]"
        )
        if gt_stats is not None and gt_scale is not None:
            debug_msg += (
                f" gt_scale={gt_scale:.6g} "
                f"gt[min={gt_stats['min']:.6g}, max={gt_stats['max']:.6g}, "
                f"mean={gt_stats['mean']:.6g}, p95={gt_stats['p95']:.6g}, p99={gt_stats['p99']:.6g}]"
            )
        if norm_percentile is not None:
            debug_msg += f" norm_percentile={float(norm_percentile):.2f}"
        else:
            debug_msg += " norm_percentile=max"
        if invert_synth:
            debug_msg += " invert_synth=True"
        print(debug_msg, flush=True)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Also save standalone prediction image (no GT, no comparison)
    pred_only_dir = _ensure_dir(model_path, "dsa_pred")
    pred_filename = f"pred_iter_{iteration:06d}_{view_tag}.png"
    pred_filepath = os.path.join(pred_only_dir, pred_filename)
    fig_pred, ax_pred = plt.subplots(1, 1, figsize=(8, 8))
    ax_pred.imshow(image_np, cmap="gray")
    ax_pred.axis("off")
    plt.tight_layout()
    plt.savefig(pred_filepath, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig_pred)

    if wandb_run is not None:
        img = _wandb_image(filepath, caption=f"Iteration {iteration}")
        if img is not None:
            _wandb_log(wandb_run, {wandb_key: img}, step=iteration)

    return filepath


