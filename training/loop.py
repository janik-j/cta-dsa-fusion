"""
Training loop orchestration for this framework.

Training-time reporting/visualization lives in `training/reporting.py`.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

if TYPE_CHECKING:
    from scene import GaussianModel, Scene
from gaussian_renderer_voxelizer import render as _render
from scene.io import read_ply_vertices
from training.losses import LossManager
from training.reporting import log_point_cloud_init, print_vis_config, resolve_vis_config, training_report
from utils.gaussian_utils import get_scale_bound
from utils.system_utils import apply_training_state, load_training_state

# ============================================================================
# Training Loop
# ============================================================================


@dataclass
class TrainingContext:
    dataset: Any
    opt: Any
    pipe: Any
    wandb_run: Any
    gaussians: GaussianModel
    scene: Scene
    loss_manager: LossManager
    prior_xyz_torch: torch.Tensor | None
    train_cams: list
    viewpoint_stack: list
    testing_iterations: np.ndarray
    saving_iterations: np.ndarray
    vis_config: dict
    progress_bar: Any
    first_iter: int
    debug_from: int
    fn_loss_enabled: bool
    fn_loss_start_iter: int
    fn_loss_gt_min_norm: float
    fn_loss_margin: float
    fn_loss_power: float
    fn_loss_norm_percentile: float
    fn_loss_weight: float
    fn_loss_length_balance: bool
    fn_loss_length_balance_kernel: int
    fn_loss_length_balance_power: float
    fn_loss_length_balance_max: float


def _select_training_cameras(ctx: TrainingContext, iteration: int) -> list:
    """Select training camera(s) per iteration."""
    camera_sampling = str(getattr(ctx.opt, "camera_sampling", "random") or "random").lower().strip()

    if camera_sampling == "all":
        return list(ctx.train_cams)

    if camera_sampling in {"random", "stack", "random_stack"}:
        if not ctx.viewpoint_stack:
            ctx.viewpoint_stack = ctx.train_cams.copy()
        return [ctx.viewpoint_stack.pop(random.randint(0, len(ctx.viewpoint_stack) - 1))]
    return [ctx.train_cams[int((iteration - 1) % len(ctx.train_cams))]]


def _compute_fn_loss(
    gt_image: torch.Tensor,
    pred_image: torch.Tensor,
    *,
    gt_min_norm: float,
    margin: float,
    power: float,
    norm_percentile: float,
    length_balance: bool,
    length_balance_kernel: int,
    length_balance_power: float,
    length_balance_max: float,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """
    Compute an explicit false-negative loss, normalized over vessel support.

    This pushes missing vessel response directly and reduces vessel-mass bias by
    averaging over (optionally length-balanced) vessel support instead of all pixels.
    """
    gt = gt_image.detach()
    pred = pred_image

    p = float(norm_percentile)
    if 0.0 < p < 100.0:
        vals = torch.cat([gt.reshape(-1), pred.detach().reshape(-1)])
        vals = vals[torch.isfinite(vals)]
        vals = vals[vals > 0]
        if vals.numel() > 0:
            try:
                scale_t = torch.quantile(vals, q=p / 100.0)
            except Exception:
                scale_t = vals.max()
            scale = float(max(float(scale_t.item()), 0.0))
        else:
            scale = 0.0
    else:
        scale = float(max(float(gt.max().item()), float(pred.detach().max().item()), 0.0))

    if scale <= 0.0:
        return None, {"scale": scale, "fn_loss_raw": 0.0, "fn_loss_denom": 0.0}

    gt_norm = (gt / (scale + 1e-6)).clamp(0.0, 1.0)
    pred_norm = (pred / (scale + 1e-6)).clamp(0.0, 1.0)

    vessel = (gt_norm >= float(gt_min_norm)).to(dtype=pred_norm.dtype)
    missing = torch.relu(gt_norm - pred_norm - float(margin))
    if float(power) != 1.0:
        missing = missing.pow(float(max(0.1, power)))

    support = vessel
    if bool(length_balance):
        k = int(max(3, int(length_balance_kernel)))
        if k % 2 == 0:
            k += 1
        density = F.avg_pool2d(vessel, kernel_size=k, stride=1, padding=k // 2)
        lb = torch.rsqrt(density.clamp_min(1e-4)).pow(float(max(0.1, length_balance_power)))
        lb = torch.clamp(lb, max=float(max(1.0, length_balance_max)))
        support = support * lb

    denom = support.sum().clamp_min(1e-6)
    fn_loss_raw = (support * missing).sum() / denom
    return fn_loss_raw, {
        "scale": scale,
        "fn_loss_raw": float(fn_loss_raw.detach().item()),
        "fn_loss_denom": float(denom.detach().item()),
    }


def training_loop(
    dataset,
    field_conf,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    debug_from,
    wandb_run=None,
    vis_config=None,
    load_iteration=None,
) -> float:
    """Main training loop. Returns training time in seconds."""
    start_time = time.time()

    ctx = _setup_training(
        dataset=dataset,
        field_conf=field_conf,
        opt=opt,
        pipe=pipe,
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        debug_from=debug_from,
        wandb_run=wandb_run,
        vis_config=vis_config,
        load_iteration=load_iteration,
    )

    for iteration in range(ctx.first_iter + 1, int(opt.iterations) + 1):
        render_pkgs, loss_dict, total_loss = _train_step(ctx, iteration)
        _post_step(ctx, iteration, render_pkgs, loss_dict, total_loss)

    return _finalize_training(ctx, start_time)


def _setup_training(
    *,
    dataset,
    field_conf,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    debug_from,
    wandb_run,
    vis_config,
    load_iteration,
) -> TrainingContext:
    from scene import GaussianModel, Scene

    vis_config = resolve_vis_config(vis_config, opt.iterations, testing_iterations)
    print_vis_config(vis_config, dataset.model_path)

    scale_bound = get_scale_bound(dataset)
    gaussians = GaussianModel(field_conf)
    scene = Scene(dataset, gaussians, scale_bound, load_iteration=load_iteration)

    log_point_cloud_init(gaussians, dataset)
    gaussians.spatial_lr_scale = scene.cameras_extent

    first_iter = int(scene.loaded_iter) if scene.loaded_iter is not None else 0
    testing_iterations = np.array(testing_iterations)
    saving_iterations = np.array(saving_iterations)

    gaussians.training_setup(opt, 0)
    gaussians.densify_setup(opt, 0)

    fn_loss_enabled = bool(getattr(opt, "fn_loss_enabled", False))
    fn_loss_start_iter = max(0, int(getattr(opt, "fn_loss_start_iter", 0) or 0))
    fn_loss_gt_min_norm = float(getattr(opt, "fn_loss_gt_min_norm", 0.0) or 0.0)
    fn_loss_gt_min_norm = float(max(0.0, min(1.0, fn_loss_gt_min_norm)))
    fn_loss_margin = float(getattr(opt, "fn_loss_margin", 0.0) or 0.0)
    fn_loss_margin = float(max(0.0, min(1.0, fn_loss_margin)))
    fn_loss_power = float(getattr(opt, "fn_loss_power", 1.0) or 1.0)
    fn_loss_power = float(max(0.1, fn_loss_power))
    fn_loss_norm_percentile = float(getattr(opt, "fn_loss_norm_percentile", -1.0) or -1.0)
    fn_loss_weight = max(
        0.0, float(getattr(opt, "fn_loss_weight", 0.0) or 0.0)
    )
    fn_loss_length_balance = bool(getattr(opt, "fn_loss_length_balance", False))
    fn_loss_length_balance_kernel = int(
        max(3, int(getattr(opt, "fn_loss_length_balance_kernel", 15) or 15))
    )
    if fn_loss_length_balance_kernel % 2 == 0:
        fn_loss_length_balance_kernel += 1
    fn_loss_length_balance_power = float(
        getattr(opt, "fn_loss_length_balance_power", 0.5) or 0.5
    )
    fn_loss_length_balance_power = float(max(0.1, fn_loss_length_balance_power))
    fn_loss_length_balance_max = float(
        getattr(opt, "fn_loss_length_balance_max", 3.0) or 3.0
    )
    fn_loss_length_balance_max = float(max(1.0, fn_loss_length_balance_max))
    if fn_loss_enabled and fn_loss_weight <= 0.0:
        fn_loss_enabled = False
    if fn_loss_enabled:
        print(
            "[FN] False-negative loss enabled: "
            f"start_iter={fn_loss_start_iter}, "
            f"fn_loss_weight={fn_loss_weight:.3f}, "
            f"gt_min_norm={fn_loss_gt_min_norm:.3f}, "
            f"margin={fn_loss_margin:.3f}, "
            f"power={fn_loss_power:.3f}, "
            f"norm_pct={fn_loss_norm_percentile:.2f}, "
            f"length_balance={int(fn_loss_length_balance)}, "
            f"lb_kernel={fn_loss_length_balance_kernel}, "
            f"lb_power={fn_loss_length_balance_power:.3f}, "
            f"lb_max={fn_loss_length_balance_max:.3f}",
            flush=True,
        )

    if scene.loaded_iter is not None:
        state = load_training_state(model_path=scene.model_path, iteration=int(scene.loaded_iter))
        if int(state.get("iteration", -1)) != int(scene.loaded_iter):
            raise ValueError(f"Training state iteration mismatch: expected {scene.loaded_iter}, got {state.get('iteration')}")
        apply_training_state(gaussians=gaussians, state=state)

    if 0 in set(vis_config.get("vis_iterations", [])):
        training_report(
            wandb_run,
            0,
            {},
            None,
            testing_iterations,
            scene,
            gaussians,
            _render,
            pipe,
            vis_config,
        )

    # Support saving at iteration 0 (before any training)
    if 0 in set(saving_iterations):
        print("\n[ITER 0] Saving Gaussians (pre-training)")
        scene.save(0)

    loss_manager = LossManager.from_opt_params(opt, dataset)
    print(f"\n{loss_manager}\n")

    prior_xyz_torch = None
    if float(getattr(opt, "lambda_prior_anchoring", 0.0)) > 0.0:
        prior_path_raw = getattr(opt, "prior_path", "") or getattr(dataset, "ply_path", "")
        if not prior_path_raw:
            raise ValueError("lambda_prior_anchoring > 0 requires --prior_path or --ply_path")

        prior_path = Path(str(prior_path_raw)).expanduser().resolve()
        if not prior_path.exists():
            raise FileNotFoundError(f"Prior PLY not found: {prior_path} (cwd={Path.cwd()})")

        pts, _opacities, _normals = read_ply_vertices(prior_path)
        print(f"[Prior] Loaded {pts.shape[0]} points from {prior_path}")
        prior_xyz_torch = torch.from_numpy(pts).to(gaussians.get_xyz.device)

    train_cams = scene.getTrainCameras().copy()
    if not train_cams:
        raise ValueError("No training cameras loaded.")

    progress_bar = tqdm(
        range(first_iter, opt.iterations),
        desc="Training progress",
        initial=first_iter,
        total=opt.iterations,
    )

    return TrainingContext(
        dataset=dataset,
        opt=opt,
        pipe=pipe,
        wandb_run=wandb_run,
        gaussians=gaussians,
        scene=scene,
        loss_manager=loss_manager,
        prior_xyz_torch=prior_xyz_torch,
        train_cams=train_cams,
        viewpoint_stack=[],
        testing_iterations=testing_iterations,
        saving_iterations=saving_iterations,
        vis_config=vis_config,
        progress_bar=progress_bar,
        first_iter=first_iter,
        debug_from=int(debug_from),
        fn_loss_enabled=fn_loss_enabled,
        fn_loss_start_iter=fn_loss_start_iter,
        fn_loss_gt_min_norm=fn_loss_gt_min_norm,
        fn_loss_margin=fn_loss_margin,
        fn_loss_power=fn_loss_power,
        fn_loss_norm_percentile=fn_loss_norm_percentile,
        fn_loss_weight=fn_loss_weight,
        fn_loss_length_balance=fn_loss_length_balance,
        fn_loss_length_balance_kernel=fn_loss_length_balance_kernel,
        fn_loss_length_balance_power=fn_loss_length_balance_power,
        fn_loss_length_balance_max=fn_loss_length_balance_max,
    )


def _train_step(ctx: TrainingContext, iteration: int) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor]:
    gaussians = ctx.gaussians
    gaussians.update_learning_rate(iteration)

    viewpoint_cams = _select_training_cameras(ctx, iteration)
    if not viewpoint_cams:
        raise RuntimeError("No training cameras selected for current iteration.")

    if (iteration - 1) == int(ctx.debug_from):
        ctx.pipe.debug = True

    device = gaussians.get_xyz.device
    render_pkgs: list[dict[str, Any]] = []
    images: list[torch.Tensor] = []
    gt_images: list[torch.Tensor] = []
    fn_raw_values: list[torch.Tensor] = []
    fn_weighted_values: list[torch.Tensor] = []

    for viewpoint_cam in viewpoint_cams:
        try:
            render_pkg = _render(viewpoint_cam, gaussians, ctx.scene.recon_args, ctx.pipe)
        except Exception as e:
            print(
                "[RenderError] "
                f"iter={iteration} uid={getattr(viewpoint_cam, 'uid', None)} "
                f"err={type(e).__name__}: {e}",
                flush=True,
            )
            raise

        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.to(device)

        if ctx.fn_loss_enabled and int(iteration) >= int(ctx.fn_loss_start_iter):
            fn_loss_raw, fn_stats = _compute_fn_loss(
                gt_image,
                image,
                gt_min_norm=ctx.fn_loss_gt_min_norm,
                margin=ctx.fn_loss_margin,
                power=ctx.fn_loss_power,
                norm_percentile=ctx.fn_loss_norm_percentile,
                length_balance=ctx.fn_loss_length_balance,
                length_balance_kernel=ctx.fn_loss_length_balance_kernel,
                length_balance_power=ctx.fn_loss_length_balance_power,
                length_balance_max=ctx.fn_loss_length_balance_max,
            )
            if fn_loss_raw is not None:
                fn_raw_values.append(fn_loss_raw)
                fn_weighted_values.append(fn_loss_raw * float(ctx.fn_loss_weight))

            if iteration == 1 or iteration % 1000 == 0:
                fn_w = 0.0
                if fn_weighted_values:
                    fn_w = float(fn_weighted_values[-1].detach().item())
                print(
                    "[FN][DEBUG] "
                    f"iter={iteration} uid={getattr(viewpoint_cam, 'uid', None)} "
                    f"fn_loss_raw={fn_stats.get('fn_loss_raw', 0.0):.6f} "
                    f"fn_loss_w={fn_w:.6f}",
                    flush=True,
                )

        render_pkgs.append(render_pkg)
        images.append(image)
        gt_images.append(gt_image)

    if len(images) == 1:
        image_input = images[0]
        gt_input = gt_images[0]
    else:
        image_input = torch.stack(images, dim=0)
        gt_input = torch.stack(gt_images, dim=0)

    total_loss, loss_dict = ctx.loss_manager.compute_total_loss(
        image_input,
        gt_input,
        gaussians,
        ctx.prior_xyz_torch,
        iteration,
        recon_args=ctx.scene.recon_args,
        pipe=ctx.pipe,
    )

    if fn_weighted_values:
        fn_raw_mean = torch.stack(fn_raw_values, dim=0).mean()
        fn_weighted_mean = torch.stack(fn_weighted_values, dim=0).mean()
        total_loss = total_loss + fn_weighted_mean
        loss_dict["Lfn_raw"] = fn_raw_mean.detach()
        loss_dict["Lfn"] = fn_weighted_mean.detach()
        loss_dict["Ltotal"] = total_loss.detach()

    total_loss.backward()

    return render_pkgs, loss_dict, total_loss


def _post_step(
    ctx: TrainingContext,
    iteration: int,
    render_pkgs: list[dict[str, Any]],
    loss_dict: dict[str, Any],
    total_loss: torch.Tensor,
) -> None:
    with torch.no_grad():
        num_points = int(ctx.gaussians._xyz.shape[0])
        if iteration % 10 == 0:
            info_dict = {"Loss": f"{total_loss.item():.{5}f}", "point": f"{num_points}"}
            ctx.progress_bar.set_postfix(info_dict)
            ctx.progress_bar.update(10)
        if iteration == int(ctx.opt.iterations):
            ctx.progress_bar.close()

        prune_num_record = None
        if iteration < int(ctx.opt.densify_until_iter):
            for rp in render_pkgs:
                ctx.gaussians.max_radii2D[rp["visibility_filter"]] = torch.max(
                    ctx.gaussians.max_radii2D[rp["visibility_filter"]],
                    rp["radii"][rp["visibility_filter"]],
                )
                ctx.gaussians.add_densification_stats(
                    rp["viewspace_points"],
                    rp["dummy_opacity"],
                    rp["visibility_filter"],
                )

            if iteration > int(ctx.opt.densify_from_iter) and (iteration % int(ctx.opt.densification_interval) == 0):
                d_clone, d_split, split_prune_mask = ctx.gaussians.densify(ctx.scene.scene_spacing, iteration)
                # For multi-view steps, pruning on a single view can bias opacity
                # pruning toward that camera. Aggregate per-point
                # dummy opacity across all rendered views for a stable prune criterion.
                if len(render_pkgs) == 1:
                    prune_dummy_opacity = render_pkgs[0]["dummy_opacity"]
                else:
                    prune_dummy_opacity = torch.stack(
                        [rp["dummy_opacity"] for rp in render_pkgs], dim=0
                    ).mean(dim=0)
                prune_num_record = ctx.gaussians.prune(
                    split_prune_mask,
                    ctx.opt.random_prune,
                    ctx.opt.avgopacity_prune,
                    ctx.opt.opacity_prune,
                    prune_dummy_opacity,
                    ctx.opt.max_screen_size,
                    ctx.scene.recon_args,
                    iteration,
                )
                ctx.scene.gaussians.densification_postfix(d_clone)
                ctx.scene.gaussians.densification_postfix(d_split)

        training_report(
            ctx.wandb_run,
            iteration,
            loss_dict,
            prune_num_record,
            ctx.testing_iterations,
            ctx.scene,
            ctx.gaussians,
            _render,
            ctx.pipe,
            ctx.vis_config,
        )

        if iteration < int(ctx.opt.iterations):
            # Zero geometry gradients for fixed (M1) points before optimizer step.
            fm = getattr(ctx.gaussians, "fixed_mask", None)
            if fm is not None and torch.is_tensor(fm) and fm.numel() == ctx.gaussians.get_xyz.shape[0] and fm.any():
                for attr in ("_xyz", "_scaling", "_rotation"):
                    param = getattr(ctx.gaussians, attr, None)
                    if param is not None and param.grad is not None:
                        param.grad[fm] = 0

            ctx.gaussians.optimizer.step()
            ctx.gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in ctx.saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            ctx.scene.save(iteration)


def _finalize_training(ctx: TrainingContext, start_time: float) -> float:
    elapsed_time = time.time() - start_time
    total_seconds = max(0, int(round(elapsed_time)))
    minutes, seconds = divmod(total_seconds, 60)

    eval_results: dict[str, Any] = {}
    with torch.no_grad():
        try:
            from evaluation.pipeline import render_set, voxel_query_reconstruction

            eval_results["vqr"] = voxel_query_reconstruction(ctx.scene, ctx.gaussians, ctx.pipe, ctx.dataset, "train")
            eval_results["render"] = render_set(ctx.scene, ctx.gaussians, ctx.pipe, ctx.dataset, "train")
        except Exception as e:
            print(f"[WARN] Post-training eval skipped: {type(e).__name__}: {e}", flush=True)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Time: {minutes}m {seconds}s")
    print(f"Output: {ctx.scene.model_path}")
    print(f"Gaussians: {ctx.gaussians._xyz.shape[0]}")

    active = ctx.loss_manager.get_active_losses(ctx.loss_manager.total_iterations)
    all_active = active.get("rendering", []) + active.get("regularization", [])
    if all_active:
        print(f"Losses: {', '.join(all_active)}")

    if eval_results:
        if "render" in eval_results:
            r = eval_results["render"]
            print("\n2D Metrics:")
            print(f"  Train PSNR: {r.get('train_psnr', '-')} dB")
            print(f"  Eval  PSNR: {r.get('eval_psnr', '-')} dB")

        if "vqr" in eval_results:
            v = eval_results["vqr"]

            global_m = (v.get("metrics") or {}).get("global") if isinstance(v, dict) else {}
            roi_m = (v.get("metrics") or {}).get("roi") if isinstance(v, dict) else {}
            if not isinstance(global_m, dict):
                global_m = {}
            if not isinstance(roi_m, dict):
                roi_m = {}

            def _fmt_num(val: Any, decimals: int = 4) -> str:
                if val is None:
                    return "-"
                try:
                    f = float(val)
                    if math.isnan(f):
                        return "-"
                    return f"{f:.{decimals}f}"
                except Exception:
                    return str(val)

            print("\n3D Metrics:")
            print(f"  CD:   {_fmt_num(global_m.get('cd_mm', v.get('cd_mm')))} mm")
            print(f"  HD95: {_fmt_num(global_m.get('hd95_mm', v.get('hd95_mm')))} mm")
            # Always report ROI splits (may be '-' if meshes are unavailable).
            print(f"  CD (missing):   {_fmt_num(roi_m.get('missing_cd_mm', v.get('cd_missing_mm')))} mm")
            print(f"  HD95 (missing): {_fmt_num(roi_m.get('missing_hd95_mm', v.get('hd95_missing_mm')))} mm")
            print(f"  CD (prior):     {_fmt_num(roi_m.get('prior_cd_mm', v.get('cd_prior_mm')))} mm")
            print(f"  HD95 (prior):   {_fmt_num(roi_m.get('prior_hd95_mm', v.get('hd95_prior_mm')))} mm")

    print(f"{'=' * 60}\n")
    return elapsed_time
