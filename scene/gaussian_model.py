"""
3D Gaussian Model for X-ray Imaging.

This module implements the core GaussianModel class which represents a scene
as a collection of 3D Gaussians. Each Gaussian has:
- Position (xyz): 3D location in world coordinates
- Scale: 3D anisotropic scale factors
- Rotation: Quaternion rotation (w, x, y, z)
- Opacity: Base opacity value (processed through neural field)

The model supports:
- Prior-guided initialization (M1 anatomical + M2 residual)
- Adaptive densification and pruning during training
- Checkpoint save/restore for training resumption
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn

from scene.field import Field, detect_field_type_from_state_dict
from utils.gaussian_utils import (
    build_rotation,
    build_scaling_rotation,
    exp_decay,
    inverse_sigmoid,
    inverse_sigmoid_clamp,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p


def _dist2_fallback_ckdtree(points: torch.Tensor) -> torch.Tensor:
    """Fallback for `simple_knn._C.distCUDA2` (squared 2-NN distance).

    Returns per-point squared distance to the nearest *other* point (k=2, skip self).
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "simple_knn is not available and scipy is missing, so we cannot compute initial point scales. "
            "Install the repo environment with `uv sync`."
        ) from e

    pts = points.detach().to(dtype=torch.float32, device="cpu").numpy()
    n = int(pts.shape[0])
    if n <= 1:
        out = np.full((n,), 1e-7, dtype=np.float32)
        return torch.from_numpy(out).to(device=points.device)

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2, workers=-1)
    # d[:, 0] is self (0). d[:, 1] is nearest neighbor.
    d2 = np.square(d[:, 1]).astype(np.float32)
    return torch.from_numpy(d2).to(device=points.device)


try:
    from simple_knn._C import distCUDA2  # type: ignore
except Exception:  # pragma: no cover
    distCUDA2 = _dist2_fallback_ckdtree


class GaussianModel:
    """
    3D Gaussian representation for differentiable X-ray rendering.

    A scene is represented as a collection of anisotropic 3D Gaussians,
    each parameterized by position, scale, rotation, and opacity. The
    opacity is modulated by a neural field that encodes spatial variations.

    Attributes:
        _xyz: Gaussian positions (N, 3)
        _scaling: Log-space scale factors (N, 3)
        _rotation: Quaternions (N, 4) in (w, x, y, z) order
        _opacity: Base opacity values (N, 1)
        _field: Neural field for opacity modulation
        is_m1: Boolean mask indicating M1 (prior) points vs M2 (residual)

    Example:
        >>> gaussians = GaussianModel(field_conf)
        >>> gaussians.create_from_pcd(point_cloud, spatial_lr_scale=1.0)
        >>> gaussians.training_setup(opt_params)
    """

    def setup_functions(self, scale_bound: tuple[float, float] | None = None) -> None:
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        if scale_bound is not None:
            scale_min_bound, scale_max_bound = scale_bound
            assert scale_min_bound < scale_max_bound, "scale_min_bound should be less than scale_max_bound"
            self.scaling_activation = lambda x: torch.sigmoid(x) * (scale_max_bound - scale_min_bound) + scale_min_bound
            self.scaling_inverse_activation = lambda x: inverse_sigmoid_clamp(
                (x - scale_min_bound) / (scale_max_bound - scale_min_bound)
            )
        else:
            self.scaling_activation = torch.exp  # 保证 scale 是正数
            self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = F.relu
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, field_conf):
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._field_conf = dict(field_conf)  # Store for potential field recreation
        self._field = Field(field_conf)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.avgopacity_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        # M1 mask: True if point belongs to the trusted prior (M1).
        self.is_m1 = None
        self.fixed_mask = None  # bool tensor: True = frozen geometry (set by use_ply_mask)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def _filter_invalid_points(self, tensors_dict, context_name):
        """Filter out NaN/Inf points from a densification dict and return cleaned version."""
        xyz, opacity = tensors_dict["xyz"], tensors_dict["opacity"]
        scaling, rotation = tensors_dict["scaling"], tensors_dict["rotation"]

        valid = (
            torch.isfinite(xyz).all(dim=1)
            & torch.isfinite(opacity).all(dim=1)
            & torch.isfinite(scaling).all(dim=1)
            & torch.isfinite(rotation).all(dim=1)
        )
        if valid.all():
            return tensors_dict

        n_bad = int((~valid).sum().item())
        print(f"WARNING: Dropping {n_bad} invalid {context_name} points (NaN/Inf).", flush=True)
        return {
            "xyz": xyz[valid],
            "opacity": opacity[valid],
            "scaling": scaling[valid],
            "rotation": rotation[valid],
        }

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
    ) -> None:
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)

        # Default initialization (isotropic)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # NOTE: We intentionally do not use PLY normals for initialization.
        # For strict comparability with the upstream implementation, all points
        # use isotropic scale initialization from nearest-neighbor distances and an
        # identity rotation.

        opacities = torch.tensor(np.asarray(pcd.opacities)).float().cuda()

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # Store M1 Mask
        if hasattr(pcd, "is_m1") and pcd.is_m1 is not None:
            self.is_m1 = torch.tensor(pcd.is_m1).bool().cuda()
            print(f"Loaded M1 Mask: {self.is_m1.sum()} M1 points, {(~self.is_m1).sum()} M2 points.")
        else:
            self.is_m1 = None

        self._field = self._field.to("cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _reset_densification_stats(self):
        """Reset gradient accumulators and statistics for densification."""
        n_points = self.get_xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((n_points, 1), device="cuda")
        self.avgopacity_accum = torch.zeros((n_points, 1), device="cuda")
        self.denom = torch.zeros((n_points, 1), device="cuda")

    def training_setup(self, training_args, start_iter=0):
        self._reset_densification_stats()
        param_groups = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._opacity], "lr": training_args.opacity_lr_init, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr_init, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr_init, "name": "rotation"},
            {
                "params": list(self._field.parameters()),
                "lr": training_args.field_lr_init,
                "name": "field",
                "weight_decay": training_args.field_decay,
            },
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

        end_iter = training_args.iterations + start_iter
        scale = self.spatial_lr_scale
        self.xyz_scheduler_args = exp_decay(
            training_args.position_lr_init * scale, training_args.position_lr_final * scale, start_iter, end_iter
        )
        self.opacity_scheduler_args = exp_decay(
            training_args.opacity_lr_init, training_args.opacity_lr_final, start_iter, end_iter
        )
        self.scaling_scheduler_args = exp_decay(
            training_args.scaling_lr_init, training_args.scaling_lr_final, start_iter, end_iter
        )
        self.rotation_scheduler_args = exp_decay(
            training_args.rotation_lr_init, training_args.rotation_lr_final, start_iter, end_iter
        )
        self.field_scheduler_args = exp_decay(
            training_args.field_lr_init, training_args.field_lr_final, start_iter, end_iter
        )

    def densify_setup(self, densify_args, start_iter=0):
        from_iter = densify_args.densify_from_iter + start_iter
        until_iter = densify_args.densify_until_iter + start_iter
        self.percent_dense_args = exp_decay(
            densify_args.percent_dense_init, densify_args.percent_dense_final, from_iter, until_iter
        )
        self.densify_grad_threshold_args = exp_decay(
            densify_args.densify_grad_threshold_init, densify_args.densify_grad_threshold_final, from_iter, until_iter
        )
        self.percent_random_prune_args = exp_decay(
            densify_args.percent_random_prune_init, densify_args.percent_random_prune_final, from_iter, until_iter
        )
        self.min_avgopacity_args = exp_decay(
            densify_args.min_avgopacity_init, densify_args.min_avgopacity_final, from_iter, until_iter
        )
        self.min_opacity_args = exp_decay(
            densify_args.min_opacity_init, densify_args.min_opacity_final, from_iter, until_iter
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step."""
        schedulers = {
            "xyz": self.xyz_scheduler_args,
            "opacity": self.opacity_scheduler_args,
            "scaling": self.scaling_scheduler_args,
            "rotation": self.rotation_scheduler_args,
            "field": self.field_scheduler_args,
        }
        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in schedulers:
                param_group["lr"] = schedulers[name](iteration)

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "opacity"]
        attrs += [f"scale_{i}" for i in range(self._scaling.shape[1])]
        attrs += [f"rot_{i}" for i in range(self._rotation.shape[1])]
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        attrs_list = [xyz, opacities, scale, rotation]
        attr_names = self.construct_list_of_attributes()

        if self.is_m1 is not None:
            attrs_list.append(self.is_m1.detach().cpu().numpy()[:, None].astype(np.float32))
            attr_names.append("is_m1")

        dtype_full = [(n, "f4") for n in attr_names]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, np.concatenate(attrs_list, axis=1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def save_field(self, path):
        torch.save(self._field.state_dict(), path)

    def _load_ply_attributes(self, plydata, prefix, n_points):
        """Load PLY attributes with a given prefix (e.g., 'scale_' or 'rot_')."""
        names = [p.name for p in plydata.elements[0].properties if p.name.startswith(prefix)]
        names = sorted(names, key=lambda x: int(x.split("_")[-1]))
        data = np.zeros((n_points, len(names)))
        for idx, attr_name in enumerate(names):
            data[:, idx] = np.asarray(plydata.elements[0][attr_name])
        return data

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        scales = self._load_ply_attributes(plydata, "scale_", xyz.shape[0])
        rots = self._load_ply_attributes(plydata, "rot_", xyz.shape[0])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        prop_names = [p.name for p in plydata.elements[0].properties]
        if "is_m1" in prop_names:
            is_m1 = np.asarray(plydata.elements[0]["is_m1"])
            self.is_m1 = torch.tensor(is_m1, dtype=torch.float, device="cuda") > 0.5

    def load_field(self, path):
        print(f"loading field from exists {path}")
        state_dict = torch.load(path, map_location="cuda")

        # Detect field type from saved checkpoint
        saved_type = detect_field_type_from_state_dict(state_dict)
        current_type = self._field.select_field

        if saved_type != current_type:
            print(f"Field type mismatch: checkpoint={saved_type}, config={current_type}")
            print(f"Recreating field with type={saved_type} to match checkpoint")
            # Recreate field with the type from the checkpoint
            new_conf = dict(self._field_conf)
            new_conf["select_field"] = saved_type
            self._field = Field(new_conf)

        self._field.load_state_dict(state_dict)
        self._field = self._field.to("cuda")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                old_param = group["params"][0]
                stored_state = self.optimizer.state.get(old_param, None)

                # Create new optimizer state
                new_state = {"exp_avg": torch.zeros_like(tensor), "exp_avg_sq": torch.zeros_like(tensor)}

                # Copy step count if exists (for Adam)
                if stored_state is not None and "step" in stored_state:
                    new_state["step"] = stored_state["step"]

                # Remove old state if exists
                if old_param in self.optimizer.state:
                    del self.optimizer.state[old_param]

                # Update parameter and state
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = new_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "field":
                continue
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]

            group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _reset_to_empty(self):
        """Reset all tensors to empty when all points are pruned."""
        self._xyz = torch.empty((0, 3), device="cuda", requires_grad=True)
        self._opacity = torch.empty((0, 1), device="cuda", requires_grad=True)
        self._scaling = torch.empty((0, 3), device="cuda", requires_grad=True)
        self._rotation = torch.empty((0, 4), device="cuda", requires_grad=True)
        self.xyz_gradient_accum = torch.empty((0, 1), device="cuda")
        self.avgopacity_accum = torch.empty((0, 1), device="cuda")
        self.denom = torch.empty((0, 1), device="cuda")
        self.max_radii2D = torch.empty((0), device="cuda")

    def prune_points(self, mask):
        # Never prune fixed (M1) points.
        if self.fixed_mask is not None and self.fixed_mask.numel() == mask.numel():
            mask = mask.clone()
            mask[self.fixed_mask] = False

        valid_points_mask = ~mask
        if valid_points_mask.sum() == 0:
            self._reset_to_empty()
            return

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.avgopacity_accum = self.avgopacity_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.is_m1 is not None:
            self.is_m1 = self.is_m1[valid_points_mask]
        if self.fixed_mask is not None:
            self.fixed_mask = self.fixed_mask[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "field":
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)

            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )

                # Zero out Adam state for fixed points to prevent numerical drift.
                if self.fixed_mask is not None:
                    fm = self.fixed_mask[: len(stored_state["exp_avg"])]
                    if fm.numel() > 0 and fm.numel() == stored_state["exp_avg"].size(0):
                        stored_state["exp_avg"][fm] = 0
                        stored_state["exp_avg_sq"][fm] = 0

                del self.optimizer.state[group["params"][0]]

            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
            )
            if stored_state is not None:
                self.optimizer.state[group["params"][0]] = stored_state
            optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, d):
        # NOTE: Densification bookkeeping is reset at every densification step,
        # even if no new points are added. Do NOT early-return before resetting.
        n_new = int(d["xyz"].shape[0])

        if n_new > 0:
            # Extend fixed_mask (new points are not fixed)
            if self.fixed_mask is not None:
                ext = torch.zeros((n_new,), device="cuda", dtype=torch.bool)
                self.fixed_mask = torch.cat([self.fixed_mask, ext], dim=0)

            # Extend is_m1 mask (use provided mask or default to False)
            if self.is_m1 is not None:
                new_is_m1 = d.get("is_m1", torch.zeros((n_new,), device="cuda", dtype=torch.bool))
                self.is_m1 = torch.cat([self.is_m1, new_is_m1], dim=0)

            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

        self._reset_densification_stats()
        # Match upstream: reset image-space radii stats after every densification step.
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def _finalize_densification_dict(self, d, context_name):
        """Filter invalid points and add is_m1 mask (new points are never M1)."""
        d = self._filter_invalid_points(d, context_name)
        if self.is_m1 is not None:
            d["is_m1"] = torch.zeros((d["xyz"].shape[0],), device="cuda", dtype=torch.bool)
        return d

    def densify_and_split(self, grads, grad_threshold, scene_extent):
        N = 2
        n_init_points = self.get_xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        grad_mask = padded_grad >= grad_threshold
        scale_mask = torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        selected_pts_mask = grad_mask & scale_mask

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        # Use normalized rotations to avoid invalid rotation matrices and NaNs
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        d = self._finalize_densification_dict(
            {
                "xyz": new_xyz,
                "opacity": new_opacity,
                "scaling": new_scaling,
                "rotation": new_rotation,
            },
            "split",
        )

        return d, selected_pts_mask

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Points with high gradient but small scale are cloned (not split)
        grad_mask = torch.norm(grads, dim=-1) >= grad_threshold
        scale_mask = torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        selected_pts_mask = grad_mask & scale_mask

        return self._finalize_densification_dict(
            {
                "xyz": self._xyz[selected_pts_mask],
                "opacity": self._opacity[selected_pts_mask],
                "scaling": self._scaling[selected_pts_mask],
                "rotation": self._rotation[selected_pts_mask],
            },
            "clone",
        )

    def densify(self, extent, iteration):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.percent_dense = self.percent_dense_args(iteration)
        max_grad = self.densify_grad_threshold_args(iteration)

        d_clone = self.densify_and_clone(grads, max_grad, extent)
        d_split, split_prune_mask = self.densify_and_split(grads, max_grad, extent)

        # Protect fixed (M1) points from split-pruning.
        if (
            split_prune_mask is not None
            and self.fixed_mask is not None
            and self.fixed_mask.numel() == split_prune_mask.numel()
        ):
            split_prune_mask = split_prune_mask.clone()
            split_prune_mask[self.fixed_mask] = False

        torch.cuda.empty_cache()

        return d_clone, d_split, split_prune_mask

    def prune(
        self,
        split_prune_mask,
        random_prune,
        avgopacity_prune,
        opacity_prune,
        dummy_opacity,
        max_screen_size,
        recon_args,
        iteration,
    ):
        prune_mask = self.exclude_outbbx(self.get_xyz, recon_args)
        if split_prune_mask is not None:
            prune_mask = prune_mask | split_prune_mask

        prune_num_record = {}

        def add_prune_condition(condition_mask, record_key):
            nonlocal prune_mask
            prune_num_record[record_key] = condition_mask.sum().item()
            prune_mask = prune_mask | condition_mask

        if random_prune:
            percent = self.percent_random_prune_args(iteration)
            add_prune_condition(self.get_random_prune_mask(percent), "random_prune_num")

        if avgopacity_prune:
            avgopacity = self.avgopacity_accum / self.denom
            avgopacity[avgopacity.isnan()] = 0.0
            threshold = self.min_avgopacity_args(iteration)
            add_prune_condition(avgopacity.squeeze() < threshold, "min_avgopacity_prune_num")

        if opacity_prune:
            threshold = self.min_opacity_args(iteration)
            add_prune_condition(dummy_opacity.squeeze() < threshold, "min_opacity_prune_num")

        if max_screen_size:
            add_prune_condition(self.max_radii2D > max_screen_size, "max_screen_prune_num")

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

        return prune_num_record

    def get_random_prune_mask(self, percent_random_prune):
        n_points = self.get_xyz.shape[0]
        n_prune = int(n_points * percent_random_prune)
        prune_mask = torch.zeros(n_points, dtype=torch.bool, device="cuda")
        prune_mask[torch.randperm(n_points, device="cuda")[:n_prune]] = True
        return prune_mask

    def exclude_outbbx(self, pts, recon_args):
        bbx_min = torch.tensor(recon_args["volume_origin"], dtype=torch.float32, device=pts.device)
        bbx_max = bbx_min + torch.tensor(recon_args["volume_phy"], dtype=torch.float32, device=pts.device)
        return ((pts < bbx_min) | (pts > bbx_max)).any(dim=1)

    def add_densification_stats(self, viewspace_point_tensor, dummy_opacity, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.avgopacity_accum[update_filter] += dummy_opacity[update_filter]
        self.denom[update_filter] += 1
