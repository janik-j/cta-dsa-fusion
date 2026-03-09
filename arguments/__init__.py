"""
Argument definitions for this framework training and evaluation.

This file keeps the upstream-style grouped CLI structure while exposing the
public paper-facing options used in this repo.
"""

import json
import os
from argparse import ArgumentParser, BooleanOptionalAction


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if isinstance(value, list):
                # List args: accept `--arg 1 2 3` (nargs="+") with element type inferred from default.
                elem_t = type(value[0]) if value else str
                flags = ("--" + key, "-" + key[0:1]) if shorthand else ("--" + key,)
                group.add_argument(*flags, default=value, type=elem_t, nargs="+")
                continue
            flags = ("--" + key, "-" + key[0:1]) if shorthand else ("--" + key,)
            if shorthand:
                if t is bool:
                    group.add_argument(*flags, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument(*flags, default=value, type=t)
            else:
                if t is bool:
                    group.add_argument(*flags, default=value, action=BooleanOptionalAction)
                else:
                    group.add_argument(*flags, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        # Base loading parameters inherited from the upstream training structure.
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"

        # Initialization defaults are explicit for standalone runs:
        # - `--ply_path` auto-enables PLY initialization
        # - `--fdk_initial` must be opted into explicitly
        self.fdk_initial = False
        self.ply_initial = False
        self.ply_path = ""  # path to ply file for initialization
        self.use_ply_mask = False  # whether to preserve PLY points during training (true inpainting)


        # VQR evaluation: fixed threshold for mesh extraction (CD/HD metrics).
        # Output: `<model>/.../VQR_output/vqr_metrics.json`
        self.eval_threshold = 0.008

        self.M1 = 30_000  # initial sample points for main body
        self.M2 = 30_000  # context points (M2 initialization)
        # Residuals prior: pre-computed PLY for deterministic M2 initialization.
        self.residuals_path = ""  # path to residuals.ply file for M2 initialization

        # EXTENSION: M2 initialization modes (used by PLY init).
        # - residual: load from `residuals_path` (deterministic subsample)
        # - uniform: uniform random voxel centers in the reconstruction volume
        # - fdk: sample top voxels from an FDK reconstruction
        self.m2_mode = "residual"
        self.m2_seed = 0
        self.m2_uniform_opacity = 0.1

        # ORIGINAL: Scale bounds
        self.use_scale_bound = True  # whether use scale bound
        self.scale_min = 0.1  #  scale min voxels
        self.scale_max = 10.0  #  scale max voxels

        self.thres_percent_fdk = 0.016

        # Opacity field selection: "identity_field" or "single_3d_field"
        self.select_field = "single_3d_field"

        super().__init__(parser, "Loading Parameters", sentinel)

    @staticmethod
    def _infer_paths(source_path: str, g) -> None:
        """Auto-infer ply_path and residuals_path from source_path convention.

        Expected layout:
            <case>/views/<view_name>/        → source_path
            <case>/priors/p_gt1p5mm/vessels.ply
            <case>/residuals/<view_name>__p_gt1p5mm/intersected_fdk_residual.ply

        Also supported:
            <dataset_root>/                 → source_path
            <dataset_root>/transforms.json
            <dataset_root>/priors/*.ply
            <dataset_root>/residuals/intersected_fdk_residual.ply
        """
        from pathlib import Path

        src = Path(source_path)
        if src.parent.name == "views":
            case_dir = src.parent.parent
            view_name = src.name

            if not g.ply_path:
                candidate = case_dir / "priors" / "p_gt1p5mm" / "vessels.ply"
                if candidate.exists():
                    g.ply_path = str(candidate)
                    print(f"[auto] --ply_path inferred: {g.ply_path}")

            if not g.residuals_path:
                candidate = (
                    case_dir
                    / "residuals"
                    / f"{view_name}__p_gt1p5mm"
                    / "intersected_fdk_residual.ply"
                )
                if candidate.exists():
                    g.residuals_path = str(candidate)
                    print(f"[auto] --residuals_path inferred: {g.residuals_path}")
            return

        transforms_path = src / "transforms.json"
        if not transforms_path.exists():
            return

        try:
            transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
        except Exception:
            transforms = {}

        if not g.ply_path:
            priors = transforms.get("priors") or {}
            for info in priors.values():
                rel_path = (info or {}).get("path")
                if not rel_path:
                    continue
                candidate = src / rel_path
                if candidate.exists():
                    g.ply_path = str(candidate)
                    print(f"[auto] --ply_path inferred: {g.ply_path}")
                    break
            if not g.ply_path:
                priors_dir = src / "priors"
                if priors_dir.exists():
                    candidates = sorted(priors_dir.glob("*_prior.ply"))
                    if candidates:
                        g.ply_path = str(candidates[0])
                        print(f"[auto] --ply_path inferred: {g.ply_path}")

        if not g.residuals_path:
            residuals = transforms.get("residuals") or {}
            files = residuals.get("files") or {}
            for key in ("intersected_fdk_residual", "fdk_residual"):
                rel_path = (files.get(key) or {}).get("path")
                if not rel_path:
                    continue
                candidate = src / rel_path
                if candidate.exists():
                    g.residuals_path = str(candidate)
                    print(f"[auto] --residuals_path inferred: {g.residuals_path}")
                    break
            if not g.residuals_path:
                for name in ("intersected_fdk_residual.ply", "fdk_residual.ply"):
                    candidate = src / "residuals" / name
                    if candidate.exists():
                        g.residuals_path = str(candidate)
                        print(f"[auto] --residuals_path inferred: {g.residuals_path}")
                        break

    def extract(self, args):
        g = super().extract(args)
        if g.source_path:
            g.source_path = os.path.abspath(g.source_path)
        # Upstream training uses `--Nviews` as the single source of truth for how many
        # training views to subsample from the dataset.
        g.train_views = int(getattr(args, "Nviews", 0) or 0)

        # Auto-infer ply_path / residuals_path from source_path if not provided.
        if g.source_path:
            self._infer_paths(g.source_path, g)
            # Propagate inferred paths back to args so downstream extractors can see them.
            if g.ply_path and not getattr(args, "ply_path", ""):
                args.ply_path = g.ply_path
            if g.residuals_path and not getattr(args, "residuals_path", ""):
                args.residuals_path = g.residuals_path

        # Handle PLY vs FDK initialization (mutually exclusive)
        has_ply_path = bool(g.ply_path)
        if has_ply_path:
            g.ply_initial = True
            g.fdk_initial = False
            args.ply_initial = True
            args.fdk_initial = False
            print(f"PLY path provided ({g.ply_path}), using PLY initialization.")
        elif g.ply_initial:
            raise ValueError("--ply_initial requires --ply_path.")

        m2_mode = str(getattr(g, "m2_mode", "residual") or "residual").lower().strip()
        if int(getattr(g, "M2", 0) or 0) > 0 and m2_mode in {"residual", "ply", "residual_ply", "ply_residual"}:
            if not bool(getattr(g, "residuals_path", "")):
                raise ValueError("M2 residual initialization requires --residuals_path.")

        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ORIGINAL: Base training parameters
        self.iterations = 30_000
        self.log_interval = 5000  # Print loss summary every N iterations
        self.lambda_dssim = 0.07

        # ORIGINAL-like camera sampling during training.
        # - "random": random order without replacement (epoch-style)
        # - "sequential": deterministic cycling
        # - "all": render all training views per step (multi-view consistency)
        self.camera_sampling = "random"
        # ORIGINAL: Learning rates
        self.position_lr_init = 0.0001
        self.position_lr_final = 0.00001
        self.opacity_lr_init = 0.001
        self.opacity_lr_final = 0.0001
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.field_lr_init = 0.001
        self.field_lr_final = 0.0001
        self.field_decay = 0.00005  # 0.00005

        # ORIGINAL: Densification parameters
        self.densification_interval = 200
        self.densify_until_iter = 15000
        self.densify_from_iter = 500
        self.densify_grad_threshold_init = 0.0001  # 0.0001
        self.densify_grad_threshold_final = 0.0001  # 0.00006
        self.percent_dense_init = 2.5  # num of voxels
        self.percent_dense_final = 0.5  # num of voxels

        # ORIGINAL: Pruning parameters
        self.random_prune = False
        self.percent_random_prune_init = 0.08  # 0.1
        self.percent_random_prune_final = 0.08  # 0.06
        self.opacity_prune = False
        self.min_opacity_init = 1e-6
        self.min_opacity_final = 1e-6
        self.avgopacity_prune = True
        self.min_avgopacity_init = 1e-6
        self.min_avgopacity_final = 1e-6
        self.max_screen_size = None

        # EXTENSION: False-negative vessel loss (image-space)
        # Explicit FN term that pushes missing vessel response.
        # Normalized over vessel support to reduce bias toward large vessel mass.
        # Public reference default: enable FN loss; path-dependent components like
        # prior anchoring and residual M2 stay opt-in via config or CLI inputs.
        self.fn_loss_enabled = True
        self.fn_loss_start_iter = 4000
        self.fn_loss_gt_min_norm = 0.01
        self.fn_loss_margin = 0.005
        self.fn_loss_power = 1.0
        # Robust normalization percentile for FN extraction.
        # <=0 or >=100 disables percentile mode and falls back to max scaling.
        self.fn_loss_norm_percentile = 99.5
        self.fn_loss_weight = 1.5
        # Optional length-balancing: approximate local vessel width from GT mask
        # and inversely weight missing-pixel errors so thick trunks do not dominate.
        self.fn_loss_length_balance = True
        self.fn_loss_length_balance_kernel = 15
        self.fn_loss_length_balance_power = 0.5
        self.fn_loss_length_balance_max = 3.0

        # EXTENSION: Prior anchoring (anchor near prior but allow far outpainting)
        # This is opt-in because it requires an explicit prior path.
        self.lambda_prior_anchoring = 0.0025
        self.prior_anchoring_start_iter = 1000  # When to start prior anchoring loss
        self.prior_anchor_radius = 0.5
        self.prior_anchor_k = 8
        self.prior_anchor_sample_size = 2048
        self.prior_path = ""  # Path to the prior vessel skeleton/centerline (PLY file)

        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        g = super().extract(args)

        has_prior_path = bool(getattr(g, "prior_path", "")) or bool(getattr(args, "ply_path", ""))
        if float(getattr(g, "lambda_prior_anchoring", 0.0) or 0.0) > 0.0 and not has_prior_path:
            raise ValueError("Prior anchoring requires --prior_path or --ply_path.")

        return g
