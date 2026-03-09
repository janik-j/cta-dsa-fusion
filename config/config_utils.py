"""
Configuration utilities for this framework.

This module centralizes:
- project.yaml path resolution
- YAML config parsing and application
- run_config.yaml persistence
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATH_KEYS = ("datasets_dir", "raw_datasets_dir", "results_dir")

_CACHED_PROJECT_CONFIG: dict[str, Any] | None = None


# ============================================================================
# Project paths (project.yaml)
# ============================================================================


def load_project_config(*, config_path: str | Path | None = None, refresh: bool = False) -> dict[str, Any]:
    global _CACHED_PROJECT_CONFIG
    if _CACHED_PROJECT_CONFIG is not None and not refresh:
        return _CACHED_PROJECT_CONFIG

    path = Path(config_path).expanduser() if config_path else (REPO_ROOT / "project.yaml")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing project config: {path}\n"
            "Create `project.yaml` at the repo root with:\n"
            "paths: {datasets_dir, raw_datasets_dir, results_dir}"
        )

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Project config must be a YAML mapping/object, got {type(cfg).__name__}: {path}")

    _CACHED_PROJECT_CONFIG = cfg
    return cfg


def _resolve_repo_relative(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


@dataclass(frozen=True)
class ProjectPaths:
    datasets_dir: Path
    raw_datasets_dir: Path
    results_dir: Path


def get_project_paths(*, config_path: str | Path | None = None) -> ProjectPaths:
    cfg = load_project_config(config_path=config_path)
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("project.yaml must contain a top-level `paths:` mapping.")

    missing = [k for k in REQUIRED_PATH_KEYS if not paths.get(k)]
    if missing:
        raise ValueError("project.yaml is missing required keys under `paths:`: " + ", ".join(missing))
    return ProjectPaths(
        datasets_dir=_resolve_repo_relative(str(paths["datasets_dir"])),
        raw_datasets_dir=_resolve_repo_relative(str(paths["raw_datasets_dir"])),
        results_dir=_resolve_repo_relative(str(paths["results_dir"])),
    )


# ============================================================================
# YAML experiment config (experiments/*.yaml)
# ============================================================================


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict, returning a new dict."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _as_dict(value: Any, *, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(f"{where} must be a mapping/object, got {type(value).__name__}")


def _validate_allowed_keys(cfg: dict[str, Any], *, allowed: set[str], where: str) -> None:
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Unknown {where} keys: {sorted(unknown)}. Allowed: {sorted(allowed)}")


def _parse_int_list(value: Any) -> list[int]:
    """Parse a value into a list of integers (from list or comma-separated string)."""
    if isinstance(value, list):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if str(x).strip()]


def _set_if_exists(args: Any, key: str, value: Any) -> None:
    """Set attribute on args only if it exists."""
    if hasattr(args, key):
        setattr(args, key, value)


def _apply_config_mapping(
    args: Any,
    cfg: dict[str, Any],
    mapping: dict[str, tuple[str, type, Any]],
) -> None:
    """
    Apply a config dict to args using a mapping.

    mapping: {config_key: (args_attr, type_converter, default_or_None)}
    If default is None, the key must exist in cfg to be applied.
    """
    for cfg_key, (attr, converter, default) in mapping.items():
        if default is None:
            if cfg_key in cfg:
                _set_if_exists(args, attr, converter(cfg[cfg_key]))
        else:
            _set_if_exists(args, attr, converter(cfg.get(cfg_key, default)))


def _extract_args_dict(args: Any, keys: list[str]) -> dict[str, Any]:
    """Extract a dict of attributes from args for the given keys."""
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def _looks_like_path_key(key: str) -> bool:
    key_l = key.lower()
    return "path" in key_l or key_l.endswith(("_dir", "_file")) or "ply" in key_l


def _resolve_rel_path(value: str, *, config_dir: Path, source_path: str | None) -> str:
    del source_path  # explicit: path semantics do not depend on dataset existence

    if not value or os.path.isabs(value):
        return value

    # Explicit, unambiguous rule:
    # - paths starting with ./ or ../ are resolved relative to the YAML file directory
    # - other relative paths are resolved relative to the repo root
    if value.startswith(("./", "../")):
        return str((config_dir / value).resolve())

    return str((REPO_ROOT / value).resolve())


_ALLOWED_TOP_LEVEL_KEYS = {"dataset", "training", "losses", "optimization", "visualization", "field"}
_ALLOWED_VISUALIZATION_KEYS = {
    "save_dsa",
    "save_dsa_max_views",
    "save_dsa_debug_stats",
    "save_dsa_norm_percentile",
    "save_dsa_invert",
    "interval",
    "vis_iterations",
}
_ALLOWED_TRAINING_KEYS = {"Nviews", "seed", "test_iterations", "save_iterations"}
_ALLOWED_LOSSES_KEYS = {
    "standard",
    "prior_anchoring",
}
_ALLOWED_FIELD_KEYS = {"select_field"}


def _apply_visualization_section(config: dict[str, Any]) -> dict[str, Any] | None:
    if "visualization" not in config:
        return None

    vcfg = _as_dict(config.get("visualization"), where="visualization")
    _validate_allowed_keys(vcfg, allowed=_ALLOWED_VISUALIZATION_KEYS, where="visualization")

    vis_config: dict[str, Any] = {
        "save_dsa": bool(vcfg.get("save_dsa", False)),
    }
    if "save_dsa_max_views" in vcfg:
        vis_config["save_dsa_max_views"] = int(vcfg["save_dsa_max_views"])
    if "save_dsa_debug_stats" in vcfg:
        vis_config["save_dsa_debug_stats"] = bool(vcfg["save_dsa_debug_stats"])
    if "save_dsa_norm_percentile" in vcfg:
        vis_config["save_dsa_norm_percentile"] = float(vcfg["save_dsa_norm_percentile"])
    if "save_dsa_invert" in vcfg:
        vis_config["save_dsa_invert"] = bool(vcfg["save_dsa_invert"])
    if "vis_iterations" in vcfg:
        vis_config["vis_iterations"] = _parse_int_list(vcfg["vis_iterations"])
    elif "interval" in vcfg:
        vis_config["interval"] = int(vcfg["interval"])

    return vis_config


def _migrate_deprecated_dataset_keys(dcfg: dict[str, Any]) -> dict[str, Any]:
    """Migrate deprecated dataset keys to current names for backwards compatibility."""
    dcfg = dict(dcfg)

    # Removed custom parameters - silently strip them.
    for obsolete_key in (
        "focus_missing_alpha",
        "focus_missing_dilation_kernel",
        "init_aniso_long_mult",
        "init_aniso_thin_mult",
        "force_timestamp",
        "eval_threshold_mode",
        "eval_threshold_scale",
    ):
        dcfg.pop(obsolete_key, None)

    return dcfg


def _apply_dataset_section(*, args: Any, config: dict[str, Any], config_dir: Path) -> None:
    if "dataset" not in config:
        return

    dcfg = _as_dict(config.get("dataset"), where="dataset")
    dcfg = _migrate_deprecated_dataset_keys(dcfg)
    source_path = getattr(args, "source_path", None)
    for key, value in dcfg.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown dataset key: {key!r}")
        if _looks_like_path_key(key) and isinstance(value, str):
            value = _resolve_rel_path(value, config_dir=config_dir, source_path=source_path)
        setattr(args, key, value)


def _apply_training_section(*, args: Any, config: dict[str, Any]) -> None:
    if "training" not in config:
        return

    tcfg = _as_dict(config.get("training"), where="training")
    _validate_allowed_keys(tcfg, allowed=_ALLOWED_TRAINING_KEYS, where="training")
    if "Nviews" in tcfg:
        _set_if_exists(args, "Nviews", int(tcfg["Nviews"]))
    if "seed" in tcfg:
        _set_if_exists(args, "seed", int(tcfg["seed"]))
    if "test_iterations" in tcfg:
        _set_if_exists(args, "test_iterations", _parse_int_list(tcfg["test_iterations"]))
    if "save_iterations" in tcfg:
        _set_if_exists(args, "save_iterations", _parse_int_list(tcfg["save_iterations"]))


def _apply_field_section(*, args: Any, config: dict[str, Any]) -> None:
    """Apply field config section overrides."""
    if "field" not in config:
        return

    fcfg = _as_dict(config.get("field"), where="field")
    _validate_allowed_keys(fcfg, allowed=_ALLOWED_FIELD_KEYS, where="field")

    if "select_field" in fcfg:
        _set_if_exists(args, "select_field", str(fcfg["select_field"]))


def _apply_losses_standard(*, args: Any, cfg: dict[str, Any]) -> None:
    unknown = set(cfg) - {"weight_l1", "weight_dssim"}
    if unknown:
        raise ValueError(f"Unknown losses.standard keys: {sorted(unknown)}. Allowed: ['weight_l1', 'weight_dssim']")

    if "weight_l1" not in cfg and "weight_dssim" not in cfg:
        return

    l1_w = float(cfg.get("weight_l1", 1.0))
    dssim_w = float(cfg.get("weight_dssim", 1.0))
    total = l1_w + dssim_w
    if hasattr(args, "lambda_dssim"):
        args.lambda_dssim = dssim_w / total if total > 0 else 0.07


def _apply_losses_prior_anchoring(*, args: Any, cfg: dict[str, Any]) -> None:
    unknown = set(cfg) - {"weight", "start_iter", "anchor_radius", "k", "sample_size"}
    if unknown:
        raise ValueError(
            "Unknown losses.prior_anchoring keys: "
            + str(sorted(unknown))
            + ". Allowed: ['weight','start_iter','anchor_radius','k','sample_size']"
        )
    _apply_config_mapping(
        args,
        cfg,
        {
            "weight": ("lambda_prior_anchoring", float, 0.0),
            "start_iter": ("prior_anchoring_start_iter", int, 1000),
            "anchor_radius": ("prior_anchor_radius", float, None),
            "k": ("prior_anchor_k", int, None),
            "sample_size": ("prior_anchor_sample_size", int, None),
        },
    )


def _apply_losses_section(*, args: Any, config: dict[str, Any]) -> None:
    if "losses" not in config:
        return

    lcfg = _as_dict(config.get("losses"), where="losses")
    _validate_allowed_keys(lcfg, allowed=_ALLOWED_LOSSES_KEYS, where="losses")

    _apply_losses_standard(args=args, cfg=_as_dict(lcfg.get("standard"), where="losses.standard"))
    _apply_losses_prior_anchoring(args=args, cfg=_as_dict(lcfg.get("prior_anchoring"), where="losses.prior_anchoring"))


def _apply_optimization_section(
    *,
    args: Any,
    config: dict[str, Any],
    allow_unknown_keys: bool,
) -> None:
    if "optimization" not in config:
        return

    ocfg = _as_dict(config.get("optimization"), where="optimization")
    removed_keys = {
        "thin_vessel_rescue",
        "thin_vessel_rescue_warmup_iters",
        "thin_vessel_rescue_fg_weight",
        "thin_vessel_rescue_fg_min_norm",
        "thin_vessel_rescue_static_blend",
    }
    removed_present = sorted(k for k in ocfg if k in removed_keys)
    if removed_present:
        raise ValueError(
            "Removed optimization keys are not supported: "
            f"{removed_present}. Delete them from config."
        )
    # Silently strip deprecated keys.
    for obsolete_key in ("flow_consistency", "TP_std"):
        ocfg.pop(obsolete_key, None)
    unknown: list[str] = []
    for key, value in ocfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            unknown.append(str(key))
    if unknown:
        if not allow_unknown_keys:
            raise ValueError(
                "Unknown optimization keys: "
                f"{sorted(unknown)}. Fix the YAML instead of relying on silent ignores."
            )
        import warnings

        warnings.warn(f"Ignoring unknown optimization keys for this CLI: {sorted(unknown)}", stacklevel=2)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    p = Path(config_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"Config must be YAML (.yaml/.yml), got '{p.suffix}' for: {p}")

    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a mapping/object at the top level, got {type(cfg).__name__}")
    return cfg


def apply_yaml_config(
    config: dict[str, Any],
    args,
    *,
    config_dir: Path,
    allow_unknown_optimization_keys: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Apply a YAML config dict onto an argparse namespace.

    Returns:
        (args, vis_config)
    """
    _validate_allowed_keys(config, allowed=_ALLOWED_TOP_LEVEL_KEYS, where="top-level config")

    vis_config = _apply_visualization_section(config)
    _apply_dataset_section(args=args, config=config, config_dir=config_dir)
    _apply_training_section(args=args, config=config)
    _apply_field_section(args=args, config=config)
    _apply_losses_section(args=args, config=config)
    _apply_optimization_section(
        args=args,
        config=config,
        allow_unknown_keys=allow_unknown_optimization_keys,
    )

    return args, vis_config


def load_config_from_file(
    config_path: str | Path,
    args,
    *,
    allow_unknown_optimization_keys: bool = False,
) -> tuple[Any, dict[str, Any] | None]:
    config = load_yaml_config(config_path)
    cfg_dir = Path(config_path).expanduser().resolve().parent
    return apply_yaml_config(
        config,
        args,
        config_dir=cfg_dir,
        allow_unknown_optimization_keys=allow_unknown_optimization_keys,
    )


def build_run_config(args, *, vis_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build a minimal, schema-valid YAML config from an argparse namespace.

    This is meant for persisting the *effective* configuration of a run so that
    `test.py` can be invoked later without any unsafe `eval`-based merging.
    """
    cfg: dict[str, Any] = {}

    dataset = _extract_args_dict(
        args,
        [
            "source_path",
            "data_device",
            "fdk_initial",
            "ply_initial",
            "ply_path",
            "residuals_path",
            "M1",
            "M2",
            "m2_mode",
            "m2_seed",
            "m2_uniform_opacity",
            "thres_percent_fdk",
            "use_scale_bound",
            "scale_min",
            "scale_max",
            "eval_threshold",
        ],
    )
    if dataset:
        cfg["dataset"] = dataset

    training = _extract_args_dict(args, ["Nviews", "seed", "test_iterations", "save_iterations"])
    if training:
        cfg["training"] = training

    losses = _build_losses_run_config(args)
    if losses:
        cfg["losses"] = losses

    optimization = _extract_args_dict(
        args,
        [
            "iterations",
            "densify_from_iter",
            "densify_until_iter",
            "densification_interval",
            "camera_sampling",
            "fn_loss_enabled",
            "fn_loss_start_iter",
            "fn_loss_gt_min_norm",
            "fn_loss_margin",
            "fn_loss_power",
            "fn_loss_norm_percentile",
            "fn_loss_weight",
            "fn_loss_length_balance",
            "fn_loss_length_balance_kernel",
            "fn_loss_length_balance_power",
            "fn_loss_length_balance_max",
        ],
    )
    if optimization:
        cfg["optimization"] = optimization

    if vis_config:
        cfg["visualization"] = dict(vis_config)

    # Field config (only include if non-empty)
    select_field = getattr(args, "select_field", "")
    if select_field:
        cfg["field"] = {"select_field": select_field}

    return cfg


def _build_losses_run_config(args: Any) -> dict[str, Any]:
    losses: dict[str, Any] = {}
    if hasattr(args, "lambda_dssim"):
        dssim = float(args.lambda_dssim)
        losses["standard"] = {"weight_l1": float(1.0 - dssim), "weight_dssim": dssim}
    if hasattr(args, "lambda_prior_anchoring"):
        losses["prior_anchoring"] = {
            "weight": float(args.lambda_prior_anchoring),
            "start_iter": int(getattr(args, "prior_anchoring_start_iter", 0)),
            "anchor_radius": float(getattr(args, "prior_anchor_radius", 0.5)),
            "k": int(getattr(args, "prior_anchor_k", 8)),
            "sample_size": int(getattr(args, "prior_anchor_sample_size", 2048)),
        }
    return losses


def save_run_config_yaml(
    out_path: str | Path,
    args,
    *,
    vis_config: dict[str, Any] | None = None,
) -> Path:
    p = Path(out_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = build_run_config(args, vis_config=vis_config)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return p



