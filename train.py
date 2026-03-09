"""
Training Script

Goals:
- YAML-only configs (no JSON).
- Keep the CLI close to the original repo, but minimal.
- Keep only paper-relevant functionality for this framework.
"""

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

# Standard file-relative root resolution
REPO_ROOT = Path(__file__).resolve().parent

from pyhocon import ConfigFactory  # noqa: E402

# Import core parameter groups used by training/evaluation.
from arguments import ModelParams, OptimizationParams, PipelineParams  # noqa: E402

# EXTENSION: Import refactored utilities
from utils.config_utils import (  # noqa: E402
    get_project_paths,
    load_config_from_file,
    save_run_config_yaml,
)
from utils.system_utils import seed_everything  # noqa: E402


def _resolve_dataset_path(path_str: str) -> str:
    """
    Resolve `--source_path` robustly when running from inside the repo.

    Common gotcha: users pass `processed/<case>/...` while their CWD is
    the repo root, which would resolve incorrectly. In that case, we
    reinterpret it relative to the workspace root.
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # First try relative to current working directory (default argparse behavior).
    cwd_abs = Path(os.path.abspath(path_str))
    if cwd_abs.exists():
        return str(cwd_abs)

    # Try relative to the repo root (works even if CWD isn't REPO_ROOT).
    repo_rel = (REPO_ROOT / p).resolve()
    if repo_rel.exists():
        return str(repo_rel)

    # If user provided `processed/<...>`, interpret it as workspace-relative.
    parts = p.parts
    if parts and parts[0] == "processed":
        workspace_rel = (REPO_ROOT.parent.parent / p).resolve()
        if workspace_rel.exists():
            return str(workspace_rel)

    # Fall back to cwd-based absolute path (for better error messages downstream).
    return str(cwd_abs)


def _prepare_results_dir(model_path: str, args, vis_config) -> None:
    """
    Create results directory and persist the effective CLI/config args as YAML.
    """
    os.makedirs(model_path, exist_ok=True)
    save_run_config_yaml(Path(model_path) / "run_config.yaml", args, vis_config=vis_config)


def main():
    """
    Main training entry point.

    This function:
    1. Parses arguments (CLI + YAML config)
    2. Creates loss manager
    3. Calls training loop from utils/train.py
    """
    # ========================================================================
    # ORIGINAL: Argument Parsing
    # ========================================================================
    parser = ArgumentParser(description="Training Script")

    # Parameter groups
    lp_group = ModelParams(parser)
    pp_group = PipelineParams(parser)
    op_group = OptimizationParams(parser)
    parser.add_argument("--Nviews", type=int, default=30, help="Number of training views")
    parser.add_argument("--debug_from", type=int, default=-1, help="Enable debug from iteration")
    parser.add_argument(
        "--detect_anomaly",
        action="store_true",
        default=False,
        help="Detect autograd anomalies",
    )
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[30_000],
        help="Iterations to run testing",
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[30_000],
        help="Iterations to save checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (explicit; no env fallback).")

    # ORIGINAL: Field configuration
    parser.add_argument(
        "--field_conf",
        type=str,
        default=str(REPO_ROOT / "arguments" / "nerf.conf"),
        help="Field config file (HOCON)",
    )

    # YAML configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file for training",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=None,
        help="Iteration to load for resuming",
    )

    # Parse arguments
    args = parser.parse_args(sys.argv[1:])

    # Extract parameter groups
    lp = lp_group.extract(args)
    pp = pp_group.extract(args)
    op = op_group.extract(args)
    # Fix common path mistake (see _resolve_dataset_path docstring)
    if getattr(args, "source_path", None):
        resolved = _resolve_dataset_path(str(args.source_path))
        args.source_path = resolved
        lp.source_path = resolved

    # ========================================================================
    # Load YAML config (if provided)
    # ========================================================================
    vis_config = None
    if args.config is not None:
        args, vis_config = load_config_from_file(args.config, args)
        lp = lp_group.extract(args)
        pp = pp_group.extract(args)
        op = op_group.extract(args)

    # ========================================================================
    # Setup
    # ========================================================================
    if not getattr(lp, "model_path", ""):
        base = get_project_paths().results_dir
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        auto_path = (base / run_id).resolve()
        args.model_path = str(auto_path)
        lp.model_path = str(auto_path)
        print(f"[auto] --model_path not set; using {lp.model_path}")

    print("=" * 80)
    print("Training")
    print("=" * 80)
    print(f"Model path: {lp.model_path}")
    print(f"Source path: {lp.source_path}")
    print(f"Iterations: {op.iterations}")
    print(f"Densify until: {op.densify_until_iter}")
    print(f"Training views: {lp.train_views}")
    print(f"Seed: {args.seed}")

    if args.config:
        print(f"Config file: {args.config}")
    print("=" * 80)

    _prepare_results_dir(lp.model_path, args, vis_config)

    # Initialize RNGs.
    seed_everything(int(args.seed))

    # Load field configuration (HOCON)
    field_conf = ConfigFactory.parse_file(args.field_conf)

    # Apply YAML override for field selection if specified
    if hasattr(lp, "select_field") and lp.select_field:
        field_conf["select_field"] = lp.select_field
        print(f"Field override from YAML: select_field={lp.select_field}")

    # ========================================================================
    # Run Training
    # ========================================================================
    import torch

    from training.loop import training_loop

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print("\nStarting training...")
    training_time = training_loop(
        dataset=lp,
        field_conf=field_conf,
        opt=op,
        pipe=pp,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        debug_from=args.debug_from,
        vis_config=vis_config,
        load_iteration=args.load_iteration,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Total time: {int(training_time // 60)} minutes {int(training_time % 60)} seconds")
    print(f"Output saved to: {lp.model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
