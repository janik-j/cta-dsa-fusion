"""
Evaluation Script

Thin CLI wrapper around evaluation.pipeline.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    from pyhocon import ConfigFactory

    from arguments import ModelParams, PipelineParams
    from utils.config_utils import load_config_from_file
    from utils.system_utils import seed_everything

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--Nviews", type=int, default=30)
    parser.add_argument(
        "--field_conf",
        type=str,
        default=str(REPO_ROOT / "arguments" / "nerf.conf"),
        help="field config file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (same schema as train). If omitted, will try <model_path>/run_config.yaml.",
    )
    parser.add_argument("--render_2d", action="store_true")
    parser.add_argument("--VQR", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (explicit; no env fallback).")
    parser.add_argument(
        "--phase",
        type=str,
        default="test",
        help="Output phase subdir for render/VQR (default: test). Use 'train' to overwrite training-phase outputs.",
    )

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.config is None and getattr(args, "model_path", None):
        cand = Path(str(args.model_path)) / "run_config.yaml"
        if cand.exists():
            args.config = str(cand)

    if args.config is not None:
        args, _ = load_config_from_file(
            args.config,
            args,
            allow_unknown_optimization_keys=True,
        )

    field_conf = ConfigFactory.parse_file(args.field_conf)

    # Apply YAML override for field selection if specified
    select_field = getattr(args, "select_field", "")
    if select_field:
        field_conf["select_field"] = select_field
        print(f"Field override from YAML: select_field={select_field}")

    seed_everything(int(args.seed))

    # Import evaluation code only after parsing args so `--help` works even when
    # optional CUDA extensions (e.g. simple_knn) are not installed.
    from evaluation.pipeline import print_results_summary, run_evaluation

    model = model.extract(args)
    results = run_evaluation(model, pipeline.extract(args), field_conf, args)
    print_results_summary(results, args.model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
