#!/usr/bin/env python3
"""TopBrain dataset generation CLI.

Run from the repo root:
  python -m data_generation.topbrain.cli --case topcow_ct_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_generation.topbrain.config import (
    LOCKED_VIEW_NAMES,
    PRIOR_CONFIGS,
    TOPBRAIN_DEFAULT_TRANSFORMS_TEMPLATE,
    TopbrainConfig,
    print_generation_summary,
)
from data_generation.topbrain.loaders import cases as topbrain_cases
from data_generation.topbrain.pipeline import TopbrainPipeline
from utils.config_utils import get_project_paths


def _build_parser(*, default_dataset_root: str, default_output_root: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TopBrain Dataset Generator - Generates experimental data from CTA/segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate everything for all cases
    python -m data_generation.topbrain.cli --batch

    # Generate everything for a single case
    python -m data_generation.topbrain.cli --case topcow_ct_001

    # Generate for specific cases
    python -m data_generation.topbrain.cli --batch --cases topcow_ct_001,topcow_ct_002

    # Limit number of cases (for testing)
    python -m data_generation.topbrain.cli --batch --limit 3
""",
    )

    # Case selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=str, help="Single case name (e.g., topcow_ct_001)")
    group.add_argument("--batch", action="store_true", help="Process all cases in dataset root")

    # Batch options
    parser.add_argument("--cases", type=str, help="Comma-separated case names to process (with --batch)")
    parser.add_argument("--limit", type=int, help="Limit number of cases (with --batch)")

    # Paths
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=default_dataset_root,
        help="Root directory with raw CTA data",
    )
    parser.add_argument("--output-root", type=str, default=default_output_root, help="Output root directory")
    parser.add_argument(
        "--transforms-template",
        type=str,
        default=str(TOPBRAIN_DEFAULT_TRANSFORMS_TEMPLATE),
        help="Transforms.json template",
    )

    # Options
    parser.add_argument("--contrast-enhancement", type=float, default=1200.0, help="Contrast enhancement in HU")
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum points per prior PLY (0 = no cap; use all vessel voxels).",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=",".join(LOCKED_VIEW_NAMES),
        help="Comma-separated view config names to generate",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default=",".join(PRIOR_CONFIGS.keys()),
        help="Comma-separated prior config names to generate",
    )
    parser.add_argument(
        "--prior-exclude-labels",
        type=str,
        default="",
        help="Prior filtering: comma-separated label ids to exclude (optional)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    paths = get_project_paths()
    parser = _build_parser(
        default_dataset_root=str(paths.raw_datasets_dir), default_output_root=str(paths.datasets_dir)
    )
    args = parser.parse_args(argv)

    generator = TopbrainPipeline(
        TopbrainConfig(
            dataset_root=Path(args.dataset_root),
            output_root=Path(args.output_root),
            transforms_template=Path(args.transforms_template),
            contrast_enhancement=float(args.contrast_enhancement),
            max_points=int(args.max_points),
            view_names=[s.strip() for s in str(args.views).split(",") if s.strip()],
            prior_names=[s.strip() for s in str(args.priors).split(",") if s.strip()],
            prior_excluded_labels=[
                int(x)
                for x in args.prior_exclude_labels.split(",")
                if str(x).strip() and str(x).strip().lstrip("-").isdigit()
            ],
        )
    )

    print(f"\n{'=' * 70}")
    print("TopBrain Dataset Generator")
    print(f"{'=' * 70}")
    print_generation_summary()

    if args.batch:
        case_names = args.cases.split(",") if args.cases else None
        summaries = generator.generate_batch(case_names, args.limit)

        failed: list[str] = []
        for s in summaries:
            case = str(s.get("case", "UNKNOWN"))
            errors = s.get("errors", None)
            hard_error = s.get("error", None)
            if hard_error:
                print(f"[FAIL] {case}: {hard_error}")
                failed.append(case)
                continue
            if isinstance(errors, list) and errors:
                print(f"[WARN] {case}: {len(errors)} error(s)")
                failed.append(case)
                continue
            print(f"[OK]   {case}")

        if failed:
            print("\nRerun only failed IDs:")
            print(
                "python -m data_generation.topbrain.cli --batch --cases "
                + ",".join(failed)
                + f" --views {args.views} --priors {args.priors} --max-points {int(args.max_points)}"
            )
            sys.exit(1)
        return

    dataset_root = Path(args.dataset_root)
    all_cases = topbrain_cases.discover_cases(dataset_root)

    ct_path, seg_path = None, None
    for ct, seg in all_cases:
        if topbrain_cases.extract_case_name(ct) == args.case:
            ct_path, seg_path = ct, seg
            break

    if not ct_path:
        print(f"ERROR: Case '{args.case}' not found in {dataset_root}")
        sys.exit(1)

    generator.generate_case(ct_path, seg_path)


if __name__ == "__main__":
    main()
