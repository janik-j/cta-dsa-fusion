#!/usr/bin/env python3
"""Clinical DSA dataset generation CLI.

Run from the repo root::

    python -m data_generation.clinical.cli \
        --patient-id case_1 \
        --artery lica --views ap lat \
        --mip-offset-start 3 --mip-offset-end 15
"""

from __future__ import annotations

import argparse
import shlex

from data_generation.clinical.config import ClinicalConfig
from data_generation.clinical.pipeline import ClinicalPipeline
from utils.config_utils import get_project_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clinical DSA Dataset Generator — export registered DSA data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Two views, frames +3..+15 from first fill
    python -m data_generation.clinical.cli \\
        --patient-id case_1 \\
        --artery lica --views ap lat \\
        --mip-offset-start 3 --mip-offset-end 15
""",
    )

    # Required arguments
    parser.add_argument(
        "--patient-id",
        type=str,
        required=True,
        help="Patient identifier (directory name under raw-root)",
    )
    parser.add_argument(
        "--artery",
        type=str,
        required=True,
        help="Artery name (e.g., lica, lvert)",
    )
    parser.add_argument(
        "--views",
        type=str,
        nargs="+",
        required=True,
        help="View names (e.g., ap lat obl)",
    )

    # MIP frame selection (offsets relative to first fill frame)
    parser.add_argument(
        "--mip-offset-start",
        type=int,
        default=3,
        help="Start offset from first fill frame for MIP (default: 3)",
    )
    parser.add_argument(
        "--mip-offset-end",
        type=int,
        default=15,
        help="End offset from first fill frame for MIP; 0 = all remaining (default: 15)",
    )

    # Output overrides
    parser.add_argument(
        "--export-subdir",
        type=str,
        default=None,
        help="Override subdirectory name (default: artery name)",
    )

    # Visual hull settings
    parser.add_argument(
        "--vh-mask-dilate-px",
        type=int,
        default=0,
        help="Dilate 2D masks by N pixels before visual hull filtering",
    )
    # Prior settings
    parser.add_argument(
        "--vessel-label",
        type=int,
        default=2,
        help="Label ID in vessel segmentation (default: 2)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=100_000,
        help="Maximum prior points (default: 100000)",
    )
    # Mesh reference export
    parser.add_argument(
        "--export-filtered-mesh-ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a visual-hull-filtered version of mesh_ref.obj (default: enabled)",
    )
    parser.add_argument(
        "--mesh-ref-remove-islands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove small disconnected components from mesh_ref_filtered (default: enabled)",
    )
    parser.add_argument(
        "--prior-opacity",
        type=float,
        default=0.1,
        help="Uniform opacity for prior PLY points (default: 0.1)",
    )

    # Residual (M2) export
    parser.add_argument(
        "--export-residuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate residual (M2) PLYs from DSA-based error maps (default: enabled)",
    )
    parser.add_argument(
        "--residual-voxel-size-mm",
        type=float,
        default=2.0,
        help="Residual backprojection grid voxel size in mm (default: 2.0)",
    )

    # Other settings
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.0,
        help="Zoom factor for projections (default: 1.0)",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite existing output (default: enabled)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    paths = get_project_paths()
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = ClinicalConfig(
        patient_id=args.patient_id,
        artery=args.artery,
        views=args.views,
        raw_root=paths.raw_datasets_dir,
        export_root=paths.datasets_dir / "clinical",
        mip_offset_start=args.mip_offset_start,
        mip_offset_end=args.mip_offset_end,
        zoom_factor=args.zoom_factor,
        vh_mask_dilate_px=args.vh_mask_dilate_px,
        max_points=args.max_points,
        overwrite=args.overwrite,
        vessel_label=args.vessel_label,

        export_filtered_mesh_ref=args.export_filtered_mesh_ref,
        mesh_ref_remove_islands=args.mesh_ref_remove_islands,
        prior_opacity=args.prior_opacity,
        export_residuals=args.export_residuals,
        residual_voxel_size_mm=args.residual_voxel_size_mm,
        export_subdir=args.export_subdir,
    )

    print(f"\n{'=' * 70}")
    print("Clinical DSA Dataset Generator")
    print(f"{'=' * 70}")

    pipeline = ClinicalPipeline(config)
    summary = pipeline.generate_case()

    train_cmd = [
        "uv run python train.py",
        f"-s {shlex.quote(summary['output_dir'])}",
        "-m results/clinical_experiment",
        f"--Nviews {summary['n_views']} --seed 42",
    ]
    if summary.get("prior_path"):
        train_cmd.append(f"--ply_path {shlex.quote(summary['prior_path'])}")
    if summary.get("residuals_path"):
        train_cmd.append(f"--residuals_path {shlex.quote(summary['residuals_path'])}")

    print(f"\n{'=' * 70}")
    print("Done.")
    print(f"  Output: {summary['output_dir']}")
    print(f"  Views:  {summary['n_views']}")
    print(f"  Prior:  {summary['n_prior_points']} points")
    if summary.get("prior_path"):
        print(f"  PLY:    {summary['prior_path']}")
    if summary.get("residuals_path"):
        print(f"  M2:     {summary['residuals_path']}")
    print("\nSuggested train command:")
    print("  " + " \\\n  ".join(train_cmd))
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
