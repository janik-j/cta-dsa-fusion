from __future__ import annotations

import json
import shutil
from pathlib import Path

from data_generation.topbrain.config import (
    PRIOR_CONFIGS,
    VIEW_CONFIGS,
    PriorConfig,
    TopbrainConfig,
    ViewConfig,
)
from data_generation.topbrain.exporters.priors import generate_prior
from data_generation.topbrain.exporters.residuals import generate_residual
from data_generation.topbrain.exporters.views import generate_view
from data_generation.topbrain.loaders import cases as topbrain_cases
from data_generation.topbrain.processors import cta as topbrain_cta
from data_generation.topbrain.processors import geometry as topbrain_geometry


class TopbrainPipeline:
    """Generate TopBrain synthetic datasets for this framework."""

    def __init__(self, config: TopbrainConfig) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.output_root = Path(config.output_root)
        self.transforms_template = Path(config.transforms_template)
        self.contrast_enhancement = float(config.contrast_enhancement)
        self.max_points = int(config.max_points)
        self.overwrite = bool(config.overwrite)

    def generate_base(self, ct_path: Path, seg_path: Path) -> tuple[Path, dict]:
        """Generate base data shared across all experiments."""
        case_name = topbrain_cases.extract_case_name(ct_path)
        output_dir = self.output_root / case_name

        ct_volume, seg_volume, ct_metadata = topbrain_cta.load_cta_and_segmentation(str(ct_path), str(seg_path))
        (output_dir / "base").mkdir(parents=True, exist_ok=True)
        (output_dir / "geometry").mkdir(parents=True, exist_ok=True)

        metadata = {
            "case_name": case_name,
            "ct_path": str(ct_path),
            "seg_path": str(seg_path),
            "spacing": list(ct_metadata["spacing"]),
            "origin": list(ct_metadata["origin"]),
            "size": list(ct_metadata["size"]),
            "contrast_enhancement": self.contrast_enhancement,
            "_ct_shape": list(ct_volume.shape),
            "_seg_shape": list(seg_volume.shape),
        }
        (output_dir / "base" / "ct_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

        from data_generation.common.transforms import load_transforms

        base_transforms = load_transforms(self.transforms_template)
        base_transforms = topbrain_cta.apply_cta_volume_to_transforms(base_transforms, ct_metadata, case_name=case_name)
        transforms = topbrain_cases.create_view_transforms(base_transforms, [-90, -45, 0])
        topbrain_geometry.create_ply_from_segmentation(
            seg_volume,
            transforms,
            output_dir / "geometry" / "complete_vessels.ply",
            excluded_labels=None,
            max_points=self.max_points,
        )
        topbrain_geometry.export_mesh_obj(
            seg_volume, transforms, output_dir / "geometry" / "mesh_ref.obj", excluded_labels=None
        )
        return output_dir, metadata

    def generate_view(self, ct_path: Path, seg_path: Path, view_config: ViewConfig) -> Path:
        return generate_view(
            ct_path=ct_path,
            seg_path=seg_path,
            view_config=view_config,
            output_root=self.output_root,
            transforms_template=self.transforms_template,
            contrast_enhancement=self.contrast_enhancement,
        )

    def generate_prior(self, ct_path: Path, seg_path: Path, prior_config: PriorConfig) -> Path:
        return generate_prior(
            ct_path=ct_path,
            seg_path=seg_path,
            prior_config=prior_config,
            output_root=self.output_root,
            transforms_template=self.transforms_template,
            max_points=self.max_points,
        )

    def generate_residual(self, ct_path: Path, seg_path: Path, view_name: str, prior_name: str) -> Path | None:
        view_config = VIEW_CONFIGS[view_name]
        prior_config = PRIOR_CONFIGS[prior_name]

        # If priors are uncapped (max_points=0), keep residual PLYs capped to a
        # reasonable budget (matches common training defaults for M2).
        residual_max_points = int(self.max_points) if int(self.max_points) > 0 else 30_000

        return generate_residual(
            ct_path=ct_path,
            seg_path=seg_path,
            view_config=view_config,
            prior_config=prior_config,
            output_root=self.output_root,
            contrast_enhancement=self.contrast_enhancement,
            max_points=residual_max_points,
        )

    def generate_case(self, ct_path: Path, seg_path: Path) -> dict:
        """Generate ALL experimental data for a single case."""
        case_name = topbrain_cases.extract_case_name(ct_path)
        output_dir = self.output_root / case_name

        if output_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"{output_dir} exists (set overwrite=True to regenerate)")
            shutil.rmtree(output_dir)

        views = list(self.config.view_names)
        for v in views:
            if v not in VIEW_CONFIGS:
                raise KeyError(f"Unknown view config '{v}'. Available: {sorted(VIEW_CONFIGS.keys())}")

        priors: list[PriorConfig] = []
        for prior_name in list(self.config.prior_names):
            if prior_name not in PRIOR_CONFIGS:
                raise KeyError(f"Unknown prior config '{prior_name}'. Available: {sorted(PRIOR_CONFIGS.keys())}")
            base = PRIOR_CONFIGS[prior_name]
            priors.append(
                PriorConfig(
                    name=topbrain_cases.prior_name(base.name, excluded_labels=self.config.prior_excluded_labels),
                    excluded_labels=list(self.config.prior_excluded_labels),
                    min_diameter_mm=base.min_diameter_mm,
                    description=base.description,
                )
            )

        summary = {
            "case": case_name,
            "output_dir": str(output_dir),
            "views": {},
            "priors": {},
            "residuals": {},
            "errors": [],
        }

        try:
            self.generate_base(ct_path, seg_path)
        except Exception as e:
            summary["errors"].append(f"Base generation failed: {e}")
            return summary

        for view_name in views:
            try:
                view_dir = self.generate_view(ct_path, seg_path, VIEW_CONFIGS[view_name])
                summary["views"][view_name] = str(view_dir)
            except Exception as e:
                summary["errors"].append(f"View {view_name}: {e}")

        for prior_cfg in priors:
            try:
                prior_dir = self.generate_prior(ct_path, seg_path, prior_cfg)
                summary["priors"][prior_cfg.name] = str(prior_dir)
            except Exception as e:
                summary["errors"].append(f"Prior {prior_cfg.name}: {e}")

        # Residuals: for each (view, prior) combo, attempt to generate if applicable.
        # (p_all typically returns None; filtered priors may produce residual PLYs.)
        for view_name in views:
            for prior_cfg in priors:
                key = f"{view_name}__{prior_cfg.name}"
                try:
                    residual_dir = self.generate_residual(ct_path, seg_path, view_name, prior_cfg.name)
                    if residual_dir is not None:
                        summary["residuals"][key] = str(residual_dir)
                except Exception as e:
                    summary["errors"].append(f"Residual {key}: {e}")

        (output_dir / "experiment_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def generate_batch(self, case_names: list[str] | None = None, limit: int | None = None) -> list[dict]:
        """Generate datasets for multiple cases."""
        all_cases = topbrain_cases.discover_cases(self.dataset_root)
        if not all_cases:
            return []
        selected_cases = topbrain_cases.filter_cases(all_cases, case_names, limit)

        summaries = []
        for ct_path, seg_path in selected_cases:
            try:
                summaries.append(self.generate_case(ct_path, seg_path))
            except Exception as e:
                case_name = topbrain_cases.extract_case_name(ct_path)
                summaries.append({"case": case_name, "error": str(e)})
        return summaries
