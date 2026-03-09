"""Configuration for the clinical DSA export pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class ClinicalConfig:
    """Immutable configuration for a clinical dataset export run.

    Required fields
    ---------------
    patient_id : str
        Patient identifier (directory name under *raw_root*).
    artery : str
        Artery name, e.g. ``"lica"`` or ``"lvert"``.
    views : list[str]
        View names, e.g. ``["ap", "lat"]``.
    raw_root : Path
        Root directory containing patient data.
    export_root : Path
        Root directory for exported datasets.

    Derived fields (computed in ``__post_init__``)
    -----------------------------------------------
    output_dir : Path
        Full output path: ``export_root / patient_id / <subdir>``.
    """

    patient_id: str
    artery: str
    views: list[str]
    raw_root: Path
    export_root: Path

    mip_offset_start: int = 3
    mip_offset_end: int = 15
    zoom_factor: float = 1.0
    vh_mask_dilate_px: int = 0
    max_points: int = 100_000
    overwrite: bool = True

    # Segmentation
    vessel_label: int = 2

    # Mesh reference export
    export_filtered_mesh_ref: bool = True
    mesh_ref_remove_islands: bool = True
    mesh_ref_island_min_size_ratio: float = 0.01

    # Prior
    prior_opacity: float = 0.1

    # Residuals (M2)
    export_residuals: bool = True
    residual_voxel_size_mm: float = 2.0

    # Optional overrides
    export_subdir: str | None = None

    # Derived (computed)
    output_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        subdir = self.export_subdir if self.export_subdir else self.artery
        object.__setattr__(self, "output_dir", self.export_root / self.patient_id / subdir)
