"""TopBrain generation configuration (views, priors, residual combos)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ViewConfig:
    """View configuration for projection generation."""

    name: str
    angles_deg: list[float]
    secondary_angles_deg: list[float] | None = None  # Cranio-caudal tilt
    description: str = ""

    def __post_init__(self) -> None:
        if self.secondary_angles_deg is None:
            object.__setattr__(self, "secondary_angles_deg", [0.0] * len(self.angles_deg))


@dataclass(frozen=True)
class PriorConfig:
    """Prior configuration for vessel filtering.

    Priors can be made incomplete in two independent ways:
      1) `excluded_labels`: hard-remove anatomical segments by label id.
      2) `min_diameter_mm`: keep only voxels with local diameter >= threshold (mm).
    """

    name: str
    excluded_labels: list[int] = field(default_factory=list)
    min_diameter_mm: float | None = None
    description: str = ""

# -----------------------------------------------------------------------------
# View Configurations (LOCKED FOR PAPER)
# -----------------------------------------------------------------------------

LOCKED_VIEW_NAMES: list[str] = ["v2_ap_lat", "v3_standard", "v3_ap_lat_obl", "v4_quadrant"]

VIEW_CONFIGS: dict[str, ViewConfig] = {
    "v2_ap_lat": ViewConfig("v2_ap_lat", [-90, 0], description="2 views: AP + Lateral (-90°, 0°)"),
    "v3_standard": ViewConfig("v3_standard", [-90, -45, 0], description="3 views: Standard (-90°, -45°, 0°)"),
    "v3_ap_lat_obl": ViewConfig(
        "v3_ap_lat_obl",
        [-90, -21, 0],
        description="3 views: AP + mild oblique + Lateral (-90°, -21°, 0°)",
    ),
    "v4_quadrant": ViewConfig(
        "v4_quadrant", [-90, -45, 0, 45], description="4 views: Quadrant (-90°, -45°, 0°, 45°)"
    ),
    # Augmented 2-view pairs for denoise training data (90° internal separation, irregular offsets).
    # Each pair has unique effective projection directions (mod 180°) — no duplicates across pairs.
    "v2_aug_15_105": ViewConfig("v2_aug_15_105", [15, 105], description="2 views: augmented (15°, 105°)"),
    "v2_aug_45_135": ViewConfig("v2_aug_45_135", [45, 135], description="2 views: augmented (45°, 135°)"),
    "v2_aug_n60_30": ViewConfig("v2_aug_n60_30", [-60, 30], description="2 views: augmented (-60°, 30°)"),
    "v2_aug_n15_75": ViewConfig("v2_aug_n15_75", [-15, 75], description="2 views: augmented (-15°, 75°)"),
}


# -----------------------------------------------------------------------------
# Prior Configurations
# TopBrain label reference:
#   1-4: ICA (R-C1, R-C7, L-C1, L-C7)
#   5-7: MCA M1 (R, Acom, L)
#   8-10: ACA A1/A2 (R, Acom, L)
#   11-12: PCA P1 (R, L)
#   13-16: BA, VA
#   17-20: MCA M2 (R-sup, R-inf, L-sup, L-inf)
#   21-24: PCA P2 (similar)
#   25+: Distal branches
# -----------------------------------------------------------------------------

PRIOR_CONFIGS: dict[str, PriorConfig] = {
    "p_all": PriorConfig(name="p_all", min_diameter_mm=None, description="All vessels (no thickness filtering)"),
    "p_gt1p0mm": PriorConfig(name="p_gt1p0mm", min_diameter_mm=1.0, description="Keep diameter ≥ 1.0 mm"),
    "p_gt1p5mm": PriorConfig(name="p_gt1p5mm", min_diameter_mm=1.5, description="Keep diameter ≥ 1.5 mm"),
    "p_gt2p0mm": PriorConfig(name="p_gt2p0mm", min_diameter_mm=2.0, description="Keep diameter ≥ 2.0 mm"),
}


def print_generation_summary() -> None:
    print(f"Views: {len(VIEW_CONFIGS)}")
    for v in LOCKED_VIEW_NAMES:
        print(f"  - {v}: {VIEW_CONFIGS[v].description}")

    print("\nPriors (thickness-based):")
    for p in PRIOR_CONFIGS.values():
        print(f"  - {p.name}: {p.description}")


TOPBRAIN_DEFAULT_TRANSFORMS_TEMPLATE = (
    Path(__file__).resolve().parent / "templates" / "transforms_topbrain_template.json"
)


@dataclass(frozen=True)
class TopbrainConfig:
    dataset_root: Path
    output_root: Path
    transforms_template: Path = TOPBRAIN_DEFAULT_TRANSFORMS_TEMPLATE
    contrast_enhancement: float = 1200.0
    # Maximum points written to each prior PLY. 0 means "no cap".
    max_points: int = 0
    overwrite: bool = True
    prior_excluded_labels: list[int] = field(default_factory=list)
    view_names: list[str] = field(default_factory=lambda: list(LOCKED_VIEW_NAMES))
    prior_names: list[str] = field(default_factory=lambda: list(PRIOR_CONFIGS.keys()))
