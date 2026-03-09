"""TopBrain case discovery and view transform helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ============================================================================
# Batch Processing
# ============================================================================


def discover_cases(dataset_root: Path) -> list[tuple[Path, Path]]:
    """Find all CT/segmentation pairs in TopBrain directory structure.

    Supports:
    - TopBrain structure: imagesTr_topbrain_ct/ and labelsTr_topbrain_ct/
    - Flat structure with _0000 suffix convention

    Returns:
        List of (ct_path, seg_path) tuples
    """
    cases = []

    # Try TopBrain structure
    images_dir = dataset_root / "imagesTr_topbrain_ct"
    labels_dir = dataset_root / "labelsTr_topbrain_ct"

    if images_dir.exists() and labels_dir.exists():
        print(f"Detected TopBrain structure in {dataset_root}")
        ct_files = sorted(images_dir.glob("*.nii.gz"))
        for ct_file in ct_files:
            # topcow_ct_001_0000.nii.gz → topcow_ct_001.nii.gz
            seg_name = ct_file.name.replace("_0000.nii.gz", ".nii.gz")
            seg_file = labels_dir / seg_name

            if seg_file.exists():
                cases.append((ct_file, seg_file))
            else:
                print(f"Warning: No segmentation for {ct_file.name}")
    else:
        # Recursive search
        print(f"Scanning {dataset_root} for CT/Seg pairs...")
        all_nii = sorted(dataset_root.rglob("*.nii.gz"))
        potential_cts = [
            f
            for f in all_nii
            if "label" not in f.name.lower() and "seg" not in f.name.lower() and "mask" not in f.name.lower()
        ]

        for ct_file in potential_cts:
            parent = ct_file.parent
            name = ct_file.name

            seg_name = name.replace("_0000.nii.gz", ".nii.gz")

            # Check parallel labels folder
            if parent.name.startswith("images"):
                labels_parent_name = parent.name.replace("images", "labels")
                seg_file = parent.parent / labels_parent_name / seg_name
                if seg_file.exists():
                    cases.append((ct_file, seg_file))
                    continue

            # Check same folder
            seg_file = parent / seg_name
            if seg_file.exists() and seg_file != ct_file:
                cases.append((ct_file, seg_file))

    return cases


def filter_cases(
    all_cases: list[tuple[Path, Path]], case_names: list[str] | None = None, limit: int | None = None
) -> list[tuple[Path, Path]]:
    """Filter cases by name list or limit.

    Args:
        all_cases: List of (ct, seg) tuples
        case_names: Optional list of case names to include
        limit: Optional max number of cases

    Returns:
        Filtered list of cases
    """
    filtered = all_cases

    if case_names:
        filtered = []
        for ct, seg in all_cases:
            case_name = extract_case_name(ct)
            if case_name in case_names:
                filtered.append((ct, seg))

    if limit and limit > 0:
        filtered = filtered[:limit]

    return filtered


def extract_case_name(ct_path: Path) -> str:
    """Extract case name from CT file path.

    topcow_ct_001_0000.nii.gz → topcow_ct_001
    """
    return ct_path.name.replace(".nii.gz", "").replace("_0000", "")


def prior_name(base_name: str, *, excluded_labels: list[int] | None = None) -> str:
    """Create a stable prior directory name from a base name + optional label exclusions."""
    name = str(base_name).strip()
    if not name:
        raise ValueError("base_name must be non-empty")
    excluded_labels = excluded_labels or []
    if not excluded_labels:
        return name
    excl = "-".join(str(int(x)) for x in excluded_labels)
    return f"{name}_excl{excl}"


# ============================================================================
# Transform Manipulation
# ============================================================================


def create_view_transforms(base_transforms: dict, angles_deg: list[float]) -> dict:
    """Create transforms dict with specified viewing angles.

    Args:
        base_transforms: Base transforms template
        angles_deg: List of angles in degrees

    Returns:
        Modified transforms with frames for each angle
    """
    transforms = dict(base_transforms)

    # Use first frame as template or create default
    base_frame = (
        dict(transforms["frames"][0]) if transforms.get("frames") else {"PrimaryAngle": 0.0, "SecondaryAngle": 0.0}
    )

    # Create frames
    frames = []
    for i, angle_deg in enumerate(angles_deg):
        frame = dict(base_frame)
        frame["PrimaryAngle"] = float(np.deg2rad(angle_deg))
        frame["SecondaryAngle"] = 0.0
        frame["file"] = f"frame_{i:04d}"
        frames.append(frame)

    transforms["frames"] = frames
    transforms["N_views"] = len(frames)

    return transforms
