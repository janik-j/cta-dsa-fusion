"""Dataset conventions (DSA subtraction, pose types)."""

from __future__ import annotations

import numpy as np

POSE_TYPES = {"angles_rad", "world2cam_4x4"}
# NOTE: Datasets in this repo may store projections in different domains:
# - linear: already in an attenuation-like domain (can be negative), so subtraction is linear.
# - log: raw intensity domain, where the upstream method uses log-subtraction: log(mask) - log(fill).
DSA_CONVENTIONS = {
    "fill_minus_mask",
    "mask_minus_fill",
    "log_fill_minus_mask",
    "log_mask_minus_fill",
}


def compute_dsa(mask_run: np.ndarray, fill_run: np.ndarray, *, convention: str) -> tuple[np.ndarray, str]:
    """Compute DSA projection from mask/fill runs (explicit convention)."""
    conv = str(convention).strip().lower()
    if conv == "log_fill_minus_mask":
        proj = np.log(fill_run) - np.log(mask_run)
        proj = np.clip(proj, 0, proj.max())
        return proj, "log subtraction (log(fill) - log(mask))"
    if conv == "log_mask_minus_fill":
        # Upstream convention.
        proj = np.log(mask_run) - np.log(fill_run)
        proj = np.clip(proj, 0, proj.max())
        return proj, "log subtraction (log(mask) - log(fill))"

    if conv == "fill_minus_mask":
        proj = fill_run - mask_run
        method = "linear subtraction (fill - mask)"
    elif conv == "mask_minus_fill":
        proj = mask_run - fill_run
        method = "linear subtraction (mask - fill)"
    else:
        raise ValueError(
            f"Unsupported dsa_convention={convention!r} (expected: "
            "fill_minus_mask|mask_minus_fill|log_fill_minus_mask|log_mask_minus_fill)"
        )

    proj = np.clip(proj, 0, None)
    proj_max = proj.max()
    if proj_max > 0:
        proj = proj / proj_max
    return proj, method
