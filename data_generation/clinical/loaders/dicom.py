"""DICOM data loading for clinical DSA sequences.

Loads a DICOM stack and its associated DiffDRR registration parameters
for a given artery and view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_generation.clinical.utils.image import load_pt_fixed


@dataclass
class DicomData:
    """Loaded DICOM stack with preprocessing metadata."""

    stack: np.ndarray  # [T, H, W] pixel data
    mask_frames: list[int]  # baseline (pre-contrast) frame indices
    fill_frames: list[int]  # contrast-enhanced frame indices
    pf_to_af: bool  # LAT-view horizontal flip applied?
    camera_params: dict[str, Any]  # contents of parameters.pt
    view_id: str = ""
    artery: str = ""
    n_frames: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_frames = int(self.stack.shape[0])


class DicomLoader:
    """Load DICOM pixel data and DiffDRR registration parameters.

    Expected filesystem layout::

        <raw_root>/<patient_id>/
            <artery>/<view>.dcm
            <artery>/<view>/parameters.pt
    """

    def __init__(self, raw_root: Path, patient_id: str) -> None:
        self.raw_root = Path(raw_root)
        self.patient_id = str(patient_id)
        self.base_dir = self.raw_root / self.patient_id

    def load(
        self,
        artery: str,
        view: str,
        *,
        crop_total: int | None = None,
    ) -> DicomData:
        """Load DICOM + registration for *artery* / *view*.

        Args:
            artery: e.g. ``"lica"``.
            view: e.g. ``"ap"``.
            crop_total: Symmetric edge crop (px).  Defaults to the value
                stored in ``camera_params["xray"]["crop"]`` (typically 100).
        """
        import pydicom

        # ---- Load DICOM pixels ------------------------------------------------
        dcm_path = self.base_dir / artery / f"{view}.dcm"
        if not dcm_path.exists():
            raise FileNotFoundError(f"DICOM not found: {dcm_path}")

        ds = pydicom.dcmread(str(dcm_path))
        pixel = ds.pixel_array
        stack = pixel[None, ...] if pixel.ndim == 2 else pixel
        stack = stack.astype(np.float32, copy=False)

        pf_to_af = _needs_pf_to_af_flip(ds)
        if pf_to_af:
            stack = stack[..., ::-1].copy()

        # ---- Load registration parameters ------------------------------------
        pt_path = self.base_dir / artery / view / "parameters.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Registration parameters not found: {pt_path}")
        camera_params = load_pt_fixed(pt_path)

        # ---- Crop -------------------------------------------------------------
        if crop_total is None:
            crop_total = int(camera_params.get("xray", {}).get("crop", 100))
        if crop_total > 0:
            h, w = int(stack.shape[1]), int(stack.shape[2])
            half = crop_total // 2
            stack = stack[:, half : h - half, half : w - half]

        # ---- Frame selection --------------------------------------------------
        n_frames = int(stack.shape[0])
        mask_frames, fill_frames = _determine_frame_selections(
            stack, n_frames, camera_params
        )

        data = DicomData(
            stack=stack,
            mask_frames=mask_frames,
            fill_frames=fill_frames,
            pf_to_af=pf_to_af,
            camera_params=camera_params,
            view_id=view,
            artery=artery,
        )
        return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _needs_pf_to_af_flip(ds: object) -> bool:
    """Detect Patient-Foot → Anterior-Foot orientation requiring a flip."""
    try:
        if (
            list(getattr(ds, "PatientOrientation", [])) == ["P", "F"]
            and float(getattr(ds, "PositionerPrimaryAngle", 0.0)) < 0
        ):
            return True
    except Exception:
        return False
    return False


def _determine_frame_selections(
    stack: np.ndarray,
    n_frames: int,
    camera_params: dict[str, Any],
) -> tuple[list[int], list[int]]:
    """Pick baseline and fill frame indices."""
    mask_frames = list(range(min(4, n_frames)))

    means = stack.reshape(n_frames, -1).mean(axis=1)
    centre = int(means.argmin())
    fill_frames = list(range(max(0, centre - 2), min(n_frames, centre + 3)))

    xray_cfg = camera_params.get("xray", {})
    if "mask_frames" in xray_cfg:
        mask_frames = list(xray_cfg["mask_frames"])
    if "fill_frames" in xray_cfg:
        fill_frames = list(xray_cfg["fill_frames"])

    return mask_frames, fill_frames
