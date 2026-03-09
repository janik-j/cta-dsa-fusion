"""Per-frame DSA segmentation mask loading.

Loads binary vessel segmentation masks stored as per-frame NIfTI files::

    <raw_root>/<patient_id>/<artery>/<view>_masks/
        frame_000.nii.gz
        frame_001.nii.gz
        ...
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


class SegmentationLoader:
    """Load per-frame binary segmentation masks for a DSA view."""

    def __init__(self, raw_root: Path, patient_id: str) -> None:
        self.raw_root = Path(raw_root)
        self.patient_id = str(patient_id)
        self.base_dir = self.raw_root / self.patient_id

    def get_masks_dir(self, artery: str, view_id: str) -> Path:
        return self.base_dir / artery / f"{view_id}_masks"

    def get_available_frames(self, artery: str, view_id: str) -> list[int]:
        """Return sorted 0-indexed frame numbers for which masks exist."""
        masks_dir = self.get_masks_dir(artery, view_id)
        if not masks_dir.exists():
            return []
        available: list[int] = []
        for p in sorted(masks_dir.glob("frame_*.nii.gz")):
            m = re.search(r"frame_(\d+)", p.name)
            if m:
                available.append(int(m.group(1)))
        return sorted(available)

    def load(
        self,
        artery: str,
        view_id: str,
        *,
        frame_indices: list[int],
        crop_total: int = 0,
        pf_to_af: bool = False,
        target_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray | None, list[int]]:
        """Load a stack of masks ``[T, H, W]`` for the requested frames.

        Returns ``(stack, loaded_indices)`` or ``(None, [])`` if nothing found.
        """
        masks_dir = self.get_masks_dir(artery, view_id)
        if not masks_dir.exists():
            return None, []

        masks: list[np.ndarray] = []
        loaded: list[int] = []
        for idx in frame_indices:
            path = masks_dir / f"frame_{idx:03d}.nii.gz"
            if not path.exists():
                continue
            try:
                arr = _load_single_mask(
                    path, crop_total=crop_total, pf_to_af=pf_to_af, target_shape=target_shape
                )
                masks.append(arr)
                loaded.append(idx)
            except Exception as exc:
                print(f"  [WARN] Failed to load mask {path}: {exc}")

        if not masks:
            return None, []
        return np.stack(masks, axis=0).astype(np.float32), loaded

    def load_mip(
        self,
        artery: str,
        view_id: str,
        *,
        frame_indices: list[int],
        crop_total: int = 0,
        pf_to_af: bool = False,
        target_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray | None, list[int]]:
        """Load masks and return their max-projection ``[H, W]``."""
        stack, loaded = self.load(
            artery,
            view_id,
            frame_indices=frame_indices,
            crop_total=crop_total,
            pf_to_af=pf_to_af,
            target_shape=target_shape,
        )
        if stack is None:
            return None, []
        return stack.max(axis=0).astype(np.float32), loaded


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _load_single_mask(
    path: Path,
    *,
    crop_total: int,
    pf_to_af: bool,
    target_shape: tuple[int, int] | None,
) -> np.ndarray:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.max(axis=0)
    arr = arr.astype(np.float32)

    if pf_to_af:
        arr = arr[..., ::-1].copy()

    if crop_total > 0:
        h, w = arr.shape
        half = crop_total // 2
        arr = arr[half : h - half, half : w - half]

    if target_shape is not None:
        th, tw = target_shape
        h, w = arr.shape
        if (h, w) != (th, tw):
            arr = zoom(arr, (th / h, tw / w), order=0)

    return arr.astype(np.float32)
