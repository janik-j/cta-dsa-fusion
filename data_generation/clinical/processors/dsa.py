"""DSA computation from DICOM stacks.

Computes Digital Subtraction Angiography (DSA) MIP images from raw
DICOM pixel stacks.  Per-frame DSA is computed internally but only the
**MIP** (maximum-intensity projection over selected frames) is
exposed as the final output — dynamic/sequence mode is not supported.
"""

from __future__ import annotations

import numpy as np

from data_generation.clinical.processors.phase_filter import FrameFilter


def _compute_per_frame_dsa(
    stack: np.ndarray,
    mask_frames: list[int],
    fill_frames: list[int],
) -> np.ndarray:
    """Per-frame DSA (internal): ``baseline − fill`` for each fill frame.

    Returns ``[len(fill_frames), H, W]`` normalised to ``[0, 1]``.
    """
    if not fill_frames:
        raise ValueError("fill_frames cannot be empty")

    raw = stack.astype(np.float32)
    vmin, vmax = float(raw.min()), float(raw.max())
    raw = np.clip((raw - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)

    baseline = raw[mask_frames].mean(axis=0)
    dsa_frames = [np.clip(baseline - raw[fi], 0.0, None) for fi in fill_frames]
    dsa_stack = np.stack(dsa_frames, axis=0).astype(np.float32)

    dsa_max = float(dsa_stack.max())
    if dsa_max > 0:
        dsa_stack /= dsa_max + 1e-6

    return dsa_stack


def compute_mip(
    stack: np.ndarray,
    mask_frames: list[int],
    fill_frames: list[int],
    frame_filter: FrameFilter,
    *,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Frame-filtered MIP DSA.

    Args:
        stack: DICOM pixel stack ``[T, H, W]``.
        mask_frames: Baseline (pre-contrast) frame indices.
        fill_frames: Contrast-enhanced frame indices.
        frame_filter: Selects which fill frames to include.
        seg_mask: Optional ``[H, W]`` binary mask to bake in.

    Returns:
        MIP DSA ``[H, W]`` normalised to ``[0, 1]``.
    """
    selected_frames = frame_filter.filter_frames(fill_frames)
    if not selected_frames:
        selected_frames = fill_frames

    dsa_seq = _compute_per_frame_dsa(stack, mask_frames, selected_frames)
    dsa_mip = dsa_seq.max(axis=0).astype(np.float32)

    mip_max = float(dsa_mip.max())
    if mip_max > 0:
        dsa_mip /= mip_max + 1e-6

    if seg_mask is not None:
        dsa_mip *= (seg_mask > 0.5).astype(np.float32)

    return dsa_mip
