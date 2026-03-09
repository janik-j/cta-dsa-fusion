"""Coordinate transforms for DiffDRR-registered clinical camera poses.

Converts between DiffDRR camera conventions and the world2cam_4x4 format
expected by the Gaussian renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# View geometry (used by visual hull filtering in prior loader + mesh utils)
# ---------------------------------------------------------------------------


@dataclass
class ViewMaskGeometry:
    """2D mask + calibrated camera intrinsics/extrinsics for one view.

    Used by visual hull filtering to test 3D point visibility.
    """

    mask_2d: np.ndarray  # [H, W] binary vessel mask
    cam_pos: np.ndarray  # [3] world-space camera centre
    cam_forward: np.ndarray  # [3] normalised look direction
    cam_right: np.ndarray  # [3] normalised (image +u)
    cam_up: np.ndarray  # [3] normalised (image +v)
    fx: float  # focal length in pixels (horizontal)
    fy: float  # focal length in pixels (vertical)
    cx: float  # principal point x
    cy: float  # principal point y
    source_file: str = ""  # debug identifier


# ---------------------------------------------------------------------------
# Camera-axis conversion
# ---------------------------------------------------------------------------


def diffdrr_cam_to_4drgs_cam_axes() -> np.ndarray:
    """Camera-axis rotation (det = +1) from DiffDRR to renderer convention.

    Transform: x' = x,  y' = -z,  z' = y
    """
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def as_world2cam(final_pose_cam2world_4x4: np.ndarray) -> np.ndarray:
    """Convert DiffDRR ``cam2world`` (4x4) to renderer ``world2cam`` (4x4).

    Steps:
      1. Invert cam2world → world2cam_diffdrr
      2. Apply axis rotation C so camera axes match the Gaussian renderer.
    """
    cam2world = np.asarray(final_pose_cam2world_4x4, dtype=np.float32)
    world2cam = np.linalg.inv(cam2world).astype(np.float32)

    C = diffdrr_cam_to_4drgs_cam_axes()
    R = world2cam[:3, :3]
    T = world2cam[:3, 3]

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = C @ R
    out[:3, 3] = C @ T
    return out


def apply_reverse_x_axis(
    world2cam_4x4: np.ndarray,
    *,
    reverse_x_axis: bool,
) -> np.ndarray:
    """Fold DiffDRR's ``reverse_x_axis`` flag into a world2cam matrix.

    DiffDRR can mirror the detector X axis.  The Gaussian renderer has no
    such toggle, so we bake it by flipping camera X::

        world2cam' = diag([-1, 1, 1, 1]) @ world2cam
    """
    world2cam = np.asarray(world2cam_4x4, dtype=np.float32)
    if not reverse_x_axis:
        return world2cam
    S = np.eye(4, dtype=np.float32)
    S[0, 0] = -1.0
    return (S @ world2cam).astype(np.float32, copy=False)


def convert_pose(
    camera_params: dict,
    drr_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cam2world from DiffDRR params and return ``(world2cam, cam2world)``.

    Handles both ``torch.Tensor`` and plain array ``final_pose`` values.
    """
    import torch

    final_pose = camera_params.get("final_pose")
    if isinstance(final_pose, torch.Tensor):
        cam2world = final_pose.squeeze(0).detach().cpu().numpy().astype(np.float32)
    else:
        cam2world = np.asarray(final_pose, dtype=np.float32).squeeze()

    reverse_x = bool(drr_cfg.get("reverse_x_axis", True))
    world2cam = as_world2cam(cam2world)
    world2cam = apply_reverse_x_axis(world2cam, reverse_x_axis=reverse_x)
    return world2cam, cam2world
