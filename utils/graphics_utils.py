"""
Graphics and camera-geometry utilities for this framework.

Contains small helpers for camera transforms, projection matrices, and point clouds.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch


class BasicPointCloud(NamedTuple):
    """Point cloud representation for X-ray imaging."""

    points: np.ndarray
    opacities: np.ndarray
    normals: np.ndarray = None
    is_m1: np.ndarray = None


def get_extrinsic(primary_angle: float, secondary_angle: float, sad: float) -> np.ndarray:
    """Build extrinsic camera matrix from angles and source-axis distance."""
    cp, sp = np.cos(primary_angle), np.sin(primary_angle)
    cs, ss = np.cos(secondary_angle), np.sin(secondary_angle)
    return np.array(
        [[-sp, cp, 0, 0], [ss * cp, ss * sp, -cs, 0], [-cs * cp, -cs * sp, -ss, sad], [0, 0, 0, 1]],
        dtype=np.float64,
    )


def focal2fov(focal: float, d: float) -> float:
    """Convert focal length to field of view."""
    return 2 * math.atan(d / (2 * focal))


def getWorld2View(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build world-to-view transformation matrix."""
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1
    return Rt


def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    """Build projection matrix from near/far planes and field of view."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)

    top = tanHalfFovY * znear
    right = tanHalfFovX * znear

    P = torch.zeros(4, 4)
    P[0, 0] = znear / right
    P[1, 1] = znear / top
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def make_coords(volume_resolution, volume_phy, volume_origin) -> np.ndarray:
    """Create coordinate grid for volume."""
    s1, s2, s3 = volume_phy
    o1, o2, o3 = volume_origin
    n1, n2, n3 = volume_resolution

    def axis_coords(origin, size, n):
        return np.linspace(origin, size + origin, n, endpoint=False) + size / (2 * n)

    xyz = np.meshgrid(axis_coords(o1, s1, n1), axis_coords(o2, s2, n2), axis_coords(o3, s3, n3), indexing="ij")
    return np.asarray(xyz).transpose([1, 2, 3, 0])
