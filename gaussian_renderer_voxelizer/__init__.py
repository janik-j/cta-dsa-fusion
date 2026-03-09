#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import contextlib
import importlib
import math
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel


_EXT: Any | None = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    try:
        _EXT = importlib.import_module("diff_Xray_gaussian_rasterization_voxelization")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing `diff_Xray_gaussian_rasterization_voxelization` extension.\n"
            "Install/build it (CUDA) before calling `render()` or `query()`.\n"
            "If you use the uv-managed env, run: `uv sync --extra gpu`."
        ) from e
    return _EXT


def query(
    pc: GaussianModel,
    recon_args,
    nVoxel,
    sVoxel,
    center,
    pipe,
    scaling_modifier=1.0,
    opacity_precomp=None,
    opacity_type="final_opacity",
):
    """
    Query a volume with voxelization.
    """

    ext = _load_ext()

    voxel_settings = ext.GaussianVoxelizationSettings(
        scale_modifier=scaling_modifier,
        nVoxel_x=int(nVoxel[0]),
        nVoxel_y=int(nVoxel[1]),
        nVoxel_z=int(nVoxel[2]),
        sVoxel_x=float(sVoxel[0]),
        sVoxel_y=float(sVoxel[1]),
        sVoxel_z=float(sVoxel[2]),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        prefiltered=False,
        debug=pipe.debug,
    )
    voxelizer = ext.GaussianVoxelizer(voxel_settings=voxel_settings)

    means3D = pc.get_xyz

    if opacity_precomp is not None:
        opacity = opacity_precomp
    else:
        ret = pc._field(pc, recon_args)
        opacity = ret[opacity_type]

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    vol_pred, radii, vol_geo = voxelizer(
        means3D=means3D,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "vol": vol_pred,
        "radii": radii,
        "vol_geo": vol_geo,
    }


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    recon_args,
    pipe,
    scaling_modifier=1.0,
    opacity_precomp=None,
    opacity_type="final_opacity",
):
    """
    Render the scene.
    """

    ext = _load_ext()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    with contextlib.suppress(BaseException):
        screenspace_points.retain_grad()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = ext.GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        mode=1,  # mode = 1, default cone beam mode;  mode = 0 represents parallel beam mode
        debug=pipe.debug,
    )

    rasterizer = ext.GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    if opacity_precomp is not None:
        opacity = opacity_precomp
    else:
        ret = pc._field(pc, recon_args)
        opacity = ret[opacity_type]

    # dummy_opacity = torch.zeros_like(opacity, dtype=opacity.dtype, requires_grad=True, device="cuda")
    dummy_opacity = opacity.clone().detach().requires_grad_(True)
    with contextlib.suppress(BaseException):
        dummy_opacity.retain_grad()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, render_others = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacity,
        dummy_opacities=dummy_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "render_geo": render_others[0:1],
        "dummy_opacity": dummy_opacity,
    }
