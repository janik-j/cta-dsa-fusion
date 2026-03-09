"""
Dataset readers for this framework training.

This module handles loading camera parameters, DSA images, and point cloud
initialization from various sources.

ORIGINAL: Core camera/DSA loading from the upstream codebase.
EXTENSION: PLY initialization, M2 modes, dual-prior support added for vessel outpainting.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import SimpleITK as sitk
from plyfile import PlyData, PlyElement

from ct.tigre_ct import TigreCT
from scene.camera_build import CameraInfo, build_camera_infos, default_geometry
from scene.conventions import compute_dsa
from scene.io import (
    load_mask_fill,
    load_transforms_json,
    read_ply_vertices,
    resolve_projection_paths,
    subsample_deterministic,
)
from scene.validation import validate_dataset_layout, validate_transforms
from utils.graphics_utils import BasicPointCloud, make_coords


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    recon_args: dict
    train_indice: np.array
    eval_indice: np.array
    all_indice: np.array


def readCameras(camera_file, datapath):
    """Load camera information and DSA projections."""
    validate_dataset_layout(datapath)
    paths = resolve_projection_paths(datapath)
    camera_paras = load_transforms_json(camera_file)
    camera_paras = validate_transforms(camera_paras)

    pose_type = str(camera_paras.get("pose_type", "") or "").strip()
    dsa_convention = str(camera_paras.get("dsa_convention", "") or "").strip()

    volume_phy = np.asarray(camera_paras["volume_phy"])
    volume_origin = np.asarray(camera_paras.get("volume_origin", (-volume_phy / 2).tolist()), dtype=np.float32)
    recon_args = {
        "volume_resolution": camera_paras["volume_resolution"],
        "volume_phy": volume_phy.tolist(),
        "volume_spacing": camera_paras["volume_spacing"],
        "volume_origin": volume_origin.tolist(),
    }

    defaults = default_geometry(camera_paras, volume_phy)
    N_views = int(camera_paras["N_views"])

    mask_run, fill_run = load_mask_fill(paths)
    proj, dsa_method = compute_dsa(mask_run, fill_run, convention=dsa_convention)
    print(f"DSA: {dsa_method} (dsa_convention={dsa_convention})")

    cam_infos = build_camera_infos(
        frames=camera_paras["frames"],
        proj=proj,
        pose_type=pose_type,
        defaults=defaults,
        n_views=N_views,
    )

    return cam_infos, recon_args


def get_indice(N_views, train_views):
    """Split view indices into train/eval sets.

    For strict comparability with the upstream implementation, we follow its view subsampling:
      train_indice = np.arange(0, N_views, N_views/train_views).astype(int)

    This differs from a linspace-based selection (which tends to include the last view).
    """
    n_views = int(N_views)
    n_train = int(train_views)
    if n_views <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    all_indice = np.arange(n_views, dtype=int)
    if n_train <= 0 or n_train >= n_views:
        return all_indice, np.array([], dtype=int), all_indice

    step = float(n_views) / float(n_train)
    train_indice = np.arange(0, n_views, step).astype(int)
    if train_indice.size > 0 and int(train_indice[-1]) >= n_views:
        train_indice[-1] = n_views - 1

    eval_indice = np.delete(all_indice, train_indice).astype(int)
    return train_indice.astype(int), eval_indice.astype(int), all_indice


def vol_initializor(recon_vol, recon_args, init_args):
    """Initialize point cloud from FDK reconstruction volume."""
    recon_vol = np.clip(recon_vol, a_min=0, a_max=recon_vol.max())

    M1 = init_args["M1"]
    M2 = init_args["M2"]
    thres_percent = init_args["thres_percent_fdk"]

    n1, n2, n3 = (int(x) for x in recon_args["volume_resolution"])
    total = int(n1 * n2 * n3)

    def _coords_from_linear_idx(idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        i = idx // int(n2 * n3)
        rem = idx % int(n2 * n3)
        j = rem // int(n3)
        k = rem % int(n3)

        origin = np.asarray(recon_args["volume_origin"], dtype=np.float32).reshape(3)
        phy = np.asarray(recon_args["volume_phy"], dtype=np.float32).reshape(3)
        spacing = phy / np.asarray([n1, n2, n3], dtype=np.float32)

        pts = np.stack(
            [
                origin[0] + (i.astype(np.float32) + 0.5) * float(spacing[0]),
                origin[1] + (j.astype(np.float32) + 0.5) * float(spacing[1]),
                origin[2] + (k.astype(np.float32) + 0.5) * float(spacing[2]),
            ],
            axis=1,
        ).astype(np.float32)
        return pts

    # ORIGINAL-like behavior: randomly sample M1 points from above-threshold voxels.
    mask_idx = np.flatnonzero((recon_vol.reshape(-1) > float(thres_percent)).reshape(-1))
    if mask_idx.shape[0] < int(M1):
        raise ValueError(
            f"FDK init: only {mask_idx.shape[0]} voxels above threshold={thres_percent} "
            f"but M1={M1} requested. Lower --thres_percent_fdk or reduce M1."
        )
    sel_idx = np.random.choice(mask_idx, int(M1), replace=False)
    points = _coords_from_linear_idx(sel_idx)
    opacities = recon_vol.reshape(-1, 1)[sel_idx].astype(np.float32)

    # ORIGINAL-like behavior: optionally add M2 random context points from the whole volume.
    if int(M2) > 0:
        if int(M2) > total:
            raise ValueError(f"M2={M2} exceeds total voxels={total} for volume_resolution={recon_args['volume_resolution']}")
        rand_idx = np.random.choice(total, int(M2), replace=False)
        rand_points = _coords_from_linear_idx(rand_idx)
        rand_opacities = recon_vol.reshape(-1, 1)[rand_idx].astype(np.float32)
        points = np.concatenate([points, rand_points], axis=0)
        opacities = np.concatenate([opacities, rand_opacities], axis=0)

    return BasicPointCloud(points=points, opacities=opacities)


def _fdk_reconstruct(cam_infos: list, recon_args: dict) -> np.ndarray:
    reconstructor = TigreCT(cam_infos, recon_args)
    recon_vol_zyx = reconstructor.fdk(reconstructor.projs, reconstructor.PrimaryAngles)  # (Z, Y, X)
    recon_vol_zyx = np.clip(recon_vol_zyx, a_min=0, a_max=recon_vol_zyx.max())
    # Upstream convention: use (X, Y, Z) ordering for voxel-to-world coordinate mapping.
    recon_vol_xyz = recon_vol_zyx.transpose(2, 1, 0)
    return recon_vol_xyz


def _sample_top_from_volume(
    recon_vol: np.ndarray,
    recon_args: dict,
    *,
    max_points: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (recon_vol > float(threshold)).reshape(-1)
    coords = make_coords(
        recon_args["volume_resolution"], recon_args["volume_phy"], recon_args["volume_origin"]
    ).reshape(-1, 3)
    coords_mask = coords[mask]
    vals = recon_vol.reshape(-1, 1)[mask].reshape(-1)

    if coords_mask.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    k = int(min(int(max_points), int(coords_mask.shape[0])))
    if k <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    top_idx = np.argpartition(-vals, k - 1)[:k]
    top_idx = top_idx[np.lexsort((top_idx, -vals[top_idx]))]  # stable: (-val, idx)
    pts = coords_mask[top_idx].astype(np.float32)
    opac = vals[top_idx].reshape(-1, 1).astype(np.float32)
    return pts, opac


def _generate_m2_points(
    init_args: dict,
    recon_args: dict,
    max_m2: int,
    *,
    cam_infos: list | None = None,
) -> tuple:
    """Generate M2 context points.

    Supported modes (dataset key `m2_mode`):
    - "residual_ply" (default): load from `dataset.residuals_path`
    - "uniform": sample uniformly in the reconstruction volume bounds
    - "fdk": sample top voxels from an FDK reconstruction (computed from train cameras)

    Returns (points, opacities, normals) or (None, None, None) if disabled.
    """
    if int(max_m2) <= 0:
        return None, None, None

    m2_mode = str(init_args.get("m2_mode", "residual_ply") or "residual_ply").lower().strip()
    if m2_mode in {"residual", "ply", "residual_ply", "ply_residual"}:
        residuals_path = str(init_args.get("residuals_path", "") or "")
        if not residuals_path:
            raise ValueError("M2 > 0 with m2_mode=residual_ply requires dataset.residuals_path.")
        if not os.path.exists(residuals_path):
            raise FileNotFoundError(f"residuals_path not found: {residuals_path}")
        pts, opac, nrm = read_ply_vertices(residuals_path)
        return subsample_deterministic(pts, opac, nrm, max_m2)

    if m2_mode in {"uniform", "uniform_random"}:
        # ORIGINAL-like: sample voxel centers uniformly at random from the reconstruction grid.
        n1, n2, n3 = (int(x) for x in recon_args["volume_resolution"])
        total = int(n1 * n2 * n3)
        if int(max_m2) > total:
            raise ValueError(
                f"uniform M2 requires M2 <= total voxels ({total}) for volume_resolution={recon_args['volume_resolution']}"
            )

        seed = int(init_args.get("m2_seed", 0) or 0)
        rng = np.random.default_rng(seed)
        idx = rng.choice(total, size=int(max_m2), replace=False).astype(np.int64)

        i = idx // int(n2 * n3)
        rem = idx % int(n2 * n3)
        j = rem // int(n3)
        k = rem % int(n3)

        origin = np.asarray(recon_args["volume_origin"], dtype=np.float32).reshape(3)
        phy = np.asarray(recon_args["volume_phy"], dtype=np.float32).reshape(3)
        spacing = phy / np.asarray([n1, n2, n3], dtype=np.float32)

        pts = np.stack(
            [
                origin[0] + (i.astype(np.float32) + 0.5) * float(spacing[0]),
                origin[1] + (j.astype(np.float32) + 0.5) * float(spacing[1]),
                origin[2] + (k.astype(np.float32) + 0.5) * float(spacing[2]),
            ],
            axis=1,
        ).astype(np.float32)

        init_opacity = float(init_args.get("m2_uniform_opacity", 0.1) or 0.1)
        opac = (np.ones((pts.shape[0], 1), dtype=np.float32) * init_opacity).astype(np.float32)
        nrm = np.zeros((pts.shape[0], 3), dtype=np.float32)
        return pts, opac, nrm

    if m2_mode == "fdk":
        if cam_infos is None:
            raise ValueError("m2_mode=fdk requires camera infos (train cameras).")
        recon_vol = _fdk_reconstruct(cam_infos, recon_args)
        thres = float(init_args.get("thres_percent_fdk", 0.0) or 0.0)
        pts, opac = _sample_top_from_volume(recon_vol, recon_args, max_points=int(max_m2), threshold=thres)
        nrm = np.zeros((pts.shape[0], 3), dtype=np.float32)
        return pts, opac, nrm

    raise ValueError(f"Unknown m2_mode={m2_mode!r}. Supported: residual_ply, uniform, fdk.")


def _merge_normals(
    m1_normals: np.ndarray | None, m2_normals: np.ndarray | None, m1_count: int, m2_count: int
) -> np.ndarray | None:
    """Merge normals from M1 and M2, filling missing with zeros."""
    if m1_normals is None and m2_normals is None:
        return None
    n1 = m1_normals if m1_normals is not None else np.zeros((m1_count, 3), dtype=np.float32)
    n2 = m2_normals if m2_normals is not None else np.zeros((m2_count, 3), dtype=np.float32)
    return np.concatenate([n1, n2], axis=0)


def ply_initializor(ply_path, recon_args, init_args, cam_infos=None):
    """
    Deterministic PLY initialization.

    - M1: points from `ply_path` (optionally capped by `M1` deterministically).
    - M2 (optional): explicit `residuals_path` PLY (deterministic).
    """
    max_m1 = init_args.get("M1", None)
    max_m2 = int(init_args.get("M2", 0) or 0)

    m1_points, m1_opacities, m1_normals = read_ply_vertices(ply_path)
    m1_points, m1_opacities, m1_normals = subsample_deterministic(
        m1_points, m1_opacities, m1_normals, max_m1
    )

    points = m1_points
    opacities = m1_opacities
    normals = m1_normals
    is_m1 = np.ones((m1_points.shape[0],), dtype=np.bool_)

    if max_m2 > 0:
        m2_points, m2_opacities, m2_normals = _generate_m2_points(
            init_args, recon_args, max_m2, cam_infos=cam_infos
        )
        if m2_points is not None and m2_points.shape[0] > 0:
            points = np.concatenate([points, m2_points], axis=0)
            opacities = np.concatenate([opacities, m2_opacities], axis=0)
            is_m1 = np.concatenate([is_m1, np.zeros(m2_points.shape[0], dtype=np.bool_)])
            normals = _merge_normals(m1_normals, m2_normals, m1_points.shape[0], m2_points.shape[0])

    m1_count = m1_points.shape[0]
    m2_count = points.shape[0] - m1_count
    print(f"[Init] PLY init: total={points.shape[0]} (M1={m1_count}, M2={m2_count})")

    return BasicPointCloud(points=points, opacities=opacities, normals=normals, is_m1=is_m1)


def storeply(path, pcd):
    """Save point cloud to PLY file."""
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
    elements = np.empty(pcd.points.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate((pcd.points, pcd.opacities), axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def readSceneInfo(datapath, outpath, train_views, init_args, loaded_iter):
    camera_file = os.path.join(datapath, "transforms.json")

    cam_infos, recon_args = readCameras(camera_file, datapath)

    train_indice, eval_indice, all_indice = get_indice(len(cam_infos), train_views)
    print("Training cameras:", train_indice)
    print("Testing cameras:", eval_indice)

    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_indice]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in eval_indice]

    if not loaded_iter:
        if init_args.get("ply_initial"):
            print("PLY initialization")
            ply_path = str(init_args.get("ply_path", "") or "")
            if not ply_path:
                raise ValueError("ply_initial=True but ply_path is empty")
            pcd = ply_initializor(ply_path, recon_args, init_args, cam_infos=train_cam_infos)
        elif init_args["fdk_initial"]:
            print("FDK initialization")
            fdk_file = os.path.join(outpath, "fdk_recon.nii.gz")
            CT_reconstructor = TigreCT(train_cam_infos, recon_args)
            # CT toolboxes typically return volumes in (Z, Y, X) ordering.
            # Upstream convention converts to (X, Y, Z) before sampling voxel centers.
            FDK_recon_zyx = CT_reconstructor.fdk(CT_reconstructor.projs, CT_reconstructor.PrimaryAngles)
            FDK_recon_zyx = np.clip(FDK_recon_zyx, a_min=0, a_max=FDK_recon_zyx.max())
            FDK_recon_xyz = FDK_recon_zyx.transpose(2, 1, 0)

            pcd = vol_initializor(FDK_recon_xyz, recon_args, init_args)

            # Save the raw toolbox output (Z, Y, X) like upstream for debugging/inspection.
            FDK_recon_img = sitk.GetImageFromArray(FDK_recon_zyx)
            FDK_recon_img.SetSpacing(recon_args["volume_spacing"])
            sitk.WriteImage(FDK_recon_img, fdk_file)
        else:
            raise ValueError("Initialization must be either FDK (--fdk_initial) or PLY (--ply_path).")
    else:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        recon_args=recon_args,
        train_indice=train_indice,
        eval_indice=eval_indice,
        all_indice=all_indice,
    )
    return scene_info
