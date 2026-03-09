"""Export clinical DSA datasets to the training format.

Writes ``projections/``, ``priors/``, ``geometry/``, and
``transforms.json`` in the layout consumed by the training pipeline.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from data_generation.clinical.utils.geometry import convert_pose
from data_generation.clinical.utils.image import zoom_centered
from data_generation.common.geometry import (
    voxel_to_world_coords,
    write_ply_xyz_opacity,
)
from data_generation.common.transforms import write_transforms


class DatasetExporter:
    """Write a complete clinical dataset to disk."""

    def __init__(
        self,
        output_dir: Path,
        *,
        patient_id: str,
        zoom_factor: float = 1.0,
        mip_offset_start: int = 3,
        mip_offset_end: int = 15,
        overwrite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.patient_id = patient_id
        self.zoom_factor = zoom_factor
        self.mip_offset_start = mip_offset_start
        self.mip_offset_end = mip_offset_end
        self.overwrite = overwrite

    def export(
        self,
        views: list[dict[str, Any]],
        prior_points: np.ndarray | None,
        *,
        cta_info: dict[str, Any],
        mesh_ref_config: dict[str, Any] | None = None,
    ) -> Path:
        """Export dataset.

        Args:
            views: Per-view dicts with ``dicom`` (:class:`DicomData`),
                ``dsa_mip`` ``[H, W]``, and optional ``seg_mask``.
            prior_points: ``[N, 3]`` vessel prior or ``None``.
            cta_info: Volume metadata (resolution, spacing, phy).
            mesh_ref_config: If given, exports ``geometry/mesh_ref.obj``.

        Returns:
            Path to the output directory.
        """
        if self.output_dir.exists():
            if not self.overwrite:
                raise FileExistsError(f"{self.output_dir} exists (use --overwrite)")
            shutil.rmtree(self.output_dir)

        proj_dir = self.output_dir / "projections"
        proj_dir.mkdir(parents=True, exist_ok=True)

        frames, arrays = self._process_views(views, cta_info)

        # Write projections
        for name, arr in arrays.items():
            path = proj_dir / f"{name}.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.float32)), str(path))
            print(f"  Wrote {name}: {arr.shape}")

        # Write priors
        priors_info: dict[str, Any] = {}
        if prior_points is not None and len(prior_points) > 0:
            priors_info = self._write_priors(prior_points, views)

        # Export mesh reference
        mesh_ref_info = None
        if mesh_ref_config is not None:
            print("  Exporting reference mesh...")
            mesh_ref_info = export_mesh_ref(**mesh_ref_config, out_dir=self.output_dir)
            if mesh_ref_info:
                print(f"  Exported mesh_ref.obj ({mesh_ref_info['n_vertices']} vertices)")

        # Write transforms.json
        transforms = self._build_transforms(
            frames=frames,
            cta_info=cta_info,
            priors_info=priors_info,
            mesh_ref_info=mesh_ref_info,
            arrays=arrays,
        )
        write_transforms(self.output_dir / "transforms.json", transforms)

        # Visualisation
        self._write_visualization(arrays, frames)

        print(f"[OK] Exported dataset: {self.output_dir}")
        return self.output_dir

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_views(
        self,
        views: list[dict[str, Any]],
        cta_info: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        from scipy.ndimage import zoom as ndi_zoom

        volume_phy = np.array(cta_info["volume_phy"], dtype=np.float32)
        target_h = max(v["dsa_mip"].shape[0] for v in views)
        target_w = max(v["dsa_mip"].shape[1] for v in views)

        frames: list[dict[str, Any]] = []
        mask_runs: list[np.ndarray] = []
        fill_runs: list[np.ndarray] = []
        dsa_mips: list[np.ndarray] = []

        for view_idx, view in enumerate(views):
            dicom = view["dicom"]
            dsa_mip: np.ndarray = view["dsa_mip"]
            camera_params = dicom.camera_params
            drr_cfg = camera_params.get("drr", {})
            xray_cfg = camera_params.get("xray", {})

            world2cam, cam2world = _convert_pose(camera_params, drr_cfg)
            orig_h, orig_w = dsa_mip.shape
            fov_info = _compute_fov_geometry(drr_cfg, orig_h, orig_w, self.zoom_factor)
            out_h, out_w = orig_h, orig_w

            if (orig_h, orig_w) != (target_h, target_w):
                zh, zw = target_h / orig_h, target_w / orig_w
                dsa_mip = ndi_zoom(dsa_mip, (zh, zw), order=1)
                fov_info["delx_out"] /= zw
                fov_info["dely_out"] /= zh
                out_h, out_w = target_h, target_w

            if self.zoom_factor > 1.0:
                dsa_mip = zoom_centered(dsa_mip, self.zoom_factor, order=1)
                fov_info["delx_out"] /= self.zoom_factor
                fov_info["dely_out"] /= self.zoom_factor

            fov_info["out_h"] = out_h
            fov_info["out_w"] = out_w

            mask_runs.append(np.ones_like(dsa_mip, dtype=np.float32))
            fill_runs.append((1.0 - dsa_mip).astype(np.float32))
            dsa_mips.append(dsa_mip)

            frames.append(
                _build_frame_dict(
                    view_id=view_idx,
                    file_name=dicom.view_id,
                    world2cam=world2cam,
                    cam2world=cam2world,
                    fov_info=fov_info,
                    drr_cfg=drr_cfg,
                    xray_cfg=xray_cfg,
                    dicom=dicom,
                    volume_phy=volume_phy,
                    zoom_factor=self.zoom_factor,
                )
            )

        arrays = {
            "mask_run": np.stack(mask_runs, axis=0),
            "fill_run": np.stack(fill_runs, axis=0),
            "DSA": np.stack(dsa_mips, axis=0),
        }
        return frames, arrays

    def _write_priors(
        self,
        prior_points: np.ndarray,
        views: list[dict[str, Any]],
    ) -> dict[str, Any]:
        priors_dir = self.output_dir / "priors"
        priors_dir.mkdir(parents=True, exist_ok=True)

        artery = views[0]["dicom"].artery if views else "vessel"
        name = f"vessel_{artery}"
        out_ply = priors_dir / f"{name}_prior.ply"
        write_ply_xyz_opacity(out_ply, prior_points, opacity=1.0)

        print(f"  Wrote prior: {out_ply.name} ({prior_points.shape[0]} points)")
        return {
            name: {
                "path": str(out_ply.relative_to(self.output_dir)),
                "n_points": int(prior_points.shape[0]),
                "filter": "visual_hull",
            }
        }

    def _build_transforms(
        self,
        *,
        frames: list[dict[str, Any]],
        cta_info: dict[str, Any],
        priors_info: dict[str, Any],
        mesh_ref_info: dict[str, Any] | None,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        volume_phy = cta_info["volume_phy"]
        n_views = len(frames)

        dsa_arr = arrays.get("DSA", arrays.get("mask_run"))
        proj_h, proj_w = int(dsa_arr.shape[1]), int(dsa_arr.shape[2])

        first_frame = frames[0] if frames else {}
        transforms: dict[str, Any] = {
            "pose_type": "world2cam_4x4",
            "dsa_convention": "mask_minus_fill",
            "obj_index": self.patient_id,
            "N_views": n_views,
            "proj_resolution": [proj_w, proj_h],
            "proj_phy": first_frame.get("proj_phy", [0.0, 0.0]),
            "proj_spacing": first_frame.get("proj_spacing", [0.0, 0.0]),
            "volume_resolution": cta_info["volume_resolution"],
            "volume_spacing": cta_info["volume_spacing"],
            "volume_phy": volume_phy,
            "volume_origin": [-v / 2.0 for v in volume_phy],
            "sid": float(np.mean([f["sid"] for f in frames])) if frames else 0.0,
            "sad": float(np.mean([f["sad"] for f in frames])) if frames else 0.0,
            "coordinate_system": "volume_centered",
            "source": {
                "export_zoom_factor": self.zoom_factor,
                "mip_offset_start": self.mip_offset_start,
                "mip_offset_end": self.mip_offset_end,
            },
            "geometry": {},
            "priors": priors_info,
            "frames": frames,
        }
        if mesh_ref_info:
            transforms["geometry"]["mesh_ref"] = mesh_ref_info
        return transforms

    def _write_visualization(
        self,
        arrays: dict[str, np.ndarray],
        frames: list[dict[str, Any]],
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        dsa_mips = arrays.get("DSA")
        if dsa_mips is None or len(dsa_mips) == 0:
            return

        n = len(dsa_mips)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]

        for i, (mip, frame) in enumerate(zip(dsa_mips, frames)):
            axes[i].imshow(mip, cmap="gray")
            vid = frame.get("file", f"view_{i}")
            vtype = "LAT" if frame.get("pf_to_af") else "AP"
            axes[i].set_title(f"{vid}\n({vtype}, {mip.shape[0]}x{mip.shape[1]})")
            axes[i].axis("off")

        end_tag = "end" if self.mip_offset_end <= 0 else str(self.mip_offset_end)
        fig.suptitle(
            f"DSA MIP Export (frames +{self.mip_offset_start}..+{end_tag})\n{self.patient_id}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(str(self.output_dir / "visualization.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _convert_pose(
    camera_params: dict[str, Any],
    drr_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    return convert_pose(camera_params, drr_cfg)


def _compute_fov_geometry(
    drr_cfg: dict[str, Any],
    target_h: int,
    target_w: int,
    zoom_factor: float,
) -> dict[str, float]:
    sid = float(drr_cfg.get("sdd", 1200.0))
    orig_h = int(drr_cfg.get("height", target_h))
    orig_w = int(drr_cfg.get("width", target_w))
    delx = float(drr_cfg.get("delx", 0.2))
    dely = float(drr_cfg.get("dely", 0.2))

    det_x = float(orig_w) * delx
    det_y = float(orig_h) * dely
    if zoom_factor > 1.0:
        det_x /= zoom_factor
        det_y /= zoom_factor

    FovX = float(2.0 * np.arctan(det_x / (2.0 * sid)))
    FovY = float(2.0 * np.arctan(det_y / (2.0 * sid)))

    scale_h = float(target_h) / float(orig_h)
    scale_w = float(target_w) / float(orig_w)

    return {
        "sid": sid,
        "FovX": FovX,
        "FovY": FovY,
        "det_phy_x": det_x,
        "det_phy_y": det_y,
        "delx_out": delx / scale_w,
        "dely_out": dely / scale_h,
        "orig_h": orig_h,
        "orig_w": orig_w,
    }


def _build_frame_dict(
    *,
    view_id: int,
    file_name: str,
    world2cam: np.ndarray,
    cam2world: np.ndarray,
    fov_info: dict[str, float],
    drr_cfg: dict[str, Any],
    xray_cfg: dict[str, Any],
    dicom: Any,
    volume_phy: np.ndarray,
    zoom_factor: float,
) -> dict[str, Any]:
    R = world2cam[:3, :3].astype(np.float32)
    T = world2cam[:3, 3].astype(np.float32)
    cam_centre = (-R.T @ T).astype(np.float32)
    sad = float(np.linalg.norm(cam_centre))

    d = float(np.linalg.norm(cam_centre))
    radius = float(np.linalg.norm(volume_phy) / 2.0)
    near = max(d - radius - 50.0, 1.0)
    far = d + radius + 50.0

    return {
        "file": file_name,
        "view_id": view_id,
        "sid": fov_info["sid"],
        "sad": sad,
        "width": int(fov_info.get("out_w", fov_info["orig_w"])),
        "height": int(fov_info.get("out_h", fov_info["orig_h"])),
        "proj_phy": [fov_info["det_phy_x"], fov_info["det_phy_y"]],
        "proj_spacing": [fov_info["delx_out"], fov_info["dely_out"]],
        "FovX": fov_info["FovX"],
        "FovY": fov_info["FovY"],
        "near": near,
        "far": far,
        "world2cam": world2cam.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "mask_frames": dicom.mask_frames,
        "fill_frames": dicom.fill_frames,
        "pf_to_af": dicom.pf_to_af,
        "n_dicom_frames": dicom.n_frames,
        "xray_crop_total": int(xray_cfg.get("crop", 100)),
        "export_zoom_factor": float(zoom_factor),
        "diffdrr_final_pose_cam2world": cam2world.tolist(),
        "reverse_x_axis": bool(drr_cfg.get("reverse_x_axis", True)),
    }


def export_mesh_ref(
    *,
    cta_path: Path,
    seg_path: Path,
    label: int,
    out_dir: Path,
    volume_resolution_xyz: np.ndarray,
    volume_origin_xyz: np.ndarray,
    volume_spacing_xyz: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any] | None:
    """Export reference mesh from CTA segmentation via marching cubes.

    Uses SimpleITK instead of DiffDRR for volume loading.
    """
    try:
        import mcubes
        from scipy.ndimage import zoom as ndi_zoom
    except ImportError as exc:
        print(f"[WARN] Missing dependency for mesh_ref: {exc}")
        return None

    try:
        seg_img = sitk.ReadImage(str(seg_path))
        seg_arr = sitk.GetArrayFromImage(seg_img)  # [Z, Y, X]
    except Exception as exc:
        print(f"[WARN] Failed to load segmentation for mesh_ref: {exc}")
        return None

    vessel_mask = (seg_arr == int(label))
    if not vessel_mask.any():
        print(f"[WARN] No voxels for label={label}")
        return None

    # Transpose from [Z, Y, X] to [X, Y, Z] for marching cubes in world order
    vessel_mask = vessel_mask.transpose(2, 1, 0)

    # Flip X and Y to match DiffDRR orientation="PA" convention.
    # Camera poses were registered in DiffDRR coordinates, so the mesh must match.
    vessel_mask = vessel_mask[::-1, :, :]
    vessel_mask = vessel_mask[:, ::-1, :]

    target_shape = tuple(int(x) for x in volume_resolution_xyz)
    if vessel_mask.shape != target_shape:
        factors = [target_shape[i] / vessel_mask.shape[i] for i in range(3)]
        vessel_mask = ndi_zoom(vessel_mask.astype(np.float32), factors, order=0) > 0.5

    vertices_vox, faces = mcubes.marching_cubes(
        vessel_mask.astype(np.float32), float(threshold)
    )

    geom_dir = out_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)

    vertices_world = voxel_to_world_coords(
        vertices_vox,
        volume_origin_xyz=np.asarray(volume_origin_xyz, dtype=np.float32),
        volume_spacing_xyz=np.asarray(volume_spacing_xyz, dtype=np.float32),
    )

    out_path = geom_dir / "mesh_ref.obj"
    mcubes.export_obj(vertices_world, faces, str(out_path))

    return {
        "path": str(out_path.relative_to(out_dir)),
        "space": "world",
        "n_vertices": int(vertices_vox.shape[0]),
        "n_faces": int(faces.shape[0]),
    }
