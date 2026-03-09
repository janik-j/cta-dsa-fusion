"""Clinical DSA dataset generation pipeline.

Orchestrates DICOM loading, DSA computation, prior filtering, and
dataset export for real clinical angiography data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from data_generation.clinical.config import ClinicalConfig
from data_generation.clinical.exporters.dataset import DatasetExporter
from data_generation.clinical.exporters.residuals import generate_residual_plys
from data_generation.clinical.loaders.dicom import DicomLoader
from data_generation.clinical.loaders.prior import PriorLoader
from data_generation.clinical.loaders.segmentation import SegmentationLoader
from data_generation.clinical.processors import dsa as clinical_dsa
from data_generation.clinical.processors.phase_filter import FrameFilter
from data_generation.clinical.utils.geometry import convert_pose
from data_generation.clinical.utils.mesh import (
    filter_mesh_by_visual_hull,
    load_obj,
    remove_mesh_islands,
    sample_points_on_mesh,
    write_obj,
)
from data_generation.common.geometry import write_ply_xyz_opacity


class ClinicalPipeline:
    """End-to-end clinical DSA dataset generation.

    Usage::

        config = ClinicalConfig(...)
        pipeline = ClinicalPipeline(config)
        summary = pipeline.generate_case()
    """

    def __init__(self, config: ClinicalConfig) -> None:
        self.config = config

    def generate_case(self) -> dict[str, Any]:
        """Run the full pipeline for the configured patient/targets.

        Returns a summary dict with output paths and point counts.
        """
        cfg = self.config
        print(f"Exporting dataset for patient {cfg.patient_id}")
        print(f"  Artery: {cfg.artery}, Views: {cfg.views}")
        end_tag = "end" if cfg.mip_offset_end <= 0 else str(cfg.mip_offset_end)
        print(f"  MIP frame offsets: +{cfg.mip_offset_start}..+{end_tag}")
        print(f"  Output: {cfg.output_dir}")

        # ---- Initialise loaders / processors ---------------------------------
        dicom_loader = DicomLoader(cfg.raw_root, cfg.patient_id)
        seg_loader = SegmentationLoader(cfg.raw_root, cfg.patient_id)
        prior_loader = PriorLoader(cfg.raw_root, cfg.patient_id)
        frame_filter = FrameFilter(cfg.mip_offset_start, cfg.mip_offset_end)

        # ---- CTA metadata ----------------------------------------------------
        cta_path = cfg.raw_root / cfg.patient_id / "cta.nii.gz"
        cta_img = sitk.ReadImage(str(cta_path))
        cta_info = {
            "volume_resolution": list(cta_img.GetSize()),
            "volume_spacing": list(cta_img.GetSpacing()),
            "volume_phy": [
                s * r for s, r in zip(cta_img.GetSize(), cta_img.GetSpacing())
            ],
        }
        volume_phy = np.array(cta_info["volume_phy"], dtype=np.float32)

        # ---- Process each view (DICOM → DSA MIP) -----------------------------
        views: list[dict[str, Any]] = []
        for view_name in cfg.views:
            print(f"\nProcessing view: {cfg.artery}/{view_name}")
            view = self._process_view(
                cfg.artery,
                view_name,
                dicom_loader=dicom_loader,
                seg_loader=seg_loader,
                frame_filter=frame_filter,
            )
            views.append(view)

        # ---- Build view geometries for visual hull ---------------------------
        view_geometries = self._build_view_geometries(views, prior_loader)

        # ---- Load and filter prior -------------------------------------------
        prior_points, prior_points_raw = self._load_prior()

        # ---- Prepare mesh_ref config -----------------------------------------
        mesh_ref_config = self._mesh_ref_config(cfg, cta_path, cta_info, volume_phy)

        # ---- Export dataset ---------------------------------------------------
        print("\nExporting dataset...")
        exporter = DatasetExporter(
            cfg.output_dir,
            patient_id=cfg.patient_id,
            zoom_factor=cfg.zoom_factor,
            mip_offset_start=cfg.mip_offset_start,
            mip_offset_end=cfg.mip_offset_end,
            overwrite=cfg.overwrite,
        )
        output_path = exporter.export(
            views, prior_points, cta_info=cta_info, mesh_ref_config=mesh_ref_config
        )

        # ---- Optional post-export: filtered mesh, mesh-sampled priors --------
        post_export_prior_points = self._post_export(
            cfg, output_path, view_geometries, mesh_ref_config
        )

        # ---- Optional residuals ----------------------------------------------
        if cfg.export_residuals:
            pts_for_residual = post_export_prior_points
            if pts_for_residual is None:
                pts_for_residual = (
                    prior_points_raw if prior_points_raw is not None else prior_points
                )
            if pts_for_residual is not None:
                print("\nGenerating residual PLYs...")
                generate_residual_plys(
                    dataset_dir=output_path,
                    prior_pts_world_xyz=pts_for_residual,
                    max_points=30_000,
                    voxel_size_mm=cfg.residual_voxel_size_mm,
                    prior_proj_dilate_px=3,
                    residual_intensity_threshold=0.01,
                    residual_support_threshold=0.01,
                    residual_mask_dilate_px=0,
                    volume_threshold_rel=0.02,
                )
            else:
                print("[WARN] Cannot generate residuals: no prior points loaded")

        final_prior_points = (
            post_export_prior_points
            if post_export_prior_points is not None
            else prior_points
        )
        prior_path = output_path / "priors" / f"vessel_{cfg.artery}_prior.ply"
        residuals_path = output_path / "residuals" / "intersected_fdk_residual.ply"
        return {
            "output_dir": str(output_path),
            "n_views": len(views),
            "n_prior_points": (
                int(final_prior_points.shape[0])
                if final_prior_points is not None
                else 0
            ),
            "prior_path": str(prior_path) if prior_path.exists() else None,
            "residuals_path": str(residuals_path) if residuals_path.exists() else None,
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _process_view(
        self,
        artery: str,
        view: str,
        *,
        dicom_loader: DicomLoader,
        seg_loader: SegmentationLoader,
        frame_filter: FrameFilter,
    ) -> dict[str, Any]:
        cfg = self.config

        dicom_data = dicom_loader.load(artery, view)
        print(f"  DICOM: {dicom_data.n_frames} frames, pf_to_af={dicom_data.pf_to_af}")

        # Update fill_frames from segmentation availability
        available = seg_loader.get_available_frames(dicom_data.artery, dicom_data.view_id)
        if available:
            dicom_data.fill_frames = available
            print(f"  Seg masks: frames {min(available)}-{max(available)} ({len(available)})")

        # Load seg mask MIP for selected frames
        selected_frames = frame_filter.filter_frames(dicom_data.fill_frames)
        crop = int(dicom_data.camera_params.get("xray", {}).get("crop", 100))
        seg_mask_mip, loaded = seg_loader.load_mip(
            dicom_data.artery,
            dicom_data.view_id,
            frame_indices=selected_frames,
            crop_total=crop,
            pf_to_af=dicom_data.pf_to_af,
        )
        if seg_mask_mip is not None:
            print(f"  MIP seg mask: {len(loaded)} frames ({frame_filter})")

        # Compute DSA MIP
        dsa_mip = clinical_dsa.compute_mip(
            dicom_data.stack,
            dicom_data.mask_frames,
            dicom_data.fill_frames,
            frame_filter,
            seg_mask=seg_mask_mip,
        )
        print(f"  DSA MIP: {dsa_mip.shape}")

        return {
            "artery": artery,
            "view": view,
            "dicom": dicom_data,
            "dsa_mip": dsa_mip,
            "seg_mask_mip": seg_mask_mip,
        }

    def _build_view_geometries(
        self, views: list[dict[str, Any]], prior_loader: PriorLoader
    ) -> list:
        cfg = self.config
        geoms = []
        for view in views:
            seg_mask = view.get("seg_mask_mip")
            if seg_mask is None:
                continue
            cam_params = view["dicom"].camera_params
            drr_cfg = cam_params.get("drr", {})

            world2cam, _ = convert_pose(cam_params, drr_cfg)

            h, w = seg_mask.shape
            sid = float(drr_cfg.get("sdd", 1200.0))
            delx = float(drr_cfg.get("delx", 0.2))
            dely = float(drr_cfg.get("dely", delx))
            det_x = float(w) * delx
            det_y = float(h) * dely
            z = float(cfg.zoom_factor)
            if z > 1.0:
                det_x /= z
                det_y /= z
            FovX = float(2.0 * np.arctan(det_x / (2.0 * sid)))
            FovY = float(2.0 * np.arctan(det_y / (2.0 * sid)))

            frame_info = {"world2cam": world2cam.tolist(), "FovX": FovX, "FovY": FovY}
            geoms.append(
                prior_loader.build_view_geometry(seg_mask, frame_info, view["dicom"].view_id)
            )
        return geoms

    def _load_prior(self) -> tuple[None, None]:
        # Prior is always mesh_ref_filtered, handled in _post_export
        return None, None

    def _find_vessel_seg(self, cfg: ClinicalConfig) -> Path | None:
        seg_path = cfg.raw_root / cfg.patient_id / "segmentation" / "vessel.seg.nrrd"
        if seg_path.exists():
            return seg_path
        print("[WARN] Vessel segmentation not found")
        return None

    def _mesh_ref_config(
        self,
        cfg: ClinicalConfig,
        cta_path: Path,
        cta_info: dict[str, Any],
        volume_phy: np.ndarray,
    ) -> dict[str, Any] | None:
        seg_path = self._find_vessel_seg(cfg)
        if seg_path is None:
            return None
        return {
            "cta_path": cta_path,
            "seg_path": seg_path,
            "label": cfg.vessel_label,
            "volume_resolution_xyz": np.array(cta_info["volume_resolution"]),
            "volume_origin_xyz": -volume_phy / 2.0,
            "volume_spacing_xyz": np.array(cta_info["volume_spacing"]),
        }

    def _post_export(
        self,
        cfg: ClinicalConfig,
        output_path: Path,
        view_geometries: list,
        mesh_ref_config: dict[str, Any] | None,
    ) -> np.ndarray | None:
        need_filtered_mesh = cfg.export_filtered_mesh_ref
        need_mesh_prior = True  # always mesh_ref_filtered

        if need_filtered_mesh:
            if not (view_geometries and mesh_ref_config):
                print(
                    "  [WARN] Cannot export mesh_ref_filtered.obj without visual hull geometry"
                )
            else:
                mesh_in = output_path / "geometry" / "mesh_ref.obj"
                if not mesh_in.exists():
                    print("  [WARN] geometry/mesh_ref.obj not found; skipping filtered mesh")
                else:
                    try:
                        v, f = load_obj(mesh_in)
                        v2, f2 = filter_mesh_by_visual_hull(
                            mesh_vertices_world_xyz=v,
                            mesh_faces=f,
                            masks_with_geometry=view_geometries,
                            mask_dilate_px=cfg.vh_mask_dilate_px,
                        )

                        if cfg.mesh_ref_remove_islands and f2.shape[0] > 0:
                            n_before = int(v2.shape[0])
                            v2, f2 = remove_mesh_islands(
                                vertices_world_xyz=v2,
                                faces=f2,
                                min_size_ratio=cfg.mesh_ref_island_min_size_ratio,
                            )
                            print(
                                f"  Island removal: {n_before} -> {v2.shape[0]} verts"
                            )

                        mesh_out = output_path / "geometry" / "mesh_ref_filtered.obj"
                        write_obj(mesh_out, v2, f2)
                        print(
                            f"  Exported mesh_ref_filtered.obj ({v2.shape[0]} verts, {f2.shape[0]} faces)"
                        )

                        tf_path = output_path / "transforms.json"
                        with open(tf_path, encoding="utf-8") as fj:
                            tf = json.load(fj)
                        tf.setdefault("geometry", {})
                        tf["geometry"]["mesh_ref_filtered"] = {
                            "path": str(mesh_out.relative_to(output_path)),
                            "space": "world",
                            "n_vertices": int(v2.shape[0]),
                            "n_faces": int(f2.shape[0]),
                        }
                        tf_path.write_text(
                            json.dumps(tf, indent=2) + "\n", encoding="utf-8"
                        )
                    except Exception as exc:
                        print(f"  [WARN] Failed to export filtered mesh: {exc}")

        if need_mesh_prior:
            return self._export_mesh_sampled_prior(cfg, output_path)
        return None

    def _export_mesh_sampled_prior(
        self, cfg: ClinicalConfig, output_path: Path
    ) -> np.ndarray | None:
        geom_dir = output_path / "geometry"
        preferred = "mesh_ref_filtered.obj"
        mesh_path = geom_dir / preferred
        if not mesh_path.exists():
            mesh_path = geom_dir / "mesh_ref.obj"
        if not mesh_path.exists():
            print("  [WARN] No mesh found for mesh-sampled prior")
            return None

        try:
            v, f = load_obj(mesh_path)
            pts = sample_points_on_mesh(
                vertices_world_xyz=v,
                faces=f,
                n_points=cfg.max_points,
                seed=0,
            )
            artery = cfg.artery
            priors_dir = output_path / "priors"
            priors_dir.mkdir(parents=True, exist_ok=True)
            out_ply = priors_dir / f"vessel_{artery}_prior.ply"
            write_ply_xyz_opacity(out_ply, pts, opacity=cfg.prior_opacity)
            print(f"  Wrote mesh-sampled prior: {out_ply.name} ({pts.shape[0]} points)")

            # Update transforms.json
            tf_path = output_path / "transforms.json"
            with open(tf_path, encoding="utf-8") as fj:
                tf = json.load(fj)
            tf.setdefault("priors", {})
            tf["priors"][f"vessel_{artery}"] = {
                "path": str(out_ply.relative_to(output_path)),
                "n_points": int(pts.shape[0]),
                "filter": "mesh_ref_filtered",
                "source_mesh": str(mesh_path.relative_to(output_path)),
            }
            tf_path.write_text(json.dumps(tf, indent=2) + "\n", encoding="utf-8")
            return pts
        except Exception as exc:
            print(f"  [WARN] Failed to export mesh-sampled prior: {exc}")
            return None
