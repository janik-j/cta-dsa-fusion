"""
Scene manager for this framework.

Single responsibility:
- Load cameras + reconstruction args for a dataset.
- Load/create the Gaussian model state for training/evaluation.
- Save checkpoints (PLY + field + deterministic training_state).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from arguments import ModelParams

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


def _resolve_loaded_iter(model_path: Path, load_iteration: int | None) -> int | None:
    if load_iteration is None:
        return None
    load_iteration = int(load_iteration)
    if load_iteration == -1:
        from utils.system_utils import searchForMaxIteration

        return searchForMaxIteration(model_path / "point_cloud")
    return load_iteration


class Scene:
    """
    Scene container managing cameras, point clouds, and Gaussian model.

    This mirrors the upstream repo structure: a small class that owns cameras and
    model state, without hiding control flow behind frameworks.
    """

    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        scale_bound=None,
        load_iteration: int | None = None,
        shuffle: bool = True,
    ) -> None:
        self.model_path = str(args.model_path)
        self.gaussians = gaussians
        self.scale_bound = scale_bound

        model_dir = Path(self.model_path)
        self.loaded_iter = _resolve_loaded_iter(model_dir, load_iteration)
        if self.loaded_iter is not None:
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras: list[Any] = []
        self.test_cameras: list[Any] = []
        self.all_cameras: list[Any] = []

        self.init_args = self._build_init_args(args)
        from scene.dataset_readers import readSceneInfo

        scene_info = readSceneInfo(args.source_path, self.model_path, args.train_views, self.init_args, self.loaded_iter)

        self.recon_args = scene_info.recon_args
        self.train_indice = scene_info.train_indice
        self.eval_indice = scene_info.eval_indice
        self.all_indice = scene_info.all_indice

        self._write_input_ply(scene_info, model_dir=model_dir)
        if shuffle:
            # The upstream implementation shuffles camera infos once at load time.
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self._setup_scene_geometry()
        self._setup_camera_lists(scene_info, args)
        self.all_cameras = self.camera_merge()
        self._load_or_create_gaussians(scene_info, args)

    def _build_init_args(self, args: ModelParams) -> dict[str, Any]:
        return {
            "M1": args.M1,
            "M2": args.M2,
            "fdk_initial": args.fdk_initial,
            "ply_initial": args.ply_initial,
            "ply_path": args.ply_path,
            "residuals_path": getattr(args, "residuals_path", ""),
            "m2_mode": getattr(args, "m2_mode", "residual"),
            "m2_seed": int(getattr(args, "m2_seed", 0) or 0),
            "m2_uniform_opacity": float(getattr(args, "m2_uniform_opacity", 0.1) or 0.1),
            "thres_percent_fdk": args.thres_percent_fdk,
        }

    def _write_input_ply(self, scene_info, *, model_dir: Path) -> None:
        if scene_info.point_cloud is None:
            return
        from scene.dataset_readers import storeply

        storeply(str(model_dir / "input.ply"), scene_info.point_cloud)

    def _setup_scene_geometry(self) -> None:
        self.cameras_extent = float(np.asarray(self.recon_args["volume_phy"]).max())
        self.scene_spacing = float(np.asarray(self.recon_args["volume_spacing"]).min())
        if self.scale_bound is not None:
            self.gaussians.setup_functions(self.scale_bound * self.scene_spacing)
        else:
            self.gaussians.setup_functions()

    def _setup_camera_lists(self, scene_info, args: ModelParams) -> None:
        from scene.cameras import cameraList_from_camInfos

        print("Load Training Cameras: ", len(scene_info.train_cameras))
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Load Testing Cameras: ", len(scene_info.test_cameras))
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

    def _lock_ply_backbone_if_enabled(self, args: ModelParams) -> None:
        if not (bool(getattr(args, "use_ply_mask", False)) and bool(getattr(args, "ply_initial", False))):
            return
        if self.gaussians.is_m1 is None:
            print("[PLY_MASK] Requested, but no `is_m1` mask available from initialization.")
            return
        self.gaussians.fixed_mask = self.gaussians.is_m1.detach().clone()
        n_fixed = int(self.gaussians.fixed_mask.sum().item())
        n_total = int(self.gaussians.fixed_mask.numel())
        print(f"[PLY_MASK] Locked {n_fixed}/{n_total} init points (M1) as fixed backbone.")

    def _load_or_create_gaussians(self, scene_info, args: ModelParams) -> None:
        model_dir = Path(self.model_path)
        if self.loaded_iter is not None:
            ckpt_dir = model_dir / "point_cloud" / f"iteration_{int(self.loaded_iter)}"
            self.gaussians.load_ply(str(ckpt_dir / "point_cloud.ply"))
            self.gaussians.load_field(str(ckpt_dir / "field.pth"))
            self._lock_ply_backbone_if_enabled(args)
            return

        self.gaussians.create_from_pcd(
            scene_info.point_cloud,
            self.cameras_extent,
        )
        self._lock_ply_backbone_if_enabled(args)

    def save(self, iteration: int) -> None:
        from utils.system_utils import save_training_state

        point_cloud_path = Path(self.model_path) / "point_cloud" / f"iteration_{int(iteration)}"
        self.gaussians.save_ply(str(point_cloud_path / "point_cloud.ply"))
        self.gaussians.save_field(str(point_cloud_path / "field.pth"))
        save_training_state(model_path=self.model_path, iteration=int(iteration), gaussians=self.gaussians)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras

    def getAllCameras(self):
        return self.all_cameras

    def camera_merge(self):
        all_cameras = self.train_cameras + self.test_cameras
        return sorted(all_cameras, key=lambda cam: cam.frame_id)

    def get_indice(self):
        return {
            "train_indice": self.train_indice,
            "eval_indice": self.eval_indice,
            "all_indice": self.all_indice,
        }


def __getattr__(name: str):
    if name == "GaussianModel":
        from scene.gaussian_model import GaussianModel as _GaussianModel

        return _GaussianModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
