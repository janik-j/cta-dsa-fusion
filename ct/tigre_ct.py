import copy

import numpy as np

try:
    import tigre  # type: ignore

    _TIGRE_IMPORT_ERROR = None
except ModuleNotFoundError as e:
    tigre = None  # type: ignore
    _TIGRE_IMPORT_ERROR = e


class TigreCT:
    def __init__(self, cam_infos, recon_args):
        if tigre is None:
            raise ModuleNotFoundError(
                "The optional dependency 'tigre' is required for TIGRE/FDK reconstruction. "
                "Install the `synthetic` or `full` uv extra, or disable FDK-based initialization."
            ) from _TIGRE_IMPORT_ERROR

        geo = tigre.geometry(mode="cone")

        geo.nVoxel = np.asarray(recon_args["volume_resolution"][::-1])  # x-axis, y-axis, z-axis
        geo.sVoxel = np.asarray(recon_args["volume_phy"][::-1])
        geo.dVoxel = np.asarray(recon_args["volume_spacing"][::-1])

        projs_cols, projs_rows = cam_infos[0].width, cam_infos[0].height
        self.W = projs_cols
        self.H = projs_rows

        self.proj_spacing_x = cam_infos[0].sx
        self.proj_spacing_y = cam_infos[0].sy

        self.sid = cam_infos[0].sid
        self.sad = cam_infos[0].sad

        geo.DSD = self.sid
        geo.DSO = self.sad

        geo.nDetector = np.asarray([self.H, self.W])
        geo.sDetector = np.asarray([self.H * self.proj_spacing_y, self.W * self.proj_spacing_x])
        geo.dDetector = np.asarray([self.proj_spacing_y, self.proj_spacing_x])

        geo.offOrigin = np.asarray([0, 0, 0])
        geo.offDetector = np.asarray([0, 0, 0])

        geo.accuracy = 0.5
        self.geo = geo

        projs = []
        PrimaryAngles = []
        for cam in cam_infos:
            projs.append(cam.image.squeeze())
            PrimaryAngles.append(cam.PrimaryAngle)
        projs = np.stack(projs, axis=0)
        PrimaryAngles = np.stack(PrimaryAngles, axis=0).astype(np.float32, copy=False)

        # Some exported datasets store PrimaryAngle as a placeholder (often all zeros).
        # Recover a usable angular ordering from camera centers if needed.
        try:
            finite = np.isfinite(PrimaryAngles)
            uniq = np.unique(PrimaryAngles[finite]) if finite.any() else np.asarray([])
            if uniq.size <= 1:
                centers = []
                for cam in cam_infos:
                    R = np.asarray(cam.R, dtype=np.float32)
                    T = np.asarray(cam.T, dtype=np.float32).reshape(3)
                    center = (-R.T @ T).astype(np.float32, copy=False)
                    centers.append(center)
                centers = np.stack(centers, axis=0)
                PrimaryAngles = np.arctan2(centers[:, 1], centers[:, 0]).astype(np.float32, copy=False)
        except Exception as e:
            import warnings

            warnings.warn(f"Could not recover camera angles from centers: {e}", stacklevel=2)
        self.projs = projs
        self.PrimaryAngles = PrimaryAngles

    def fdk(self, projs, PrimaryAngles, filter="hann"):
        recon = tigre.algorithms.fdk(projs[:, ::-1, :], copy.deepcopy(self.geo), PrimaryAngles, filter=filter)
        return recon
