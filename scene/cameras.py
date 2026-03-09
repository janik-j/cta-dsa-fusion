import torch
from torch import nn

from utils.graphics_utils import getProjectionMatrix, getWorld2View


class Camera(nn.Module):
    def __init__(
        self,
        frame_id,
        timestamp,
        R,
        T,
        PrimaryAngle,
        SecondaryAngle,
        FoVx,
        FoVy,
        sid,
        sad,
        near,
        far,
        width,
        height,
        sx,
        sy,
        image,
        image_name,
        uid,
        dicom_frame=None,
        data_device="cuda",
    ):
        super().__init__()

        self.uid = uid
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.dicom_frame = dicom_frame
        self.R = R
        self.T = T
        self.PrimaryAngle = PrimaryAngle
        self.SecondaryAngle = SecondaryAngle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.sid = sid
        self.sad = sad
        self.zfar = far  # zfar, znear
        self.znear = near
        self.width = width
        self.height = height
        self.sx = sx
        self.sy = sy
        self.image = image  # save for fdk recon
        self.image_name = image_name

        self.data_device = _resolve_data_device(data_device)

        self.original_image = torch.from_numpy(image).to(self.data_device)
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.as_tensor(
            getWorld2View(R, T),
            dtype=torch.float32,
            device=self.data_device,
        ).transpose(0, 1)  # world2camera
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
            ).transpose(0, 1).to(self.data_device)
        )  # camera2NDC
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)  # world2NDC # world coords * world2NDC = NDC (P, 4) * (4, 4) -> (P, 4)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


# --- Camera loading utilities (merged from camera_utils.py) ---


def _resolve_data_device(data_device):
    try:
        device = torch.device(data_device)
    except Exception as e:
        raise ValueError(f"Invalid data_device={data_device!r}") from e

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested data_device={data_device!r}, but CUDA is unavailable.")
        device_index = device.index if device.index is not None else 0
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested data_device={data_device!r}, but only {torch.cuda.device_count()} CUDA device(s) are available."
            )

    return device


def loadCam(args, cam_info):
    """Load a Camera object from CameraInfo."""
    data_device = "cuda" if args is None else args.data_device
    return Camera(
        frame_id=cam_info.uid,
        timestamp=cam_info.timestamp,
        R=cam_info.R,
        T=cam_info.T,
        PrimaryAngle=cam_info.PrimaryAngle,
        SecondaryAngle=cam_info.SecondaryAngle,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        sid=cam_info.sid,
        sad=cam_info.sad,
        near=cam_info.near,
        far=cam_info.far,
        width=cam_info.width,
        height=cam_info.height,
        sx=cam_info.sx,
        sy=cam_info.sy,
        image=cam_info.image,
        image_name=cam_info.image_name,
        uid=cam_info.uid,
        dicom_frame=getattr(cam_info, "dicom_frame", None),
        data_device=data_device,
    )


def cameraList_from_camInfos(cam_infos, args=None):
    """Convert list of CameraInfo to list of Camera objects."""
    return [loadCam(args, c) for c in cam_infos]
