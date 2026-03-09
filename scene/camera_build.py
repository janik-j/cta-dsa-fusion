"""Camera construction helpers for dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from tqdm import tqdm

from utils.graphics_utils import focal2fov, get_extrinsic


class CameraInfo(NamedTuple):
    uid: int
    timestamp: float
    R: np.array
    T: np.array
    PrimaryAngle: np.array
    SecondaryAngle: np.array
    sad: np.array
    sid: np.array
    FovY: np.array
    FovX: np.array
    sx: np.array
    sy: np.array
    near: np.array
    far: np.array
    image: np.array
    image_name: str
    width: int
    height: int
    dicom_frame: int | None = None


def get_near_far(sad: float, volume_phy: np.ndarray) -> tuple[float, float]:
    dist = np.linalg.norm([volume_phy[0] / 2, volume_phy[1] / 2])
    near = float(sad - dist)
    far = float(sad + dist)
    return near, far


@dataclass(frozen=True)
class DefaultGeometry:
    sad: float
    sid: float
    width: int
    height: int
    fov_x: float
    fov_y: float
    sx: float
    sy: float
    near: float
    far: float


def default_geometry(camera_paras: dict[str, Any], volume_phy: np.ndarray) -> DefaultGeometry:
    sad = float(camera_paras["sad"])
    sid = float(camera_paras["sid"])
    width, height = camera_paras["proj_resolution"]
    dx, dy = camera_paras["proj_phy"]
    fov_x = float(focal2fov(sid, dx))
    fov_y = float(focal2fov(sid, dy))
    sx, sy = camera_paras["proj_spacing"]
    near, far = get_near_far(sad, volume_phy)
    return DefaultGeometry(
        sad=sad,
        sid=sid,
        width=int(width),
        height=int(height),
        fov_x=fov_x,
        fov_y=fov_y,
        sx=float(sx),
        sy=float(sy),
        near=float(near),
        far=float(far),
    )


def _as_int_or_none(value: object) -> int | None:
    return int(value) if value is not None else None


def _world2cam_from_frame(frame: dict[str, Any]) -> np.ndarray | None:
    w2c = frame.get("world2cam")
    if w2c is None:
        return None
    arr = np.asarray(w2c, dtype=np.float32)
    if arr.shape != (4, 4):
        raise ValueError("frame.world2cam must be 4x4")
    return arr


def frame_extrinsics_and_geometry(
    frame: dict[str, Any],
    *,
    pose_type: str,
    defaults: DefaultGeometry,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    pt = str(pose_type).strip().lower()
    if pt == "world2cam_4x4":
        w2c = _world2cam_from_frame(frame)
        if w2c is not None:
            r = w2c[:3, :3]
            t = w2c[:3, 3]
        else:
            r = np.asarray(frame["R"], dtype=np.float32)
            t = np.asarray(frame["T"], dtype=np.float32)

        sad = float(frame.get("sad", defaults.sad))
        sid = float(frame.get("sid", defaults.sid))
        width = int(frame.get("width", defaults.width))
        height = int(frame.get("height", defaults.height))
        fov_x = float(frame.get("FovX", defaults.fov_x))
        fov_y = float(frame.get("FovY", defaults.fov_y))
        proj_spacing = frame.get("proj_spacing", [defaults.sx, defaults.sy])
        sx = float(proj_spacing[0])
        sy = float(proj_spacing[1])
        near = float(frame.get("near", defaults.near))
        far = float(frame.get("far", defaults.far))
        primary_angle = float(frame.get("PrimaryAngle", 0.0))
        secondary_angle = float(frame.get("SecondaryAngle", 0.0))
        return (
            r,
            t,
            primary_angle,
            secondary_angle,
            sad,
            sid,
            width,
            height,
            fov_x,
            fov_y,
            sx,
            sy,
            near,
            far,
        )

    if pt != "angles_rad":
        raise ValueError(f"Unsupported pose_type={pose_type!r} (expected: angles_rad|world2cam_4x4)")

    primary_angle = float(frame["PrimaryAngle"])
    secondary_angle = -float(frame.get("SecondaryAngle", 0.0))
    extr = get_extrinsic(primary_angle, secondary_angle, defaults.sad)
    r = extr[:3, :3]
    t = extr[:3, 3]
    return (
        r,
        t,
        primary_angle,
        secondary_angle,
        defaults.sad,
        defaults.sid,
        defaults.width,
        defaults.height,
        defaults.fov_x,
        defaults.fov_y,
        defaults.sx,
        defaults.sy,
        defaults.near,
        defaults.far,
    )


def build_camera_infos(
    *,
    frames: list[dict[str, Any]],
    proj: np.ndarray,
    pose_type: str,
    defaults: DefaultGeometry,
    n_views: int,
) -> list[CameraInfo]:
    cam_infos: list[CameraInfo] = []
    if proj.shape[0] < n_views:
        raise ValueError(f"DSA has {proj.shape[0]} views but transforms.json declares N_views={n_views}")

    for idx in tqdm(range(n_views), desc="camera loading"):
        frame = frames[idx]
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{idx}] must be an object in transforms.json")

        (
            R,
            T,
            PrimaryAngle,
            SecondaryAngle,
            sad,
            sid,
            width,
            height,
            FovX,
            FovY,
            sx,
            sy,
            near,
            far,
        ) = frame_extrinsics_and_geometry(frame, pose_type=pose_type, defaults=defaults)

        uid = idx
        # Upstream convention: assign a normalized timestamp based on view order.
        # Frame can override with an explicit "timestamp" field if needed.
        default_ts = float(uid) / float(max(int(n_views), 1))
        timestamp = float(frame.get("timestamp", default_ts))
        dicom_frame = _as_int_or_none(frame.get("dicom_frame", frame.get("dicom_frame_index", None)))

        image = proj[idx][np.newaxis, :, :]
        image_name = frame["file"]

        cam_infos.append(
            CameraInfo(
                uid=uid,
                timestamp=timestamp,
                R=R,
                T=T,
                PrimaryAngle=PrimaryAngle,
                SecondaryAngle=SecondaryAngle,
                sad=sad,
                sid=sid,
                FovY=FovY,
                FovX=FovX,
                sx=sx,
                sy=sy,
                near=near,
                far=far,
                image=image,
                image_name=image_name,
                width=width,
                height=height,
                dicom_frame=dicom_frame,
            )
        )

    return cam_infos
