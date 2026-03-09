"""Opacity field models used by this framework."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

_SELECT_FIELD_KEYS = {"identity_field", "single_3d_field", "dual_3d_field"}


def detect_field_type_from_state_dict(state_dict: dict) -> str:
    """Detect field type from saved state dict keys.

    Returns one of the keys in ``_SELECT_FIELD_KEYS``.
    """
    keys = set(state_dict.keys())

    # dual_3d_field has fine_grid + coarse_grid
    if any("fine_grid" in k for k in keys) and any("coarse_grid" in k for k in keys):
        return "dual_3d_field"

    # single_3d_field has grid key but no fine_grid
    if any("grid" in k for k in keys):
        return "single_3d_field"

    # identity_field has no parameters (empty state dict or only buffer keys)
    return "identity_field"


class IdentityField(nn.Module):
    """Identity mapping (static opacity only)."""

    def __init__(self, conf: dict[str, Any]):
        super().__init__()
        self.conf = conf

    def forward(self, opacity: torch.Tensor) -> torch.Tensor:
        return opacity


class Single3DField(nn.Module):
    """Single 3D hash-grid opacity field."""

    def __init__(self, conf: dict[str, Any]):
        super().__init__()
        import tinycudann as tcnn

        self.conf = conf
        self.grid = tcnn.Encoding(n_input_dims=3, encoding_config=conf["hash3dgrid"], dtype=torch.float32)
        self.net = tcnn.Network(
            n_input_dims=self.grid.n_output_dims,
            n_output_dims=1,
            network_config=conf["net"],
        )
        if hasattr(tcnn, "supports_jit_fusion"):
            self.grid.jit_fusion = tcnn.supports_jit_fusion()
            self.net.jit_fusion = tcnn.supports_jit_fusion()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        feat = self.grid(xyz)
        return self.net(feat).to(torch.float32)


class Dual3DField(nn.Module):
    """Two 3D hash-grids (fine + coarse) fused through MLP.

    Static equivalent of the original 4DRGS ``static_dynamic_field`` at t=0.
    Both grids operate on xyz only (no temporal dimension).
    """

    def __init__(self, conf: dict[str, Any]):
        super().__init__()
        import tinycudann as tcnn

        self.conf = conf
        self.fine_grid = tcnn.Encoding(n_input_dims=3, encoding_config=conf["hash3dgrid_fine"], dtype=torch.float32)
        self.coarse_grid = tcnn.Encoding(n_input_dims=3, encoding_config=conf["hash3dgrid_coarse"], dtype=torch.float32)
        self.net = tcnn.Network(
            n_input_dims=self.fine_grid.n_output_dims + self.coarse_grid.n_output_dims,
            n_output_dims=1,
            network_config=conf["net"],
        )
        if hasattr(tcnn, "supports_jit_fusion"):
            self.fine_grid.jit_fusion = tcnn.supports_jit_fusion()
            self.coarse_grid.jit_fusion = tcnn.supports_jit_fusion()
            self.net.jit_fusion = tcnn.supports_jit_fusion()

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        fine_feat = self.fine_grid(xyz)
        coarse_feat = self.coarse_grid(xyz)
        feat = torch.cat([fine_feat, coarse_feat], dim=1)
        return self.net(feat).to(torch.float32)


class Field(nn.Module):
    """Opacity field wrapper producing {static,final}_opacity tensors."""

    def __init__(self, conf: dict[str, Any]):
        super().__init__()
        select = str(conf.get("select_field", "")).strip()
        if select not in _SELECT_FIELD_KEYS:
            raise ValueError(f"select_field must be one of {_SELECT_FIELD_KEYS}, got {select!r}")
        self.select_field = select

        if select == "identity_field":
            self.model: nn.Module | None = IdentityField(conf.get("identity_field", {}))
        elif select == "single_3d_field":
            if not torch.cuda.is_available():
                raise OSError("single_3d_field requires CUDA (torch.cuda.is_available() is False).")
            self.model = Single3DField(conf["single_3d_field"])
        elif select == "dual_3d_field":
            if not torch.cuda.is_available():
                raise OSError("dual_3d_field requires CUDA (torch.cuda.is_available() is False).")
            self.model = Dual3DField(conf["dual_3d_field"])

    def forward(self, gaussians, recon_args: dict[str, Any]) -> dict[str, torch.Tensor]:
        xyz = gaussians.get_xyz
        static_opacity = gaussians.get_opacity

        if self.select_field == "identity_field":
            return {"final_opacity": static_opacity}

        if self.model is None:
            raise RuntimeError("Internal error: select_field requires a model but none was constructed.")

        device = xyz.device
        volume_origin = torch.as_tensor(recon_args["volume_origin"], dtype=torch.float32, device=device).reshape(3)
        volume_phy = torch.as_tensor(recon_args["volume_phy"], dtype=torch.float32, device=device).reshape(3)
        xyz_norm = (xyz - volume_origin) / volume_phy

        field_opacity = self.model(xyz_norm)
        return {"final_opacity": field_opacity}
