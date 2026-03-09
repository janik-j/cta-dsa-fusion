"""Loss orchestration for training (LossManager)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch

from training.losses.base import BaseLoss, RegularizationLoss, RenderingLoss
from training.losses.regularization import (
    PriorAnchoringLoss,
)
from training.losses.rendering import DSSIMLoss, L1Loss


class LossManager:
    """Unified loss manager for this framework training."""

    def __init__(self, total_iterations: int = 30000, log_interval: int = 1000):
        self.total_iterations = int(total_iterations)
        self.log_interval = int(log_interval)
        self._rendering_losses: OrderedDict[str, RenderingLoss] = OrderedDict()
        self._regularization_losses: OrderedDict[str, RegularizationLoss] = OrderedDict()

    def add_loss(self, name: str, loss: BaseLoss):
        loss.config.name = str(name)
        if isinstance(loss, RenderingLoss):
            self._rendering_losses[name] = loss
        elif isinstance(loss, RegularizationLoss):
            self._regularization_losses[name] = loss
        else:
            raise TypeError(f"Unknown loss type: {type(loss)}")
        if hasattr(loss, "set_total_iterations"):
            loss.set_total_iterations(self.total_iterations)

    def set_iteration(self, iteration: int):
        for loss in [*self._rendering_losses.values(), *self._regularization_losses.values()]:
            loss.set_iteration(iteration)

    def _extract_scalar_info(self, info: dict[str, Any] | None) -> dict[str, Any]:
        if not info:
            return {}
        skip = {"vis"}
        return {k: v for k, v in info.items() if k not in skip and not (isinstance(v, torch.Tensor) and v.numel() > 1)}

    def compute_rendering_loss(
        self,
        image: torch.Tensor,
        gt_image: torch.Tensor,
        iteration: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.set_iteration(iteration)
        total_loss = torch.tensor(0.0, device=image.device)
        loss_dict: dict[str, Any] = {}

        for _name, loss_fn in self._rendering_losses.items():
            loss_val, info = loss_fn(image, gt_image, iteration=iteration)
            total_loss = total_loss + loss_val

            loss_dict.update(self._extract_scalar_info(info))

        return total_loss, loss_dict

    def compute_regularization_loss(
        self,
        gaussians,
        prior_xyz: torch.Tensor | None,
        iteration: int,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.set_iteration(iteration)
        device = gaussians.get_xyz.device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: dict[str, Any] = {}

        for _name, loss_fn in self._regularization_losses.items():
            loss_val, info = loss_fn(
                gaussians,
                prior_xyz=prior_xyz,
                iteration=iteration,
                **kwargs,
            )
            total_loss = total_loss + loss_val

            loss_dict.update(self._extract_scalar_info(info))

        return total_loss, loss_dict

    def compute_total_loss(
        self,
        image: torch.Tensor,
        gt_image: torch.Tensor,
        gaussians,
        prior_xyz: torch.Tensor | None,
        iteration: int,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        render_loss, render_dict = self.compute_rendering_loss(image, gt_image, iteration)
        reg_loss, reg_dict = self.compute_regularization_loss(
            gaussians, prior_xyz, iteration, **kwargs
        )
        total_loss = render_loss + reg_loss

        loss_dict = {
            **render_dict,
            **reg_dict,
            "Ltotal": total_loss,
            "Lrender": render_loss,
            "Lreg": reg_loss,
        }

        self._log_diagnostics(loss_dict, iteration)
        return total_loss, loss_dict

    def _log_diagnostics(self, loss_dict: dict[str, Any], iteration: int) -> None:
        if iteration % self.log_interval != 0:
            return

        skip = {"Ltotal", "Lrender", "Lreg"}
        active = [
            f"{k}={v.item() if hasattr(v, 'item') else v:.6f}"
            for k, v in loss_dict.items()
            if k.startswith("L") and k not in skip
        ]
        if active:
            print(f"\n[Loss:Iter {iteration}] {' | '.join(active)}\n", flush=True)

    def get_active_losses(self, iteration: int) -> dict[str, list[str]]:
        self.set_iteration(iteration)

        def active_names(losses: OrderedDict) -> list[str]:
            return [name for name, loss in losses.items() if loss.is_active(iteration)]

        return {
            "rendering": active_names(self._rendering_losses),
            "regularization": active_names(self._regularization_losses),
        }

    @classmethod
    def from_opt_params(cls, opt_params, dataset_params=None) -> LossManager:
        def opt(name: str, default):
            return getattr(opt_params, name, default)

        manager = cls(
            total_iterations=opt("iterations", 30000),
            log_interval=opt("log_interval", 1000),
        )

        lambda_dssim = float(opt("lambda_dssim", 0.07))
        manager.add_loss("l1", L1Loss(weight=1.0 - lambda_dssim))
        manager.add_loss("dssim", DSSIMLoss(weight=lambda_dssim))

        if (lambda_prior := float(opt("lambda_prior_anchoring", 0.0))) > 0:
            manager.add_loss(
                "prior_anchoring",
                PriorAnchoringLoss(
                    weight=lambda_prior,
                    start_iter=opt("prior_anchoring_start_iter", 0),
                    anchor_radius=opt("prior_anchor_radius", 0.5),
                    k=opt("prior_anchor_k", 8),
                    sample_size=opt("prior_anchor_sample_size", 2048),
                ),
            )

        return manager

    def __repr__(self) -> str:
        lines = [
            "LossManager Configuration:",
            f"  Total iterations: {self.total_iterations}",
            "  Rendering Losses:",
            *[f"    - {loss}" for loss in self._rendering_losses.values()],
            "  Regularization Losses:",
            *[f"    - {loss}" for loss in self._regularization_losses.values()],
        ]
        if not self._rendering_losses and not self._regularization_losses:
            lines.append("  (no losses configured)")
        return "\n".join(lines)


__all__ = ["LossManager"]
