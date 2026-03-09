"""Base classes for all training losses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LossConfig:
    weight: float = 1.0
    start_iter: int = 0
    end_iter: int | None = None
    enabled: bool = True
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def is_active(self, iteration: int) -> bool:
        if not self.enabled or self.weight <= 0:
            return False
        if iteration < self.start_iter:
            return False
        return not (self.end_iter is not None and iteration > self.end_iter)


class BaseLoss(nn.Module, ABC):
    """Loss base class with scheduling and weight scaling."""

    def __init__(
        self,
        weight: float = 1.0,
        start_iter: int = 0,
        end_iter: int | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = LossConfig(
            weight=float(weight),
            start_iter=int(start_iter),
            end_iter=int(end_iter) if end_iter is not None else None,
            name=name or self.__class__.__name__,
            extra=kwargs,
        )
        self._iteration = 0

    @property
    def name(self) -> str:
        return str(self.config.name)

    @property
    def weight(self) -> float:
        return float(self.config.weight)

    @weight.setter
    def weight(self, value: float):
        self.config.weight = float(value)

    def is_active(self, iteration: int) -> bool:
        return self.config.is_active(int(iteration))

    def set_iteration(self, iteration: int):
        self._iteration = int(iteration)

    @abstractmethod
    def _compute(self, *args, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        ...

    def forward(self, *args, iteration: int | None = None, **kwargs) -> tuple[torch.Tensor, dict[str, Any]]:
        iter_val = int(iteration) if iteration is not None else int(self._iteration)
        if not self.is_active(iter_val):
            device = None
            if args and hasattr(args[0], "device"):
                device = args[0].device
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.tensor(0.0, device=device), {}

        raw_loss, info = self._compute(*args, **kwargs)
        weighted_loss = float(self.config.weight) * raw_loss

        loss_key = f"L{self.name.lower()}"
        info[loss_key] = raw_loss
        info[f"{loss_key}_weighted"] = weighted_loss
        return weighted_loss, info

    def __repr__(self) -> str:
        status = "active" if self.config.enabled else "disabled"
        schedule = f"start={self.config.start_iter}"
        if self.config.end_iter:
            schedule += f", end={self.config.end_iter}"
        return f"{self.__class__.__name__}(w={self.weight:.4f}, {schedule}, {status})"


class RenderingLoss(BaseLoss):
    @abstractmethod
    def _compute(
        self,
        image: torch.Tensor,
        gt_image: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ...


class RegularizationLoss(BaseLoss):
    @abstractmethod
    def _compute(
        self,
        gaussians,
        prior_xyz: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ...


__all__ = ["BaseLoss", "LossConfig", "RegularizationLoss", "RenderingLoss"]
