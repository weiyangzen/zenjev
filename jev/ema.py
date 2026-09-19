from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EMAState:
    """EMA of adapter parameters only. Initialize once, update once per step."""
    decay: float = 0.999
    values: dict[str, Any] = field(default_factory=dict)
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")

    @staticmethod
    def _clone(value: Any) -> Any:
        return value.detach().float().clone() if hasattr(value, "detach") else float(value)

    def initialize(self, current: dict[str, Any]) -> None:
        self.values = {key: self._clone(value) for key, value in current.items()}
        self.updates = 0
        self._validate(current)

    def _validate(self, current: dict[str, Any]) -> None:
        if set(current) != set(self.values):
            raise ValueError("EMA adapter parameter keys changed")
        for key, value in current.items():
            if hasattr(value, "detach"):
                import torch
                if value.shape != self.values[key].shape or not torch.isfinite(value).all():
                    raise ValueError("EMA requires finite tensors with unchanged shapes")
            elif not math.isfinite(float(value)):
                raise ValueError("EMA requires finite values")

    def update(self, current: dict[str, Any]) -> None:
        if not self.values:
            self.initialize(current)
        self._validate(current)
        for key, value in current.items():
            old = self.values[key]
            if hasattr(value, "detach"):
                old.mul_(self.decay).add_(value.detach().float(), alpha=1 - self.decay)
            else:
                self.values[key] = self.decay * old + (1 - self.decay) * value
        self.updates += 1

    def norm(self) -> float:
        return math.sqrt(sum(float(value.detach().float().pow(2).sum().item())
                             if hasattr(value, "detach") else float(value) ** 2
                             for value in self.values.values()))
