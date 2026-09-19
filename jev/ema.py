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

    def state_dict(self) -> dict[str, Any]:
        """Return a serialization-safe copy of the shadow adapter."""
        out: dict[str, Any] = {}
        for key, value in self.values.items():
            out[key] = value.detach().cpu().clone() if hasattr(value, "detach") else float(value)
        return out

    def restore(self, current: dict[str, Any], values: dict[str, Any], updates: int) -> None:
        """Restore a checkpointed EMA after validating the live adapter shape.

        EMA tensors are moved to the live parameter device so publication can
        apply them without an implicit cross-device copy.  The key and shape
        checks prevent accidentally loading an adapter from another LoRA
        configuration.
        """
        if updates < 0:
            raise ValueError("EMA update count cannot be negative")
        if set(current) != set(values):
            raise ValueError("EMA checkpoint keys do not match adapter parameters")
        restored: dict[str, Any] = {}
        for key, live in current.items():
            value = values[key]
            if hasattr(live, "detach"):
                import torch
                if not hasattr(value, "shape") or value.shape != live.shape:
                    raise ValueError(f"EMA checkpoint shape mismatch for {key}")
                value = value.detach().to(device=live.device, dtype=torch.float32).clone()
                if not torch.isfinite(value).all():
                    raise ValueError("EMA checkpoint contains non-finite values")
            elif not math.isfinite(float(value)):
                raise ValueError("EMA checkpoint contains non-finite values")
            restored[key] = value
        self.values = restored
        self.updates = int(updates)
