from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import MutableMapping


@dataclass
class EMAState:
    """Numerically stable exponential moving average for LoRA tensors/metrics."""

    decay: float = 0.999
    values: MutableMapping[str, object] = field(default_factory=dict)
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.decay < 1:
            raise ValueError("EMA decay must be between 0 and 1")

    def update(self, current: MutableMapping[str, object]) -> None:
        self.updates += 1
        for key, value in current.items():
            old = self.values.get(key)
            if old is None:
                self.values[key] = value.detach().clone() if hasattr(value, "detach") else value
            elif hasattr(value, "detach") and hasattr(old, "mul_"):
                old.mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.values[key] = self.decay * old + (1 - self.decay) * value

    def norm(self) -> float:
        total = 0.0
        for value in self.values.values():
            if hasattr(value, "detach"):
                total += float(value.detach().float().pow(2).sum().item())
            else:
                total += float(value) ** 2
        return math.sqrt(total)

