from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from .config import JevConfig
from .drift import DriftMonitor, DriftState
from .ema import EMAState


@dataclass
class RuntimeStats:
    inference_calls: int = 0
    training_steps: int = 0
    model_generation: int = 0
    last_drift: DriftState | None = None
    reset_id: int = 0


class JevRuntime:
    """Thread-safe publication boundary between continuous training and inference.

    The trainer may update its LoRA model outside the lock. Only a completed
    optimizer step is published under the lock, so inference never sees a
    half-written adapter. EMA tensors are copied to the inference model by the
    supplied publisher callback at publication time.
    """

    def __init__(self, config: JevConfig, infer: Callable[[Any, str], Any] | None = None, reset: Callable[[str], None] | None = None, apply_ema: Callable[[Any, dict[str, Any]], None] | None = None):
        self.config = config
        self._lock = threading.RLock()
        self._model: Any = None
        self._infer = infer or self._default_infer
        self._reset = reset
        self._apply_ema = apply_ema
        self.ema = EMAState()
        self.stats = RuntimeStats()
        self.monitor = DriftMonitor(config.drift, self._on_reset)

    def _default_infer(self, model: Any, text: str) -> Any:
        if model is None:
            raise RuntimeError("model is not loaded")
        return model.extract(text, self.config.schema.as_teacher_contract())

    def publish(self, model: Any, lora_state: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._model = model
            if lora_state:
                self.ema.update(lora_state)
                if self._apply_ema:
                    self._apply_ema(model, dict(self.ema.values))
            self.stats.model_generation += 1

    @contextmanager
    def training_transaction(self):
        """Keep inference from observing an in-flight optimizer mutation."""
        with self._lock:
            yield

    def infer(self, text: str) -> Any:
        with self._lock:
            model = self._model
            result = self._infer(model, text)
            self.stats.inference_calls += 1
            return result

    def observe_training(self, loss: float, lora_state: dict[str, Any] | None = None) -> DriftState:
        return self.observe_metrics(loss, lora_state=lora_state)

    def observe_metrics(self, loss: float, lora_state: dict[str, Any] | None = None, **metrics: Any) -> DriftState:
        with self._lock:
            if lora_state:
                self.ema.update(lora_state)
            self.stats.training_steps += 1
            state = self.monitor.observe(loss, self.ema.norm(), **metrics)
            self.stats.last_drift = state
            if self._apply_ema and self._model is not None:
                self._apply_ema(self._model, dict(self.ema.values))
            return state

    def set_reset_handler(self, callback: Callable[[str], None]) -> None:
        self._reset = callback

    def _on_reset(self, reason: str) -> None:
        if self._reset:
            self._reset(reason)
