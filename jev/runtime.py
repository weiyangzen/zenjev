from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
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


@dataclass
class Snapshot:
    model: Any
    generation: int
    ema_step: int
    reset_id: int
    checkpoint: str | None
    # GLiNER2's processor can have request-local mutable state. Serialize requests
    # on a snapshot, while training and building its replacement remain independent.
    request_lock: Any = field(default_factory=threading.Lock)


class JevRuntime:
    """Serve immutable accepted snapshots independently of the training model."""

    def __init__(self, config: JevConfig, infer: Callable[[Any, str], Any] | None = None,
                 reset: Callable[[str], None] | None = None,
                 apply_ema: Callable[[Any, dict[str, Any]], None] | None = None):
        self.config = config
        self._lock = threading.RLock()
        self._snapshot: Snapshot | None = None
        self._infer = infer or self._default_infer
        self._reset = reset
        self._apply_ema = apply_ema
        self.ema = EMAState(decay=config.runtime.ema_decay)
        self.stats = RuntimeStats()
        self.monitor = DriftMonitor(config.drift, self._on_reset)

    def _default_infer(self, model: Any, text: str) -> Any:
        from .model import extract
        return extract(model, self.config, text)

    def publish(self, model: Any, lora_state: dict[str, Any] | None = None,
                *, ema_step: int | None = None, checkpoint: str | None = None) -> None:
        """Copy before swapping. Publication never mutates the training weights.

        The caller owns the training model during the copy; training has exactly
        one writer. Existing requests keep a reference to their old snapshot.
        lora_state is an already averaged state, not another EMA update.
        """
        candidate = copy.deepcopy(model)
        if lora_state is not None:
            if self._apply_ema:
                self._apply_ema(candidate, lora_state)
            elif hasattr(candidate, "named_parameters"):
                import torch
                parameters = dict(candidate.named_parameters())
                with torch.no_grad():
                    for key, value in lora_state.items():
                        parameters[key].copy_(value)
        if hasattr(candidate, "eval"):
            candidate.eval()
        with self._lock:
            generation = self.stats.model_generation + 1
            self._snapshot = Snapshot(candidate, generation,
                                      self.ema.updates if ema_step is None else ema_step,
                                      self.stats.reset_id, checkpoint)
            self.stats.model_generation = generation

    @contextmanager
    def training_transaction(self):
        """Training mutates only its private model; serving uses snapshots."""
        yield

    def infer_result(self, text: str) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("no accepted inference snapshot")
        with snapshot.request_lock:
            if hasattr(snapshot.model, "parameters"):
                import torch
                with torch.inference_mode():
                    output = self._infer(snapshot.model, text)
            else:
                output = self._infer(snapshot.model, text)
        with self._lock:
            self.stats.inference_calls += 1
        return {"output": output, "generation": snapshot.generation,
                "ema_step": snapshot.ema_step, "reset_id": snapshot.reset_id,
                "checkpoint": snapshot.checkpoint, "schema_digest": self.config.schema.digest(),
                "config_digest": self.config.digest(),
                "model_id": self.config.model_id}

    def infer(self, text: str) -> Any:
        return self.infer_result(text)["output"]

    def observe_training(self, loss: float, lora_state: dict[str, Any] | None = None) -> DriftState:
        return self.observe_metrics(loss, lora_state=lora_state)

    def observe_metrics(self, loss: float, lora_state: dict[str, Any] | None = None,
                        **metrics: Any) -> DriftState:
        # Called by the single training writer. No serving model is touched.
        weights_finite = bool(metrics.pop("weights_finite", True))
        if lora_state:
            weights_finite = weights_finite and self._state_is_finite(lora_state)
            # Let the drift gate decide whether a non-finite adapter requires a
            # reset. EMAState intentionally rejects NaN/Inf, so do not call it
            # with a corrupt state before the gate has observed the signal.
            if weights_finite:
                self.ema.update(lora_state)
        self.stats.training_steps += 1
        state = self.monitor.observe(loss, self.ema.norm(), weights_finite=weights_finite, **metrics)
        self.stats.last_drift = state
        return state

    @staticmethod
    def _state_is_finite(state: dict[str, Any]) -> bool:
        for value in state.values():
            if hasattr(value, "detach"):
                import torch
                if not bool(torch.isfinite(value.detach()).all()):
                    return False
            else:
                import math
                if not math.isfinite(float(value)):
                    return False
        return True

    def set_reset_handler(self, callback: Callable[[str], None]) -> None:
        self._reset = callback

    def _on_reset(self, reason: str) -> None:
        if self._reset:
            self._reset(reason)
