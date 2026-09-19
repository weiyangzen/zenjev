from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import JevConfig
from .model import apply_lora, load_gliner
from .runtime import JevRuntime


@dataclass(frozen=True)
class TrainEvent:
    step: int
    loss: float
    model_generation: int
    reset_required: bool
    reset_reason: str | None


class ContinuousLoRATrainer:
    """Online adapter trainer with atomic publication, EMA and hard reset gate.

    `step_fn` owns GLiNER2's task-specific loss construction. It receives the
    live PEFT model and one JSONL example, and returns a scalar torch loss. A
    caller can use the same function with a replay buffer or a stream of newly
    distilled examples. No checkpoint is published until the optimizer step,
    EMA update and drift decision have completed.
    """

    def __init__(self, config: JevConfig, runtime: JevRuntime, step_fn: Callable[[Any, dict[str, Any]], Any], model_factory: Callable[[], Any] | None = None, checkpoint_dir: str | Path = "runs/jev"):
        self.config, self.runtime, self.step_fn = config, runtime, step_fn
        self.model_factory = model_factory or (lambda: apply_lora(load_gliner(config), config))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model = self.model_factory()
        self.runtime.set_reset_handler(self._reset_model)
        self.runtime.publish(self.model, self._state())
        self._optimizer = None

    def _state(self) -> dict[str, Any]:
        return {name: p for name, p in self.model.named_parameters() if p.requires_grad}

    def _ensure_optimizer(self) -> Any:
        if self._optimizer is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("continuous training requires torch") from exc
            params = [p for p in self.model.parameters() if p.requires_grad]
            if not params:
                raise RuntimeError("no trainable LoRA parameters found")
            self._optimizer = torch.optim.AdamW(params, lr=1e-4)
        return self._optimizer

    def _reset_model(self, reason: str) -> None:
        self.model = self.model_factory()
        self._optimizer = None
        self.runtime.publish(self.model, self._state())
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.checkpoint_dir / "reset-events.jsonl").open("a", encoding="utf-8").write(json.dumps({"reason": reason, "step": self.runtime.stats.training_steps}) + "\n")

    def train(self, examples: Iterable[dict[str, Any]], max_steps: int | None = None) -> list[TrainEvent]:
        optimizer = self._ensure_optimizer()
        events: list[TrainEvent] = []
        for step, example in enumerate(examples, start=1):
            if max_steps is not None and step > max_steps:
                break
            optimizer.zero_grad(set_to_none=True)
            loss_obj = self.step_fn(self.model, example)
            loss_obj.backward()
            optimizer.step()
            value = float(loss_obj.detach().item())
            state = self.runtime.observe_training(value, self._state())
            if step % self.config.runtime.publish_every_steps == 0 or state.reset_required:
                self.runtime.publish(self.model, self._state())
            event = TrainEvent(step, value, self.runtime.stats.model_generation, state.reset_required, state.reason)
            events.append(event)
            if state.reset_required:
                # DriftMonitor invokes this synchronously via runtime callback.
                optimizer = self._ensure_optimizer()
        self._save_metadata(events)
        return events

    def _save_metadata(self, events: list[TrainEvent]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {"config_digest": self.config.digest(), "events": [vars(x) for x in events]}
        fd, temp = tempfile.mkstemp(prefix="jev-metrics-", dir=self.checkpoint_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temp, self.checkpoint_dir / "metrics.json")
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
