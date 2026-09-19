from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import JevConfig
from .ema import EMAState
from .model import apply_lora, gliner2_step, load_gliner
from .runtime import JevRuntime


@dataclass(frozen=True)
class TrainEvent:
    step: int
    loss: float
    model_generation: int
    reset_required: bool
    reset_reason: str | None
    checkpoint: str | None = None


class ContinuousLoRATrainer:
    """Online adapter trainer with EMA, durable checkpoints and reset gates.

    Only parameters marked trainable by PEFT are persisted. The immutable
    GLiNER2 base is reconstructed by ``model_factory`` on resume and after a
    collapse reset. Serving keeps the last accepted snapshot while a reset
    adapter completes its configurable warm-up period.
    """

    def __init__(
        self,
        config: JevConfig,
        runtime: JevRuntime,
        step_fn: Callable[[Any, dict[str, Any]], Any] | None = None,
        model_factory: Callable[[], Any] | None = None,
        checkpoint_dir: str | Path = "runs/jev",
        resume_from: str | Path | None = None,
        warmup_evaluator: Callable[[Any], bool] | None = None,
    ):
        self.config, self.runtime = config, runtime
        self.step_fn = step_fn or (lambda model, example: gliner2_step(model, config, example))
        self.model_factory = model_factory or (lambda: apply_lora(load_gliner(config), config))
        self.checkpoint_dir = Path(checkpoint_dir)
        self.warmup_evaluator = warmup_evaluator
        self._warmup_remaining = 0
        self._warmup_pending = False
        self.model = self.model_factory()
        if hasattr(self.model, "train"):
            self.model.train()
        self.runtime.set_reset_handler(self._reset_model)
        # The trainer owns the configured EMA policy even when a runtime was
        # constructed before the config was parsed.
        self.runtime.ema = EMAState(decay=config.runtime.ema_decay)
        self._optimizer = None
        if resume_from is not None:
            self.resume(resume_from)
        else:
            self.runtime.ema.initialize(self._state())
            self.runtime.publish(self.model, dict(self.runtime.ema.values), ema_step=0)

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
            self._optimizer = torch.optim.AdamW(params, lr=self.config.runtime.learning_rate)
        return self._optimizer

    @staticmethod
    def _finite_gradients(parameters: list[Any]) -> bool:
        import torch

        return all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in parameters
        )

    def _clip_gradients(self, optimizer: Any) -> float | None:
        """Validate gradients and apply the configured global norm clip."""
        import torch

        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        if not self._finite_gradients(parameters):
            raise FloatingPointError("non-finite LoRA gradient")
        max_norm = self.config.runtime.grad_clip_norm
        if max_norm is None:
            return None
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        norm_value = float(norm.detach().item() if hasattr(norm, "detach") else norm)
        if not torch.isfinite(torch.as_tensor(norm_value)):
            raise FloatingPointError("non-finite LoRA gradient norm")
        return norm_value

    @staticmethod
    def _cpu_copy(value: Any) -> Any:
        return value.detach().cpu().clone() if hasattr(value, "detach") else float(value)

    def _checkpoint_payload(self, *, reason: str | None = None) -> dict[str, Any]:
        adapter = {key: self._cpu_copy(value) for key, value in self._state().items()}
        payload: dict[str, Any] = {
            "format": 1,
            "kind": "adapter-only",
            "config_digest": self.config.digest(),
            "schema_digest": self.config.schema.digest(),
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "adapter_state": adapter,
            "ema_state": self.runtime.ema.state_dict(),
            "ema_decay": self.runtime.ema.decay,
            "ema_updates": self.runtime.ema.updates,
            "training_steps": self.runtime.stats.training_steps,
            "reset_id": self.runtime.stats.reset_id,
            "created_at": time.time(),
        }
        if self._optimizer is not None:
            payload["optimizer_state"] = self._optimizer.state_dict()
        if reason is not None:
            payload["failure_reason"] = reason
        return payload

    def save_checkpoint(self, *, archive_reason: str | None = None) -> Path:
        """Atomically persist adapter, EMA and optimizer state."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("checkpoint persistence requires torch") from exc
        target_dir = self.checkpoint_dir / "archives" if archive_reason else self.checkpoint_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        if archive_reason:
            filename = (
                f"reset-{self.runtime.stats.reset_id:06d}-"
                f"step-{self.runtime.stats.training_steps:08d}-"
                f"{int(time.time())}-{uuid.uuid4().hex[:8]}.pt"
            )
        else:
            filename = f"adapter-step-{self.runtime.stats.training_steps:08d}.pt"
        target = target_dir / filename
        fd, temporary = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp", dir=target_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                torch.save(self._checkpoint_payload(reason=archive_reason), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

        if not archive_reason:
            latest = self.checkpoint_dir / "latest.pt"
            fd, temporary = tempfile.mkstemp(prefix=".latest.", suffix=".tmp", dir=self.checkpoint_dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save(self._checkpoint_payload(), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, latest)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            files = sorted(
                self.checkpoint_dir.glob("adapter-step-*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for stale in files[self.config.runtime.max_checkpoints :]:
                stale.unlink(missing_ok=True)
        return target

    def _resolve_checkpoint(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_dir():
            candidate = candidate / "latest.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"checkpoint not found: {candidate}")
        return candidate

    def resume(self, path: str | Path) -> dict[str, Any]:
        """Restore a checkpoint into a freshly constructed base + LoRA model."""
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("checkpoint resume requires torch") from exc
        checkpoint = self._resolve_checkpoint(path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or payload.get("kind") != "adapter-only":
            raise ValueError("unsupported Jev checkpoint")
        if payload.get("config_digest") != self.config.digest():
            raise ValueError("checkpoint config digest does not match current config")
        state = self._state()
        adapter = payload.get("adapter_state", {})
        if set(adapter) != set(state):
            raise ValueError("checkpoint adapter keys do not match current LoRA model")
        with torch.no_grad():
            for key, parameter in state.items():
                value = adapter[key]
                if value.shape != parameter.shape:
                    raise ValueError(f"checkpoint shape mismatch for {key}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.runtime.ema = EMAState(decay=float(payload.get("ema_decay", self.config.runtime.ema_decay)))
        self.runtime.ema.restore(state, payload.get("ema_state", {}), int(payload.get("ema_updates", 0)))
        self.runtime.stats.training_steps = int(payload.get("training_steps", 0))
        self.runtime.stats.reset_id = int(payload.get("reset_id", 0))
        self._optimizer = self._ensure_optimizer()
        if payload.get("optimizer_state"):
            self._optimizer.load_state_dict(payload["optimizer_state"])
        self.runtime.publish(
            self.model,
            dict(self.runtime.ema.values),
            ema_step=self.runtime.ema.updates,
            checkpoint=str(checkpoint),
        )
        return {
            "path": str(checkpoint),
            "training_steps": self.runtime.stats.training_steps,
            "ema_updates": self.runtime.ema.updates,
            "reset_id": self.runtime.stats.reset_id,
        }

    def _reset_model(self, reason: str) -> None:
        # Archive the failed lineage before replacing any state. The old
        # serving snapshot remains live until the warm-up adapter is promoted.
        archive_path = self.save_checkpoint(archive_reason=reason)
        self.runtime.stats.reset_id += 1
        self.runtime.ema = EMAState(decay=self.config.runtime.ema_decay)
        self.model = self.model_factory()
        if hasattr(self.model, "train"):
            self.model.train()
        self._optimizer = None
        self.runtime.ema.initialize(self._state())
        self._warmup_remaining = self.config.runtime.reset_warmup_steps
        self._warmup_pending = self._warmup_remaining > 0
        if not self._warmup_pending:
            self.runtime.publish(self.model, dict(self.runtime.ema.values), ema_step=0)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (self.checkpoint_dir / "reset-events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "reset_id": self.runtime.stats.reset_id,
                        "reason": reason,
                        "step": self.runtime.stats.training_steps,
                        "archive": str(archive_path),
                        "warmup_steps": self.config.runtime.reset_warmup_steps,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _advance_warmup(self) -> Path | None:
        if not self._warmup_pending:
            return None
        self._warmup_remaining -= 1
        if self._warmup_remaining > 0:
            return None
        if self.warmup_evaluator is not None and not self.warmup_evaluator(self.model):
            self._warmup_remaining = max(1, self.config.runtime.reset_warmup_steps)
            return None
        checkpoint = self.save_checkpoint()
        self.runtime.publish(
            self.model,
            dict(self.runtime.ema.values),
            ema_step=self.runtime.ema.updates,
            checkpoint=str(checkpoint),
        )
        self._warmup_pending = False
        return checkpoint

    def train(self, examples: Iterable[dict[str, Any]], max_steps: int | None = None) -> list[TrainEvent]:
        optimizer = self._ensure_optimizer()
        if hasattr(self.model, "train"):
            self.model.train()
        events: list[TrainEvent] = []
        for local_step, example in enumerate(examples, start=1):
            if max_steps is not None and local_step > max_steps:
                break
            step = self.runtime.stats.training_steps + 1
            with self.runtime.training_transaction():
                optimizer.zero_grad(set_to_none=True)
                loss_obj = self.step_fn(self.model, example)
                loss_obj.backward()
                try:
                    self._clip_gradients(optimizer)
                except FloatingPointError:
                    optimizer.zero_grad(set_to_none=True)
                    self._reset_model("non_finite_gradients")
                    events.append(
                        TrainEvent(
                            step,
                            float("nan"),
                            self.runtime.stats.model_generation,
                            True,
                            "non_finite_gradients",
                        )
                    )
                    optimizer = self._ensure_optimizer()
                    continue
                optimizer.step()
                value = float(loss_obj.detach().item())
                state = self.runtime.observe_training(value, self._state())
                checkpoint_path: Path | None = None
                if not state.reset_required:
                    if step % self.config.runtime.checkpoint_every_steps == 0:
                        checkpoint_path = self.save_checkpoint()
                    if self._warmup_pending:
                        checkpoint_path = self._advance_warmup() or checkpoint_path
                    elif step % self.config.runtime.publish_every_steps == 0:
                        self.runtime.publish(
                            self.model,
                            dict(self.runtime.ema.values),
                            ema_step=self.runtime.ema.updates,
                            checkpoint=str(checkpoint_path) if checkpoint_path else None,
                        )
            event = TrainEvent(
                step,
                value,
                self.runtime.stats.model_generation,
                state.reset_required,
                state.reason,
                str(checkpoint_path) if checkpoint_path else None,
            )
            events.append(event)
            if state.reset_required:
                # DriftMonitor invokes the reset synchronously via runtime.
                optimizer = self._ensure_optimizer()
        self._save_metadata(events)
        return events

    def _save_metadata(self, events: list[TrainEvent]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_digest": self.config.digest(),
            "ema_decay": self.runtime.ema.decay,
            "events": [vars(x) for x in events],
        }
        fd, temporary = tempfile.mkstemp(prefix="jev-metrics-", dir=self.checkpoint_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, self.checkpoint_dir / "metrics.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class ContinuousLoRAStream:
    """Bounded background producer for training beside live inference.

    The trainer remains the sole writer of live LoRA/EMA state. ``submit``
    blocks when the replay queue is full, making backpressure explicit; the
    runtime's immutable snapshots continue serving inference concurrently.
    """

    _STOP = object()

    def __init__(self, trainer: ContinuousLoRATrainer, max_queue_size: int = 128):
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.trainer = trainer
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._error: BaseException | None = None
        self.events: list[TrainEvent] = []

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> "ContinuousLoRAStream":
        if self.running or self._thread is not None:
            raise RuntimeError("stream has already started")
        self._thread = threading.Thread(target=self._run, name="jev-lora-trainer", daemon=True)
        self._thread.start()
        return self

    def submit(self, example: dict[str, Any], *, timeout: float | None = None) -> None:
        if self._thread is None or self._stop_requested:
            raise RuntimeError("stream is not accepting examples")
        self._queue.put(example, timeout=timeout)

    def stop(self, *, wait: bool = True, timeout: float | None = None) -> list[TrainEvent]:
        if self._thread is None:
            return list(self.events)
        self._stop_requested = True
        self._queue.put(self._STOP, timeout=timeout)
        if wait:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                raise TimeoutError("timed out waiting for Jev training stream")
            if self._error is not None:
                raise RuntimeError("Jev training stream failed") from self._error
        return list(self.events)

    def _examples(self):
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            yield item

    def _run(self) -> None:
        try:
            self.events = self.trainer.train(self._examples())
        except BaseException as exc:  # surfaced by stop()
            self._error = exc
