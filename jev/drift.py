from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .config import DriftPolicy


@dataclass(frozen=True)
class DriftState:
    step: int
    baseline_loss: float | None
    rolling_loss: float | None
    ema_norm: float
    consecutive_bad_windows: int
    resets: int
    reset_required: bool
    reason: str | None = None


class DriftMonitor:
    """Fail-closed LoRA health gate. A reset is required, never silently ignored."""

    def __init__(self, policy: DriftPolicy, on_reset: Callable[[str], None] | None = None):
        self.policy, self.on_reset = policy, on_reset
        self.losses: deque[float] = deque(maxlen=policy.eval_window)
        self.baseline_loss: float | None = None
        self.bad_windows = 0
        self.step = self.resets = 0
        self.last_reset_step = -policy.reset_cooldown_steps
        self.baseline_ema_norm: float | None = None
        self._reason: str | None = None

    def observe(
        self,
        loss: float,
        ema_norm: float,
        *,
        f1: float | None = None,
        baseline_f1: float | None = None,
        val_loss: float | None = None,
        baseline_val_loss: float | None = None,
        invalid_output: bool = False,
        confidence_ok: bool = True,
        weights_finite: bool = True,
    ) -> DriftState:
        self.step += 1
        reason = None
        if not weights_finite:
            reason = "non_finite_weights"
        elif loss != loss or loss in (float("inf"), float("-inf")):
            reason = "non_finite_loss"
        else:
            self.losses.append(float(loss))
            if self.baseline_ema_norm is None and ema_norm > 0:
                self.baseline_ema_norm = ema_norm
            if self.baseline_loss is None and len(self.losses) == self.policy.eval_window:
                self.baseline_loss = sum(self.losses) / len(self.losses)
            if self.baseline_loss and len(self.losses) == self.policy.eval_window:
                rolling = sum(self.losses) / len(self.losses)
                if rolling > self.baseline_loss * self.policy.loss_ratio + self.policy.loss_margin:
                    self.bad_windows += 1
                else:
                    self.bad_windows = 0
                if f1 is not None and baseline_f1 is not None and (f1 <= baseline_f1 - 0.15 or f1 <= baseline_f1 * 0.80):
                    self.bad_windows += 1
                    reason = "f1_drop"
                if val_loss is not None and baseline_val_loss is not None and val_loss > baseline_val_loss * 2.0:
                    self.bad_windows += 1
                    reason = reason or "validation_loss_exceeded"
                if invalid_output:
                    self.bad_windows += 1
                    reason = reason or "invalid_output"
                if not confidence_ok:
                    self.bad_windows += 1
                    reason = reason or "confidence_coverage_floor"
                if self.bad_windows >= self.policy.min_windows:
                    reason = reason or "loss_window_exceeded"
            if self.baseline_ema_norm and ema_norm > self.policy.ema_norm_ratio * self.baseline_ema_norm:
                reason = reason or "ema_norm_exceeded"
        self._reason = reason
        reset_required = reason is not None and self.step - self.last_reset_step >= self.policy.reset_cooldown_steps
        if reset_required:
            self.resets += 1
            self.last_reset_step = self.step
            self.bad_windows = 0
            self.baseline_loss = None
            self.baseline_ema_norm = None
            self.losses.clear()
            if self.on_reset:
                self.on_reset(reason)
        return DriftState(self.step, self.baseline_loss, sum(self.losses) / len(self.losses) if self.losses else None, ema_norm, self.bad_windows, self.resets, reset_required, reason)
