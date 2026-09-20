# Collapse detection and recovery runbook

This runbook covers the frozen §1.1 collapse guard (G5, items ZJ-024/ZJ-025/
ZJ-033). The control-plane implementation is `jev/drift.py` (`DriftMonitor`)
driven by `JevRuntime.observe_metrics`; the reset transaction is
`ContinuousLoRATrainer._reset_model` in `jev/training.py`.

## 1. Detection thresholds

A collapse event is raised when any signal persists for `consecutive_windows=3`
(`drift.min_windows`). Every threshold below is a config field with a checked-in
example in `configs/example.yaml`; changing the config starts a new lineage.

| §1.1 condition | Config field(s) | Stage 0 default |
|---|---|---|
| (a) entity/span F1 below baseline by absolute drop | `drift.f1_absolute_drop` | `0.15` |
| (a) relative F1 drop of 20% (F1 at or below 80% of baseline) | `drift.f1_relative_floor` | `0.80` |
| (b) validation loss above baseline | `drift.validation_loss_ratio` | `2.0` |
| (b) rolling training-loss window above baseline | `drift.loss_ratio` (+ `drift.loss_margin`) | `2.0` (+ `0.0`) |
| (c) NaN/Inf or invalid span/schema output | `drift.max_non_finite` | `0` (a single non-finite weight/loss is terminal) |
| (d) confidence calibration/coverage floor | `confidence_ok` signal; `drift.ema_norm_ratio` | `3.0` for EMA norm |
| Window length | `drift.eval_window` | `256` |
| Consecutive windows before reset | `drift.min_windows` | `3` |
| Minimum steps between resets | `drift.reset_cooldown_steps` | `100` |

Signals are evaluated when a full `eval_window` of finite losses has been
collected. `invalid_output` and `confidence_ok` are windowed flags; the loss,
F1, validation-loss, and EMA-norm checks are numeric. A reset clears the
baseline and window so the fresh lineage re-establishes its own baseline.

## 2. Reset transaction

When the guard raises `reset_required`, the runtime synchronously invokes the
trainer's reset:

1. **Stop the adapter update.** The current step is abandoned; no new
   publication occurs from the failed lineage.
2. **Archive before replacing.** `save_checkpoint(archive_reason=...)` writes
   the failing adapter, EMA, optimizer state, counters, and `failure_reason` to
   `runs/jev/archives/reset-<reset_id>-step-<step>-<epoch>-<uuid>.pt`.
3. **Increment the reset id.** `runtime.stats.reset_id` increases
   monotonically; the previous serving snapshot keeps answering requests.
4. **Rebuild from the immutable base.** The trainer constructs base + a fresh
   zeroed LoRA and a new EMA initialized from that state; the optimizer is
   discarded.
5. **Warm-up gate.** `runtime.reset_warmup_steps` clean steps must complete
   (and any configured warm-up evaluator must pass) before the new lineage is
   checkpointed and published. A failed evaluation restarts the warm-up window.
6. **Audit event.** `runs/jev/reset-events.jsonl` receives a JSON object with
   `reset_id`, `reason`, `step`, `archive`, and `warmup_steps`.

Reasons observed in practice include `non_finite_weights` (weights),
`non_finite_loss`, `loss_window_exceeded`, `f1_drop`,
`validation_loss_exceeded`, `invalid_output`, `confidence_coverage_floor`,
`ema_norm_exceeded`, and `non_finite_gradients` (gradient clip).

To inspect or trigger a reset manually:

```bash
python -m jev.cli metrics --checkpoint-dir runs/jev
python -m jev.cli reset --config configs/example.yaml --checkpoint-dir runs/jev --reason operator_requested
```

## 3. Recovery drill

Item ZJ-033 requires a collapse-and-recovery drill that proves the thresholds,
the archive, the reset id, and the warm-up gate. `scripts/recovery_drill.py` is
provided by the evaluation worker (ZJ-033); once present, the intended CLI is:

```bash
python3 scripts/recovery_drill.py \
  --config configs/example.yaml \
  --model-path /home/sansha/jev-model-base \
  --output artifacts/recovery/collapse_recovery.json
```

Expected evidence in `artifacts/recovery/collapse_recovery.json`:

- the injected degradation that triggered each guard signal and the before/
  after metric values;
- the archive checkpoint path under `runs/jev/archives/` and the incremented
  `reset_id`;
- the emitted `reset-events.jsonl` record;
- proof that the old snapshot kept serving during warm-up and that the fresh
  lineage was published only after the warm-up gate passed;
- the `--model-path` snapshot identity (revision and parameter count) used for
  the rebuild.

Run the drill on the designated NVIDIA GPU host (or as a faithful simulator
plus one hardware smoke) and keep the JSON next to the checkpoint directory.
The related GPU harness `scripts/smoke_nvidia_gpu.py` exercises the same
train/EMA/publish path and writes
`artifacts/nvidia_gpu_train_ema_smoke.json`.

## 4. Operator checklist after a reset

1. Confirm `python -m jev.cli metrics --checkpoint-dir runs/jev` shows the new
   `reset_id`, the archive, and the reset event.
2. Confirm the serving snapshot still reports the previous generation until the
   next warm-up publication.
3. Inspect the archive with the training owner before deleting anything;
   archives are never pruned automatically.
4. If resets recur, treat the data/teacher pipeline as suspect before lowering
   thresholds: the guard is intentionally fail-closed, and schema changes start
   a new lineage rather than reset an adapter.
