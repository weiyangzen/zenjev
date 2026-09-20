# Training and serving runbook

Operator surface for the continual LoRA/EMA loop (ZJ-040, ZJ-021/ZJ-022,
ZJ-024/ZJ-025). All commands run from the repository root; `python` is
`.venv/bin/python` on the GPU host. Every command prints JSON to stdout.

## 1. Validate and inspect configuration

```bash
python -m jev.cli validate --config configs/example.yaml
```

```json
{"valid": true, "config_digest": "…", "model_id": "fastino/gliner2-base-v1", "schema_digest": "…", "mq_enabled": false}
```

`manifest` prints the canonical config, both digests, the pinned model manifest
at `artifacts/model_manifest.json`, and (when MQ is enabled) the rendered Rust
bridge config:

```bash
python -m jev.cli manifest --config configs/example.yaml
python -m jev.cli manifest --config configs/jev_tool_task.yaml
```

## 2. Produce training data

Training consumes the GLiNER2 supervised JSONL contract produced by
distillation: one object per line with `input` and `output` (plus provenance
fields added by `jev.distill`).

```bash
python -m jev.cli distill --config configs/example.yaml --output data/distilled.jsonl
```

Distillation requires the teacher route and the `JEV_TEACHER_API_KEY`
environment variable named by `teacher.api_key_env`; the request/response cache
lives under `runs/jev/teacher-cache` and never stores credentials. Set
`teacher.cache_dir: null` to disable it.

For an offline smoke or a hand-built batch, write the same contract yourself:

```bash
cat > /tmp/jev-train.jsonl <<'JSONL'
{"input": "2026 stack: Python, PostgreSQL, and PyTorch.", "output": {"entities": {"technology": ["Python", "PostgreSQL", "PyTorch"]}}}
JSONL
```

## 3. Train

```bash
python -m jev.cli train \
  --config configs/example.yaml \
  --data /tmp/jev-train.jsonl \
  --checkpoint-dir runs/jev \
  --steps 8
```

Output:

```json
{"events": [{"step": 1, "loss": 0.83, "model_generation": 1, "reset_required": false, "reset_reason": null, "checkpoint": null}],
 "steps": 8, "ema_updates": 8, "model_generation": 1, "checkpoint": "runs/jev/latest.pt"}
```

- `--steps` caps the number of records consumed; omit it to consume the file.
- `--resume runs/jev/latest.pt` (or a directory; `latest.pt` is implied)
  restores adapter, EMA, optimizer, step counters, and reset id. The resume
  fails closed when the stored `config_digest` differs from the current config.
- The default step function is GLiNER2's official `total_loss`; the base weights
  stay frozen and only LoRA parameters are persisted.

## 4. Serve from an immutable EMA snapshot

`serve` reconstructs the runtime, optionally resumes the checkpoint, and answers
one text or a JSONL batch. Serving reads the published snapshot; training keeps
mutating its private model.

```bash
python -m jev.cli serve \
  --config configs/example.yaml \
  --checkpoint-dir runs/jev \
  --text '2026最佳技术栈包括 Python 和 PostgreSQL'

printf '%s\n' '{"record_id": "r1", "text": "Rust owns the bridge"}' > /tmp/jev-serve.jsonl
python -m jev.cli serve \
  --config configs/example.yaml \
  --checkpoint-dir runs/jev \
  --input-jsonl /tmp/jev-serve.jsonl
```

Each result carries `output`, `generation`, `ema_step`, `reset_id`,
`checkpoint`, `schema_digest`, `config_digest`, and `model_id` so an operator
can trace exactly which accepted snapshot answered.

## 5. Metrics, checkpoints, and reset

```bash
python -m jev.cli metrics --checkpoint-dir runs/jev
```

```json
{"metrics": {"config_digest": "…", "ema_decay": 0.999, "events": [{"step": 8, "loss": 0.71, "…": "…"}]},
 "reset_events": [], "archives": [], "latest_checkpoint": "runs/jev/latest.pt"}
```

Manual collapse reset (normally raised by the drift guard, see
[`recovery.md`](recovery.md)):

```bash
python -m jev.cli reset \
  --config configs/example.yaml \
  --checkpoint-dir runs/jev \
  --reason operator_requested
```

```json
{"reset_id": 1, "reason": "operator_requested", "archive_dir": "runs/jev/archives"}
```

### Checkpoint layout

```text
runs/jev/
  latest.pt                                   # atomically replaced serving checkpoint
  adapter-step-00000008.pt                    # rolling adapter-only checkpoints
  archives/
    reset-000001-step-00000042-1758400000-a1b2c3d4.pt
  reset-events.jsonl                          # one JSON object per reset
  metrics.json                                # last train() events and config digest
```

`latest.pt` / `adapter-step-*.pt` payload fields:

| Field | Meaning |
|---|---|
| `format`, `kind` | `1`, `adapter-only` |
| `config_digest`, `schema_digest` | lineage binding; resume rejects a config mismatch |
| `model_id`, `model_revision` | pinned base identity |
| `adapter_state`, `ema_state`, `ema_updates`, `ema_decay` | LoRA parameters and EMA shadow |
| `training_steps`, `reset_id`, `created_at` | counters and timestamp |
| `optimizer_state` | present once the optimizer has run |
| `failure_reason` | present only in archive checkpoints |

Only the most recent `runtime.max_checkpoints` rolling `adapter-step-*.pt`
files are retained; archives are never pruned automatically and live under
`runs/jev/archives/`, outside the tracked source.

## 6. EMA snapshot semantics

After each optimizer step the runtime updates the shadow as

```text
ema <- beta * ema + (1 - beta) * lora      # beta = runtime.ema_decay, default 0.999
```

Only finite adapter states are folded in; a non-finite state is reported to the
drift guard instead. On publication (`runtime.publish_every_steps`, default 10,
or every step during warm-up) the runtime deep-copies the model, applies the
averaged EMA state, switches it to `eval`, and swaps the snapshot under a lock.
Inference therefore always sees a complete immutable snapshot, never a
partially written adapter, and an in-flight request keeps the snapshot it
started with. The old snapshot remains live until a reset lineage completes
warm-up.

## 7. Collapse reset procedure

1. The drift guard raises `reset_required` (see
   [`recovery.md`](recovery.md) for thresholds).
2. The trainer archives the failing adapter, EMA, optimizer, and counters to
   `runs/jev/archives/reset-<reset_id>-step-<step>-<epoch>-<uuid>.pt`.
3. `runtime.stats.reset_id` increments monotonically and a `reset-events.jsonl`
   entry records reason, step, archive path, and warm-up length.
4. A fresh zeroed LoRA and an EMA initialized from it are built from the
   immutable base. The previous accepted snapshot keeps serving.
5. Training continues; after `runtime.reset_warmup_steps` clean steps (and an
   optional `warmup_evaluator`) the new lineage is checkpointed and published.
   A failed warm-up evaluation restarts the warm-up window.

Use `python -m jev.cli reset` only for an operator-initiated reset; the drift
guard calls the same transaction automatically.
