# Monitoring runbook

Operator surface for `zenjev-monitor`, the read-only host/service observer for
the continual LoRA/EMA loop (blueprint §5 telemetry expectations). All commands
run from the repository root unless stated otherwise; `python` is
`.venv/bin/python` on the GPU host.

`zenjev-monitor` is read-only. It never signals, stops, or mutates the service;
it only reads `/proc`, `nvidia-smi`, and the status document.

## 1. Prerequisites

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd monitor && cargo build --locked --release
```

The crate is pinned to Rust 1.98.1 (`monitor/rust-toolchain.toml`) and ships a
locked dependency graph (`monitor/Cargo.lock`). Build once per host; the binary
lands at `monitor/target/release/zenjev-monitor`.

## 2. Start the service

The Python service writes the status document atomically at
`runs/jev/status.json` (default). Until the long-lived service writer is
enabled, the existing one-shot commands produce checkpoints/metrics but not a
status document — the monitor will report `status: null` / `stale: true` and
still show host metrics.

Training/serving commands used beside the monitor are documented in
[`training_serving.md`](training_serving.md). Example service invocation:

```bash
python -m jev.cli serve \
  --config configs/example.yaml \
  --checkpoint-dir runs/jev \
  --input-jsonl /tmp/jev-serve.jsonl
```

## 3. Run the monitor

Live dashboard (default `--interval 1.0`, `Ctrl-C` to stop):

```bash
monitor/target/release/zenjev-monitor --status-path runs/jev/status.json
```

Record JSONL evidence while watching:

```bash
mkdir -p artifacts/monitor
monitor/target/release/zenjev-monitor \
  --status-path runs/jev/status.json \
  --jsonl artifacts/monitor/metrics.jsonl \
  --interval 1.0 \
  --stale-after 15
```

One-shot CI/acceptance probe (exit 0 even when the status file is absent):

```bash
monitor/target/release/zenjev-monitor \
  --once --format json --no-color \
  --status-path runs/jev/status.json
```

Bounded soak (stop after N seconds; `--duration 0` runs until interrupted):

```bash
monitor/target/release/zenjev-monitor \
  --duration 300 \
  --jsonl artifacts/monitor/metrics.jsonl
```

## 4. Reading the dashboard

```text
zenjev-monitor/0.1.0  2026-09-21T05:06:44.000Z  sample 12
status  phase=training gen=9 ema=8 steps=8 loss=1.5700 reset=0  age=0.8s fresh
cpu     [#####---------------]  25.0%  cores=16  load=0.52/0.58/0.59
gpu0    util [############-------]  61.0%  vram [####----] 20.0%  4.00GiB/32.00GiB  temp=61C  power=120.0W
mem     [##########----------]  50.0%  8.00GiB/16.00GiB
proc    pid=12345 rss=1.50GiB cpu=43.1% threads=8 state=S
```

| Field | Meaning |
|---|---|
| `phase` | Service phase from the status document: `training`, `serving`, `judging`, `idle`, or `reset` |
| `gen` / `generation` | Accepted model snapshot generation; increments on every publish |
| `ema` / `ema_step` | Number of EMA updates folded into the served snapshot |
| `steps` / `training_steps` | Optimizer steps since the last reset |
| `loss` | Most recent observed training loss |
| `reset` / `reset_id` | Monotonic collapse-reset lineage id |
| `age` / `status_age_seconds` | Seconds since the service last refreshed the status document |
| `fresh` / `STALE` | `STALE` when age exceeds `--stale-after`, or the status is missing/invalid |

Interpretation:

- `gen` and `ema` advancing with `phase=training` means the loop is publishing
  snapshots; `phase=serving` with a static `gen` means it is answering without
  training.
- `phase=reset` with `reset` incrementing indicates a collapse reset; confirm
  the archive under `runs/jev/archives/` and see
  [`recovery.md`](recovery.md).
- `loss` should trend down between resets. A rising or `n/a` loss with a
  stalling `ema` is a drift/recovery signal, not a monitor error.
- `STALE` does **not** mean the service is down by itself; it means the status
  document is older than `--stale-after` or invalid. Check the `status_error`
  field in `--format json` output for the exact cause.

## 5. JSONL evidence

Every sample is appended as one JSON object per line:

```bash
tail -n 1 artifacts/monitor/metrics.jsonl | python -m json.tool
```

Fields: `monitor_schema_version`, `sampled_at_unix_ms`, `cpu`, `load`,
`memory`, `gpu`, `process`, `status`, `status_age_seconds`, `status_error`,
`stale`. Metrics whose source was unavailable are `null` (for example
`gpu: null` on a host without `nvidia-smi`).

To attach monitoring evidence to acceptance:

1. Collect the JSONL under `artifacts/monitor/metrics.jsonl` for the run
   window (the file is append-only; use a fresh path per run when comparing).
2. Summarize with the existing evidence scripts, for example:

   ```bash
   python3 scripts/audit_stage0.py
   ```

3. Reference the path and the status contract in the acceptance record, next to
   the MQ bridge telemetry under `artifacts/mq/` (blueprint §5).

A quick freshness check suitable for scripts:

```bash
monitor/target/release/zenjev-monitor --once --format json --no-color \
  --status-path runs/jev/status.json | python -c 'import json,sys; s=json.load(sys.stdin); sys.exit(0 if not s["stale"] else 1)'
```

## 6. Troubleshooting

### `gpu unavailable (nvidia-smi missing or failed)`

The monitor degrades to `gpu: null`; host metrics continue. Verify the driver:

```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
```

If this prints `[N/A]` fields, the GPU is visible but the metric is
unsupported; those fields are `null` and the rest of the sample is kept. No GPU
is required for the monitor or its tests.

### `status unavailable (status file not found: …)`

The service has not started or the writer is not enabled. Confirm the path:

```bash
ls -l runs/jev/status.json
```

The monitor never creates, repairs, or deletes the status document. If the file
exists, validate it by hand:

```bash
python -m json.tool runs/jev/status.json
```

### `STALE` status

- Check `status_age_seconds`; if it keeps growing, the service stopped
  refreshing the document (or was killed).
- Check `pid` against the process line: `proc pid=… not running` means the
  service exited while the stale document remained.
- A malformed document yields `status: null` with `status_error` naming the
  JSON/parse cause; fix/restart the service, then the next interval recovers
  automatically.
- A `schema_version` other than `1` is rejected rather than misread; align the
  writer and monitor versions.

### High CPU/GPU while monitoring

One `nvidia-smi` process is spawned per interval. If that overhead matters,
raise `--interval` (for example `--interval 5`) or use JSONL-only collection
with a longer period; an optional NVML backend is a documented TODO in
`monitor/README.md`.

## 7. Stop and cleanup

- Stop the dashboard with `Ctrl-C`. The monitor never hides the cursor or uses
  the alternate screen, so the terminal remains usable.
- JSONL evidence is append-only; keep it under `artifacts/monitor/` and do not
  delete evidence that an acceptance record references.
- `monitor/target/` is build output and safe to remove; `runs/` and
  `artifacts/` hold service evidence and are not the monitor's to prune.
