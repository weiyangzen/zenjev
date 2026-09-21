# zenjev-monitor

`zenjev-monitor` is a small, robust, read-only observability tool for the
ZenJev train/infer service. It samples host CPU, load, memory, optional NVIDIA
GPU state, the training/serving process, and the service status document;
prints a live dashboard; and appends JSONL evidence that can be attached to
Stage 0 acceptance evidence under `artifacts/monitor/`.

It is deliberately dependency-light: `serde` + `serde_json` only, no async
runtime, no TUI framework, and no native build dependencies. Every sampler
returns `Option`/`Result` and is skipped on error, so a missing status file, a
dead process, an unreadable `/proc` entry, or a host without `nvidia-smi`
degrades to `null` instead of crashing.

## Build

The toolchain is pinned in `monitor/rust-toolchain.toml` (Rust 1.98.1 with
`rustfmt` + `clippy`) and a `Cargo.lock` is checked in.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd monitor
cargo build --locked               # debug binary: target/debug/zenjev-monitor
cargo build --locked --release     # release binary: target/release/zenjev-monitor
```

The crate is both a binary (`zenjev-monitor`) and a library
(`zenjev_monitor`) so the parsers and renderers are unit-testable.

## Usage

Live dashboard (refresh in place, `Ctrl-C` to stop):

```bash
cd monitor
./target/release/zenjev-monitor --status-path ../runs/jev/status.json
```

Dashboard plus durable JSONL evidence for acceptance:

```bash
./target/release/zenjev-monitor \
  --status-path ../runs/jev/status.json \
  --jsonl ../artifacts/monitor/metrics.jsonl \
  --interval 1.0 \
  --stale-after 15
```

One-shot, CI-friendly check (always exits 0 for a successful sample, even when
the status file is absent):

```bash
./target/release/zenjev-monitor --once --format json --no-color --status-path ../runs/jev/status.json
```

Bounded soak (0 means run until interrupted):

```bash
./target/release/zenjev-monitor --duration 300 --jsonl ../artifacts/monitor/metrics.jsonl
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--status-path PATH` | `runs/jev/status.json` | Status document written atomically by the Python service |
| `--interval SECONDS` | `1.0` | Sampling period; must be positive |
| `--stale-after SECONDS` | `15.0` | Age above which the status is marked `stale: true` |
| `--format dashboard\|json` | `dashboard` | Dashboard block, or one JSON object per line |
| `--jsonl PATH` | none | Append every sample as one JSON line; parent dirs are created |
| `--once` | off | Sample exactly once and exit 0 |
| `--duration SECONDS` | `0` (until interrupted) | Stop after this many seconds |
| `--no-color` | off | Strip ANSI escapes from the dashboard |
| `-h`, `--help` / `-V`, `--version` | | Help / version |

Exit codes: `0` for a normal stop or a successful `--once` sample; `2` only for
invalid arguments.

## Status contract

The Python service writes the status document atomically (temporary file +
rename). Only `schema_version` and `updated_at_unix_ms` are required; every
other field may be absent, and unknown fields are ignored:

```json
{
  "schema_version": 1,
  "updated_at_unix_ms": 1789938404000,
  "pid": 12345,
  "phase": "training|serving|judging|idle|reset",
  "generation": 9,
  "ema_step": 8,
  "training_steps": 8,
  "reset_id": 0,
  "loss": 1.57,
  "learning_rate": 0.0001,
  "checkpoint": "runs/jev/jevraw/adapter-step-00000008.pt",
  "config_digest": "…64 hex…",
  "model_id": "fastino/gliner2-base-v1",
  "device": "cuda",
  "note": "optional"
}
```

- `status_age_seconds = max(0, now - updated_at_unix_ms) / 1000`; a timestamp in
  the future clamps to `0.0`.
- `stale: true` when the age exceeds `--stale-after`, **or** when the file is
  missing, unreadable, malformed, or has an unsupported `schema_version`.
- On any of those failures the sample carries `status: null` plus a
  `status_error` string explaining the cause.
- The Python-side writer that lands this document is the service integration
  point; until it is enabled, the monitor reports `status: null` / `stale: true`
  and still prints all host metrics.

## Output

`--format json` prints the monitor's own envelope, one object per line:

```json
{"monitor_schema_version":1,"sampled_at_unix_ms":1789938404000,
 "cpu":{"percent":12.3,"cores":16,"per_core_percent":[10.0,null]},
 "load":{"one":0.52,"five":0.58,"fifteen":0.59,"running":1,"total":789},
 "memory":{"total_bytes":17179869184,"used_bytes":8589934592,"available_bytes":8589934592,"used_percent":50.0},
 "gpu":[{"index":0,"utilization_percent":15.0,"memory_used_bytes":34225520640,"memory_total_bytes":34359738368,"temperature_c":50.0,"power_w":26.33}],
 "process":{"pid":12345,"rss_bytes":1572864000,"cpu_percent":43.1,"threads":8,"state":"S"},
 "status":{"schema_version":1,"updated_at_unix_ms":1789938404000,"phase":"training","generation":9,"ema_step":8,"training_steps":8,"reset_id":0,"loss":1.57,"learning_rate":0.0001,"checkpoint":"…","config_digest":"…","model_id":"…","device":"cuda","note":null},
 "status_age_seconds":0.8,"status_error":null,"stale":false}
```

`--jsonl PATH` appends the same object per interval (create + append + flush +
`fsync`, parent directories created) so evidence accumulates at
`artifacts/monitor/metrics.jsonl`.

The dashboard is one compact ASCII block: service phase/generation/EMA
step/training steps/loss/age/stale flag, CPU bar and load, one GPU line per
device (utilization bar, VRAM bar with used/total, temperature, power), system
memory bar, and the service process RSS/CPU/threads/state. `--no-color` yields
the same text with no ANSI escapes.

## Degradation behavior

| Failure | Result |
|---|---|
| Status file missing / malformed / wrong schema | `status: null`, `stale: true`, `status_error` populated |
| `nvidia-smi` missing, failing, or reporting no GPUs | `gpu: null` |
| `[N/A]` / `[Not Supported]` GPU fields | that field is `null` |
| Process from `pid` exited mid-sample | `process: null` |
| `/proc` entry unreadable or counters wrapped | that metric is `null`; the loop continues |
| First CPU sample (no previous delta) | a 100 ms probe read establishes the delta instead of emitting noise |

No I/O path uses `unwrap`/`expect`; malformed input is skipped, never fatal.

## Why `nvidia-smi`?

GPU sampling shells out to `nvidia-smi --query-gpu=… --format=csv,noheader,nounits`
and parses the CSV. This keeps the crate free of native build dependencies
(NVML would require a C toolchain and driver headers) and works in the pinned
container/GPU host as long as the driver CLI is installed.

TODO: if per-sample `nvidia-smi` process spawn overhead matters for
high-frequency sampling, add an optional NVML backend behind a feature flag
(maintaining the no-native-deps default).

## Terminal behavior

The dashboard refreshes in place by moving the cursor back over its previous
block and clearing each line (`ESC[<n>A` + `ESC[2K`). It never hides the cursor
and never enters the alternate screen. A `TerminalGuard` resets ANSI attributes
and prints a final newline on normal exit and on panic. There is intentionally
no signal handler (that would need `libc`/`unsafe` or a signal crate): a direct
`SIGINT`/`SIGTERM` kill skips destructors, and because the cursor is never
hidden the terminal stays usable.

## Tests

```bash
cd monitor
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --locked
```

Parsers are fed synthetic `/proc/stat`, `/proc/meminfo`, `/proc/loadavg`,
`/proc/<pid>/stat|status`, and `nvidia-smi` CSV text, so the suite is CPU-only,
deterministic, and needs no GPU. Status and JSONL tests use temp files under
`std::env::temp_dir()` and clean up after themselves.
