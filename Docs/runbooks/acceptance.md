# Stage 0 acceptance runbook

The exact validation profile from the authoritative blueprint, plus the Stage 0
evidence items. Run every command from the repository root on the GPU host; on
that host `python3` is `.venv/bin/python`. A missing applicable validator is a
failure, not a skip. A worker self-test is provisional until Master reruns the
gate.

## 1. Validation profile commands

| # | Command | Evidence |
|---|---|---|
| 1 | `python3 -m compileall -q jev scripts` | exit code 0; no artifact |
| 2 | `python3 -m pytest -q` | pytest summary; covers schema/config, model/distill, EMA/drift, and `tests/test_mq.py` replay/fault tests |
| 3 | `python3 scripts/validate_blueprint.py --check` | stdout `blueprint_ok ids=<n> digest=<sha256> controller=disabled` |
| 4 | `python3 scripts/smoke.py` | stdout `jev_smoke_ok` (control-plane distill/drift/runtime smoke) |
| 5 | `cd mq && cargo fmt --check` | exit code 0 |
| 6 | `cd mq && cargo clippy --locked -- -D warnings` | exit code 0 |
| 7 | `cd mq && cargo test --locked` | Rust unit/bridge test summary |
| 8 | `python3 scripts/smoke_mq.py` | `artifacts/mq/smoke.json` and stdout evidence + `jev_mq_smoke_ok` |
| 9 | `python3 -m jev.cli validate --config configs/example.yaml` | stdout JSON (`valid`, `config_digest`, `schema_digest`) |
| 10 | `python3 -m jev.cli validate --config configs/jev_tool_task.yaml` | stdout JSON for the tool-task schema |
| 11 | `python3 -m jev.cli extract --config configs/example.yaml --text '2026最佳技术栈包括 Python 和 PostgreSQL'` | stdout inference JSON with generation/EMA/schema digests |
| 12 | `python3 scripts/recovery_drill.py --config configs/example.yaml --model-path /home/sansha/jev-model-base --output artifacts/recovery/collapse_recovery.json` | `artifacts/recovery/collapse_recovery.json` (train/EMA/collapse-recovery simulation; script supplied by ZJ-033) |
| 13 | `python3 -m pytest -q tests/test_mq.py` | MQ replay/fault assertions (duplicate suppression, crash-before-ack, poison-to-DLQ, quarantine, offset resume, credit backpressure, graceful drain, no-ack-before-durable-ack) |
| 14 | `python3 scripts/validate_nvidia_gpu.py` | stdout JSON `pass`/`reason`; fail-closed CUDA gate |
| 15 | `JEV_MODEL_PATH=/home/sansha/jev-model-base python3 scripts/smoke_nvidia_gpu.py --output artifacts/nvidia_gpu_train_ema_smoke.json` | `artifacts/nvidia_gpu_train_ema_smoke.json` (base load, LoRA train, EMA, concurrent inference, p50/p95, peak VRAM, thermal/power) |
| 16 | `python3 scripts/audit_stage0.py` | `artifacts/audits/stage0_audit.json` (source/license/provenance, secret scan, secret references, MQ paths, manifest presence, checklist counts) |
| 17 | `python3 scripts/build_repro_bundle.py` | `artifacts/repro/repro_manifest.json` (commit/status, config/model/blueprint digests, versions, lockfiles, exact commands) |
| 18 | `cd mq && cargo build --locked && cargo build --locked --release` | bridge binaries used by MQ smoke/soak |

The Rust steps run in `mq/` with the pinned `mq/rust-toolchain.toml`. The
`mock` adapter needs no broker; the default binary builds only the `nats`
feature. `--features kafka`/`--features iggy` builds are opt-in and require
system `librdkafka`/`cmake` (Kafka) as documented in
[`mq_operations.md`](mq_operations.md#7-build-the-rust-bridge).

## 2. NVIDIA GPU gate

A CPU-only result cannot satisfy G6. The hardware gate requires:

1. `scripts/validate_nvidia_gpu.py` to report `"pass": true` with a CUDA build
   of PyTorch, an importable `gliner2`/`peft`, a visible device, and a working
   `nvidia-smi`.
2. `scripts/smoke_nvidia_gpu.py` against the staged snapshot
   (`JEV_MODEL_PATH=/home/sansha/jev-model-base`) with non-empty losses, EMA
   updates equal to steps, concurrent inference observations, and recorded
   p50/p95 latency, throughput, and peak VRAM.

This environment recorded `NVIDIA GeForce RTX 5090 D`, capability 12.0, driver
595.84, `torch 2.9.1+cu128`, CUDA 12.8; evidence lives in
`artifacts/nvidia_gpu_model_smoke.json` and
`artifacts/nvidia_gpu_train_ema_smoke.json`.

## 3. Live MQ soak (deferred)

`broker-connected MQ pull-train soak` (G11, ZJ-054) was **not performed in this
environment**; `artifacts/mq/smoke.json` is mock-adapter evidence with
`"live_broker": false` and does not satisfy G11. Run the live soak later per
[`mq_operations.md`](mq_operations.md#9-live-broker-soak-not-yet-performed) and
store throughput, bounded-lag recovery, peak VRAM, ack latency, and
DLQ/quarantine counts under `artifacts/benchmarks/mq/`. Repeat the MQ smoke
after the remote transfer for G7/G11.

## 4. Ordering and reconciliation

1. Run commands 1–8 (fast, no GPU) after every merge; they are the repository
   gates for G0–G5 and G10.
2. Run 9–12 for schema/lineage and recovery evidence.
3. Run 13–15 on the GPU host; 13 also runs inside 2.
4. Run 16–17 to freeze the audit and reproducibility bundle last, so their
   digests cover the accepted tree.
5. Reconcile the blueprint and Gantt
   (`python3 scripts/validate_blueprint.py --check`) and record the actual
   command outcomes in the relevant handoff/manifest. Only Master may advance
   checklist marks; these scripts never modify the blueprint.
