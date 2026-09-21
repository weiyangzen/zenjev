# ZenJev

ZenJev is a schema-driven information extraction runtime built around the 205M
parameter GLiNER2 checkpoint and PEFT LoRA adapters. It lets a user define a
schema, distil labelled examples from a configured teacher and configured
source set, train an adapter continuously, and serve extraction from the last
accepted model while an EMA shadow is maintained.

The implementation is deliberately dependency-light at the control-plane
boundary. `jev-core` validates configuration, provenance, drift/reset policy,
EMA state and concurrent publication without importing PyTorch. Install the
optional `train` extra on the NVIDIA GPU host for GLiNER2, PEFT and CUDA.

## Quick start

```bash
python -m jev.cli validate --config configs/example.yaml
python -m jev.cli distill --config configs/example.yaml --output data/distilled.jsonl
python -m jev.cli extract --config configs/example.yaml --text '2026最佳技术栈包括 Python 和 PostgreSQL'
```

`configs/example.yaml` is an example only. Put API keys in the environment
named by the config; never commit credentials or raw provider responses.
For an offline NVIDIA GPU deployment, stage the pinned model snapshot and set
`JEV_MODEL_PATH=/home/sansha/jev-model-base` (or `runtime.model_path`) so model
loading does not depend on a live Hub connection.

Distillation writes the official GLiNER2 JSONL `input`/`output` contract and
keeps an access-controlled request/response cache under
`runs/jev/teacher-cache`; API credentials are never persisted (set
`teacher.cache_dir: null` to disable it). The
default trainer consumes those records with GLiNER2's supervised `total_loss`:

```python
from jev.config import load_config
from jev.runtime import JevRuntime
from jev.training import ContinuousLoRATrainer

config = load_config("configs/example.yaml")
runtime = JevRuntime(config)
trainer = ContinuousLoRATrainer(config, runtime, checkpoint_dir="runs/jev")
trainer.train([{"input": "Python", "output": {"entities": {"technology": ["Python"]}}}])
```

Checkpoints are adapter-only and include optimizer/EMA/config lineage in
`runs/jev/latest.pt`; collapse resets archive the failed lineage under
`runs/jev/archives/` and keep the last serving snapshot until warm-up passes.
The reproducible hardware gate is
`JEV_MODEL_PATH=/home/sansha/jev-model-base python scripts/smoke_nvidia_gpu.py`.

The authoritative implementation plan is
[`Docs/stage0_zenjev_blueprint.md`](Docs/stage0_zenjev_blueprint.md); research
notes live under [`Docs/researches/`](Docs/researches/).

For request/response decision analysis, use
[`configs/jev_tool_task.yaml`](configs/jev_tool_task.yaml). GLiNER2 classifies
finite task labels and returns a probability-bearing `decision_choices` list
for the configured technology stack while extracting tools/languages/technology
spans; ZenJev's allowlist router returns `allow`, `review`, or `reject`. It
never executes a tool from model output. The existing runtime supports training beside
inference through immutable EMA snapshots; `ContinuousLoRAStream` provides a
bounded background queue with explicit backpressure around the trainer.

Live training input can also arrive continuously from a configurable external
standard message queue (NATS JetStream durable pull by default, Kafka/Redpanda,
optional Apache Iggy) through the Rust `mq` bridge: it validates the versioned
envelope, deduplicates by `record_id` plus content hash, spools to a bounded
local WAL, and acks or commits the broker offset only after the trainer durably
accepts the record. The contract is frozen in §1.5 of
[`Docs/stage0_zenjev_blueprint.md`](Docs/stage0_zenjev_blueprint.md).

## Perpetual runtime (Stage 0.1)

The accepted Stage 0 pipeline now runs unstopped on the GPU host: a read-only
feeder normalizes `/home/sansha/data/jevraw` into §1.5 envelopes, the Rust
`dir-spool` MQ bridge delivers them with byte-offset watermarks and
ack-after-durable-acceptance, `zenjev-loop` self-labels with the live LoRA
judge (`run_jevraw_loop.py` recipe), trains, publishes monotonic LoRA
generations and serves the EMA snapshot, and `zenjev-console` renders the live
panel.

```bash
bash scripts/install_zenjev_services.sh          # install + start + linger note
scripts/zenjev_services.sh status               # units + console URL
scripts/zenjev_services.sh health               # heartbeats, gaps, sessions
scripts/zenjev_services.sh logs zenjev-loop 80
scripts/zenjev_services.sh evidence             # freeze acceptance evidence
```

Live console: `http://<host>:8790/`. Full operations guide:
[`Docs/runbooks/perpetual_operations.md`](Docs/runbooks/perpetual_operations.md).
