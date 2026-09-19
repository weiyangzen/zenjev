# Jev

Jev is a schema-driven information extraction runtime built around the 205M
parameter GLiNER2 checkpoint and PEFT LoRA adapters. It lets a user define a
schema, distil labelled examples from a configured teacher and configured
source set, train an adapter continuously, and serve extraction from the last
accepted model while an EMA shadow is maintained.

The implementation is deliberately dependency-light at the control-plane
boundary. `jev-core` validates configuration, provenance, drift/reset policy,
EMA state and concurrent publication without importing PyTorch. Install the
optional `train` extra on the RTX 5090 host for GLiNER2, PEFT and CUDA.

## Quick start

```bash
python -m jev.cli validate --config configs/example.yaml
python -m jev.cli distill --config configs/example.yaml --output data/distilled.jsonl
python -m jev.cli extract --config configs/example.yaml --text '2026最佳技术栈包括 Python 和 PostgreSQL'
```

`configs/example.yaml` is an example only. Put API keys in the environment
named by the config; never commit credentials or raw provider responses.
For an offline 5090 deployment, stage the pinned model snapshot and set
`JEV_MODEL_PATH=/home/sansha/jev-model-base` (or `runtime.model_path`) so model
loading does not depend on a live Hub connection.

The authoritative implementation plan is
[`Docs/stage0_zenjev_blueprint.md`](Docs/stage0_zenjev_blueprint.md); research
notes live under [`Docs/researches/`](Docs/researches/).
