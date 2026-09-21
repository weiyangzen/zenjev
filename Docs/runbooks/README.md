# ZenJev operator runbooks

Operator runbooks for the Stage 0 ZenJev pipeline (checklist items ZJ-040,
ZJ-034, ZJ-041, ZJ-055). The authoritative plan is
[`../stage0_zenjev_blueprint.md`](../stage0_zenjev_blueprint.md); these runbooks
are operating instructions and never change checklist state or acceptance.

| Runbook | Purpose |
|---|---|
| [`deployment.md`](deployment.md) | Stage the offline GLiNER2 snapshot, set `JEV_MODEL_PATH`/`JEV_MODEL_MANIFEST`/`JEV_MQ_BRIDGE_BIN`, install the runtime or train extra, and run the fail-closed NVIDIA GPU validator. |
| [`training_serving.md`](training_serving.md) | Operate `train`, `serve`, `metrics`, `reset`, and `manifest`; understand the `runs/jev` checkpoint layout, EMA snapshot semantics, and the collapse reset procedure. |
| [`mq_operations.md`](mq_operations.md) | The frozen §1.5 external-MQ contract (adapter selection, envelope, idempotency, ack ordering, credits, WAL, DLQ, quarantine, secret references) and the `python -m jev.cli mq ...` operator commands. |
| [`recovery.md`](recovery.md) | Collapse detection thresholds, the reset transaction (archive, reset id, warm-up), and how to run the collapse recovery drill. |
| [`monitoring.md`](monitoring.md) | `zenjev-monitor`: CPU/GPU/GPU-memory sampling, training-generation/EMA/loss status contract, JSONL evidence, degradation behavior. |
| [`infinite_soak.md`](infinite_soak.md) | Endless head-to-tail MQ replay (`loop: true`) and the bounded/unbounded train-infer-update soak with monitoring evidence. |
| [`acceptance.md`](acceptance.md) | The exact Stage 0 validation profile commands and the evidence file each one produces. |

Common conventions:

- Run from the repository root. On the GPU host the virtual environment is
  `.venv`, so `python3` means `.venv/bin/python`.
- Every command prints JSON to stdout unless stated otherwise; an exit code of
  `2` is a fail-closed rejection.
- Never put credential values in configs, logs, or the repository. Configs may
  contain secret *references* only: `env:NAME`, `file:/path`, or an
  `api_key_env` variable name resolved from the process environment.
