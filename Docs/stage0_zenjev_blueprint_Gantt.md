# Stage 0 ZenJev Gantt / Kanban projection

Generated read-only projection. The sole authority is [stage0_zenjev_blueprint.md](stage0_zenjev_blueprint.md).
No implementation acceptance is implied by generating this file.

**Source blueprint:** `Docs/stage0_zenjev_blueprint.md`
**Source SHA-256:** `b3da2a6c30359083e617b012134a9527fd5b9533097178119e355948f316eacd`
**Specification SHA-256:** `7d570930847fc149df13738063837df3fe3b4fe20db1cec55bbe49327cc53508`
**Generated at (UTC):** `2026-09-21T11:13:56Z`
**Generation command:** `python3 scripts/validate_blueprint.py --generate`
**Check command:** `python3 scripts/validate_blueprint.py --check`

## Monitoring summary

| Surface | Value |
|---|---|
| Checklist items | 60 |
| Todo / self-tested / Master-accepted | 0 / 0 / 60 |
| Controller | disabled; operator concurrency/lifecycle/route unresolved |
| Logical claims / service records / admitted executions | not applicable; no controller ledger |
| Startup reservations / live transports / authenticated goals / running turns | not applicable; no controller ledger |
| Request starts/window / in-flight / outstanding requests | not applicable; no controller ledger |
| Validator / integration leases; handoff / integration / repair backlog | not applicable; no controller ledger |
| Breaker / saturation / underfill | not applicable; activation disabled, no capacity target authorized |
| Telemetry scope | future repository controller only; does not count the current interactive session |
| Timing | every checklist item is unscheduled |

## Recorded event Gantt

Only the recorded governance edit is plotted. It is a zero-duration event, not a task estimate or completion date.

```mermaid
gantt
    title Recorded governance event (UTC)
    dateFormat YYYY-MM-DDTHH:mm:ss[Z]
    axisFormat %Y-%m-%d %H:%M
    section Recorded events
    Governance policy correction :milestone, governance-policy-correction, 2026-09-19T18:08:30Z, 0d
```

## Unscheduled monitoring index

Each stable checklist ID has exactly one row. Owners are planned roles; no row claims a live execution.
The validation-preparation frontier consists of unfinished items; it authorizes preparing checks, never early acceptance.

| ID | State | Depends on | Planned owner | Owned paths | Claim / run | Frontier | Validation preparation | Startup / live / handoff / integration / repair | Blocked | Timing | Gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZJ-001 | accepted | — | Master | `configs/example.yaml`, `artifacts/model_manifest.json`, `Docs/researches/**` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G0,G3,G4,G5 |
| ZJ-002 | accepted | ZJ-001 | Master | `.gitignore`, `jev/`, `tests/`, `scripts/`, `Docs/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G0 |
| ZJ-003 | accepted | ZJ-002 | Master | `scripts/validate_blueprint.py`, `tests/test_governance.py`, `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G0 |
| ZJ-004 | accepted | ZJ-002 | Schema worker | `jev/config.py`, `jev/distill.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G2,G3,G4 |
| ZJ-010 | accepted | ZJ-004 | Schema worker | `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1 |
| ZJ-011 | accepted | ZJ-004 | Data worker | `jev/sources.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G2 |
| ZJ-012 | accepted | ZJ-004,ZJ-011 | Distillation worker | `jev/distill.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G2 |
| ZJ-013 | accepted | ZJ-011 | Data worker | `jev/sources.py`, `jev/distill.py`, `artifacts/provenance/`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G2 |
| ZJ-014 | accepted | ZJ-010,ZJ-012,ZJ-013 | Distillation worker | `jev/distill.py`, `artifacts/datasets/`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G2 |
| ZJ-020 | accepted | ZJ-001,ZJ-002 | Model worker | `jev/model.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G3,G6 |
| ZJ-021 | accepted | ZJ-020 | Training worker | `jev/training.py`, `jev/model.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G3 |
| ZJ-022 | accepted | ZJ-021 | Training worker | `jev/ema.py`, `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G4 |
| ZJ-023 | accepted | ZJ-022,ZJ-020 | Runtime worker | `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G4 |
| ZJ-024 | accepted | ZJ-021,ZJ-022,ZJ-004 | Evaluation worker | `jev/drift.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G5 |
| ZJ-025 | accepted | ZJ-024,ZJ-020 | Training worker | `jev/training.py`, `jev/runtime.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G5 |
| ZJ-026 | accepted | ZJ-010,ZJ-020,ZJ-022 | Inference worker | `jev/model.py`, `jev/runtime.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G4 |
| ZJ-027 | accepted | ZJ-014,ZJ-023,ZJ-025,ZJ-026 | Master | `jev/runtime.py`, `jev/training.py`, `jev/cli.py`, `tests/test_jev.py`, `scripts/smoke.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G2,G4,G5 |
| ZJ-028 | accepted | ZJ-026 | Schema worker | `configs/jev_tool_task.yaml`, `jev/config.py`, `jev/tool_task.py`, `tests/test_model_config_distill.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G4,G9 |
| ZJ-029 | accepted | ZJ-011,ZJ-028 | Data worker | `jev/sources.py`, `jev/tool_task.py`, `configs/jev_tool_task.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G2,G9 |
| ZJ-030 | accepted | ZJ-001,ZJ-020 | Hardware worker | `scripts/validate_nvidia_gpu.py`, `artifacts/hardware/`, `Docs/runbooks/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G6 |
| ZJ-031 | accepted | ZJ-027,ZJ-030 | Hardware worker | `scripts/smoke_nvidia_gpu.py`, `benchmarks/`, `artifacts/benchmarks/`, `tests/benchmarks/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G6 |
| ZJ-032 | accepted | ZJ-014,ZJ-027 | Evaluation worker | `artifacts/evaluations/`, `jev/drift.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G2,G4,G5 |
| ZJ-033 | accepted | ZJ-025,ZJ-031 | Evaluation worker | `artifacts/recovery/`, `tests/recovery/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G5,G6 |
| ZJ-034 | accepted | ZJ-011,ZJ-012,ZJ-013,ZJ-014,ZJ-051 | Master | `artifacts/audits/`, `Docs/runbooks/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G2,G7,G10 |
| ZJ-035 | accepted | ZJ-026,ZJ-028,ZJ-029 | Inference worker | `jev/model.py`, `jev/tool_task.py`, `jev/runtime.py`, `tests/test_model_config_distill.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G4,G9 |
| ZJ-036 | accepted | ZJ-032,ZJ-035 | Evaluation worker | `artifacts/evaluations/`, `tests/test_jev.py`, `tests/test_model_config_distill.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G5,G9 |
| ZJ-037 | accepted | ZJ-023,ZJ-035 | Runtime worker | `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G4,G9 |
| ZJ-040 | accepted | ZJ-010,ZJ-011,ZJ-027 | Docs worker | `jev/cli.py`, `Docs/runbooks/`, `README.md`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G1,G4,G5 |
| ZJ-041 | accepted | ZJ-031,ZJ-032,ZJ-034,ZJ-040,ZJ-051 | Master | `artifacts/repro/`, `Docs/runbooks/`, `requirements*.txt`, `pyproject.toml` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G0,G6,G7,G10 |
| ZJ-042 | accepted | ZJ-003,ZJ-027,ZJ-031,ZJ-033,ZJ-041,ZJ-053,ZJ-055 | Master | `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G0–G11 as applicable |
| ZJ-043 | accepted | ZJ-042 | Master | `.git/`, `artifacts/delivery/github_receipt.json` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G7 |
| ZJ-044 | accepted | ZJ-042,ZJ-043,ZJ-054 | Hardware worker/Master | `artifacts/delivery/remote_transfer_receipt.json`, `artifacts/hardware/remote_*.json` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G6,G7,G11 |
| ZJ-045 | accepted | ZJ-043,ZJ-044 | Master | `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G8 |
| ZJ-050 | accepted | ZJ-004,ZJ-011 | Schema worker | `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G10 |
| ZJ-051 | accepted | ZJ-050 | MQ worker | `mq/`, `scripts/smoke_mq.py`, `tests/test_mq.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G10 |
| ZJ-052 | accepted | ZJ-037,ZJ-050 | Runtime worker | `jev/mq.py`, `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G10 |
| ZJ-053 | accepted | ZJ-051,ZJ-052 | Evaluation worker | `tests/test_mq.py`, `mq/tests/`, `scripts/smoke_mq.py`, `artifacts/mq/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G10 |
| ZJ-054 | accepted | ZJ-031,ZJ-053 | Hardware worker | `scripts/smoke_mq_nvidia_gpu.py`, `benchmarks/mq/`, `artifacts/benchmarks/`, `Docs/runbooks/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G6,G11 |
| ZJ-055 | accepted | ZJ-040,ZJ-052 | Docs worker | `jev/cli.py`, `Docs/runbooks/`, `README.md`, `tests/test_jev.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G10,G11 |
| ZJ-060 | accepted | ZJ-055 | Schema worker | `Docs/researches/`, `configs/zenjev_perpetual.yaml`, `jev/config.py`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12 |
| ZJ-061 | accepted | ZJ-027,ZJ-060 | Runtime worker | `scripts/install_zenjev_services.sh`, `scripts/`, `Docs/runbooks/`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12 |
| ZJ-062 | accepted | ZJ-060 | Runtime worker | `jev/runtime.py`, `jev/cli.py`, `runs/jev/status.json`, `artifacts/perpetual/`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12 |
| ZJ-063 | accepted | ZJ-051,ZJ-053 | MQ worker | `mq/src/source/`, `mq/tests/`, `tests/test_mq.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G13 |
| ZJ-064 | accepted | ZJ-050,ZJ-063 | Schema worker | `jev/config.py`, `jev/mq.py`, `configs/zenjev_perpetual.yaml`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G13 |
| ZJ-065 | accepted | ZJ-052,ZJ-061,ZJ-063 | Runtime worker | `jev/mq.py`, `jev/runtime.py`, `scripts/zenjev_loop.py`, `tests/test_mq.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G13 |
| ZJ-066 | accepted | ZJ-021,ZJ-025 | Training worker | `jev/training.py`, `jev/ema.py`, `scripts/zenjev_loop.py`, `artifacts/model_manifest.json` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-067 | accepted | ZJ-023,ZJ-031,ZJ-061,ZJ-066 | Training worker | `jev/training.py`, `jev/runtime.py`, `scripts/zenjev_loop.py`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-068 | accepted | ZJ-052,ZJ-062 | Runtime worker | `scripts/zenjev_console.py`, `jev/cli.py`, `artifacts/console/`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G15 |
| ZJ-069 | accepted | ZJ-068 | Docs worker | `scripts/console/`, `Docs/runbooks/`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G15 |
| ZJ-070 | accepted | ZJ-052,ZJ-060,ZJ-068 | Data worker | `scripts/zenjev_fabricator.py`, `configs/zenjev_perpetual.yaml`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G15 |
| ZJ-071 | accepted | ZJ-065,ZJ-067,ZJ-069,ZJ-070 | Evaluation worker | `scripts/run_infinite_soak.py`, `scripts/perpetual_evidence.py`, `artifacts/perpetual/`, `artifacts/soak/`, `Docs/runbooks/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12,G13,G14,G15 |
| ZJ-073 | accepted | ZJ-067 | Runtime worker | `scripts/zenjev_serve.py`, `scripts/zenjev_services.sh`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12,G14 |
| ZJ-074 | accepted | ZJ-066,ZJ-073 | Runtime worker | `scripts/zenjev_deploy.py`, `artifacts/perpetual/`, `scripts/console/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-075 | accepted | ZJ-021,ZJ-067 | Training worker | `scripts/zenjev_loop.py`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-077 | accepted | ZJ-035,ZJ-028 | Schema worker | `jev/tool_task.py`, `configs/zenjev_perpetual.yaml`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G9 |
| ZJ-078 | accepted | ZJ-063,ZJ-069 | Runtime worker | `scripts/zenjev_lib.py`, `scripts/zenjev_feeder.py`, `scripts/zenjev_console.py`, `scripts/console/` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G13,G15 |
| ZJ-079 | accepted | ZJ-021,ZJ-066 | Training worker | `jev/training.py`, `scripts/zenjev_loop.py`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-080 | accepted | ZJ-035,ZJ-079 | Distillation worker | `scripts/zenjev_loop.py`, `runs/jev/perpetual/pairs.jsonl`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G14 |
| ZJ-076 | accepted | ZJ-011,ZJ-063 | Data worker | `scripts/zenjev_feeder.py`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G13 |
| ZJ-072 | accepted | ZJ-065,ZJ-067,ZJ-069,ZJ-070 | Docs worker | `jev/cli.py`, `scripts/zenjev_services.sh`, `Docs/runbooks/perpetual_operations.md`, `README.md`, `tests/test_perpetual.py` | unclaimed / none | complete | complete | not applicable (controller disabled) | none from DAG | Unscheduled | G12,G15 |

## Reconciliation

The checker reconstructs this entire file from the authoritative source and recorded generation time. Any stale digest, count, dependency, owner, owned path, state, frontier, omitted/duplicate ID or invented timing fails verification. Generation uses an atomic replacement and never edits authoritative checklist state. Future controller activation requires a policy migration and ledger-aware projection; this tool cannot activate it.
