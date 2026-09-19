# Stage 0 ZenJev Blueprint (authoritative)

**Status:** execution specification and master checklist for Stage 0  
**Authority:** This file is the only mutable requirement/checklist authority for ZenJev. `Docs/researches/**` explains decisions but cannot change checklist state or acceptance. `Docs/stage0_zenjev_gantt.md` is a generated, read-only projection of this file and the durable execution ledgers.  
**Scope:** build the first usable ZenJev pipeline: configurable schema and sources, trusted-teacher distillation, GLiNER2 205M with LoRA continual training, EMA-backed live inference, collapse detection/restart, 5090 validation, public repository publication, and verified transfer to the 5090 host.

## 1. Frozen repository and execution specification

The following values are frozen for this Stage 0 run. A change is a named policy migration that updates this file, its specification digest, claims, state, and Gantt together; no environment probe or scheduler may silently substitute a value.

| Field | Frozen value |
|---|---|
| Canonical repository root | `/home/sansha/Github/zenjev` |
| Authoritative blueprint | `Docs/stage0_zenjev_blueprint.md` |
| Same-name Gantt companion | `Docs/stage0_zenjev_gantt.md` (replace terminal `Blueprint` token only) |
| Checklist grammar | Markdown task rows beginning exactly `[ ]`, `[_]`, or `[x]`; stable ID `ZJ-###` immediately after the mark; duplicate IDs, other marks, and missing IDs fail closed |
| Dependency authority | `Depends on` column in the checklist below; edges must resolve to existing IDs and be acyclic; document order is never an edge |
| Runtime root | `.zenjev/runtime/` (controller-owned and ignored by source packaging) |
| Task root | `.zenjev/runtime/tasks/<claim-id>/<run-id>/` with `work/`, private `codex-home/`, `tmux.sock`, immutable `claim.json`, and `result.json` |
| Worker writable scope | Claim-card-declared repository-relative paths only; workers may not write the canonical checkout or this blueprint |
| Handoff store | `.zenjev/runtime/handoffs/<claim-id>/<run-id>/` with checksum-bound result and patch; harvested before process pruning |
| Selected platform/transport | Codex interactive TUI in one task-local tmux server per claim; `WORKER_TRANSPORT=tmux_codex_tui`, `WORKER_GOAL_COMMAND=/goal` |
| Forbidden transports | Codex app-server, app-server JSON-RPC, `codex exec`, shared Codex daemons, shared tmux servers, shared writable Codex state, and no-tmux Codex workers |
| Lifecycle | `bounded`; one authenticated goal per admitted generation; terminal result stops its exact tmux transport; no automatic post-terminal continuation |
| Nested agents | Forbidden (`NESTED_AGENTS=forbidden`) |
| Route policy | Use the installed Codex configuration unless a route is explicitly recorded in the claim; route, cwd, thread, goal, and private `CODEX_HOME` must authenticate before a lane is live |
| Request contract | Acquire atomic running-turn and outbound-request leases before one Enter; one outstanding request per execution; request starts and in-flight requests are separate counters |
| Result schema | `schema_version`, `claim_id`, `run_id`, `status=self_tested`, `baseline_digest`, `spec_digest`, `changed_paths`, `patch_sha256`, `commands`, `artifacts`, `dependencies`, `failure_class`, and `handoff_timestamp` |
| Master rule | Only canonical Master integrates patches, runs gates, reconciles completion surfaces, and advances `[_]` to `[x]`; workers can only produce a checksum-valid `self_tested` handoff and never write `[x]` |
| Completion surfaces | This blueprint, the generated Gantt, research documents, implementation/tests, model/data/run manifests, GitHub publication receipt, and remote transfer receipt |
| Source policy | No secrets, provider keys, or plaintext remote password in the repository; source records contain URI, retrieval time, content hash, license/terms, and user approval |
| Validation profile | `python -m compileall` for Python sources; `pytest -q` when tests exist; schema/config lint; deterministic smoke inference; train/EMA/collapse-recovery simulation; 5090 CUDA/VRAM/latency benchmark; manifest checksum audit. A missing applicable validator is a failure, not a skip. |
| Git policy | Master may initialize/commit/push only after required gates. Public remote must be `weiyangzen/zenjev`; the publication receipt records URL, commit, branch, and visibility. |
| Remote delivery | Target `sansha@192.168.50.38`, destination `/home/sansha/zenjev`; transfer through an authenticated channel supplied at run time, never by committing the provided password; verify a full file manifest and rerun smoke/5090 checks remotely. |

### 1.1 Stage 0 model and data contract (frozen)

* **Base model:** GLiNER2 205M. The exact model identifier, revision, tokenizer files, parameter count, and license are recorded in `artifacts/model_manifest.json`; startup rejects a revision whose parameter count or architecture does not match the manifest. Model downloads are pinned by revision/hash and cached outside the source tree.
* **Trainable state:** LoRA adapters only in Stage 0. Base weights are immutable. Adapter rank, alpha, dropout, target modules, optimizer, learning rate, gradient clipping, precision, and checkpoint cadence are configuration fields with a checked-in example and run manifest.
* **Continual loop:** each accepted labelled/distilled batch updates LoRA, writes a checksum-bound checkpoint, then updates the EMA shadow adapter. EMA is `ema <- beta * ema + (1 - beta) * lora` after each optimizer step; the inference reader uses an atomically swapped, immutable EMA snapshot and never observes a partially written adapter.
* **Collapse guard:** maintain a frozen baseline evaluation set and a rolling window of labelled/distilled examples. A collapse event is raised when any one of these persists for `consecutive_windows=3`: (a) entity/span F1 falls below baseline by at least `absolute_f1_drop=0.15` or relative drop `0.20`; (b) validation loss exceeds `baseline_loss * 2.0`; (c) NaN/Inf or invalid span/schema output occurs; or (d) confidence calibration/coverage violates the configured floor. One NaN/Inf in weights is immediately terminal. On collapse, stop the adapter update, archive the failing adapter/EMA/optimizer and metrics, reload the immutable base plus a fresh zeroed LoRA and EMA initialized from it, and require a clean warm-up evaluation before resuming. Every reset receives a monotonic `reset_id` and reason.
* **Inference:** user-supplied schema labels and descriptions constrain extraction. Output is versioned JSON with source span offsets, label, normalized value, confidence, model revision, adapter checkpoint, EMA step, schema digest, and provenance. Unknown labels are rejected or explicitly reported as `unmapped` according to schema policy.
* **Distillation:** the operator selects one or more trusted teacher routes (for example GPT-6 Astra or DeepSeek 4.1f) and a user-configured source set. The request contains the schema, source excerpt, provenance, and a strict JSONL contract. Raw request/response, teacher model/revision, timestamp, source hash, prompt/template hash, and validation result are retained in a local, access-controlled cache. Teacher output is never treated as ground truth until schema, span, provenance, duplicate, safety, and confidence checks pass.
* **Schema:** a user-editable YAML/JSON document defines `schema_id`, version, entity types, descriptions, aliases, normalization, required/optional fields, relations, allowed source domains, language, and unknown-label policy. The schema digest is embedded in every distilled example, checkpoint, inference result, and evaluation report. Changing schema version starts a new adapter/data lineage unless an explicit migration is accepted.
* **Sources:** a user-editable source manifest lists URLs/files/connectors, retrieval policy, time range, license/terms, parser, inclusion/exclusion filters, and refresh cadence. Retrieval is reproducible and content-addressed; unapproved or unverifiable sources are excluded from distillation.
* **5090 target:** CUDA availability, compute capability, driver/toolkit, BF16/FP16 support, VRAM headroom, throughput, p50/p95 latency, and thermal/power observations are captured on the test host. CPU fallback may support development smoke tests but cannot satisfy the Stage 0 5090 gate.

### 1.2 Explicit Stage 0 concurrency vector

This is a bounded documentation-and-implementation run with one worker lane. Values are explicit and immutable for this stage; changing them requires a policy migration.

```yaml
logical_claim_cap: 1
persistent_service_record_cap: not_applicable
agent_execution_cap: 1
startup_reservation_cap: 1
launch_fanout_per_wave: 1
live_transport_cap: 1
authenticated_goal_cap: 1
running_turn_cap: 1
outbound_request_rate: 1
outbound_request_window_seconds: 60
in_flight_request_cap: 1
max_outstanding_requests_per_execution: 1
integration_cap: 1
validator_cap: 1
exact_path_conflict_cap: 1
desired_live_target: 1
hard_worker_cap: 1
accelerator_validator_leases: 1
nested_agents: forbidden
worker_lifecycle: bounded
```

Host observations may reduce admission and must persist a concrete reason; they may not raise, infer, or replace a value. Underfill reasons are dependency, path conflict, startup, host-resource, external-limit, route, validator, or breaker records. The request-start breaker opens on a rate/in-flight violation, host pressure, or unauthorized continuation and can be reset only by an audited operator action.

## 2. Acceptance gates

A gate is passable only with recorded evidence in the relevant handoff/manifest and a Master rerun. A worker self-test is provisional.

| Gate | Required evidence |
|---|---|
| G0 Specification | Blueprint parses; all IDs unique; all dependencies resolve and are acyclic; frozen spec, concurrency vector, authority paths, and transport policy are present and hashed into runtime state. |
| G1 Schema/source | Example schema and source manifest validate; schema digest and source hashes are deterministic; unsupported labels/domains and missing provenance fail closed. |
| G2 Distillation | Teacher requests honor selected route and source allowlist; cache is content-addressed; JSONL examples pass schema/span/provenance/quality checks; no secret is persisted. |
| G3 Model/LoRA | GLiNER2 manifest matches 205M architecture/revision; base is immutable; LoRA checkpoint/resume and adapter-only update are tested. |
| G4 EMA/live inference | EMA update is numerically correct; atomic snapshot swap prevents partial reads; inference exposes EMA step/checkpoint/schema digest and runs while a training update is active. |
| G5 Collapse recovery | Synthetic degradation triggers the configured guard; failing adapter/EMA/optimizer are archived; fresh LoRA/EMA starts from base; reset is logged and clean warm-up is required before serving. |
| G6 5090 | On the designated 5090 host, CUDA/model load/train/infer smoke passes, VRAM stays within budget, p50/p95 latency and throughput are recorded, and no CPU-only result is accepted as the hardware gate. |
| G7 Delivery | Public GitHub repository URL, commit/branch/visibility and manifest are recorded; remote copy has matching file manifest and runs the remote smoke/5090 checks. |
| G8 Master completion | All applicable repository validators pass; no `[ ]` or `[_]`, handoff/integration/repair queue is empty; Gantt digest and monitoring index are current and every checklist ID appears exactly once. |

## 3. Authoritative checklist and DAG

Each row is a stable claim. `Depends on` is the only execution edge source. `Owned paths` are repository-relative and are not permission to edit another claim's files. Dates are intentionally absent until a run records timestamps; the Gantt must show unscheduled items rather than inventing dates.

### 3.1 Governance and contracts

- [ ] **ZJ-001** Freeze GLiNER2 205M + LoRA + EMA architecture and collapse policy in model manifest/config. **Depends on:** —. **Owner:** Master. **Owned paths:** `config/model.yaml`, `artifacts/model_manifest.json`, `Docs/researches/**` (evidence only). **Gate:** G0,G3,G4,G5.
- [ ] **ZJ-002** Initialize repository layout, ignore/runtime policy, authority paths, checklist parser, and digest tooling. **Depends on:** ZJ-001. **Owner:** Master. **Owned paths:** `.gitignore`, `src/`, `tests/`, `scripts/`, `Docs/`. **Gate:** G0.
- [ ] **ZJ-003** Materialize the bounded execution controller contract: task isolation, tmux/Codex transport, leases, handoff schema, and cleanup checks. **Depends on:** ZJ-002. **Owner:** Master. **Owned paths:** `execution/`, `scripts/`, `tests/execution/`. **Gate:** G0,G8.
- [ ] **ZJ-004** Define versioned schema, source-manifest, provenance, teacher-request, distilled-example, checkpoint, inference, and evaluation contracts. **Depends on:** ZJ-002. **Owner:** Schema worker. **Owned paths:** `schemas/`, `config/`, `src/contracts/`, `tests/contracts/`. **Gate:** G1,G2,G3,G4.

### 3.2 Configurable schema and source ingestion

- [ ] **ZJ-010** Implement schema loader/validator, digesting, migrations, unknown-label policy, and example “2026 best tech stack” schema. **Depends on:** ZJ-004. **Owner:** Schema worker. **Owned paths:** `src/schema/`, `schemas/`, `tests/schema/`, `examples/schema_2026_best_tech_stack.yaml`. **Gate:** G1.
- [ ] **ZJ-011** Implement user-selectable source manifest and reproducible retrieval/parser connectors with license/domain/time filters. **Depends on:** ZJ-004. **Owner:** Data worker. **Owned paths:** `src/sources/`, `config/sources.example.yaml`, `tests/sources/`. **Gate:** G1,G2.
- [ ] **ZJ-012** Implement trusted-teacher routing (Astra/DeepSeek-compatible adapters), prompt/template versioning, rate/error handling, and secret-free request logging. **Depends on:** ZJ-004,ZJ-011. **Owner:** Distillation worker. **Owned paths:** `src/distill/teachers/`, `config/teachers.example.yaml`, `tests/distill/`. **Gate:** G2.
- [ ] **ZJ-013** Implement content-addressed provenance store and dataset lineage manifest. **Depends on:** ZJ-011. **Owner:** Data worker. **Owned paths:** `src/data/`, `artifacts/provenance/`, `tests/data/`. **Gate:** G1,G2.
- [ ] **ZJ-014** Generate and validate distilled JSONL from selected sources and teacher route, including span/schema/quality checks and a reproducible sample. **Depends on:** ZJ-010,ZJ-012,ZJ-013. **Owner:** Distillation worker. **Owned paths:** `src/distill/`, `artifacts/datasets/`, `tests/distill/`. **Gate:** G2.

### 3.3 GLiNER2, LoRA, EMA, and live inference

- [ ] **ZJ-020** Implement pinned GLiNER2 205M loader, tokenizer, device/precision checks, and immutable base-weight policy. **Depends on:** ZJ-001,ZJ-002. **Owner:** Model worker. **Owned paths:** `src/model/`, `config/model.yaml`, `tests/model/`. **Gate:** G3,G6.
- [ ] **ZJ-021** Implement LoRA adapter construction, optimizer/checkpoint/resume, gradient clipping, and adapter-only persistence. **Depends on:** ZJ-020. **Owner:** Training worker. **Owned paths:** `src/train/`, `config/lora.yaml`, `tests/train/`. **Gate:** G3.
- [ ] **ZJ-022** Implement EMA shadow-adapter update, checksum snapshots, atomic publication, restore, and step metadata. **Depends on:** ZJ-021. **Owner:** Training worker. **Owned paths:** `src/train/ema.py`, `src/runtime/checkpoints.py`, `tests/train/test_ema.py`. **Gate:** G4.
- [ ] **ZJ-023** Implement concurrent continual-training/inference service using immutable EMA snapshots and bounded queues/backpressure. **Depends on:** ZJ-022,ZJ-020. **Owner:** Runtime worker. **Owned paths:** `src/runtime/`, `tests/runtime/`. **Gate:** G4.
- [ ] **ZJ-024** Implement baseline/rolling metrics, confidence/coverage checks, and configurable collapse detector with the frozen Stage 0 defaults. **Depends on:** ZJ-021,ZJ-022,ZJ-004. **Owner:** Evaluation worker. **Owned paths:** `src/eval/`, `config/collapse.yaml`, `tests/eval/`. **Gate:** G5.
- [ ] **ZJ-025** Implement collapse transaction: stop updates, archive failing state, reset LoRA/EMA/optimizer from base, increment reset ID, warm-up gate, and audit event. **Depends on:** ZJ-024,ZJ-020. **Owner:** Training worker. **Owned paths:** `src/train/recovery.py`, `src/runtime/`, `tests/train/test_recovery.py`. **Gate:** G5.
- [ ] **ZJ-026** Implement schema-constrained inference output, span offsets, normalization, confidence, provenance, adapter/EMA metadata, and unmapped-label behavior. **Depends on:** ZJ-010,ZJ-020,ZJ-022. **Owner:** Inference worker. **Owned paths:** `src/infer/`, `tests/infer/`. **Gate:** G1,G4.
- [ ] **ZJ-027** Integrate source/distillation batches, LoRA training, EMA publication, inference, metrics, and recovery in one deterministic end-to-end harness. **Depends on:** ZJ-014,ZJ-023,ZJ-025,ZJ-026. **Owner:** Master. **Owned paths:** `src/zenjev/`, `tests/e2e/`, `scripts/smoke.py`. **Gate:** G2,G4,G5.

### 3.4 Validation, observability, and operator surfaces

- [ ] **ZJ-030** Provision and validate the 5090 test environment, CUDA/toolkit/driver, model cache, precision support, and VRAM budget. **Depends on:** ZJ-001,ZJ-020. **Owner:** Hardware worker. **Owned paths:** `scripts/validate_5090.py`, `artifacts/hardware/`, `Docs/runbooks/`. **Gate:** G6.
- [ ] **ZJ-031** Measure 5090 train/infer throughput, p50/p95 latency, peak VRAM, thermal/power observations, and concurrent train/infer stability. **Depends on:** ZJ-027,ZJ-030. **Owner:** Hardware worker. **Owned paths:** `benchmarks/`, `artifacts/benchmarks/`, `tests/benchmarks/`. **Gate:** G6.
- [ ] **ZJ-032** Run quality evaluation on held-out human/distilled data, compare base/LoRA/EMA, and publish error/confidence/calibration report. **Depends on:** ZJ-014,ZJ-027. **Owner:** Evaluation worker. **Owned paths:** `artifacts/evaluations/`, `src/eval/`, `tests/eval/`. **Gate:** G2,G4,G5.
- [ ] **ZJ-033** Execute a collapse-and-recovery drill on the 5090 (or a faithful simulator plus one hardware smoke), prove reset thresholds and warm-up gate. **Depends on:** ZJ-025,ZJ-031. **Owner:** Evaluation worker. **Owned paths:** `artifacts/recovery/`, `tests/recovery/`. **Gate:** G5,G6.
- [ ] **ZJ-034** Audit source/license/provenance, teacher request redaction, secret handling, artifact retention, and schema lineage. **Depends on:** ZJ-011,ZJ-012,ZJ-013,ZJ-014. **Owner:** Master. **Owned paths:** `artifacts/audits/`, `Docs/runbooks/`. **Gate:** G1,G2,G7.
- [ ] **ZJ-040** Provide CLI/API and operator runbook for schema/source selection, distillation, train/serve, metrics, reset, and manifest inspection. **Depends on:** ZJ-010,ZJ-011,ZJ-027. **Owner:** Docs worker. **Owned paths:** `src/cli/`, `Docs/runbooks/`, `README.md`, `tests/cli/`. **Gate:** G1,G4,G5.
- [ ] **ZJ-041** Produce reproducibility bundle: lockfiles, configs, model/data/checkpoint manifests, evaluation reports, and exact validation commands. **Depends on:** ZJ-031,ZJ-032,ZJ-034,ZJ-040. **Owner:** Master. **Owned paths:** `artifacts/repro/`, `Docs/runbooks/`, `requirements*.txt`, `pyproject.toml`. **Gate:** G0,G6,G7.

### 3.5 Publication, remote move, and final acceptance

- [ ] **ZJ-042** Run the complete local acceptance profile and reconcile blueprint, handoffs, manifests, and generated Gantt. **Depends on:** ZJ-003,ZJ-027,ZJ-031,ZJ-033,ZJ-041. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_gantt.md`, `.zenjev/runtime/`. **Gate:** G0–G8 as applicable.
- [ ] **ZJ-043** Initialize/verify public `weiyangzen/zenjev`, commit the accepted tree, push the recorded branch, and store URL/commit/visibility receipt. **Depends on:** ZJ-042. **Owner:** Master. **Owned paths:** `.git/` (metadata), `artifacts/delivery/github_receipt.json`. **Gate:** G7.
- [ ] **ZJ-044** Transfer the complete accepted tree to `sansha@192.168.50.38:/home/sansha/zenjev`, verify a full manifest/checksum, and run remote smoke plus 5090 checks. **Depends on:** ZJ-042,ZJ-043. **Owner:** Hardware worker/Master. **Owned paths:** `artifacts/delivery/remote_receipt.json`, `artifacts/hardware/remote_*.json`. **Gate:** G6,G7.
- [ ] **ZJ-045** Master final acceptance: all validators and delivery receipts pass; no unfinished checklist/handoff/repair item; atomically regenerate and digest-check the final Gantt; perform scoped cleanup. **Depends on:** ZJ-043,ZJ-044. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_gantt.md`, `.zenjev/runtime/`. **Gate:** G8.

## 4. Handoff, integration, and cleanup protocol

1. A claim is reserved only after dependency, exact-path conflict, and the frozen concurrency vector admit it. Its immutable claim card includes the item ID, run ID, baseline/spec digests, owned paths, commands, deadline, and result path.
2. The worker receives one task-local tmux/Codex TUI lane and one short `/goal`. It acquires both running-turn and outbound-request leases before Enter. Authentication must prove tmux socket/session, pane PID/start time, cwd, private `CODEX_HOME`, route, thread, goal, and claim objective.
3. The worker writes only its declared paths and emits `status=self_tested` with checksums. The controller harvests the handoff before pruning any process and marks the item `[_]`; this is still unfinished.
4. Master applies a conflict-safe patch to the canonical checkout, runs the item and dependency gates, updates manifests and the Gantt projection, and only then changes `[_]` to `[x]`. Failed validation remains a durable repair entry and does not erase the handoff.
5. Cleanup is scoped to controller-owned runtime and task descendants. Completion cleanup requires zero `[ ]`, zero `[_]`, no handoff/integration/repair backlog, all gates pass, final Gantt freshness, no live task transport, and preserved source/delivery receipts.

## 5. Required status and Gantt projection fields

At every scheduler tick, the generated Gantt must expose checklist counts (`[ ]`, `[_]`, `[x]`), each ID exactly once, dependencies, owner/claim, owned paths, implementation/validation/integration frontiers, and durable startup/live/handoff/integration/repair/blocked state. It must separately report logical claims, admitted executions, startup reservations, live tmux transports, authenticated goals, running turns, request starts/window, in-flight and outstanding requests, breaker state, validator/integration leases, and every persisted underfill reason. Unknown timing remains in an **Unscheduled** section; fabricated dates are forbidden. The Gantt is never parsed as authority and contains no mutable checklist marks.

## 6. Policy migration record

No migration is active for the initial Stage 0 specification. A future migration must state the old/new specification digests, affected checklist IDs and claims, reason, operator, timestamp, compatibility decision, and regenerated Gantt digest before any new launch.
