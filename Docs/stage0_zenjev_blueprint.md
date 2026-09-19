# Stage 0 ZenJev Blueprint (authoritative)

**Status:** authoritative implementation plan; automated execution controller disabled
**Authority:** This file is the only mutable requirement/checklist authority for ZenJev. `Docs/researches/**` explains decisions but cannot change checklist state or acceptance. `Docs/stage0_zenjev_blueprint_Gantt.md` is a generated, read-only projection of this file and the durable execution ledgers.
**Scope:** build the first usable ZenJev pipeline: configurable schema and sources, trusted-teacher distillation, GLiNER2 205M with LoRA continual training, EMA-backed live inference, collapse detection/restart, 5090 validation, public repository publication, and verified transfer to the 5090 host.

## 1. Frozen repository and execution specification

The repository contract below governs authorized manual implementation. The user requested the execution skill’s blueprint format, not a running execution cron. Producing this blueprint, implementing Jev, publishing the authorized repository, and transferring it do not require creating or launching a controller. Future automated execution requires the unresolved operator inputs in section 1.2; no controller may be generated, materialized, reserved, or launched until those inputs are explicitly supplied and accepted. Manual work in this session is not a claim admitted by that future controller. A change to this specification is an explicit policy migration.

| Field | Frozen value |
|---|---|
| Canonical repository root | `/home/sansha/Github/zenjev` |
| Authoritative blueprint | `Docs/stage0_zenjev_blueprint.md` |
| Same-name Gantt companion | `Docs/stage0_zenjev_blueprint_Gantt.md` (case-sensitive rule: the lowercase stem does not end in `Blueprint`, so append `_Gantt` to the entire stem) |
| Checklist grammar | Markdown task rows `- [ ] **ZJ-###**`, `- [_] **ZJ-###**`, or `- [x] **ZJ-###**`; duplicate IDs, other marks, and missing IDs fail closed |
| Dependency authority | `Depends on` field in each checklist row below; edges must resolve to existing IDs and be acyclic; document order is never an edge |
| Planned runtime root (inactive) | `.zenjev/runtime/` (controller-owned and ignored by source packaging) |
| Planned task root (inactive) | `.zenjev/runtime/tasks/<claim-id>/<run-id>/` with `work/`, private `codex-home/`, `tmux.sock`, immutable `claim.json`, and `result.json` |
| Worker writable scope | Claim-card-declared repository-relative paths only; workers may not write the canonical checkout or this blueprint |
| Planned handoff store (inactive) | `.zenjev/runtime/handoffs/<claim-id>/<run-id>/` with checksum-bound result and patch; harvested before process pruning |
| Platform/transport prerequisite | Platform remains unresolved. If Codex is explicitly selected, each admitted generation requires its own tmux server, interactive TUI, private writable `CODEX_HOME`, and exactly one authenticated `/goal`; no alternate transport is allowed. |
| Forbidden transports | Codex app-server, app-server JSON-RPC, `codex exec`, shared Codex daemons, shared tmux servers, shared writable Codex state, and no-tmux Codex workers |
| Lifecycle prerequisite | Unresolved: operator must explicitly select `bounded` or `persistent_pool`, continuation, replacement, terminal and stop policies before automation activation. |
| Nested-agent prerequisite | Unresolved: no controller or nested worker may start before explicit policy and complete accounting are accepted. |
| Route prerequisite | Unresolved: operator must select platform/route policy before activation. For Codex, every explicitly frozen route field plus cwd, thread, goal and private state identity must authenticate before a lane counts as live. |
| Request contract | Acquire atomic running-turn and outbound-request leases before one Enter; one outstanding request per execution; request starts and in-flight requests are separate counters |
| Result schema | `schema_version`, `claim_id`, `run_id`, `status=self_tested`, `baseline_digest`, `spec_digest`, `changed_paths`, `patch_sha256`, `commands`, `artifacts`, `dependencies`, `failure_class`, and `handoff_timestamp` |
| Master rule | Only canonical Master integrates patches, runs gates, reconciles completion surfaces, and advances `[_]` to `[x]`; workers can only produce a checksum-valid `self_tested` handoff and never write `[x]` |
| Completion surfaces | This blueprint, the generated Gantt, research documents, implementation/tests, model/data/run manifests, GitHub publication receipt, and remote transfer receipt |
| Source policy | No secrets, provider keys, or plaintext remote password in the repository; source records contain URI, retrieval time, content hash, license/terms, and user approval |
| Validation profile | `python3 -m compileall -q jev scripts`; `python3 -m pytest -q`; `python3 scripts/validate_blueprint.py --check`; `python3 scripts/smoke.py`; schema/config lint; deterministic smoke inference; train/EMA/collapse-recovery simulation; 5090 CUDA/VRAM/latency benchmark; manifest checksum audit. A missing applicable validator is a failure, not a skip. |
| Git policy | The user has authorized initialization, commit and push; each publication states its actual validation status and may deliver a work-in-progress plan before implementation acceptance. Public remote must be `weiyangzen/zenjev`; the publication receipt records URL, commit, branch, and visibility. |
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

### 1.2 Automated execution prerequisites (unresolved)

The current user prompt supplies no concurrency vector, lifecycle or execution route. The machine-readable block below is an inventory of missing inputs, **not an accepted execution specification**. `null` means unresolved and must never become a default, a value inferred from this session’s agent count, or `not_applicable`. Explicit `not_applicable` is acceptable only in a future operator prompt and only for dimensions the selected lifecycle/schema permits.

<!-- execution-prerequisites:start -->
```json
{
  "schema_version": 1,
  "controller_enabled": false,
  "operator_prompt": null,
  "operator_prompt_sha256": null,
  "platform": null,
  "route_policy": null,
  "worker_lifecycle": null,
  "nested_agent_policy": null,
  "continuation_policy": null,
  "replacement_policy": null,
  "terminal_stop_policy": null,
  "scheduler_cadence": null,
  "lease_policy": null,
  "budgets": null,
  "cron_marker": null,
  "cooldown_policy": null,
  "breaker_reset_policy": null,
  "concurrency": {
    "logical_claim_cap": null,
    "persistent_service_record_cap": null,
    "agent_execution_cap": null,
    "startup_reservation_cap": null,
    "launch_fanout_per_wave": null,
    "live_transport_cap": null,
    "authenticated_goal_cap": null,
    "running_turn_cap": null,
    "outbound_request_rate": null,
    "outbound_request_window_seconds": null,
    "in_flight_request_cap": null,
    "max_outstanding_requests_per_execution": null,
    "integration_cap": null,
    "validator_cap": null,
    "exact_path_conflict_cap": null,
    "desired_live_target": null,
    "hard_worker_cap": null,
    "accelerator_validator_leases": null
  }
}
```
<!-- execution-prerequisites:end -->

**Activation gate:** disabled. A future operator instruction must provide the complete vector, lifecycle, route, nested-agent/accounting policy, replacement/continuation and stop policies, cadence, leases, budgets, exact cron marker, cooldown and breaker-reset rules. Persist the exact accepted prompt bytes and SHA-256, then bind the digest to the migrated specification, claims, ledgers, admission receipts and Gantt. Missing, partial, stale, inherited or ambiguous values fail closed before controller generation or any launch side effect. Host probes can only reduce explicitly authorized capacity. The conditional Codex contract allows at most one outstanding request per execution, which still must appear explicitly in the accepted operator vector.

The current governance utility only reads the blueprint and atomically renders its companion. It creates no runtime root, claim, tmux server, cron, goal or model request. It does not implement an execution controller. If automation is later authorized, its own transport, concurrency, request-lease, lifecycle, liveness, handoff, cleanup and two-repository portability tests become activation gates in that migration.

### 1.3 Recorded governance event

<!-- governance-event:start -->
```json
{
  "event_id": "governance-policy-correction",
  "recorded_at": "2026-09-19T18:08:30Z",
  "description": "Removed inferred concurrency, disabled automated execution, and corrected the companion naming policy."
}
```
<!-- governance-event:end -->

This records a documentation edit only. It is neither an implementation acceptance timestamp nor a task-duration estimate. Checklist items remain unscheduled until their actual timing is recorded.

## 2. Acceptance gates

A gate is passable only with recorded evidence in the relevant handoff/manifest and a Master rerun. A worker self-test is provisional.

| Gate | Required evidence |
|---|---|
| G0 Specification | Blueprint parses; all IDs unique; all dependencies resolve and are acyclic; authority paths and prerequisite block validate; controller remains disabled with unresolved operator inputs; source/specification digests, all index fields, counts and derived frontiers exactly match the generated Gantt. Document delivery does not require a controller or accepted implementation. |
| G1 Schema/source | Example schema and source manifest validate; schema digest and source hashes are deterministic; unsupported labels/domains and missing provenance fail closed. |
| G2 Distillation | Teacher requests honor selected route and source allowlist; cache is content-addressed; JSONL examples pass schema/span/provenance/quality checks; no secret is persisted. |
| G3 Model/LoRA | GLiNER2 manifest matches 205M architecture/revision; base is immutable; LoRA checkpoint/resume and adapter-only update are tested. |
| G4 EMA/live inference | EMA update is numerically correct; atomic snapshot swap prevents partial reads; inference exposes EMA step/checkpoint/schema digest and runs while a training update is active. |
| G5 Collapse recovery | Synthetic degradation triggers the configured guard; failing adapter/EMA/optimizer are archived; fresh LoRA/EMA starts from base; reset is logged and clean warm-up is required before serving. |
| G6 5090 | On the designated 5090 host, CUDA/model load/train/infer smoke passes, VRAM stays within budget, p50/p95 latency and throughput are recorded, and no CPU-only result is accepted as the hardware gate. |
| G7 Delivery | Public GitHub repository URL, commit/branch/visibility and manifest are recorded; remote copy has matching file manifest and runs the remote smoke/5090 checks. |
| G8 Master completion | All applicable repository validators pass; no `[ ]` or `[_]`, handoff/integration/repair queue is empty; Gantt digest and monitoring index are current and every checklist ID appears exactly once. |

## 3. Authoritative checklist and DAG

Each row is a stable planned work item, not a live claim. Role names are responsibility labels, not agent identities. `Depends on` is the only execution edge source. `Owned paths` are repository-relative and are not permission to edit another claim's files. Dates are intentionally absent until a run records timestamps; the Gantt must show unscheduled items rather than inventing dates.

### 3.1 Governance and contracts

- [ ] **ZJ-001** Freeze GLiNER2 205M + LoRA + EMA architecture and collapse policy in model manifest/config. **Depends on:** —. **Owner:** Master. **Owned paths:** `configs/example.yaml`, `artifacts/model_manifest.json`, `Docs/researches/**` (evidence only). **Gate:** G0,G3,G4,G5.
- [ ] **ZJ-002** Initialize repository layout, ignore/runtime policy, authority paths, checklist parser, and digest tooling. **Depends on:** ZJ-001. **Owner:** Master. **Owned paths:** `.gitignore`, `jev/`, `tests/`, `scripts/`, `Docs/`. **Gate:** G0.
- [ ] **ZJ-003** Document automation prerequisites and validate the authoritative checklist/DAG plus deterministic read-only Gantt generation; controller construction and launch are outside this item. **Depends on:** ZJ-002. **Owner:** Master. **Owned paths:** `scripts/validate_blueprint.py`, `tests/test_governance.py`, `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`. **Gate:** G0.
- [ ] **ZJ-004** Define versioned schema, source-manifest, provenance, teacher-request, distilled-example, checkpoint, inference, and evaluation contracts. **Depends on:** ZJ-002. **Owner:** Schema worker. **Owned paths:** `jev/config.py`, `jev/distill.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G1,G2,G3,G4.

### 3.2 Configurable schema and source ingestion

- [ ] **ZJ-010** Implement schema loader/validator, digesting, migrations, unknown-label policy, and example “2026 best tech stack” schema. **Depends on:** ZJ-004. **Owner:** Schema worker. **Owned paths:** `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G1.
- [ ] **ZJ-011** Implement user-selectable source manifest and reproducible retrieval/parser connectors with license/domain/time filters. **Depends on:** ZJ-004. **Owner:** Data worker. **Owned paths:** `jev/sources.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G1,G2.
- [ ] **ZJ-012** Implement trusted-teacher routing (Astra/DeepSeek-compatible adapters), prompt/template versioning, rate/error handling, and secret-free request logging. **Depends on:** ZJ-004,ZJ-011. **Owner:** Distillation worker. **Owned paths:** `jev/distill.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G2.
- [ ] **ZJ-013** Implement content-addressed provenance store and dataset lineage manifest. **Depends on:** ZJ-011. **Owner:** Data worker. **Owned paths:** `jev/sources.py`, `jev/distill.py`, `artifacts/provenance/`, `tests/test_jev.py`. **Gate:** G1,G2.
- [ ] **ZJ-014** Generate and validate distilled JSONL from selected sources and teacher route, including span/schema/quality checks and a reproducible sample. **Depends on:** ZJ-010,ZJ-012,ZJ-013. **Owner:** Distillation worker. **Owned paths:** `jev/distill.py`, `artifacts/datasets/`, `tests/test_jev.py`. **Gate:** G2.

### 3.3 GLiNER2, LoRA, EMA, and live inference

- [ ] **ZJ-020** Implement pinned GLiNER2 205M loader, tokenizer, device/precision checks, and immutable base-weight policy. **Depends on:** ZJ-001,ZJ-002. **Owner:** Model worker. **Owned paths:** `jev/model.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G3,G6.
- [ ] **ZJ-021** Implement LoRA adapter construction, optimizer/checkpoint/resume, gradient clipping, and adapter-only persistence. **Depends on:** ZJ-020. **Owner:** Training worker. **Owned paths:** `jev/training.py`, `jev/model.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G3.
- [ ] **ZJ-022** Implement EMA shadow-adapter update, checksum snapshots, atomic publication, restore, and step metadata. **Depends on:** ZJ-021. **Owner:** Training worker. **Owned paths:** `jev/ema.py`, `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py`. **Gate:** G4.
- [ ] **ZJ-023** Implement concurrent continual-training/inference service using immutable EMA snapshots and bounded queues/backpressure. **Depends on:** ZJ-022,ZJ-020. **Owner:** Runtime worker. **Owned paths:** `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py`. **Gate:** G4.
- [ ] **ZJ-024** Implement baseline/rolling metrics, confidence/coverage checks, and configurable collapse detector with the frozen Stage 0 defaults. **Depends on:** ZJ-021,ZJ-022,ZJ-004. **Owner:** Evaluation worker. **Owned paths:** `jev/drift.py`, `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G5.
- [ ] **ZJ-025** Implement collapse transaction: stop updates, archive failing state, reset LoRA/EMA/optimizer from base, increment reset ID, warm-up gate, and audit event. **Depends on:** ZJ-024,ZJ-020. **Owner:** Training worker. **Owned paths:** `jev/training.py`, `jev/runtime.py`, `tests/test_jev.py`. **Gate:** G5.
- [ ] **ZJ-026** Implement schema-constrained inference output, span offsets, normalization, confidence, provenance, adapter/EMA metadata, and unmapped-label behavior. **Depends on:** ZJ-010,ZJ-020,ZJ-022. **Owner:** Inference worker. **Owned paths:** `jev/model.py`, `jev/runtime.py`, `tests/test_jev.py`. **Gate:** G1,G4.
- [ ] **ZJ-027** Integrate source/distillation batches, LoRA training, EMA publication, inference, metrics, and recovery in one deterministic end-to-end harness. **Depends on:** ZJ-014,ZJ-023,ZJ-025,ZJ-026. **Owner:** Master. **Owned paths:** `jev/runtime.py`, `jev/training.py`, `jev/cli.py`, `tests/test_jev.py`, `scripts/smoke.py`. **Gate:** G2,G4,G5.

### 3.4 Validation, observability, and operator surfaces

- [ ] **ZJ-030** Provision and validate the 5090 test environment, CUDA/toolkit/driver, model cache, precision support, and VRAM budget. **Depends on:** ZJ-001,ZJ-020. **Owner:** Hardware worker. **Owned paths:** `scripts/validate_5090.py`, `artifacts/hardware/`, `Docs/runbooks/`. **Gate:** G6.
- [ ] **ZJ-031** Measure 5090 train/infer throughput, p50/p95 latency, peak VRAM, thermal/power observations, and concurrent train/infer stability. **Depends on:** ZJ-027,ZJ-030. **Owner:** Hardware worker. **Owned paths:** `scripts/smoke_5090.py`, `benchmarks/`, `artifacts/benchmarks/`, `tests/benchmarks/`. **Gate:** G6.
- [ ] **ZJ-032** Run quality evaluation on held-out human/distilled data, compare base/LoRA/EMA, and publish error/confidence/calibration report. **Depends on:** ZJ-014,ZJ-027. **Owner:** Evaluation worker. **Owned paths:** `artifacts/evaluations/`, `jev/drift.py`, `tests/test_jev.py`. **Gate:** G2,G4,G5.
- [ ] **ZJ-033** Execute a collapse-and-recovery drill on the 5090 (or a faithful simulator plus one hardware smoke), prove reset thresholds and warm-up gate. **Depends on:** ZJ-025,ZJ-031. **Owner:** Evaluation worker. **Owned paths:** `artifacts/recovery/`, `tests/recovery/`. **Gate:** G5,G6.
- [ ] **ZJ-034** Audit source/license/provenance, teacher request redaction, secret handling, artifact retention, and schema lineage. **Depends on:** ZJ-011,ZJ-012,ZJ-013,ZJ-014. **Owner:** Master. **Owned paths:** `artifacts/audits/`, `Docs/runbooks/`. **Gate:** G1,G2,G7.
- [ ] **ZJ-040** Provide CLI/API and operator runbook for schema/source selection, distillation, train/serve, metrics, reset, and manifest inspection. **Depends on:** ZJ-010,ZJ-011,ZJ-027. **Owner:** Docs worker. **Owned paths:** `jev/cli.py`, `Docs/runbooks/`, `README.md`, `tests/test_jev.py`. **Gate:** G1,G4,G5.
- [ ] **ZJ-041** Produce reproducibility bundle: lockfiles, configs, model/data/checkpoint manifests, evaluation reports, and exact validation commands. **Depends on:** ZJ-031,ZJ-032,ZJ-034,ZJ-040. **Owner:** Master. **Owned paths:** `artifacts/repro/`, `Docs/runbooks/`, `requirements*.txt`, `pyproject.toml`. **Gate:** G0,G6,G7.

### 3.5 Publication, remote move, and final acceptance

- [ ] **ZJ-042** Run the complete local acceptance profile and reconcile blueprint, handoffs, manifests, and generated Gantt. **Depends on:** ZJ-003,ZJ-027,ZJ-031,ZJ-033,ZJ-041. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/`. **Gate:** G0–G8 as applicable.
- [ ] **ZJ-043** Initialize/verify public `weiyangzen/zenjev`, commit the accepted tree, push the recorded branch, and store URL/commit/visibility receipt. **Depends on:** ZJ-042. **Owner:** Master. **Owned paths:** `.git/` (metadata), `artifacts/delivery/github_receipt.json`. **Gate:** G7.
- [ ] **ZJ-044** Transfer the complete accepted tree to `sansha@192.168.50.38:/home/sansha/zenjev`, verify a full manifest/checksum, and run remote smoke plus 5090 checks. **Depends on:** ZJ-042,ZJ-043. **Owner:** Hardware worker/Master. **Owned paths:** `artifacts/delivery/remote_transfer_receipt.json`, `artifacts/hardware/remote_*.json`. **Gate:** G6,G7.
- [ ] **ZJ-045** Master final acceptance: all validators and delivery receipts pass; no unfinished checklist/handoff/repair item; atomically regenerate and digest-check the final Gantt; perform scoped cleanup. **Depends on:** ZJ-043,ZJ-044. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/`. **Gate:** G8.

## 4. Conditional automated handoff, integration, and cleanup protocol

This protocol applies only after a future explicit automation activation. It does not require a cron/controller for the current authorized manual work. Shared module and test paths in the checklist must be serialized or narrowed to disjoint exact paths in any future claims; broad planned ownership is not a concurrency lease. Manual integration uses the same evidence and Master acceptance discipline, without inventing worker claims or handoff ledgers.

1. A claim is reserved only after dependency, exact-path conflict, and the frozen concurrency vector admit it. Its immutable claim card includes the item ID, run ID, baseline/spec digests, owned paths, commands, deadline, and result path.
2. The worker receives one task-local tmux/Codex TUI lane and one short `/goal`. It acquires both running-turn and outbound-request leases before Enter. Authentication must prove tmux socket/session, pane PID/start time, cwd, private `CODEX_HOME`, route, thread, goal, and claim objective.
3. The worker writes only its declared paths and emits `status=self_tested` with checksums. The controller harvests the handoff before pruning any process and marks the item `[_]`; this is still unfinished.
4. Master applies a conflict-safe patch to the canonical checkout, runs the item and dependency gates, updates manifests and the Gantt projection, and only then changes `[_]` to `[x]`. Failed validation remains a durable repair entry and does not erase the handoff.
5. Cleanup is scoped to controller-owned runtime and task descendants. Completion cleanup requires zero `[ ]`, zero `[_]`, no handoff/integration/repair backlog, all gates pass, final Gantt freshness, no live task transport, and preserved source/delivery receipts.

## 5. Required status and Gantt projection fields

During current manual work, run `python3 scripts/validate_blueprint.py --generate` after a blueprint edit and `python3 scripts/validate_blueprint.py --check` before delivery. A future enabled controller must do the same atomically after every final state merge. The generated Gantt must expose checklist counts (`[ ]`, `[_]`, `[x]`), each ID exactly once, dependencies, owner/claim, owned paths, implementation/validation/integration frontiers, and durable startup/live/handoff/integration/repair/blocked state. It must separately report logical claims, admitted executions, startup reservations, live tmux transports, authenticated goals, running turns, request starts/window, in-flight and outstanding requests, breaker state, validator/integration leases, and every persisted underfill reason. Unknown timing remains in an **Unscheduled** section; fabricated dates are forbidden. The Gantt is never parsed as authority and contains no mutable checklist marks.

## 6. Policy migration record

Governance correction: the originally inferred single-worker vector and bounded lifecycle were invalid because the operator never supplied them. They are withdrawn, not accepted defaults. The companion path now follows the case-sensitive skill rule; fabricated relative task durations are removed. No implementation state was accepted by this correction. A future migration must state the old/new specification digests, affected checklist IDs and claims, reason, operator, timestamp, compatibility decision, and regenerated Gantt digest before any new launch.
