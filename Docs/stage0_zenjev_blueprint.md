# Stage 0 ZenJev Blueprint (authoritative)

**Status:** authoritative implementation plan; automated execution controller disabled
**Authority:** This file is the only mutable requirement/checklist authority for ZenJev. `Docs/researches/**` explains decisions but cannot change checklist state or acceptance. `Docs/stage0_zenjev_blueprint_Gantt.md` is a generated, read-only projection of this file and the durable execution ledgers.
**Scope:** build the first usable ZenJev pipeline: configurable schema and sources, continual pull consumption from configurable external standard message queues through a high-performance Rust bridge, trusted-teacher distillation, GLiNER2 205M with LoRA continual training, EMA-backed live inference, collapse detection/restart, the `jev-tool-task` request/response analyzer with model allowlist and deterministic routing, NVIDIA GPU validation, public repository publication, and verified transfer to the designated NVIDIA GPU host.

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
| MQ ingestion policy | The live training input is a user-configurable external standard message queue, not a fixed dataset. Adapter kind (`nats-jetstream` durable pull default, `kafka`, optional `iggy`), endpoints, stream/topic/subject, consumer group/durable name, start position, TLS, batch/byte limits, `ack_wait`, `max_ack_pending`/prefetch, retry ceiling, DLQ, spool path, and local transport path are config fields; credentials are secret references only. Unknown adapters, missing secret references, and non-TLS remote endpoints fail closed. |
| MQ delivery contract | End-to-end at-least-once with consumer-side idempotency by `record_id` plus canonical `content_sha256`. The bridge acks or commits a broker offset only after the trainer confirms durable replay-buffer/provenance acceptance. Malformed/oversized messages go to the configured DLQ; schema-digest mismatch is quarantined; unknown `observed_model` is rejected before extraction. Poison retries stop at the configured ceiling. Exactly-once is never claimed. |
| MQ bridge | Rust `mq/` crate (pinned stable toolchain, committed `Cargo.lock`) owning the broker protocol through a pluggable source trait: bounded local WAL spool, credit-based local Unix-socket delivery to the Python trainer, unified lag/ack/DLQ metrics. Python code never speaks a broker protocol directly. |
| Validation profile | `python3 -m compileall -q jev scripts`; `python3 -m pytest -q`; `python3 scripts/validate_blueprint.py --check`; `python3 scripts/smoke.py`; `cargo fmt --check`, `cargo clippy --locked -- -D warnings`, `cargo test --locked` in `mq/`; `python3 scripts/smoke_mq.py`; schema/config lint; deterministic smoke inference; train/EMA/collapse-recovery simulation; MQ replay/fault tests; broker-connected MQ pull-train soak; NVIDIA GPU CUDA/VRAM/latency benchmark; manifest checksum audit. A missing applicable validator is a failure, not a skip. |
| Git policy | The user has authorized initialization, commit and push; each publication states its actual validation status and may deliver a work-in-progress plan before implementation acceptance. Public remote must be `weiyangzen/zenjev`; the publication receipt records URL, commit, branch, and visibility. |
| Remote delivery | Target `sansha@192.168.50.38`, destination `/home/sansha/zenjev`; transfer through an authenticated channel supplied at run time, never by committing the provided password; verify a full file manifest and rerun smoke/NVIDIA GPU checks remotely. |

### 1.1 Stage 0 model and data contract (frozen)

* **Base model:** GLiNER2 205M. The exact model identifier, revision, tokenizer files, parameter count, and license are recorded in `artifacts/model_manifest.json`; startup rejects a revision whose parameter count or architecture does not match the manifest. Model downloads are pinned by revision/hash and cached outside the source tree.
* **Trainable state:** LoRA adapters only in Stage 0. Base weights are immutable. Adapter rank, alpha, dropout, target modules, optimizer, learning rate, gradient clipping, precision, and checkpoint cadence are configuration fields with a checked-in example and run manifest.
* **Continual loop:** each accepted labelled/distilled batch updates LoRA, writes a checksum-bound checkpoint, then updates the EMA shadow adapter. EMA is `ema <- beta * ema + (1 - beta) * lora` after each optimizer step; the inference reader uses an atomically swapped, immutable EMA snapshot and never observes a partially written adapter.
* **Collapse guard:** maintain a frozen baseline evaluation set and a rolling window of labelled/distilled examples. A collapse event is raised when any one of these persists for `consecutive_windows=3`: (a) entity/span F1 falls below baseline by at least `absolute_f1_drop=0.15` or relative drop `0.20`; (b) validation loss exceeds `baseline_loss * 2.0`; (c) NaN/Inf or invalid span/schema output occurs; or (d) confidence calibration/coverage violates the configured floor. One NaN/Inf in weights is immediately terminal. On collapse, stop the adapter update, archive the failing adapter/EMA/optimizer and metrics, reload the immutable base plus a fresh zeroed LoRA and EMA initialized from it, and require a clean warm-up evaluation before resuming. Every reset receives a monotonic `reset_id` and reason.
* **Inference:** user-supplied schema labels and descriptions constrain extraction. Output is versioned JSON with source span offsets, label, normalized value, confidence, model revision, adapter checkpoint, EMA step, schema digest, and provenance. Unknown labels are rejected or explicitly reported as `unmapped` according to schema policy.
* **Tool-task contract:** `configs/jev_tool_task.yaml` defines the standard `jev-tool-task` schema. A landing-pool record contains a request and response pair, source URI/hash and the observed provider/model metadata. The observed model is checked against the explicit allowlist before extraction; it is never guessed from response text. Request extraction returns finite `task_type` labels and `task_detail` spans. Response extraction returns `tool`, `programming_language`, `framework`, `technology`, and `model` spans plus `uses_*` relations. Request and response are extracted separately to prevent response leakage into task classification.
* **Decision boundary:** GLiNER2's classification API can score a finite `task_type` and `decision_choice` label set and its multi-task schema can extract records and relations in the same pass. It is a schema-conditioned extractor/classifier, not an autonomous planner or tool executor. Each `decision_choice` is retained with its probability; a deterministic ZenJev policy verifies model/tool/language/technology allowlists, confidence and evidence, then emits `allow`, `review`, or `reject`. Unknown models, unknown labels, low confidence, missing evidence and policy conflicts never invoke a tool.
* **Streaming boundary:** the current runtime already trains a private LoRA model while inference reads immutable EMA snapshots; `ContinuousLoRATrainer.train` is a synchronous iterable API. The execution target adds a bounded `submit/stop` stream adapter with one training writer, backpressure and the same atomic publication semantics. A background trainer thread/process is required for a caller that submits records while serving requests.
* **Distillation:** the operator selects one or more trusted teacher routes (for example GPT-6 Astra or DeepSeek 4.1f) and a user-configured source set. The request contains the schema, source excerpt, provenance, and a strict JSONL contract. Raw request/response, teacher model/revision, timestamp, source hash, prompt/template hash, and validation result are retained in a local, access-controlled cache. Teacher output is never treated as ground truth until schema, span, provenance, duplicate, safety, and confidence checks pass.
* **Schema:** a user-editable YAML/JSON document defines `schema_id`, version, entity types, descriptions, aliases, normalization, required/optional fields, relations, allowed source domains, language, and unknown-label policy. The schema digest is embedded in every distilled example, checkpoint, inference result, and evaluation report. Changing schema version starts a new adapter/data lineage unless an explicit migration is accepted.
* **Sources:** a user-editable source manifest lists URLs/files/connectors, retrieval policy, time range, license/terms, parser, inclusion/exclusion filters, and refresh cadence. Retrieval is reproducible and content-addressed; unapproved or unverifiable sources are excluded from distillation.
* **MQ ingestion:** training input may arrive continually from one or more user-configured external standard message queues instead of a fixed dataset; the source manifest remains the backfill/replay path and both feed the same content-addressed replay buffer. A Rust bridge (`mq/`) owns the broker protocol through a configurable adapter (default NATS JetStream durable pull; Kafka/Redpanda; optional Apache Iggy), validates a versioned envelope, deduplicates by `record_id` plus `content_sha256`, spools to a bounded local WAL, and delivers records to the trainer over a credit-based local stream. Consumption is at-least-once: the bridge acks or commits an offset only after the trainer durably accepts the record into the replay buffer or provenance store, so a slow trainer backpressures the broker instead of buffering without bound. Malformed/oversized messages go to the DLQ, schema-digest mismatch is quarantined, and poison retries stop at the configured ceiling. `kind=labelled_example` records enter continual LoRA/EMA training directly; `kind=tool_task_pair` records feed the §1.4 landing pool and must traverse the §1.1 validation/distillation gates before any training use. MQ credentials are secret references, never values.
* **NVIDIA GPU target:** CUDA availability, compute capability, driver/toolkit, BF16/FP16 support, VRAM headroom, throughput, p50/p95 latency, and thermal/power observations are captured on the test host. CPU fallback may support development smoke tests but cannot satisfy the Stage 0 NVIDIA GPU gate.

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

### 1.4 `jev-tool-task` decision contract

The standard record is deliberately split so response text cannot become an accidental feature of request classification:

```json
{
  "schema_version": 1,
  "request_id": "stable-source-key",
  "source": {"uri": "user-owned://landing/42", "content_sha256": "...", "retrieved_at": "...", "license": "..."},
  "observed_model": {"provider": "openai-compatible", "model_id": "gpt-6-astra"},
  "request": {"text": "Fix the Python parser bug and add a regression test."},
  "response": {"text": "Use pytest and Python ..."}
}
```

Admission rejects an `observed_model` that is absent from the configured allowlist before either text is sent to the extraction path. The request and response are then processed independently with the same versioned schema. The request output contains a finite `task_type` classification and evidence-bearing `task_detail` spans; the response output contains `tool`, `programming_language`, `framework`, `technology`, and `model` spans, typed fields, and `uses_*` relations. A classification schema may additionally expose the finite `decision_choice` technology-stack labels and constraints; constraints narrow labels and do not execute anything.

The decision classifier consumes a finite `decision_choices` technology-stack list from policy and returns every candidate with a probability, for example `[{"choice":"PyTorch","probability":0.91},{"choice":"Python","probability":0.82}]`. The deterministic router consumes this auditable list only after checking exact model/tool allowlists, confidence, evidence and conflict rules. It emits `{action: allow|review|reject, decision_choices: [...], reasons: [...]}`. `allow` is an authorization result for a separately implemented tool adapter; GLiNER2 never receives permission to invoke tools, shell commands or model endpoints directly. For the pinned 205M span checkpoint, classification, structured records and relations are supported by the official Schema API. Any constrained-classifier behavior used in production must be tested against the pinned GLiNER2 package and remains subordinate to the ZenJev policy layer.

### 1.5 Continual external-MQ ingestion contract

The pipeline consumes training data continually from one or more external, user-configured standard message queues. The MQ is a transport, not a fixed dataset: the source manifest of §1.1 remains valid for backfill and replay, while the live input is the configured subscription. A Rust bridge owns the broker protocol; Python training code consumes only a local, flow-controlled stream.

**Configured, not hard-coded.** The `mq` section of the user config declares the adapter kind, endpoints/DNS names, TLS material references, credential references, stream/topic/subject, consumer group or durable name, optional key/filter, start position, batch and byte limits, `ack_wait`, `max_ack_pending`/prefetch, retry ceiling, DLQ destination, dedup window, WAL spool path, and local transport path. Unknown adapter kinds, missing secret references, non-TLS remote endpoints, and empty allowlists fail closed. Changing broker or adapter must not change the training contract.

| Adapter | Role in Stage 0 | Reference implementation | Required semantics |
|---|---|---|---|
| `nats-jetstream` (default) | Durable pull consumer for a slow single-writer training loop | Rust `async-nats`; JetStream stream with limits or work-queue retention | Explicit ack; server-tracked delivery attempts; `max_ack_pending` backpressure; `ack_wait` above the slowest batch; `max_deliver` poison ceiling |
| `kafka` | Standard Kafka-protocol compatibility for existing clusters and Redpanda | Rust `rdkafka` (librdkafka) | `enable.auto.commit=false`; offset stored/committed only after downstream durable ack; in-order processing per partition; rebalance-safe offsets |
| `iggy` (optional) | Pure-Rust persistent log for explicitly configured Apache Iggy deployments | Iggy Rust SDK | Explicit `enforce_fsync`/flush thresholds and consumer-group offsets; cannot satisfy a gate until durability settings and clustering behavior are benchmarked |

The default is NATS JetStream durable pull because it maps exactly to this workload: the consumer controls the pace with `batch`/`expires`; the server retains unacked messages, redelivers after `ack_wait`, counts attempts across restarts, and applies broker-side backpressure through `max_ack_pending`. Kafka/`rdkafka` is the required compatibility path for standard clusters; Iggy is optional and is never a gate requirement. Adapter choice is a config field, and the bridge exposes one envelope/ack contract to the trainer regardless of adapter.

**Message envelope.** Every message is UTF-8 JSON with `envelope_version`, `record_id`, `content_sha256`, `schema` (`id`, `version`, `digest`), `kind` (`labelled_example` | `tool_task_pair` | `raw_document`), `source` (`uri`, `content_sha256`, `retrieved_at`, `license`), optional `observed_model`, optional `request`/`response`, optional `labels`/`example`, and `created_at`. `record_id` plus the canonical `content_sha256` is the idempotency key. CBOR is accepted only under an explicit config flag.

**Delivery and recovery semantics.**

1. The bridge pulls a bounded batch and validates envelope shape, byte size, and `schema.id`/`version`/`digest` admission. Malformed or oversized messages are published to the configured DLQ and acked; poison input never crashes the consumer.
2. A record whose schema digest does not match the active lineage is written to quarantine and reported, never trained or silently dropped. An `observed_model` absent from the §1.4 allowlist is rejected before extraction.
3. Valid records are appended to a bounded local WAL spool and forwarded to the trainer over a local Unix-domain socket with length-prefixed frames and credit-based flow control. The broker offset is committed (Kafka) or the message is acked (NATS) only after the trainer confirms durable acceptance into the replay buffer/provenance store; credits bound the in-flight window, so a slow trainer backpressures the broker.
4. Crash recovery replays the spool and resumes from the broker's committed offset. Redelivery is expected and deduplicated by the idempotency key; accepted records are never lost. Exactly-once is explicitly not claimed.
5. Retries stop at the configured ceiling (for example `max_deliver`), after which the record moves to the DLQ with its attempt count and last error class.
6. `labelled_example` records enter the replay buffer and continual LoRA/EMA training directly. `tool_task_pair` and `raw_document` records traverse validation, the §1.4 extraction/decision path, and distillation before any training use.
7. Shutdown drains in-flight credits before ack. Pause/resume, replay-from-offset, and replay-from-timestamp are operator surfaces with recorded provenance.

**Secrets and provenance.** Credentials come from environment or secret-store references; no broker password, token, or key is written to config, logs, DLQ payloads, or the repository. Every consumed batch records adapter kind/version, broker endpoint identity, stream/topic, partition/offset or stream sequence, delivery attempt, consumer group/durable name, bridge build, and the downstream acceptance receipt for audit.

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
| G6 NVIDIA GPU | On the designated NVIDIA GPU host, CUDA/model load/train/infer smoke passes, VRAM stays within budget, p50/p95 latency and throughput are recorded, and no CPU-only result is accepted as the hardware gate. |
| G7 Delivery | Public GitHub repository URL, commit/branch/visibility and manifest are recorded; remote copy has matching file manifest and runs the remote smoke/NVIDIA GPU checks. |
| G8 Master completion | All applicable repository validators pass; no `[ ]` or `[_]`, handoff/integration/repair queue is empty; Gantt digest and monitoring index are current and every checklist ID appears exactly once. |
| G9 Tool-task decision | The standard request/response record, model allowlist, section-aware extraction, finite task labels, probability-bearing technology-stack choice list, deterministic router, low-confidence fallback, unknown-model rejection, and replay/golden tests are validated; no model output can directly execute a tool. |
| G10 MQ ingestion | The configurable MQ contract validates and fails closed; the Rust bridge builds with the pinned toolchain (`cargo fmt`/`clippy`/`test --locked`); replay/fault tests prove at-least-once commit-after-durable-ack, duplicate suppression by `record_id` plus content hash, poison-to-DLQ, schema-mismatch quarantine, broker-restart offset resume, credit backpressure, graceful drain, and secret-free logs. No broker ack or offset commit precedes downstream durable acceptance. |
| G11 MQ live soak | On the designated NVIDIA GPU host with the configured broker, a sustained pull-and-train soak records ingested records/s, consumer lag bounded and returning to baseline after bursts, peak VRAM, ack latency, and DLQ/quarantine counts; the post-transfer remote verification repeats the MQ smoke. Simulated-only evidence does not satisfy this gate. |

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
- [ ] **ZJ-028** Define the standard `jev-tool-task` schema, versioned request/response contract, and explicit model/tool/language/technology allowlists. **Depends on:** ZJ-026. **Owner:** Schema worker. **Owned paths:** `configs/jev_tool_task.yaml`, `jev/config.py`, `jev/tool_task.py`, `tests/test_model_config_distill.py`. **Gate:** G1,G4,G9.
- [ ] **ZJ-029** Implement section-aware request/response landing-pool pairing, redaction/provenance metadata, model allowlist admission, and replayable records. **Depends on:** ZJ-011,ZJ-028. **Owner:** Data worker. **Owned paths:** `jev/sources.py`, `jev/tool_task.py`, `configs/jev_tool_task.yaml`, `tests/test_jev.py`. **Gate:** G1,G2,G9.

### 3.4 Validation, observability, and operator surfaces

- [ ] **ZJ-030** Provision and validate the NVIDIA GPU test environment, CUDA/toolkit/driver, model cache, precision support, and VRAM budget. **Depends on:** ZJ-001,ZJ-020. **Owner:** Hardware worker. **Owned paths:** `scripts/validate_nvidia_gpu.py`, `artifacts/hardware/`, `Docs/runbooks/`. **Gate:** G6.
- [ ] **ZJ-031** Measure NVIDIA GPU train/infer throughput, p50/p95 latency, peak VRAM, thermal/power observations, and concurrent train/infer stability. **Depends on:** ZJ-027,ZJ-030. **Owner:** Hardware worker. **Owned paths:** `scripts/smoke_nvidia_gpu.py`, `benchmarks/`, `artifacts/benchmarks/`, `tests/benchmarks/`. **Gate:** G6.
- [ ] **ZJ-032** Run quality evaluation on held-out human/distilled data, compare base/LoRA/EMA, and publish error/confidence/calibration report. **Depends on:** ZJ-014,ZJ-027. **Owner:** Evaluation worker. **Owned paths:** `artifacts/evaluations/`, `jev/drift.py`, `tests/test_jev.py`. **Gate:** G2,G4,G5.
- [ ] **ZJ-033** Execute a collapse-and-recovery drill on the NVIDIA GPU (or a faithful simulator plus one hardware smoke), prove reset thresholds and warm-up gate. **Depends on:** ZJ-025,ZJ-031. **Owner:** Evaluation worker. **Owned paths:** `artifacts/recovery/`, `tests/recovery/`. **Gate:** G5,G6.
- [ ] **ZJ-034** Audit source/license/provenance, teacher request redaction, MQ credential handling, artifact retention, DLQ/quarantine retention, and schema lineage. **Depends on:** ZJ-011,ZJ-012,ZJ-013,ZJ-014,ZJ-051. **Owner:** Master. **Owned paths:** `artifacts/audits/`, `Docs/runbooks/`. **Gate:** G1,G2,G7,G10.
- [ ] **ZJ-035** Implement request task-type/detail and response tool/language/stack extraction, probability-bearing technology-stack choices, plus finite-label deterministic routing and rejection/review fallback. **Depends on:** ZJ-026,ZJ-028,ZJ-029. **Owner:** Inference worker. **Owned paths:** `jev/model.py`, `jev/tool_task.py`, `jev/runtime.py`, `tests/test_model_config_distill.py`. **Gate:** G4,G9.
- [ ] **ZJ-036** Add task-to-route golden evaluation, unknown-model/tool rejection, confidence/evidence floors, policy conflict tests, and no-tool-execution assertions. **Depends on:** ZJ-032,ZJ-035. **Owner:** Evaluation worker. **Owned paths:** `artifacts/evaluations/`, `tests/test_jev.py`, `tests/test_model_config_distill.py`. **Gate:** G5,G9.
- [ ] **ZJ-037** Add a bounded streaming submit/stop adapter around the concurrent trainer/inference snapshots with explicit backpressure and one training writer. **Depends on:** ZJ-023,ZJ-035. **Owner:** Runtime worker. **Owned paths:** `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py`. **Gate:** G4,G9.
- [ ] **ZJ-040** Provide CLI/API and operator runbook for schema/source selection, distillation, train/serve, metrics, reset, and manifest inspection. **Depends on:** ZJ-010,ZJ-011,ZJ-027. **Owner:** Docs worker. **Owned paths:** `jev/cli.py`, `Docs/runbooks/`, `README.md`, `tests/test_jev.py`. **Gate:** G1,G4,G5.
- [ ] **ZJ-041** Produce reproducibility bundle: lockfiles (Python and `mq/Cargo.lock`), configs, model/data/checkpoint manifests, evaluation reports, and exact validation commands. **Depends on:** ZJ-031,ZJ-032,ZJ-034,ZJ-040,ZJ-051. **Owner:** Master. **Owned paths:** `artifacts/repro/`, `Docs/runbooks/`, `requirements*.txt`, `pyproject.toml`. **Gate:** G0,G6,G7,G10.

### 3.5 Publication, remote move, and final acceptance

- [ ] **ZJ-042** Run the complete local acceptance profile and reconcile blueprint, handoffs, manifests, and generated Gantt. **Depends on:** ZJ-003,ZJ-027,ZJ-031,ZJ-033,ZJ-041,ZJ-053,ZJ-055. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/`. **Gate:** G0–G11 as applicable.
- [ ] **ZJ-043** Initialize/verify public `weiyangzen/zenjev`, commit the accepted tree, push the recorded branch, and store URL/commit/visibility receipt. **Depends on:** ZJ-042. **Owner:** Master. **Owned paths:** `.git/` (metadata), `artifacts/delivery/github_receipt.json`. **Gate:** G7.
- [ ] **ZJ-044** Transfer the complete accepted tree to `sansha@192.168.50.38:/home/sansha/zenjev`, verify a full manifest/checksum, and run remote smoke plus NVIDIA GPU and MQ checks. **Depends on:** ZJ-042,ZJ-043,ZJ-054. **Owner:** Hardware worker/Master. **Owned paths:** `artifacts/delivery/remote_transfer_receipt.json`, `artifacts/hardware/remote_*.json`. **Gate:** G6,G7,G11.
- [ ] **ZJ-045** Master final acceptance: all validators and delivery receipts pass; no unfinished checklist/handoff/repair item; atomically regenerate and digest-check the final Gantt; perform scoped cleanup. **Depends on:** ZJ-043,ZJ-044. **Owner:** Master. **Owned paths:** `Docs/stage0_zenjev_blueprint.md`, `Docs/stage0_zenjev_blueprint_Gantt.md`, `.zenjev/runtime/`. **Gate:** G8.

### 3.6 Continual external-MQ ingestion

- [ ] **ZJ-050** Define the configurable MQ ingestion contract: adapter selection, envelope schema, schema-digest admission, idempotency keys, at-least-once ack ordering, DLQ/quarantine, retry ceiling, secret references, and fail-closed validation. **Depends on:** ZJ-004,ZJ-011. **Owner:** Schema worker. **Owned paths:** `jev/config.py`, `configs/example.yaml`, `tests/test_jev.py`. **Gate:** G10.
- [ ] **ZJ-051** Implement the Rust `mq` bridge crate with pluggable source adapters (`nats-jetstream` durable pull default, Kafka/Redpanda, optional Iggy), bounded WAL spool, DLQ publisher, offset/ack bookkeeping, and TLS/secret-reference handling. **Depends on:** ZJ-050. **Owner:** MQ worker. **Owned paths:** `mq/`, `scripts/smoke_mq.py`, `tests/test_mq.py`. **Gate:** G10.
- [ ] **ZJ-052** Implement the Python bridge client and trainer ingestion loop: credit-based local transport, spool replay, idempotent replay-buffer admission, pause/resume/drain, and consumer-lag/ack metrics. **Depends on:** ZJ-037,ZJ-050. **Owner:** Runtime worker. **Owned paths:** `jev/mq.py`, `jev/runtime.py`, `jev/training.py`, `tests/test_jev.py`. **Gate:** G10.
- [ ] **ZJ-053** Add MQ replay and fault tests: duplicate suppression, crash before ack, poison-to-DLQ, schema-mismatch quarantine, broker-restart offset resume, credit backpressure, graceful drain, and no-ack-before-durable-ack assertions. **Depends on:** ZJ-051,ZJ-052. **Owner:** Evaluation worker. **Owned paths:** `tests/test_mq.py`, `mq/tests/`, `scripts/smoke_mq.py`, `artifacts/mq/`. **Gate:** G10.
- [ ] **ZJ-054** Run a sustained external-MQ pull-and-train soak on the designated NVIDIA GPU host with the configured broker and record throughput, bounded lag recovery, peak VRAM, ack latency, and DLQ/quarantine counts. **Depends on:** ZJ-031,ZJ-053. **Owner:** Hardware worker. **Owned paths:** `scripts/smoke_mq_nvidia_gpu.py`, `benchmarks/mq/`, `artifacts/benchmarks/`, `Docs/runbooks/`. **Gate:** G6,G11.
- [ ] **ZJ-055** Extend the operator CLI and runbook for MQ configuration, replay from offset/timestamp, DLQ inspection and reprocess, quarantine review, lag metrics, and broker credential rotation. **Depends on:** ZJ-040,ZJ-052. **Owner:** Docs worker. **Owned paths:** `jev/cli.py`, `Docs/runbooks/`, `README.md`, `tests/test_jev.py`. **Gate:** G10,G11.

## 4. Conditional automated handoff, integration, and cleanup protocol

This protocol applies only after a future explicit automation activation. It does not require a cron/controller for the current authorized manual work. Shared module and test paths in the checklist must be serialized or narrowed to disjoint exact paths in any future claims; broad planned ownership is not a concurrency lease. Manual integration uses the same evidence and Master acceptance discipline, without inventing worker claims or handoff ledgers.

1. A claim is reserved only after dependency, exact-path conflict, and the frozen concurrency vector admit it. Its immutable claim card includes the item ID, run ID, baseline/spec digests, owned paths, commands, deadline, and result path.
2. The worker receives one task-local tmux/Codex TUI lane and one short `/goal`. It acquires both running-turn and outbound-request leases before Enter. Authentication must prove tmux socket/session, pane PID/start time, cwd, private `CODEX_HOME`, route, thread, goal, and claim objective.
3. The worker writes only its declared paths and emits `status=self_tested` with checksums. The controller harvests the handoff before pruning any process and marks the item `[_]`; this is still unfinished.
4. Master applies a conflict-safe patch to the canonical checkout, runs the item and dependency gates, updates manifests and the Gantt projection, and only then changes `[_]` to `[x]`. Failed validation remains a durable repair entry and does not erase the handoff.
5. Cleanup is scoped to controller-owned runtime and task descendants. Completion cleanup requires zero `[ ]`, zero `[_]`, no handoff/integration/repair backlog, all gates pass, final Gantt freshness, no live task transport, and preserved source/delivery receipts.

## 5. Required status and Gantt projection fields

During current manual work, run `python3 scripts/validate_blueprint.py --generate` after a blueprint edit and `python3 scripts/validate_blueprint.py --check` before delivery. A future enabled controller must do the same atomically after every final state merge. The generated Gantt must expose checklist counts (`[ ]`, `[_]`, `[x]`), each ID exactly once, dependencies, owner/claim, owned paths, implementation/validation/integration frontiers, and durable startup/live/handoff/integration/repair/blocked state. It must separately report logical claims, admitted executions, startup reservations, live tmux transports, authenticated goals, running turns, request starts/window, in-flight and outstanding requests, breaker state, validator/integration leases, and every persisted underfill reason. Unknown timing remains in an **Unscheduled** section; fabricated dates are forbidden. The Gantt is never parsed as authority and contains no mutable checklist marks. MQ ingestion telemetry (consumer lag, in-flight/credited records, ack latency, DLQ/quarantine counters, and offset checkpoints) is recorded by the bridge under `artifacts/mq/` and referenced by G10/G11 evidence; it is runtime evidence and never mutates checklist state.

## 6. Policy migration record

Governance correction: the originally inferred single-worker vector and bounded lifecycle were invalid because the operator never supplied them. They are withdrawn, not accepted defaults. The companion path now follows the case-sensitive skill rule; fabricated relative task durations are removed. No implementation state was accepted by this correction. A future migration must state the old/new specification digests, affected checklist IDs and claims, reason, operator, timestamp, compatibility decision, and regenerated Gantt digest before any new launch.

**MQ ingestion migration (2026-09-20).** Reason: the operator requested continual consumption from a configurable external standard message queue instead of a fixed dataset. Change: added §1.5, gates G10–G11, items ZJ-050–ZJ-055, MQ telemetry in §5, and MQ dependencies on ZJ-034/ZJ-041/ZJ-042/ZJ-044; the frozen table gained MQ ingestion, delivery, and bridge rows, and the validation profile gained Rust/MQ validators. Operator: repository owner via the current interactive session. Old specification digest: `63c539ad474c8f5cd51c0b4746a1965ea88c7164c35ba2c495711c026a886f5e`; new specification digest: `53cd7fad80ce52bb3e21aa4d0c1b2299a0da024cac104f9fab3e91034829d90f`. Compatibility: additive; existing IDs keep their meaning, the controller remains disabled, and every unresolved operator input in §1.2 persists. The regenerated Gantt records the new source and specification digests; no launch is authorized by this migration.
