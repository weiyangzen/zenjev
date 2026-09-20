# MQ operations runbook

Operator surface for the frozen continual external-MQ ingestion contract
(§1.5 of [`../stage0_zenjev_blueprint.md`](../stage0_zenjev_blueprint.md),
items ZJ-050 through ZJ-055). The Rust bridge owns every broker protocol;
Python training code consumes only the local, flow-controlled Unix-socket
stream.

> **Live-broker status:** no live external broker soak (G11) was performed in
> this repository environment. The end-to-end evidence available here is
> `artifacts/mq/smoke.json`, produced by `scripts/smoke_mq.py` against the
> deterministic `mock` adapter with `"live_broker": false`. Section 9 documents
> exactly how to perform the live pull-and-train soak later on the designated
> NVIDIA GPU host.

## 1. Adapter selection and fail-closed validation

The `mq` section of the user config selects the adapter; nothing is hard-coded.
Changing broker or adapter never changes the training contract.

| Adapter | Stage 0 role | Semantics |
|---|---|---|
| `nats-jetstream` (default) | Durable pull consumer for a slow single-writer trainer | explicit ack, server-tracked attempts, `max_ack_pending` backpressure, `ack_wait`, `max_deliver` poison ceiling |
| `kafka` | Standard Kafka/Redpanda compatibility | `enable.auto.commit=false`; offset committed only after downstream durable acceptance; in-order per partition |
| `iggy` (optional) | Apache Iggy deployments | explicit fsync/flush thresholds and consumer-group offsets; never a gate requirement |
| `mock` | Tests and the no-broker smoke only | deterministic replay from `source_path`; requires `source_path` |

Validation fails closed for: unknown adapter kinds; missing endpoints; remote
endpoint without TLS; `nats-jetstream` without `stream`/`durable_name`; `kafka`
without `consumer_group`; `by_offset`/`by_timestamp` without the matching start
value; non-positive `batch`/`max_bytes`/`ack_wait_seconds`/`max_ack_pending`/
`prefetch`/`max_deliver`/`dedup_window`; empty spool/socket paths; `client_cert`
without `client_key`; and any `secret_ref` that is not an `env:`/`file:`
reference. The same rules are enforced again by the Rust bridge in
`BridgeConfig::validate`.

Relevant `configs/example.yaml` fields (disabled by default):

```yaml
mq:
  enabled: false
  adapter: nats-jetstream
  endpoints: ["tls://nats.internal.example:4222"]
  stream: JEV_RECORDS
  subject: jev.records
  durable_name: jev-trainer
  consumer_group: null
  start_position: new
  batch: 64
  max_bytes: 33554432
  ack_wait_seconds: 30
  max_ack_pending: 16
  prefetch: 32
  max_deliver: 5
  dlq_path: runs/jev/mq/dlq.jsonl
  quarantine_path: runs/jev/mq/quarantine.jsonl
  wal_path: runs/jev/mq/wal.bin
  socket_path: runs/jev/mq/bridge.sock
  metrics_path: artifacts/mq/metrics.json
  replay_buffer: runs/jev/mq/replay-buffer.jsonl
  tls_required: true
  secret_ref: env:JEV_MQ_TOKEN
```

## 2. Envelope schema and content hash

Every message is UTF-8 JSON (CBOR only under an explicit config flag):

```json
{
  "envelope_version": 1,
  "record_id": "landing-42",
  "content_sha256": "<sha256 of the canonical envelope without this field>",
  "schema": {"id": "tech_stack_2026", "version": 1, "digest": "<schema digest>"},
  "kind": "labelled_example",
  "source": {
    "uri": "user-owned://landing/42",
    "content_sha256": "<sha256 of the source bytes>",
    "retrieved_at": "2026-09-21T00:00:00Z",
    "license": "user-provided"
  },
  "observed_model": {"provider": "openai-compatible", "model_id": "gpt-6-astra"},
  "labels": {"entities": {"technology": ["Python"]}},
  "created_at": "2026-09-21T00:00:01Z"
}
```

`kind` is one of `labelled_example`, `tool_task_pair`, `raw_document`;
`tool_task_pair` requires non-empty `request.text` and `response.text`;
`raw_document` requires `document` or `example`. `observed_model` is checked
against the §1.4 allowlist before extraction, and an unlisted model is rejected
to the DLQ.

The content hash is computed over the canonical form of the envelope with
`content_sha256` removed: sorted keys, compact separators, UTF-8. The Python
implementation is `jev.mq.envelope_content_hash` (used by
`jev.mq.build_envelope`); the bridge recomputes it and rejects a mismatch as
`dlq:content_hash_mismatch`.

## 3. Idempotency

`record_id` plus the canonical `content_sha256` is the idempotency key
(`jev.mq.idempotency_key`, rendered `<record_id>|<content_sha256>`). Redelivery
is expected under at-least-once; the trainer keeps a durable, bounded
`replay_buffer` and suppresses a key it has already accepted. Exactly-once is
explicitly **not** claimed.

## 4. Ack-after-durable ordering, credits, and backpressure

1. The bridge pulls a bounded batch and validates shape, byte size, and
   `schema.id`/`version`/`digest` admission.
2. Valid records are appended to the bounded local WAL and forwarded to the
   trainer over length-prefixed Unix-socket frames.
3. The trainer validates the envelope, appends it to the replay buffer, flushes,
   and `fsync`s the file; only then does `MqIngestor` send a `credit` frame with
   the delivery id in `acks`.
4. Only on that confirmation does the bridge ack (NATS) or commit the offset
   (Kafka). **No ack or offset commit precedes downstream durable acceptance.**
5. Initial credits are `min(mq.prefetch, mq.max_ack_pending)`; each granted
   credit admits one more in-flight record. A slow trainer therefore
   backpressures the broker instead of buffering without bound. Shutdown drains
   in-flight credits (`drain`) before acking.

Downtime is recovered by replaying the WAL and resuming from the broker's
committed offset; duplicate redelivery is suppressed by the idempotency key.
Accepted records are never lost.

## 5. DLQ and quarantine

| Signal | Destination | Examples |
|---|---|---|
| Malformed, non-UTF-8/oversized, unknown kind, missing source fields, bad content hash, poison past `max_deliver` | `dlq_path` (and `dlq_subject` when the adapter can publish) | `dlq:envelope_not_json`, `dlq:content_hash_mismatch`, `dlq:observed_model_not_allowlisted` |
| Schema id/version/digest does not match the active lineage | `quarantine_path` | `quarantine:schema_digest_mismatch` |
| Trainer-side invalid envelope | `quarantine_path` via the credit frame | `client_invalid_envelope` |

Poison input never crashes the consumer: malformed/oversized messages are
DLQ'd and acked, then retries stop at `max_deliver` with the attempt count and
last error class recorded. Operators review both streams with
`mq dlq`/`mq quarantine` and never lose the entries.

## 6. Secret references

Credentials are references only. `mq.secret_ref` must be `env:NAME` or
`file:/path`; `api_key_env` in the teacher section names an environment
variable. No broker password, token, or key is written to config, logs, DLQ
payloads, or the repository, and every consumed batch records adapter
kind/version, endpoint identity, stream/topic, offset or sequence, delivery
attempt, durable/group name, bridge build, and downstream acceptance receipt.

## 7. Build the Rust bridge

The pinned toolchain is declared in `mq/rust-toolchain.toml` (1.98.1). The
default feature set builds the NATS adapter only.

```bash
cd mq
cargo fmt --check
cargo clippy --locked -- -D warnings
cargo test --locked
cargo build --locked            # debug binary at mq/target/debug/jev-mq-bridge
cargo build --locked --release  # release binary at mq/target/release/jev-mq-bridge
```

`jev.mq.locate_bridge_binary` finds a build in `mq/target/{release,debug}` or a
binary named on `PATH`; `JEV_MQ_BRIDGE_BIN` overrides both.

Feature-gated adapters are **not built by default** and fail closed at runtime
when a config selects them:

| Feature | Build command | Prerequisites |
|---|---|---|
| `nats` (default) | `cargo build --locked` | none beyond the pinned toolchain (pure Rust `async-nats`) |
| `kafka` | `cargo build --locked --features kafka` | system `librdkafka` development package and `cmake` (the `rdkafka-sys` build) |
| `iggy` | `cargo build --locked --features iggy` | Iggy client crates as wired by the MQ worker; benchmark durability before use |

Example setup for the Kafka feature on Debian/Ubuntu:

```bash
sudo apt-get install -y librdkafka-dev cmake
cd mq && cargo build --locked --features kafka --release
```

The bridge CLI is:

```bash
jev-mq-bridge run --config <bridge-config.json>
jev-mq-bridge validate --config <bridge-config.json>
jev-mq-bridge version
```

The Python side renders `<bridge-config.json>` from the validated YAML plus the
active schema digest; operators do not hand-edit it.

## 8. Operator commands

All commands require `--config <file>` and a config whose `mq.enabled` is
`true`; the example config ships disabled. Copy it before enabling:

```bash
cp configs/example.yaml /tmp/jev-mq.yaml   # then set mq.enabled: true
```

### `mq validate`

Validates the Python contract and, when a bridge binary is present, the
rendered bridge config (no broker connection is opened).

```bash
python -m jev.cli mq validate --config /tmp/jev-mq.yaml
```

```json
{"mq": {"enabled": true, "adapter": "nats-jetstream", "...": "..."},
 "fail_closed": true,
 "bridge_validate": {"returncode": 0, "stdout": "{\"valid\":true,\"adapter\":\"nats-jetstream\",\"schema_digest\":\"…\"}"}}
```

### `mq bridge-config`

Renders the exact bridge JSON to `mq.bridge_config` or `--output`:

```bash
python -m jev.cli mq bridge-config --config /tmp/jev-mq.yaml --output runs/jev/mq/bridge.json
```

```json
{"path": "runs/jev/mq/bridge.json", "adapter": "nats-jetstream", "schema_digest": "…"}
```

### `mq run`

Starts the bridge (unless `--attach` reuses a running one) and the ingestion
loop. `--train` submits `labelled_example` records to a bounded
`ContinuousLoRAStream`; `--max-records` and `--idle-timeout` bound a finite
operator run; `--pause-seconds` exercises pause/resume; `--output` persists the
metrics JSON.

```bash
python -m jev.cli mq run --config /tmp/jev-mq.yaml \
  --train --checkpoint-dir runs/jev \
  --max-records 100 --idle-timeout 60 \
  --output artifacts/mq/run-2026-09-21.json
```

The printed metrics include `received`, `accepted`, `duplicates`,
`quarantined`, `dlq_rejected`, per-kind counters, ack latency samples, and the
bridge snapshot (`pulled`, `delivered`, `acked`, `consumer_lag`, `inflight`,
`credits`, `wal_records`, `offset_checkpoint`).

### `mq lag`

Reads live bridge status over the Unix socket, falling back to
`mq.metrics_path`:

```bash
python -m jev.cli mq lag --config /tmp/jev-mq.yaml
```

```json
{"source": "bridge", "metrics": {"consumer_lag": 0, "inflight": 0, "credits": 2, "offset_checkpoint": 42}}
```

### `mq dlq` and `mq quarantine`

Summarize stored entries by reason and optionally export them for reprocessing:

```bash
python -m jev.cli mq dlq --config /tmp/jev-mq.yaml --output artifacts/mq/dlq-export.jsonl
python -m jev.cli mq quarantine --config /tmp/jev-mq.yaml
```

```json
{"path": "runs/jev/mq/dlq.jsonl", "count": 3, "reasons": {"dlq:envelope_not_json": 2, "dlq:content_hash_mismatch": 1}, "exported": "artifacts/mq/dlq-export.jsonl"}
```

### `mq replay`

Replays from the first message, a broker offset, or an epoch timestamp by
rendering an override bridge config and running the ingestor to drain:

```bash
python -m jev.cli mq replay --config /tmp/jev-mq.yaml --dry-run
python -m jev.cli mq replay --config /tmp/jev-mq.yaml --from-offset 1200 --max-records 500
python -m jev.cli mq replay --config /tmp/jev-mq.yaml --from-timestamp 1789000000 --output runs/jev/mq/replay.json
```

The replay override JSON is written next to the bridge config
(`<bridge_config>.replay.json`) or to `--output`; the run records its
provenance via the bridge metrics.

### No-broker end-to-end check

```bash
python3 scripts/smoke_mq.py
```

It builds `mq/` with `cargo build --locked` when needed, then exercises the
`mock` adapter: credits, durable-accept-then-ack ordering, duplicate
suppression, poison-to-DLQ, schema-mismatch quarantine, and graceful drain.
Evidence is written to `artifacts/mq/smoke.json` and the script exits `2` when
any assertion fails.

## 9. Live broker soak (not yet performed)

A live NATS JetStream (or Kafka/Redpanda) pull-and-train soak on the designated
NVIDIA GPU host is required by gate G11 and item ZJ-054. It was **not performed
in this environment**; simulated `mock` evidence does not satisfy G11. To run
it later:

1. Build the bridge with the adapter feature you will exercise (Section 7) and
   point `JEV_MQ_BRIDGE_BIN` at the release binary.
2. Start the broker with TLS as configured. Provision the stream/topic/subject,
   durable consumer name or consumer group, and credentials in the secret
   store. Set the `env:`/`file:` referenced variable or file; never place the
   value in config.
3. Enable `mq` in the operator config and confirm the contract:
   `python -m jev.cli mq validate --config <config>`.
4. Publish a bounded burst with the broker's standard tooling, then run the
   pull-and-train loop and sample lag while it drains:

   ```bash
   python -m jev.cli mq run --config <config> --train \
     --checkpoint-dir runs/jev --idle-timeout 900 \
     --output artifacts/benchmarks/mq/soak.json
   python -m jev.cli mq lag --config <config>
   ```

5. Record ingested records/s, peak VRAM (from the GPU smoke harness), ack
   latency distribution, consumer lag returning to baseline after the burst,
   and DLQ/quarantine counts under `artifacts/benchmarks/mq/` (raw metrics in
   `artifacts/mq/`). The hardware worker's
   `scripts/smoke_mq_nvidia_gpu.py` (ZJ-054) is the intended wrapper once it
   exists.
6. Repeat the MQ smoke after the remote transfer to `sansha@192.168.50.38`
   (G7/G11) and store the receipt.
