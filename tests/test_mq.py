"""MQ contract, replay/fault and credit-ordering tests.

The Rust bridge is exercised through the same Unix-socket protocol the trainer
uses. If no bridge binary exists the integration tests build it with the pinned
toolchain or skip when cargo is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from jev.config import ConfigError, JevConfig
from jev.mq import (
    BridgeClient,
    EnvelopeError,
    ReplayBuffer,
    build_envelope,
    envelope_content_hash,
    idempotency_key,
    locate_bridge_binary,
    render_bridge_config,
    sha256_hex,
    validate_envelope,
)

ROOT = Path(__file__).resolve().parents[1]


def base_config(tmp_path: Path, **mq_overrides) -> JevConfig:
    mq = {
        "enabled": True,
        "adapter": "mock",
        "source_path": str(tmp_path / "records.jsonl"),
        "state_path": str(tmp_path / "state.json"),
        "wal_path": str(tmp_path / "wal.bin"),
        "socket_path": str(tmp_path / "bridge.sock"),
        "bridge_config": str(tmp_path / "bridge.json"),
        "dlq_path": str(tmp_path / "dlq.jsonl"),
        "quarantine_path": str(tmp_path / "quarantine.jsonl"),
        "metrics_path": str(tmp_path / "metrics.json"),
        "replay_buffer": str(tmp_path / "replay.jsonl"),
        "batch": 2,
        "max_ack_pending": 2,
        "prefetch": 2,
        "max_deliver": 3,
        "exit_after_drain": True,
    }
    mq.update(mq_overrides)
    return JevConfig.from_dict(
        {
            "model_id": "fastino/gliner2-base-v1",
            "schema": {"name": "mq_test", "version": 1, "entities": ["technology"]},
            "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
            "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
            "mq": mq,
        }
    )


def make_envelope(config: JevConfig, record_id: str, kind: str, **payload) -> bytes:
    value = build_envelope(
        record_id=record_id,
        kind=kind,
        source={
            "uri": f"test://{record_id}",
            "content_sha256": sha256_hex(record_id),
            "retrieved_at": "2026-09-21T00:00:00Z",
            "license": "test",
        },
        schema_id=config.schema.name,
        schema_version=config.schema.version,
        schema_digest=config.schema.digest(),
        **payload,
    )
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def test_mq_config_fails_closed(tmp_path):
    with pytest.raises(ConfigError):
        base_config(tmp_path, adapter="unknown")
    with pytest.raises(ConfigError):
        base_config(tmp_path, adapter="nats-jetstream", endpoints=["nats://broker.example:4222"], tls_required=False, source_path=None)
    with pytest.raises(ConfigError):
        base_config(tmp_path, secret_ref="plaintext-token")
    with pytest.raises(ConfigError):
        base_config(tmp_path, adapter="nats-jetstream", source_path=None, endpoints=[])
    with pytest.raises(ConfigError):
        base_config(tmp_path, adapter="kafka", endpoints=["tls://kafka.example:9093"], consumer_group=None, source_path=None)
    with pytest.raises(ConfigError):
        base_config(tmp_path, start_position="by_offset", start_offset=None)


def test_envelope_validation_routes_dlq_and_quarantine(tmp_path):
    config = base_config(tmp_path)
    valid = json.loads(make_envelope(config, "r1", "raw_document", document={"text": "hello"}))
    assert validate_envelope(json.dumps(valid).encode(), config, max_bytes=1_000_000)["record_id"] == "r1"

    with pytest.raises(EnvelopeError) as bad_json:
        validate_envelope(b"not json", config, max_bytes=1_000_000)
    assert bad_json.value.class_ == "dlq" and bad_json.value.reason == "envelope_not_json"

    tampered = dict(valid, record_id="tampered")
    with pytest.raises(EnvelopeError) as mismatch:
        validate_envelope(json.dumps(tampered).encode(), config, max_bytes=1_000_000)
    assert mismatch.value.reason == "content_hash_mismatch"

    schema_mismatch = dict(valid)
    schema_mismatch["schema"] = {**valid["schema"], "digest": "d" * 64}
    schema_mismatch["content_sha256"] = envelope_content_hash(schema_mismatch)
    with pytest.raises(EnvelopeError) as quarantined:
        validate_envelope(json.dumps(schema_mismatch).encode(), config, max_bytes=1_000_000)
    assert quarantined.value.class_ == "quarantine"

    with pytest.raises(EnvelopeError) as oversized:
        validate_envelope(valid, config, max_bytes=4)
    assert oversized.value.reason == "oversized_message"

    with pytest.raises(EnvelopeError) as bad_pair:
        validate_envelope(make_envelope(config, "r2", "tool_task_pair", request={"text": "a"}), config, max_bytes=1_000_000)
    assert bad_pair.value.reason == "response_requires_text"


def test_replay_buffer_is_idempotent_and_persistent(tmp_path):
    config = base_config(tmp_path)
    envelope = json.loads(make_envelope(config, "r1", "labelled_example", labels={"entities": {"technology": ["Python"]}}))
    buffer = ReplayBuffer(tmp_path / "replay.jsonl")
    assert buffer.accept(envelope) is True
    assert buffer.accept(envelope) is False
    reopened = ReplayBuffer(tmp_path / "replay.jsonl")
    assert reopened.contains(idempotency_key(envelope))
    assert reopened.count() == 1


def bridge_binary() -> Path:
    binary = locate_bridge_binary()
    if binary is not None:
        return binary
    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    if not cargo.exists():
        pytest.skip("no jev-mq-bridge binary and no cargo toolchain")
    subprocess.run([str(cargo), "build", "--locked"], cwd=ROOT / "mq", check=True)
    binary = locate_bridge_binary()
    if binary is None:
        pytest.skip("bridge binary unavailable after build")
    return binary


@pytest.fixture()
def bridge_env(tmp_path, monkeypatch):
    binary = bridge_binary()
    monkeypatch.setenv("JEV_MQ_BRIDGE_BIN", str(binary))
    return binary


def write_records(tmp_path: Path, records: list[bytes]) -> None:
    (tmp_path / "records.jsonl").write_bytes(b"\n".join(records) + b"\n")


def test_bridge_credit_bound_and_ack_after_durable(tmp_path, bridge_env):
    from jev.mq import MqIngestor

    config = base_config(tmp_path)
    write_records(tmp_path, [make_envelope(config, "r1", "raw_document", document={"text": "one"}), make_envelope(config, "r2", "raw_document", document={"text": "two"})])
    ingestor = MqIngestor(config)
    ingestor.start_bridge()
    client = BridgeClient(config.mq.socket_path)
    try:
        client.connect(credits=1)
        first = client.receive(timeout_s=5)
        assert first and first["type"] == "record" and first["envelope"]["record_id"] == "r1"
        time.sleep(0.3)
        state_path = tmp_path / "state.json"
        committed = json.loads(state_path.read_text())["committed"] if state_path.exists() else 0
        assert committed == 0
        assert client.receive(timeout_s=0.2) is None, "credit bound was ignored"
        client.credit(acks=[first["delivery_id"]], count=1)
        second = client.receive(timeout_s=5)
        assert second and second["envelope"]["record_id"] == "r2"
        client.credit(acks=[second["delivery_id"]], count=0)
        client.drain()
        assert client.receive(timeout_s=5)["type"] == "drained"
        client.goodbye()
        assert ingestor._bridge is not None
        ingestor._bridge.wait(timeout=10)
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["acked"] == 2
        assert json.loads((tmp_path / "state.json").read_text())["committed"] == 2
    finally:
        client.close()
        ingestor.stop_bridge()


def test_bridge_faults_and_duplicate_suppression(tmp_path, bridge_env):
    from jev.mq import MqIngestor

    config = base_config(tmp_path)
    schema_mismatch = json.loads(make_envelope(config, "mismatch", "raw_document", document={"text": "x"}))
    schema_mismatch["schema"] = {**schema_mismatch["schema"], "digest": "d" * 64}
    schema_mismatch["content_sha256"] = envelope_content_hash(schema_mismatch)
    records = [
        make_envelope(config, "r1", "raw_document", document={"text": "one"}),
        b'{"envelope_version": 1, "record_id":',
        json.dumps(schema_mismatch).encode(),
        make_envelope(config, "r2", "raw_document", document={"text": "two"}),
        make_envelope(config, "r1", "raw_document", document={"text": "one"}),
    ]
    write_records(tmp_path, records)
    seen: list[str] = []
    ingestor = MqIngestor(config, document_handler=lambda envelope: seen.append(envelope["record_id"]))
    metrics = ingestor.run(manage_bridge=True, max_records=2, idle_timeout_s=10.0)
    assert seen == ["r1", "r2"]
    assert metrics["accepted"] == 2
    bridge_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert bridge_metrics["dlq"] == 1
    assert bridge_metrics["quarantined"] == 1
    assert bridge_metrics["duplicates_suppressed"] == 1
    assert (tmp_path / "dlq.jsonl").read_text(encoding="utf-8")
    assert "schema_digest_mismatch" in (tmp_path / "quarantine.jsonl").read_text(encoding="utf-8")


def test_mq_config_loop_fields_and_render(tmp_path):
    config = base_config(tmp_path, loop=True, max_cycles=3)
    contract = config.mq.as_contract()
    assert contract["loop"] is True
    assert contract["max_cycles"] == 3
    rendered = render_bridge_config(config)
    assert rendered["loop"] is True
    assert rendered["max_cycles"] == 3
    default = base_config(tmp_path)
    assert default.mq.loop is False
    assert default.mq.max_cycles == 0
    assert render_bridge_config(default)["loop"] is False
    with pytest.raises(ConfigError):
        base_config(tmp_path, max_cycles=-1)


def test_bridge_loop_mode_runs_for_duration(tmp_path, bridge_env):
    from jev.mq import MqIngestor

    config = base_config(tmp_path, loop=True)
    write_records(
        tmp_path,
        [
            make_envelope(config, "r1", "raw_document", document={"text": "one"}),
            make_envelope(config, "r2", "raw_document", document={"text": "two"}),
        ],
    )
    cycles: list[int | None] = []
    ingestor = MqIngestor(config, document_handler=lambda envelope: cycles.append(envelope.get("loop_cycle")))
    metrics = ingestor.run(manage_bridge=True, duration_s=3.0)
    assert metrics["accepted"] >= 4
    assert metrics["duplicates"] == 0
    assert metrics["bridge"]["loop_cycles"] >= 1
    assert None in cycles and 1 in cycles


def test_bridge_loop_mode_dedup_suppresses_same_cycle_duplicate(tmp_path, bridge_env):
    from jev.mq import MqIngestor

    config = base_config(tmp_path, loop=True, max_cycles=2)
    first = make_envelope(config, "r1", "raw_document", document={"text": "one"})
    write_records(
        tmp_path,
        [
            first,
            make_envelope(config, "r2", "raw_document", document={"text": "two"}),
            first,
        ],
    )
    seen: list[tuple[str, int | None]] = []
    ingestor = MqIngestor(
        config,
        document_handler=lambda envelope: seen.append((str(envelope["record_id"]), envelope.get("loop_cycle"))),
    )
    metrics = ingestor.run(manage_bridge=True, idle_timeout_s=2.0)
    assert metrics["accepted"] == 4
    assert metrics["duplicates"] == 0
    assert seen == [("r1", None), ("r2", None), ("r1", 1), ("r2", 1)]
    bridge_metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert bridge_metrics["duplicates_suppressed"] == 2
    assert bridge_metrics["loop_cycles"] == 1


def test_bridge_crash_before_ack_replays(tmp_path, bridge_env):
    from jev.mq import MqIngestor

    config = base_config(tmp_path)
    write_records(tmp_path, [make_envelope(config, "r1", "raw_document", document={"text": "one"})])
    ingestor = MqIngestor(config)
    process = ingestor.start_bridge()
    client = BridgeClient(config.mq.socket_path)
    client.connect(credits=1)
    first = client.receive(timeout_s=5)
    assert first and first["envelope"]["record_id"] == "r1"
    client.close()
    process.kill()
    process.wait(timeout=10)
    ingestor._bridge = None

    ingestor2 = MqIngestor(config)
    metrics = ingestor2.run(manage_bridge=True, max_records=1, idle_timeout_s=10.0)
    assert metrics["accepted"] == 1
    assert (tmp_path / "replay.jsonl").read_text(encoding="utf-8").count("\n") == 1


def test_training_example_from_envelope_shapes(tmp_path):
    from jev.mq import training_example_from_envelope

    assert training_example_from_envelope({"example": {"input": "a", "output": {"entities": {}}}}) == {
        "input": "a",
        "output": {"entities": {}},
    }
    assert training_example_from_envelope({"text": "b", "labels": {"entities": {"x": ["y"]}}}) == {
        "input": "b",
        "output": {"entities": {"x": ["y"]}},
    }
    assert training_example_from_envelope({"document": {"text": "c"}, "labels": {"entities": {}}}) == {
        "input": "c",
        "output": {"entities": {}},
    }
    assert training_example_from_envelope({"labels": {"entities": {}}}) is None
    assert training_example_from_envelope({"example": {"output": {}}}) is None
