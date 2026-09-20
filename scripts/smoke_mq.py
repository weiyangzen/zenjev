#!/usr/bin/env python3
"""End-to-end MQ bridge smoke without a live broker.

Uses the deterministic `mock` adapter to prove the credit protocol, durable
acceptance ordering, duplicate suppression, DLQ and quarantine routing, then
records the evidence under `artifacts/mq/`.

Set `JEV_MQ_BRIDGE_BIN` to reuse a built bridge; otherwise the script builds
`mq/` with `cargo build --locked` when the binary is missing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.config import JevConfig, _load_document
from jev.mq import EnvelopeError, MqIngestor, build_envelope, locate_bridge_binary, sha256_hex


def ensure_bridge_binary() -> Path:
    binary = locate_bridge_binary()
    if binary is not None:
        return binary
    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    if not cargo.exists():
        raise SystemExit("cargo not found; install the pinned Rust toolchain to build mq/")
    subprocess.run([str(cargo), "build", "--locked"], cwd=ROOT / "mq", check=True)
    binary = locate_bridge_binary()
    if binary is None:
        raise SystemExit("bridge binary still missing after cargo build")
    return binary


def envelope(config: JevConfig, record_id: str, kind: str, payload: dict) -> bytes:
    source_text = json.dumps(payload, sort_keys=True)
    value = build_envelope(
        record_id=record_id,
        kind=kind,
        source={
            "uri": f"smoke://{record_id}",
            "content_sha256": sha256_hex(source_text),
            "retrieved_at": "2026-09-21T00:00:00Z",
            "license": "user-provided",
        },
        schema_id=config.schema.name,
        schema_version=config.schema.version,
        schema_digest=config.schema.digest(),
        **payload,
    )
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def main() -> int:
    binary = ensure_bridge_binary()
    os.environ["JEV_MQ_BRIDGE_BIN"] = str(binary)
    raw = _load_document(ROOT / "configs" / "example.yaml")
    metrics_output = ROOT / "artifacts" / "mq" / "smoke.json"
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jev-mq-smoke-") as directory:
        root = Path(directory)
        raw["mq"] = {
            "enabled": True,
            "adapter": "mock",
            "source_path": str(root / "records.jsonl"),
            "state_path": str(root / "state.json"),
            "wal_path": str(root / "wal.bin"),
            "socket_path": str(root / "bridge.sock"),
            "bridge_config": str(root / "bridge.json"),
            "dlq_path": str(root / "dlq.jsonl"),
            "quarantine_path": str(root / "quarantine.jsonl"),
            "metrics_path": str(root / "bridge-metrics.json"),
            "replay_buffer": str(root / "replay-buffer.jsonl"),
            "batch": 2,
            "max_ack_pending": 2,
            "prefetch": 2,
            "max_deliver": 3,
            "exit_after_drain": True,
        }
        config = JevConfig.from_dict(raw)
        assert config.mq is not None and config.mq.enabled

        mismatch = build_envelope(
            record_id="mismatch",
            kind="raw_document",
            source={
                "uri": "smoke://mismatch",
                "content_sha256": sha256_hex("mismatch"),
                "retrieved_at": "2026-09-21T00:00:00Z",
            },
            schema_id=config.schema.name,
            schema_version=config.schema.version,
            schema_digest="c" * 64,
            document={"text": "mismatch"},
        )
        records = [
            envelope(config, "labelled-1", "labelled_example", {"labels": {"entities": {"technology": ["Python"]}}}),
            b'{"envelope_version": 1, "record_id":',
            json.dumps(mismatch, ensure_ascii=False).encode("utf-8"),
            envelope(config, "pair-1", "tool_task_pair", {"request": {"text": "Fix the parser"}, "response": {"text": "Use pytest and Python"}}),
            envelope(config, "doc-1", "raw_document", {"document": {"text": "Python 2026 stack"}}),
            envelope(config, "labelled-1", "labelled_example", {"labels": {"entities": {"technology": ["Python"]}}}),
        ]
        (root / "records.jsonl").write_bytes(b"\n".join(records) + b"\n")

        accepted: list[str] = []
        ingestor = MqIngestor(
            config,
            labelled_handler=lambda envelope: accepted.append(str(envelope["record_id"])),
            tool_task_handler=lambda envelope: accepted.append(str(envelope["record_id"])),
            document_handler=lambda envelope: accepted.append(str(envelope["record_id"])),
        )
        metrics = ingestor.run(manage_bridge=True, max_records=3, idle_timeout_s=10.0)
        dlq_exists = (root / "dlq.jsonl").exists()
        quarantine_exists = (root / "quarantine.jsonl").exists()
        bridge_metrics = json.loads((root / "bridge-metrics.json").read_text(encoding="utf-8"))
        evidence = {
            "pass": metrics["accepted"] == 3
            and sorted(accepted) == ["doc-1", "labelled-1", "pair-1"]
            and metrics["duplicates"] == 0
            and dlq_exists
            and quarantine_exists
            and bridge_metrics["duplicates_suppressed"] >= 1
            and bridge_metrics["dlq"] >= 1
            and bridge_metrics["quarantined"] >= 1,
            "bridge_binary": str(binary),
            "ingest": metrics,
            "bridge": {
                "pulled": bridge_metrics["pulled"],
                "delivered": bridge_metrics["delivered"],
                "acked": bridge_metrics["acked"],
                "duplicates_suppressed": bridge_metrics["duplicates_suppressed"],
                "dlq": bridge_metrics["dlq"],
                "quarantined": bridge_metrics["quarantined"],
                "offset_checkpoint": bridge_metrics["offset_checkpoint"],
            },
            "accepted_record_ids": accepted,
            "dlq_entries": len((root / "dlq.jsonl").read_text(encoding="utf-8").splitlines()) if dlq_exists else 0,
            "quarantine_entries": len((root / "quarantine.jsonl").read_text(encoding="utf-8").splitlines())
            if quarantine_exists
            else 0,
            "live_broker": False,
            "note": "mock adapter; no external broker was contacted (operator waived the live soak)",
        }
        metrics_output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False))
        if not evidence["pass"]:
            return 2
    print("jev_mq_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
