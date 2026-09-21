#!/usr/bin/env python3
"""Bounded infinite-loop smoke for the external-MQ path without a live broker.

Runs the deterministic `mock` adapter with `loop: true` and `max_cycles: 3`, so
the same two-record fixture replays three times with a distinct `loop_cycle`
and recomputed `content_sha256` per pass. Proves the loop keys do not collide,
cycle 0 stays byte-identical, and the bridge stops when `max_cycles` is reached.

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
from jev.mq import MqIngestor, build_envelope, locate_bridge_binary, sha256_hex

MAX_CYCLES = 3
RECORDS_PER_CYCLE = 2


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
            "uri": f"loop-smoke://{record_id}",
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
    output = ROOT / "artifacts" / "mq" / "loop_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="jev-mq-loop-smoke-") as directory:
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
            "dedup_window": 64,
            "loop": True,
            "max_cycles": MAX_CYCLES,
            "exit_after_drain": True,
        }
        config = JevConfig.from_dict(raw)
        assert config.mq is not None and config.mq.enabled

        records = [
            envelope(config, "doc-1", "raw_document", {"document": {"text": "loop smoke"}}),
            envelope(
                config,
                "pair-1",
                "tool_task_pair",
                {"request": {"text": "Fix the parser"}, "response": {"text": "Use pytest"}},
            ),
        ]
        (root / "records.jsonl").write_bytes(b"\n".join(records) + b"\n")

        accepted: list[dict] = []

        def remember(envelope_value: dict) -> None:
            accepted.append(
                {
                    "record_id": envelope_value.get("record_id"),
                    "loop_cycle": envelope_value.get("loop_cycle"),
                    "content_sha256": envelope_value.get("content_sha256"),
                }
            )

        ingestor = MqIngestor(config, document_handler=remember, tool_task_handler=remember)
        expected = MAX_CYCLES * RECORDS_PER_CYCLE
        metrics = ingestor.run(manage_bridge=True, max_records=expected, idle_timeout_s=10.0)
        bridge_metrics = json.loads((root / "bridge-metrics.json").read_text(encoding="utf-8"))
        cycle_sequence = [item["loop_cycle"] for item in accepted]
        dlq_path = root / "dlq.jsonl"
        quarantine_path = root / "quarantine.jsonl"
        dlq_entries = len(dlq_path.read_text(encoding="utf-8").splitlines()) if dlq_path.exists() else 0
        quarantine_entries = (
            len(quarantine_path.read_text(encoding="utf-8").splitlines())
            if quarantine_path.exists()
            else 0
        )
        evidence = {
            "pass": bool(
                metrics["accepted"] == expected
                and metrics["duplicates"] == 0
                and bridge_metrics["duplicates_suppressed"] == 0
                and bridge_metrics["loop_cycles"] == MAX_CYCLES - 1
                and cycle_sequence == [None, None, 1, 1, 2, 2]
                and all(item["loop_cycle"] is None for item in accepted[:RECORDS_PER_CYCLE])
                and len({item["content_sha256"] for item in accepted}) == expected
                and dlq_entries == 0
                and quarantine_entries == 0
            ),
            "mode": "simulated_loop",
            "max_cycles": MAX_CYCLES,
            "records_expected": expected,
            "records_accepted": metrics["accepted"],
            "loop_cycles": bridge_metrics["loop_cycles"],
            "duplicates": metrics["duplicates"],
            "duplicates_suppressed": bridge_metrics["duplicates_suppressed"],
            "dlq_entries": dlq_entries,
            "quarantine_entries": quarantine_entries,
            "cycle_sequence": cycle_sequence,
            "distinct_content_hashes": len({item["content_sha256"] for item in accepted}),
            "bridge_binary": str(binary),
            "live_broker": False,
            "note": "simulated loop mode on the deterministic mock adapter; no live broker was contacted",
        }
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False))
        if not evidence["pass"]:
            return 2
    print("jev_mq_loop_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
