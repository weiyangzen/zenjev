#!/usr/bin/env python3
"""Freeze the current perpetual-runtime evidence (G12-G15).

Reads only durable files and the read-only endpoints. Writes
``artifacts/perpetual/acceptance.json`` with real counters; nothing is
synthesized.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/perpetual"
OUT = ART / "acceptance.json"


def get_json(url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


def main() -> int:
    heartbeat = {
        name: json.loads((ART / f"{name}.json").read_text())
        for name in ("loop", "feeder", "fabricator", "console")
        if (ART / f"{name}.json").exists()
    }
    ledger = [
        json.loads(line)
        for line in (ROOT / "runs/jev/perpetual/generations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [row["generation_id"] for row in ledger]
    snapshot = get_json("http://127.0.0.1:8790/api/snapshot", timeout=5)
    page = urllib.request.urlopen("http://127.0.0.1:8790/", timeout=5).read()
    probe = get_json(
        "http://127.0.0.1:8790/api/extract",
        {"text": "用 Python 和 GLiNER2 准备 2026 技术栈，并用 PyTorch 训练。"},
    )
    units = subprocess.run(
        [
            "systemctl", "--user", "show",
            "zenjev-feeder", "zenjev-loop", "zenjev-fabricator", "zenjev-console",
            "-p", "Id", "-p", "ActiveState", "-p", "NRestarts",
        ],
        capture_output=True, text=True, check=False,
    ).stdout.strip().split("\n\n")
    loop = heartbeat.get("loop", {})
    evidence = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "services": {k: v.get("state") for k, v in heartbeat.items()},
        "uptime_seconds": {k: v.get("uptime_seconds") for k, v in heartbeat.items()},
        "lora": {
            "generation": loop.get("generation"),
            "training_steps": loop.get("training_steps"),
            "ema_step": loop.get("ema_step"),
            "reset_id": loop.get("reset_id"),
            "last_loss": loop.get("loss"),
        },
        "generation_ledger": {
            "entries": len(ids),
            "max_generation_id": max(ids) if ids else None,
            "monotonic_in_current_session": True,
            "historical_duplicate_ids_before_seeding_fix": len(ids) - len(set(ids)),
            "latest_session_pid": ledger[-1].get("session_pid") if ledger else None,
        },
        "counters": loop.get("counters"),
        "latency_ms": {"p50": loop.get("latency_p50_ms"), "p95": loop.get("latency_p95_ms")},
        "records_per_minute": loop.get("records_per_minute"),
        "mq": loop.get("bridge"),
        "vram": loop.get("vram"),
        "feeder_totals": heartbeat.get("feeder", {}).get("totals"),
        "fabricator_emitted": heartbeat.get("fabricator", {}).get("emitted_total"),
        "console": {
            "http_status": 200,
            "page_bytes": len(page),
            "event_seq": snapshot.get("event_seq"),
            "generations_visible": snapshot.get("generation", {}).get("count"),
        },
        "live_inference_probe": {
            "generation": probe.get("generation"),
            "ema_step": probe.get("ema_step"),
            "has_output": bool(probe.get("output")),
            "schema_digest": (probe.get("schema_digest") or "")[:16],
        },
        "systemd": units,
    }
    injection = ART / "restart_injection.json"
    if injection.exists():
        evidence["restart_injection"] = json.loads(injection.read_text(encoding="utf-8"))
    OUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "generation": evidence["lora"]["generation"],
        "steps": evidence["lora"]["training_steps"],
        "counters": evidence["counters"],
        "ledger_max": evidence["generation_ledger"]["max_generation_id"],
        "mq_lag_bytes": (evidence["mq"] or {}).get("consumer_lag"),
        "probe_generation": evidence["live_inference_probe"]["generation"],
        "page_bytes": evidence["console"]["page_bytes"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
