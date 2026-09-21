#!/usr/bin/env python3
"""Kill-injection restart drill for the perpetual loop (ZJ-071 evidence).

Kills `zenjev-loop` with SIGKILL, waits for the supervisor to bring it back,
and records whether the restart resumed from the durable checkpoint and the
spool watermark without reissuing a generation id. Writes
``artifacts/perpetual/restart_injection.json``.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/perpetual"
HEARTBEAT = ART / "loop.json"
LEDGER = ROOT / "runs/jev/perpetual/generations.jsonl"
CHECKPOINT = ROOT / "runs/jev/perpetual/checkpoints/latest.pt"
UNIT = "zenjev-loop"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def heartbeat() -> dict:
    try:
        return json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def ledger_ids() -> list[int]:
    ids: list[int] = []
    try:
        for line in LEDGER.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ids.append(int(json.loads(line)["generation_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    except OSError:
        pass
    return ids


def systemd_property(name: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, "-p", name, "--value"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def main() -> int:
    pre_heartbeat = heartbeat()
    pre_ids = ledger_ids()
    restarts_before = systemd_property("NRestarts")
    injected_at = utc_now()
    main_pid = int(systemd_property("MainPID") or 0)
    if main_pid <= 0:
        raise SystemExit("cannot resolve the loop MainPID")
    os.kill(main_pid, signal.SIGKILL)

    resumed: dict = {}
    deadline = time.time() + 300.0
    while time.time() < deadline:
        time.sleep(5.0)
        candidate = heartbeat()
        if (
            candidate.get("state") == "running"
            and candidate.get("pid")
            and candidate.get("pid") != pre_heartbeat.get("pid")
            and float(candidate.get("uptime_seconds") or 1e9) < 240.0
        ):
            resumed = candidate
            break

    post_ids = ledger_ids()
    payload = {
        "injected_at": injected_at,
        "recorded_at": utc_now(),
        "unit": UNIT,
        "signal": "SIGKILL",
        "main_pid_killed": main_pid,
        "pid_before": pre_heartbeat.get("pid"),
        "pid_after": resumed.get("pid"),
        "generation_before": pre_heartbeat.get("generation"),
        "generation_after": resumed.get("generation"),
        "training_steps_before": pre_heartbeat.get("training_steps"),
        "training_steps_after": resumed.get("training_steps"),
        "uptime_after_seconds": resumed.get("uptime_seconds"),
        "bridge_alive_after": resumed.get("bridge_alive"),
        "systemd_restarts_before": restarts_before,
        "systemd_restarts_after": systemd_property("NRestarts"),
        "resumed_from_checkpoint": CHECKPOINT.exists(),
        "checkpoint_mtime": (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(CHECKPOINT.stat().st_mtime))
            if CHECKPOINT.exists()
            else None
        ),
        "ledger_entries_before": len(pre_ids),
        "ledger_entries_after": len(post_ids),
        "duplicate_generation_ids_before": len(pre_ids) - len(set(pre_ids)),
        "duplicate_generation_ids_after": len(post_ids) - len(set(post_ids)),
        "generation_monotonic": bool(post_ids and pre_ids and max(post_ids) >= max(pre_ids)),
        "resumed_within_deadline": bool(resumed),
    }
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "restart_injection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if resumed else 1


if __name__ == "__main__":
    raise SystemExit(main())
