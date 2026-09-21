#!/usr/bin/env python3
"""LoRA deploy gate (§1.8, ZJ-074).

Watches the generation ledger and, when a *stable* checkpointed generation is
newer than what `zenjev-serve` currently serves, restarts the service so the
externally deployed inference endpoint runs the appropriate LoRA. Deployments
are rate-limited, health-checked, and recorded in
``artifacts/perpetual/deploys.jsonl`` plus the live event feed.

Policy (environment overrides):
* ``ZENJEV_DEPLOY_RECORDS`` (default 200000): consumed records required before the
  next deployment, so the LoRA is deployed once training has learned on a
  meaningful batch of records.
* ``ZENJEV_DEPLOY_MIN_INTERVAL_S`` (default 60): minimum seconds between deploys.
* ``ZENJEV_DEPLOY_QUIET_S`` (default 20): a generation must be at least this old
  before it is deployed, so the checkpoint is settled.
* ``ZENJEV_DEPLOY_HEALTH_TIMEOUT_S`` (default 240): wait for the new service.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time
import urllib.request
from typing import Any

from zenjev_lib import (
    ARTIFACTS,
    PERPETUAL_RUNS,
    append_jsonl,
    read_json,
    utc_now,
    write_heartbeat,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
LEDGER_PATH = PERPETUAL_RUNS / "generations.jsonl"
MANIFEST_PATH = PERPETUAL_RUNS / "checkpoints/latest.json"
STATE_PATH = ARTIFACTS / "deploy_state.json"
SERVE_HEARTBEAT = ARTIFACTS / "serve.json"
DEPLOY_LOG = ARTIFACTS / "deploys.jsonl"
EVENTS_PATH = ARTIFACTS / "events.jsonl"
SERVE_HEALTH = "http://127.0.0.1:8791/health"
UNIT = "zenjev-serve"


def ledger_tail(limit: int = 200) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = LEDGER_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def newest_deployable() -> dict[str, Any] | None:
    """The checkpoint manifest written at save time; ledger is the fallback."""
    manifest = read_json(MANIFEST_PATH, default={}) or {}
    if isinstance(manifest, dict) and manifest.get("adapter_digest"):
        return {
            "generation_id": manifest.get("generation_id"),
            "adapter_digest": manifest.get("adapter_digest"),
            "created_at": manifest.get("created_at"),
            "source": "manifest",
        }
    for row in reversed(ledger_tail()):
        if row.get("adapter_digest"):
            return {**row, "source": "ledger"}
    return None


def consumed_records() -> int:
    """Cumulative records consumed: loop heartbeat, ledger fallback."""
    heartbeat = read_json(ARTIFACTS / "loop.json", default={}) or {}
    counters = heartbeat.get("counters") or {}
    value = int(counters.get("real_pairs") or 0)
    if value:
        return value
    rows = ledger_tail(5)
    for row in reversed(rows):
        try:
            return int(row.get("records_consumed_total") or 0)
        except (TypeError, ValueError):
            continue
    return 0


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default={})
    return state if isinstance(state, dict) else {}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def serve_state() -> dict[str, Any]:
    state = read_json(SERVE_HEARTBEAT, default={})
    return state if isinstance(state, dict) else {}


def health() -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(SERVE_HEALTH, timeout=5) as response:
            value = json.loads(response.read().decode("utf-8"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def restart_serve() -> tuple[int | None, int | None]:
    before = subprocess.run(
        ["systemctl", "--user", "show", UNIT, "-p", "MainPID", "--value"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    subprocess.run(["systemctl", "--user", "restart", UNIT], check=True)
    return (int(before) if before.isdigit() else None, None)


def wait_healthy(generation: int, timeout_s: float) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        body = health()
        if body and int(body.get("generation") or 0) >= generation:
            return body
        time.sleep(3.0)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--records", type=int,
                        default=int(os.environ.get("ZENJEV_DEPLOY_RECORDS", "200000")))
    parser.add_argument("--min-interval", type=float,
                        default=float(os.environ.get("ZENJEV_DEPLOY_MIN_INTERVAL_S", "60")))
    parser.add_argument("--quiet-seconds", type=float,
                        default=float(os.environ.get("ZENJEV_DEPLOY_QUIET_S", "20")))
    parser.add_argument("--health-timeout", type=float,
                        default=float(os.environ.get("ZENJEV_DEPLOY_HEALTH_TIMEOUT_S", "240")))
    parser.add_argument("--once", action="store_true", help="evaluate one deploy decision")
    parser.add_argument("--force", action="store_true",
                        help="ignore the min-interval and quiet gates (drill only)")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    state = load_state()
    last_deploy_at = float(state.get("last_deploy_at") or 0.0)
    deployed_records = int(state.get("records_at_last_deploy") or 0)
    deploys = int(state.get("deploys") or 0)
    while True:
        candidate = newest_deployable()
        state = serve_state()
        now = time.time()
        consumed = consumed_records()
        decision: str | None = None
        if candidate is None:
            decision = "no_checkpoint_generation"
        else:
            loaded = int(state.get("generation") or 0)
            target = int(candidate.get("generation_id") or 0)
            age = None
            created = candidate.get("created_at")
            if isinstance(created, str):
                try:
                    age = now - time.mktime(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) \
                        + time.localtime().tm_gmtoff
                except ValueError:
                    age = None
            if target <= loaded:
                decision = "up_to_date"
            elif not args.force and age is not None and age < args.quiet_seconds:
                decision = "waiting_for_stable_checkpoint"
            elif not args.force and last_deploy_at and now - last_deploy_at < args.min_interval:
                decision = "cooldown"
            elif not args.force and consumed < deployed_records + args.records:
                decision = "waiting_records"
            else:
                decision = "deploy"
        since_deploy = max(0, consumed - deployed_records)
        action: dict[str, Any] = {
            "checked_at": utc_now(),
            "decision": decision,
            "records_consumed": consumed,
            "records_at_last_deploy": deployed_records,
            "records_since_deploy": since_deploy,
            "records_threshold": args.records,
            "next_deploy_in": max(0, args.records - since_deploy),
            "target_generation": (candidate or {}).get("generation_id"),
            "target_digest": ((candidate or {}).get("adapter_digest") or None),
            "target_source": (candidate or {}).get("source"),
            "serve_generation": state.get("generation"),
            "serve_checkpoint_sha256_16": state.get("checkpoint_sha256_16"),
        }
        if decision == "deploy":
            before_pid, _ = restart_serve()
            began = time.perf_counter()
            body = wait_healthy(int(candidate["generation_id"]), args.health_timeout)
            elapsed = round(time.perf_counter() - began, 2)
            action.update(
                {
                    "deployed_at": utc_now(),
                    "serve_pid_before": before_pid,
                    "serve_pid_after": _pid_of_unit(),
                    "generation_deployed": int(candidate["generation_id"]),
                    "adapter_digest": candidate.get("adapter_digest"),
                    "health_ok": body is not None,
                    "deploy_seconds": elapsed,
                    "loaded_generation": (body or {}).get("generation"),
                    "loaded_checkpoint_sha256_16": (body or {}).get("checkpoint_sha256_16"),
                }
            )
            append_jsonl(DEPLOY_LOG, action)
            append_jsonl(EVENTS_PATH, {"kind": "deploy", **action})
            deploys += 1
            last_deploy_at = time.time()
            deployed_records = consumed
            save_state(
                {
                    "records_at_last_deploy": deployed_records,
                    "last_deploy_at": last_deploy_at,
                    "deploys": deploys,
                    "latest_generation": int(candidate.get("generation_id") or 0),
                }
            )
        write_heartbeat(
            "deploy",
            {
                "state": "running",
                "uptime_seconds": round(time.time() - started, 1),
                "deploys": deploys,
                "last_action": action,
                "records_threshold": args.records,
                "records_since_deploy": since_deploy,
                "next_deploy_in": max(0, args.records - since_deploy),
                "min_interval_s": args.min_interval,
                "quiet_seconds": args.quiet_seconds,
            },
        )
        if args.once:
            print(json.dumps(action, ensure_ascii=False))
            return 0
        time.sleep(max(1.0, args.interval))


def _pid_of_unit() -> int | None:
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, "-p", "MainPID", "--value"],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


if __name__ == "__main__":
    raise SystemExit(main())
