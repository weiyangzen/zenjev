"""Shared helpers for the perpetual ZenJev services (§1.6–1.9).

Every helper here is deliberately stdlib-only so the services stay runnable on
the pinned host without extra packages. Real counters come from durable files;
nothing in this module fabricates a metric value.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Iterable

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ARTIFACTS = REPO / "artifacts" / "perpetual"
# The queue lives inside the operator's data root. The feeder appends envelopes
# there and the Rust dir-spool bridge consumes them; the dump tree itself stays
# read-only and the feeder skips this directory when scanning for captures.
FEED_DIR = pathlib.Path(
    os.environ.get("ZENJEV_FEED_DIR", "/home/sansha/data/jevraw/_zenjev")
)
PERPETUAL_RUNS = REPO / "runs" / "jev" / "perpetual"
LOGS_DIR = REPO / "runs" / "jev" / "logs"
EVENTS_PATH = ARTIFACTS / "events.jsonl"
EVENTS_MAX_BYTES = 8 * 1024 * 1024
EVENTS_KEEP_LINES = 4000


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_hex(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def ensure_dirs() -> None:
    for path in (ARTIFACTS, FEED_DIR, PERPETUAL_RUNS, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def read_json(path: str | pathlib.Path, default: Any = None) -> Any:
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def append_jsonl(path: str | pathlib.Path, payload: dict[str, Any]) -> None:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def rotate_jsonl(path: str | pathlib.Path, *, max_bytes: int, keep_lines: int) -> None:
    target = pathlib.Path(path)
    try:
        if target.stat().st_size <= max_bytes:
            return
    except OSError:
        return
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    target.write_text("\n".join(lines[-keep_lines:]) + ("\n" if lines else ""), encoding="utf-8")


def tail_jsonl(path: str | pathlib.Path, limit: int) -> list[dict[str, Any]]:
    target = pathlib.Path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in raw[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def percentile(values: Iterable[float], quantile: float) -> float | None:
    data = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not data:
        return None
    index = min(len(data) - 1, max(0, int(round(quantile * (len(data) - 1)))))
    return round(float(data[index]), 3)


def heartbeat_path(service: str) -> pathlib.Path:
    return ARTIFACTS / f"{service}.json"


def write_heartbeat(service: str, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body["service"] = service
    body["written_at"] = utc_now()
    body["pid"] = os.getpid()
    atomic_write_json(heartbeat_path(service), body)


def read_heartbeats(services: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for service in services:
        out[service] = read_json(heartbeat_path(service), default={"state": "missing"})
    return out
