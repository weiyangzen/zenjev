"""Atomic service-status publisher consumed by the Rust ``zenjev-monitor``.

The training/serving process writes one small JSON document per step. Writes are
atomic (temp file + fsync + rename) so a monitor can read it at any moment
without observing a partial document. The monitor treats an old or missing file
as stale instead of crashing.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_STATUS_PATH = "runs/jev/status.json"
SCHEMA_VERSION = 1


def resolve_status_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("JEV_STATUS_PATH") or DEFAULT_STATUS_PATH)


def write_status(payload: dict[str, Any], path: str | Path | None = None) -> Path:
    """Atomically write a status document with the monitor contract fields."""
    target = resolve_status_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at_unix_ms": int(time.time() * 1000),
        "pid": os.getpid(),
    }
    body.update(payload)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def clear_status(path: str | Path | None = None) -> None:
    try:
        resolve_status_path(path).unlink(missing_ok=True)
    except OSError:
        pass
