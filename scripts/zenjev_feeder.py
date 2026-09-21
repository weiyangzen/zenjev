#!/usr/bin/env python3
"""Perpetual raw-dump feeder (§1.7).

Reads the operator's NDJSON dump tree under ``~/data/jevraw`` read-only and
normalizes each capture into a §1.5 envelope appended to the envelope spool
``runs/jev/feed/rawspool.ndjson``. The Rust ``dir-spool`` bridge adapter owns
the watermark for the spool; this feeder owns its own per-file byte offsets for
the raw tree. It never exits: an exhausted pass is an idle tick.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

from zenjev_lib import (
    FEED_DIR,
    PERPETUAL_RUNS,
    append_jsonl,
    atomic_write_json,
    ensure_dirs,
    read_json,
    sha256_hex,
    utc_now,
    write_heartbeat,
)

from jev.config import load_config
from jev.mq import build_envelope

RAW_ROOT = pathlib.Path("/home/sansha/data/jevraw")
STATE_PATH = PERPETUAL_RUNS / "feeder-state.json"
SPOOL_PATH = FEED_DIR / "rawspool.ndjson"
CONFIG_PATH = "configs/zenjev_perpetual.yaml"
LINE_CAP = 8 * 1024 * 1024
MAX_ENVELOPE_BYTES = 256 * 1024
TEXT_CAP = 6000
CHUNK = 1 << 20


def _walk_text(value: Any, sink: list[str], cap: int) -> None:
    """Collect human-visible text from a Responses/SSE shaped structure."""
    if sum(len(item) for item in sink) >= cap:
        return
    if isinstance(value, str):
        sink.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _walk_text(item, sink, cap)
        return
    if isinstance(value, dict):
        for key in ("text", "output_text", "delta", "content", "instructions", "input", "output", "value"):
            if key in value:
                _walk_text(value[key], sink, cap)


def _request_text(record: dict[str, Any]) -> str:
    body: Any = record.get("requestBody")
    if isinstance(body, dict):
        body = body.get("value")
    if body is None:
        raw = record.get("input_body")
        if isinstance(raw, str):
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return raw[:TEXT_CAP]
    if body is None:
        return ""
    sink: list[str] = []
    _walk_text(body, sink, TEXT_CAP)
    return "\n".join(sink)[:TEXT_CAP]


def _response_text(record: dict[str, Any]) -> str:
    body: Any = record.get("responseBody")
    if isinstance(body, dict):
        body = body.get("value")
    if body is None:
        raw = record.get("output_body")
        if isinstance(raw, str):
            body = raw
    if body is None:
        return ""
    if isinstance(body, str):
        sink: list[str] = []
        for line in body.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            event_type = parsed.get("type") if isinstance(parsed, dict) else None
            if isinstance(event_type, str) and event_type.endswith("delta"):
                delta = parsed.get("delta")
                if isinstance(delta, str):
                    sink.append(delta)
            elif event_type in ("response.completed", "response.done") or isinstance(parsed.get("response"), dict):
                _walk_text(parsed.get("response", parsed), sink, TEXT_CAP)
            if sum(len(item) for item in sink) >= TEXT_CAP:
                break
        if sink:
            return "".join(sink)[:TEXT_CAP]
        return body[:TEXT_CAP]
    sink = []
    _walk_text(body, sink, TEXT_CAP)
    return "\n".join(sink)[:TEXT_CAP]


def _observed_model(record: dict[str, Any]) -> str | None:
    model = record.get("model") or record.get("upstream_model") or record.get("requested_model")
    if not model:
        body = record.get("requestBody")
        if isinstance(body, dict):
            value = body.get("value")
            if isinstance(value, dict):
                model = value.get("model")
    return model if isinstance(model, str) and model else None


def _retrieved_at(record: dict[str, Any]) -> str:
    for key in ("created_at", "receivedAt", "startedAt", "finishedAt"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return utc_now()


def load_state() -> dict[str, Any]:
    state = read_json(STATE_PATH, default=None)
    if not isinstance(state, dict) or not isinstance(state.get("files"), dict):
        state = {"files": {}}
    return state


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_PATH, state)


def convert_line(
    line: bytes,
    *,
    rel: str,
    offset: int,
    config: Any,
    allowed_models: set[str],
    fallback_model: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (envelope, reason). Reason is set when the record is skipped."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None, "not_json"
    if not isinstance(record, dict):
        return None, "not_object"
    request = _request_text(record).strip()
    response = _response_text(record).strip()
    if not request or not response:
        return None, "missing_text"
    capture_error = record.get("captureError")
    if capture_error:
        return None, "capture_error"
    status = record.get("responseStatus")
    if isinstance(status, (int, float)) and int(status) != 200:
        return None, f"non_200_response:{int(status)}"
    model = _observed_model(record) or fallback_model
    if not model:
        return None, "missing_model"
    if model not in allowed_models:
        return None, f"model_not_allowlisted:{model}"
    if len(request) < 16 or len(response) < 16:
        return None, "too_short"
    record_id = "jevraw-" + sha256_hex(f"{rel}#{offset}")[:32]
    envelope = build_envelope(
        record_id=record_id,
        kind="tool_task_pair",
        source={
            "uri": f"jevraw://{rel}#{offset}",
            "content_sha256": sha256_hex(line),
            "retrieved_at": _retrieved_at(record),
            "license": "operator-owned-corpus",
        },
        schema_id=config.schema.name,
        schema_version=config.schema.version,
        schema_digest=config.schema.digest(),
        observed_model={"provider": "openai-compatible", "model_id": model},
        request={"text": request},
        response={"text": response},
        raw_path=rel,
        raw_offset=offset,
        synthetic=False,
        response_status=int(status) if isinstance(status, (int, float)) else None,
        capture_error=False,
    )
    if len(json.dumps(envelope, ensure_ascii=False)) > MAX_ENVELOPE_BYTES:
        return None, "envelope_over_cap"
    return envelope, None


_FILE_CACHE: dict[str, Any] = {"scan_at": 0.0, "files": []}


def raw_files() -> list[pathlib.Path]:
    now = time.time()
    if now - float(_FILE_CACHE["scan_at"]) > 300.0 or not _FILE_CACHE["files"]:
        _FILE_CACHE["files"] = sorted(RAW_ROOT.rglob("*.ndjson"), key=lambda path: str(path))
        _FILE_CACHE["scan_at"] = now
    return list(_FILE_CACHE["files"])


def run_tick(
    config: Any,
    allowed_models: set[str],
    state: dict[str, Any],
    *,
    max_records: int,
    max_bytes: int,
) -> dict[str, Any]:
    files = raw_files()
    written = 0
    skipped: dict[str, int] = {}
    bytes_read = 0
    spool = SPOOL_PATH.open("a", encoding="utf-8")
    touched_files = 0
    try:
        for path in files:
            if written >= max_records or bytes_read >= max_bytes:
                break
            rel = str(path.relative_to(RAW_ROOT))
            offset = int(state["files"].get(rel, 0))
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if offset >= size:
                continue
            try:
                handle = path.open("rb")
            except OSError:
                skipped["open_failed"] = skipped.get("open_failed", 0) + 1
                continue
            touched_files += 1
            with handle:
                handle.seek(offset)
                buffer = b""
                discarding = False
                discard_bytes = 0
                while True:
                    if written >= max_records or bytes_read >= max_bytes:
                        break
                    chunk = handle.read(CHUNK)
                    if not chunk:
                        break
                    bytes_read += len(chunk)
                    if discarding:
                        position = chunk.find(b"\n")
                        if position < 0:
                            discard_bytes += len(chunk)
                            continue
                        offset += discard_bytes + position + 1
                        discard_bytes = 0
                        discarding = False
                        chunk = chunk[position + 1:]
                        if not chunk:
                            continue
                    buffer += chunk
                    while True:
                        position = buffer.find(b"\n")
                        if position < 0:
                            if len(buffer) > LINE_CAP:
                                discarding = True
                                discard_bytes = len(buffer)
                                buffer = b""
                            break
                        line = buffer[:position + 1]
                        buffer = buffer[position + 1:]
                        if len(line) > LINE_CAP:
                            skipped["oversized_line"] = skipped.get("oversized_line", 0) + 1
                            offset += len(line)
                            continue
                        if not line.strip():
                            offset += len(line)
                            continue
                        envelope, reason = convert_line(
                            line,
                            rel=rel,
                            offset=offset,
                            config=config,
                            allowed_models=allowed_models,
                        )
                        offset += len(line)
                        if envelope is None:
                            skipped[reason or "unknown"] = skipped.get(reason or "unknown", 0) + 1
                            continue
                        spool.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                        written += 1
            state["files"][rel] = offset
            if written >= max_records or bytes_read >= max_bytes:
                break
        spool.flush()
        save_state(state)
    finally:
        spool.close()
    return {
        "files_scanned": len(files),
        "files_touched": touched_files,
        "bytes_read": bytes_read,
        "envelopes_written": written,
        "skipped": skipped,
        "spool_path": str(SPOOL_PATH.relative_to(pathlib.Path(__file__).resolve().parents[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="single tick (tests/smoke only)")
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--idle-seconds", type=float, default=3.0)
    parser.add_argument(
        "--tick-sleep",
        type=float,
        default=1.5,
        help="pause after every tick so the spool does not outrun training capacity",
    )
    args = parser.parse_args()

    ensure_dirs()
    config = load_config(CONFIG_PATH)
    allowed = {item.model_id for item in config.tool_task.allowed_models}
    allowed |= {alias for item in config.tool_task.allowed_models for alias in item.aliases}
    started = time.time()
    totals = {"envelopes_written": 0, "bytes_read": 0}
    state = load_state()
    while True:
        tick_started = time.time()
        try:
            result = run_tick(
                config,
                allowed,
                state,
                max_records=args.max_records,
                max_bytes=args.max_bytes,
            )
        except Exception as error:  # fail loudly, keep the loop alive
            result = {"error": f"{type(error).__name__}: {error}"}
        totals["envelopes_written"] += int(result.get("envelopes_written", 0) or 0)
        totals["bytes_read"] += int(result.get("bytes_read", 0) or 0)
        write_heartbeat(
            "feeder",
            {
                "state": "running",
                "uptime_seconds": round(time.time() - started, 1),
                "raw_root": str(RAW_ROOT),
                "tick_seconds": round(time.time() - tick_started, 3),
                "totals": totals,
                "last_tick": result,
            },
        )
        if args.once:
            print(json.dumps(result))
            return 0
        if int(result.get("envelopes_written", 0) or 0) == 0:
            time.sleep(args.idle_seconds)
        else:
            time.sleep(max(0.0, args.tick_sleep))


if __name__ == "__main__":
    raise SystemExit(main())
