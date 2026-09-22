#!/usr/bin/env python3
"""ZenJev live console (§1.9).

Read-only, offline dashboard. Every value comes from durable state files
(service heartbeats, the generation ledger, MQ metrics, the bounded event feed)
or from the deployed inference service. The page streams a compact snapshot on
connect and then deltas only, so the SSE body stays a few kilobytes instead of
re-sending the whole history every second.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from zenjev_lib import (
    ARTIFACTS,
    EVENTS_PATH,
    PERPETUAL_RUNS,
    ensure_dirs,
    read_heartbeats,
    read_json,
    tail_jsonl,
    utc_now,
    write_heartbeat,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
LEDGER_PATH = PERPETUAL_RUNS / "generations.jsonl"
DEPLOYS_PATH = ARTIFACTS / "deploys.jsonl"
MQ_METRICS = REPO / "artifacts/mq/metrics.json"
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "console"
SERVICES = ("loop", "feeder", "fabricator", "serve", "deploy", "console")
SNAPSHOT_EVENTS = 40
DELTA_EVENTS = 60
LEDGER_ROWS = 24
SERVE_HEALTH = "http://127.0.0.1:8791/health"
LOOP_HEALTH = "http://127.0.0.1:8788/health"

STARTED = time.time()
STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {"cache": {}, "cached_at": 0.0}
EVENT_LOCK = threading.Lock()
LAST_EVENT_SEQ = 0


def ledger_entries() -> int:
    try:
        with LEDGER_PATH.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def with_ages(heartbeats: dict[str, Any]) -> dict[str, Any]:
    import calendar

    now = time.time()
    for service in heartbeats.values():
        written = service.get("written_at") if isinstance(service, dict) else None
        age = None
        if isinstance(written, str):
            try:
                age = round(now - calendar.timegm(time.strptime(written, "%Y-%m-%dT%H:%M:%SZ")), 1)
            except ValueError:
                age = None
        service["age_seconds"] = age
        if service.get("state") == "running" and (age is None or age > 30.0):
            service["state"] = "stale"
    return heartbeats


def decision_mix(counters: dict[str, Any]) -> dict[str, Any]:
    allow = int(counters.get("decision_allow") or 0)
    review = int(counters.get("decision_review") or 0)
    reject = int(counters.get("decision_reject") or 0)
    total = allow + review + reject
    if not total:
        return {"allow": 0, "review": 0, "reject": 0, "total": 0}
    return {
        "allow": round(100.0 * allow / total, 1),
        "review": round(100.0 * review / total, 1),
        "reject": round(100.0 * reject / total, 1),
        "total": total,
    }


def meta_block() -> dict[str, Any]:
    """Compact scalars sent with every delta and stored in the snapshot."""
    heartbeats = with_ages(read_heartbeats(SERVICES))
    loop = heartbeats.get("loop") or {}
    serve = heartbeats.get("serve") or {}
    deploy = heartbeats.get("deploy") or {}
    mq = read_json(MQ_METRICS, default={}) or {}
    live_bridge = loop.get("bridge") or {}
    if live_bridge.get("received") is not None:
        merged = dict(mq) if isinstance(mq, dict) else {}
        merged.update({key: value for key, value in live_bridge.items() if value is not None})
        merged["adapter"] = merged.get("adapter") or "dir-spool"
        merged["live_source"] = "loop"
        mq = merged
    return {
        "now": utc_now(),
        "services": heartbeats,
        "loop": loop,
        "serve": serve,
        "deploy": deploy,
        "counters": loop.get("counters") or {},
        "generation": {
            "count": ledger_entries(),
            "latest_generation_id": (deploy.get("last_action") or {}).get("generation_deployed")
            or serve.get("generation"),
            "serve_generation": serve.get("generation"),
            "serve_checkpoint_sha256_16": serve.get("checkpoint_sha256_16"),
            "deploys": deploy.get("deploys"),
        },
        "mq": mq,
        "train_batch": loop.get("train_batch"),
        "loss_ema": loop.get("loss_ema"),
        "loss_mean_50": loop.get("loss_mean_50"),
        "instruction_sha16": loop.get("instruction_sha16"),
        "pairs_built": loop.get("pairs_built"),
        "skipped_bad_capture": (loop.get("counters") or {}).get("skipped_bad_capture"),
        "deploy_progress": {
            "next_deploy_in": deploy.get("next_deploy_in"),
            "records_since_deploy": deploy.get("records_since_deploy"),
            "records_threshold": deploy.get("records_threshold"),
        },
        "feeder": {
            "uptime_seconds": (heartbeats.get("feeder") or {}).get("uptime_seconds"),
            "totals": (heartbeats.get("feeder") or {}).get("totals"),
            "cursor": (heartbeats.get("feeder") or {}).get("cursor"),
            "queue_path": (heartbeats.get("feeder") or {}).get("queue_path"),
        },
        "decision_mix": decision_mix(loop.get("counters") or {}),
        "records_per_generation": round(
            (loop.get("counters") or {}).get("real_pairs", 0)
            / max(1, (loop.get("generation") or 1)),
            1,
        ),
        "fabricator": {
            "uptime_seconds": (heartbeats.get("fabricator") or {}).get("uptime_seconds"),
            "emitted_total": (heartbeats.get("fabricator") or {}).get("emitted_total"),
        },
    }


def snapshot() -> dict[str, Any]:
    """Full snapshot used by /api/snapshot and the SSE handshake."""
    body = meta_block()
    body["server_uptime_seconds"] = round(time.time() - STARTED, 1)
    body["ledger_tail"] = tail_jsonl(LEDGER_PATH, LEDGER_ROWS)
    body["deploy_tail"] = tail_jsonl(DEPLOYS_PATH, 8)
    body["events"] = tail_jsonl(EVENTS_PATH, SNAPSHOT_EVENTS)
    body["event_seq"] = body["events"][-1]["seq"] if body["events"] else 0
    return body


def cached_snapshot() -> dict[str, Any]:
    with STATE_LOCK:
        if time.time() - STATE["cached_at"] < 1.0 and STATE["cache"]:
            return STATE["cache"]
        STATE["cache"] = snapshot()
        STATE["cached_at"] = time.time()
        return STATE["cache"]


def events_after(seq: int, limit: int = DELTA_EVENTS) -> list[dict[str, Any]]:
    events = [event for event in tail_jsonl(EVENTS_PATH, 400) if int(event.get("seq", 0)) > seq]
    return events[-limit:]


def proxy_extract(payload: dict[str, Any]) -> dict[str, Any]:
    """Prefer the externally deployed inference service, fall back to the loop."""
    last_error: Exception | None = None
    for url, endpoint in (
        (SERVE_HEALTH.replace("/health", "/extract"), "serve"),
        (LOOP_HEALTH.replace("/health", "/extract"), "loop"),
    ):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                value = json.loads(response.read().decode("utf-8"))
            if isinstance(value, dict):
                value["served_by"] = endpoint
                return value
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(str(last_error))


def make_handler(static_dir: pathlib.Path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: pathlib.Path, content_type: str) -> None:
            if not path.exists():
                self._json(404, {"error": "missing_asset", "path": str(path.name)})
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path in ("/", "/index.html"):
                self._file(static_dir / "index.html", "text/html; charset=utf-8")
            elif self.path == "/api/snapshot":
                self._json(200, cached_snapshot())
            elif self.path == "/api/generations":
                self._json(200, {"items": tail_jsonl(LEDGER_PATH, 200)})
            elif self.path == "/api/events":
                self._json(200, {"items": tail_jsonl(EVENTS_PATH, 300)})
            elif self.path == "/api/deploys":
                self._json(200, {"items": tail_jsonl(DEPLOYS_PATH, 100)})
            elif self.path == "/events":
                self._sse()
            else:
                self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 100_000:
                    self._json(413, {"error": "invalid_size"})
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "invalid_json"})
                return
            if self.path == "/api/extract":
                text = str(payload.get("text") or "")[:2000]
                if not text:
                    self._json(400, {"error": "missing_text"})
                    return
                try:
                    self._json(200, proxy_extract({"text": text}))
                except Exception as error:  # noqa: BLE001
                    self._json(502, {"error": f"inference_unreachable:{type(error).__name__}"})
            else:
                self._json(404, {"error": "not_found"})

        def _send_event(self, payload: dict[str, Any]) -> bool:
            try:
                chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            body = cached_snapshot()
            last_seq = int(body.get("event_seq") or 0)
            if not self._send_event(
                {"type": "snapshot", "snapshot": {k: v for k, v in body.items() if k != "events"},
                 "events": body.get("events", [])}
            ):
                return
            while True:
                time.sleep(1.0)
                fresh = events_after(last_seq)
                if fresh:
                    last_seq = max(int(event.get("seq", 0)) for event in fresh)
                delta = {"type": "delta", "meta": meta_block(), "events": fresh}
                if not self._send_event(delta):
                    return

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def heartbeat_loop() -> None:
    while True:
        write_heartbeat(
            "console",
            {
                "state": "running",
                "uptime_seconds": round(time.time() - STARTED, 1),
                "static_dir": str(STATIC_DIR.relative_to(REPO)),
                "sse_mode": "snapshot+delta",
            },
        )
        time.sleep(5.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    ensure_dirs()
    threading.Thread(target=heartbeat_loop, name="heartbeat", daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(STATIC_DIR))
    print(json.dumps({"event": "console_listening", "host": args.host, "port": args.port}))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
