#!/usr/bin/env python3
"""ZenJev live console (§1.9).

Read-only, offline dashboard: every value comes from durable state files
(service heartbeats, the generation ledger, MQ metrics, the bounded event
feed). The page auto-scrolls the live task feed and the SSE stream keeps
counters fresh. Nothing here can pause or stop the pipeline.
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
MQ_METRICS = REPO / "artifacts/mq/metrics.json"
STATIC_DIR = pathlib.Path(__file__).resolve().parent / "console"
SERVICES = ("loop", "feeder", "fabricator", "console")

STARTED = time.time()


def ledger_entries() -> int:
    try:
        with LEDGER_PATH.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def snapshot() -> dict[str, Any]:
    import calendar
    import time as _time

    heartbeats = read_heartbeats(SERVICES)
    now = _time.time()
    for service in heartbeats.values():
        written = service.get("written_at") if isinstance(service, dict) else None
        age = None
        if isinstance(written, str):
            try:
                age = round(now - calendar.timegm(_time.strptime(written, "%Y-%m-%dT%H:%M:%SZ")), 1)
            except ValueError:
                age = None
        service["age_seconds"] = age
        if service.get("state") == "running" and (age is None or age > 30.0):
            service["state"] = "stale"
    ledger = tail_jsonl(LEDGER_PATH, 200)
    events = tail_jsonl(EVENTS_PATH, 400)
    mq = read_json(MQ_METRICS, default={}) or {}
    loop = heartbeats.get("loop") or {}
    counters = loop.get("counters") or {}
    live_bridge = loop.get("bridge") or {}
    if live_bridge.get("received") is not None:
        merged = dict(mq) if isinstance(mq, dict) else {}
        merged.update({key: value for key, value in live_bridge.items() if value is not None})
        merged["adapter"] = merged.get("adapter") or "dir-spool"
        merged["live_source"] = "loop"
        mq = merged
    feeder = heartbeats.get("feeder") or {}
    fabricator = heartbeats.get("fabricator") or {}
    generations = [record for record in ledger if record.get("kind") != "event"]
    last_generation = generations[-1] if generations else None
    first_at = generations[0].get("created_at") if generations else None
    return {
        "now": utc_now(),
        "server_uptime_seconds": round(time.time() - STARTED, 1),
        "services": heartbeats,
        "loop": loop,
        "counters": counters,
        "generation": {
            "count": ledger_entries(),
            "latest": last_generation,
            "since": first_at,
        },
        "ledger_tail": generations[-12:],
        "mq": mq,
        "feeder": {
            "uptime_seconds": feeder.get("uptime_seconds"),
            "totals": feeder.get("totals"),
            "last_tick": feeder.get("last_tick"),
        },
        "fabricator": {
            "uptime_seconds": fabricator.get("uptime_seconds"),
            "emitted_total": fabricator.get("emitted_total"),
        },
        "events": events,
        "event_seq": events[-1]["seq"] if events else 0,
    }


def proxy_extract(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:8788/extract",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


class ConsoleState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cache: dict[str, Any] = {}
        self.cached_at = 0.0

    def get(self) -> dict[str, Any]:
        with self.lock:
            if time.time() - self.cached_at < 0.9 and self.cache:
                return self.cache
            self.cache = snapshot()
            self.cached_at = time.time()
            return self.cache


STATE = ConsoleState()


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
                self._json(200, STATE.get())
            elif self.path == "/api/generations":
                self._json(200, {"items": tail_jsonl(LEDGER_PATH, 200)})
            elif self.path == "/api/events":
                self._json(200, {"items": tail_jsonl(EVENTS_PATH, 300)})
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
                    self._json(502, {"error": f"loop_unreachable:{type(error).__name__}"})
            else:
                self._json(404, {"error": "not_found"})

        def _sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            last_seq = 0
            try:
                while True:
                    body = STATE.get()
                    events = [event for event in body.get("events", []) if event.get("seq", 0) > last_seq]
                    if events:
                        last_seq = max(event.get("seq", 0) for event in events)
                    payload = {
                        "snapshot": {key: value for key, value in body.items() if key != "events"},
                        "events": events[-60:],
                    }
                    chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
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
