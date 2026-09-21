#!/usr/bin/env python3
"""Perpetual standalone inference service (§1.8, ZJ-073).

Serves the accepted EMA snapshot on 0.0.0.0:8791 from the latest valid LoRA
checkpoint. It never trains and never writes model state; it is redeployed onto
a newer checkpoint by `zenjev-deploy`. If no checkpoint exists yet it serves the
immutable base model and reports `model: base` until the first deploy.

Endpoints:
* ``GET  /health``  - loaded generation, checkpoint digest, uptime, request stats
* ``POST /extract`` - schema-constrained extraction
* ``POST /analyze`` - extraction plus the deterministic tool-task decision
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from zenjev_lib import (
    PERPETUAL_RUNS,
    ensure_dirs,
    percentile,
    read_json,
    sha256_hex,
    tail_jsonl,
    write_heartbeat,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/zenjev_perpetual.yaml"
CHECKPOINT_DIR = PERPETUAL_RUNS / "checkpoints"
LEDGER_PATH = PERPETUAL_RUNS / "generations.jsonl"
DEFAULT_PORT = 8791



class Server:
    def __init__(self, config: Any, *, host: str, port: int) -> None:
        from jev.runtime import JevRuntime
        from jev.tool_task import resolve_tool_task
        from jev.training import ContinuousLoRATrainer

        self.config = config
        self.host = host
        self.port = port
        self.resolve_tool_task = resolve_tool_task
        self.started = time.time()
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.latencies: collections.deque[float] = collections.deque(maxlen=500)
        self.runtime = JevRuntime(config)
        checkpoint = CHECKPOINT_DIR / "latest.pt"
        self.loaded_checkpoint = checkpoint if checkpoint.exists() else None
        self.trainer = ContinuousLoRATrainer(
            config,
            self.runtime,
            checkpoint_dir=CHECKPOINT_DIR,
            resume_from=self.loaded_checkpoint,
        )
        self.checkpoint_digest = (
            sha256_hex(self.loaded_checkpoint.read_bytes())[:16]
            if self.loaded_checkpoint is not None
            else None
        )
        # The deploy gate compares global generation ids, so report the
        # save-time manifest (authoritative) with a ledger-digest fallback.
        manifest = read_json(CHECKPOINT_DIR / "latest.json", default={}) or {}
        self.served_generation = int(manifest.get("generation_id") or 0) or self._match_generation(
            self.checkpoint_digest
        )
        self.model_source = "lora-checkpoint" if self.loaded_checkpoint else "base"

    @staticmethod
    def _match_generation(checkpoint_digest: str | None) -> int:
        if not checkpoint_digest:
            return 0
        for row in reversed(tail_jsonl(LEDGER_PATH, 400)):
            digest = row.get("adapter_digest")
            if isinstance(digest, str) and digest.startswith(checkpoint_digest):
                return int(row.get("generation_id") or 0)
        return 0

    def heartbeat(self) -> None:
        while True:
            with self.lock:
                requests, errors = self.requests, self.errors
                latencies = list(self.latencies)
            write_heartbeat(
                "serve",
                {
                    "state": "running",
                    "host": self.host,
                    "port": self.port,
                    "uptime_seconds": round(time.time() - self.started, 1),
                    "model_source": self.model_source,
                    "generation": self.served_generation,
                    "local_generation": int(self.runtime.stats.model_generation),
                    "ema_step": int(self.runtime.ema.updates),
                    "training_steps": int(self.runtime.stats.training_steps),
                    "reset_id": int(self.runtime.stats.reset_id),
                    "checkpoint": str(self.loaded_checkpoint) if self.loaded_checkpoint else None,
                    "checkpoint_sha256_16": self.checkpoint_digest,
                    "requests": requests,
                    "errors": errors,
                    "latency_p50_ms": percentile(latencies, 0.5),
                    "latency_p95_ms": percentile(latencies, 0.95),
                    "schema_digest_16": self.config.schema.digest()[:16],
                    "config_digest_16": self.config.digest()[:16],
                },
            )
            time.sleep(5.0)

    def handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path == "/health":
                    with server.lock:
                        requests, errors = server.requests, server.errors
                        latencies = list(server.latencies)
                    self._send(
                        200,
                        {
                            "state": "ok",
                            "model_source": server.model_source,
                            "generation": server.served_generation,
                            "local_generation": int(server.runtime.stats.model_generation),
                            "ema_step": int(server.runtime.ema.updates),
                            "checkpoint": str(server.loaded_checkpoint)
                            if server.loaded_checkpoint
                            else None,
                            "checkpoint_sha256_16": server.checkpoint_digest,
                            "uptime_seconds": round(time.time() - server.started, 1),
                            "requests": requests,
                            "errors": errors,
                            "latency_p50_ms": percentile(latencies, 0.5),
                            "latency_p95_ms": percentile(latencies, 0.95),
                            "schema_digest_16": server.config.schema.digest()[:16],
                        },
                    )
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 1_000_000:
                        self._send(413, {"error": "invalid_size"})
                        return
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, json.JSONDecodeError):
                    self._send(400, {"error": "invalid_json"})
                    return
                text = str(payload.get("text") or "")[:4000]
                if not text:
                    self._send(400, {"error": "missing_text"})
                    return
                started = time.perf_counter()
                try:
                    result = server.runtime.infer_result(text)
                    # Report the deployed generation, not this process's local
                    # publish counter, so clients see the rollout identity.
                    result = {**result, "generation": server.served_generation,
                              "local_generation": result.get("generation")}
                    if self.path == "/analyze":
                        model_id = str(payload.get("model_id") or "gpt-5.6-sol")
                        output = result.get("output") if isinstance(result.get("output"), dict) else {}
                        try:
                            decision = server.resolve_tool_task(
                                output, server.config.tool_task, model_id=model_id
                            )
                        except Exception as error:  # noqa: BLE001
                            decision = {
                                "action": "reject",
                                "reasons": [f"{type(error).__name__}:{error}"],
                            }
                        result = {**result, "decision": decision, "model_id": model_id}
                    elif self.path != "/extract":
                        self._send(404, {"error": "not_found"})
                        return
                except Exception as error:  # noqa: BLE001 - never kill the server
                    with server.lock:
                        server.errors += 1
                    self._send(500, {"error": f"{type(error).__name__}: {error}"})
                    return
                with server.lock:
                    server.requests += 1
                    server.latencies.append((time.perf_counter() - started) * 1000.0)
                self._send(200, result)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        return Handler


def main() -> int:
    os.environ.setdefault("JEV_MODEL_PATH", str(pathlib.Path.home() / "jev-model-base"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    ensure_dirs()
    from jev.config import load_config

    config = load_config(CONFIG_PATH)
    server = Server(config, host=args.host, port=args.port)
    threading.Thread(target=server.heartbeat, name="heartbeat", daemon=True).start()
    print(
        json.dumps(
            {
                "event": "serve_listening",
                "host": args.host,
                "port": args.port,
                "model_source": server.model_source,
                "checkpoint": str(server.loaded_checkpoint) if server.loaded_checkpoint else None,
            }
        )
    )
    httpd = ThreadingHTTPServer((args.host, args.port), server.handler())
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
