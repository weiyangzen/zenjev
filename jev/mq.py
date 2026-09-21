"""Trainer-side client for the Rust ``mq`` bridge.

The bridge owns the broker protocol and only acks/commits a broker offset after
this client confirms durable acceptance. Credit-based Unix-socket framing keeps
in-flight work bounded, so a slow trainer backpressures the broker instead of
buffering without bound. Exactly-once is never claimed: redelivery is expected
and suppressed by ``record_id`` plus the canonical ``content_sha256``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import JevConfig, MqConfig

PROTOCOL_VERSION = 1
MAX_FRAME = 64 * 1024 * 1024
KINDS = ("labelled_example", "tool_task_pair", "raw_document")


class EnvelopeError(ValueError):
    """A rejected envelope. ``class_`` matches the bridge routing contract."""

    def __init__(self, class_: str, reason: str):
        super().__init__(f"{class_}:{reason}")
        self.class_ = class_
        self.reason = reason


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def envelope_content_hash(envelope: dict[str, Any]) -> str:
    body = {key: value for key, value in envelope.items() if key != "content_sha256"}
    return sha256_hex(canonical_json(body))


def build_envelope(
    *,
    record_id: str,
    kind: str,
    source: dict[str, Any],
    schema_id: str,
    schema_version: int,
    schema_digest: str,
    observed_model: dict[str, Any] | None = None,
    **payload: Any,
) -> dict[str, Any]:
    """Build a contract-valid envelope with a canonical content hash."""
    if kind not in KINDS:
        raise ValueError(f"unknown envelope kind: {kind}")
    envelope: dict[str, Any] = {
        "envelope_version": 1,
        "record_id": record_id,
        "schema": {"id": schema_id, "version": schema_version, "digest": schema_digest},
        "kind": kind,
        "source": source,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if observed_model is not None:
        envelope["observed_model"] = observed_model
    envelope.update(payload)
    envelope["content_sha256"] = envelope_content_hash(envelope)
    return envelope


def _required(mapping: dict[str, Any], key: str, reason: str) -> Any:
    if key not in mapping or mapping[key] in ("", None):
        raise EnvelopeError("dlq", reason)
    return mapping[key]


def validate_envelope(raw: bytes, config: JevConfig, *, max_bytes: int) -> dict[str, Any]:
    """Validate one raw message exactly like the bridge does, fail closed."""
    if len(raw) > max_bytes:
        raise EnvelopeError("dlq", "oversized_message")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EnvelopeError("dlq", "envelope_not_utf8") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvelopeError("dlq", "envelope_not_json") from exc
    if not isinstance(value, dict):
        raise EnvelopeError("dlq", "envelope_not_object")
    if value.get("envelope_version") != 1:
        raise EnvelopeError("dlq", "unsupported_envelope_version")
    record_id = _required(value, "record_id", "missing_record_id")
    content_sha256 = _required(value, "content_sha256", "missing_content_sha256")
    if not isinstance(record_id, str) or not _is_sha256(content_sha256):
        raise EnvelopeError("dlq", "invalid_content_sha256")
    if envelope_content_hash(value) != content_sha256:
        raise EnvelopeError("dlq", "content_hash_mismatch")
    schema = value.get("schema")
    if not isinstance(schema, dict):
        raise EnvelopeError("dlq", "missing_schema")
    if (
        schema.get("id") != config.schema.name
        or int(schema.get("version", 0)) != config.schema.version
        or schema.get("digest") != config.schema.digest()
    ):
        raise EnvelopeError("quarantine", "schema_digest_mismatch")
    kind = _required(value, "kind", "missing_kind")
    if kind not in KINDS:
        raise EnvelopeError("dlq", "unknown_kind")
    source = value.get("source")
    if not isinstance(source, dict):
        raise EnvelopeError("dlq", "missing_source")
    for key in ("uri", "retrieved_at"):
        if not isinstance(source.get(key), str) or not source.get(key):
            raise EnvelopeError("dlq", f"source_missing_{key}")
    if not _is_sha256(source.get("content_sha256")):
        raise EnvelopeError("dlq", "invalid_source_content_sha256")
    if kind == "labelled_example" and "labels" not in value and "example" not in value:
        raise EnvelopeError("dlq", "labelled_example_requires_labels")
    if kind == "tool_task_pair":
        for section in ("request", "response"):
            payload = value.get(section)
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not payload["text"]:
                raise EnvelopeError("dlq", f"{section}_requires_text")
    if kind == "raw_document" and "document" not in value and "example" not in value:
        raise EnvelopeError("dlq", "raw_document_requires_document")
    observed = value.get("observed_model")
    if observed is not None:
        if not isinstance(observed, dict):
            raise EnvelopeError("dlq", "invalid_observed_model")
        model_id = observed.get("model_id", observed.get("id"))
        if not model_id:
            raise EnvelopeError("dlq", "observed_model_requires_model_id")
        if config.tool_task is not None and config.tool_task.allowed_models:
            allowed_ids = {item.model_id for item in config.tool_task.allowed_models}
            allowed_ids |= {alias for item in config.tool_task.allowed_models for alias in item.aliases}
            if model_id not in allowed_ids:
                raise EnvelopeError("dlq", "observed_model_not_allowlisted")
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def idempotency_key(envelope: dict[str, Any]) -> str:
    return f"{envelope.get('record_id')}|{envelope.get('content_sha256')}"


def training_example_from_envelope(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a ``labelled_example`` envelope to a GLiNER2 training record.

    Accepted shapes: ``example`` already holding ``{input, output}``, or
    ``labels`` (output only) paired with ``text``/``document.text``. Anything
    else returns None so the caller can quarantine instead of training garbage.
    """
    candidate = envelope.get("example")
    if isinstance(candidate, dict):
        text = candidate.get("input", candidate.get("text"))
        if isinstance(text, str) and text and isinstance(candidate.get("output"), dict):
            return {"input": text, "output": candidate["output"]}
    labels = envelope.get("labels")
    text = envelope.get("text")
    if not isinstance(text, str) or not text:
        document = envelope.get("document")
        text = document.get("text") if isinstance(document, dict) else None
    if isinstance(text, str) and text and isinstance(labels, dict):
        return {"input": text, "output": labels}
    return None


def _frame(payload: dict[str, Any]) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def _read_exact(sock: socket.socket, length: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < length:
        chunk = sock.recv(length - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def read_frame(sock: socket.socket) -> dict[str, Any] | None:
    header = _read_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length > MAX_FRAME:
        raise RuntimeError("bridge frame exceeds the maximum size")
    body = _read_exact(sock, length)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


class BridgeClient:
    """Credit-based Unix-socket client for ``jev-mq-bridge``."""

    def __init__(self, socket_path: str | Path, *, timeout_s: float = 30.0):
        self.socket_path = Path(socket_path)
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._send_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self, *, credits: int) -> dict[str, Any]:
        if self._socket is not None:
            raise RuntimeError("bridge client is already connected")
        deadline = time.monotonic() + self.timeout_s
        last_error: OSError | None = None
        while time.monotonic() < deadline:
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(self.timeout_s)
                sock.connect(str(self.socket_path))
                self._socket = sock
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        if self._socket is None:
            raise TimeoutError(f"cannot connect to bridge socket {self.socket_path}: {last_error}")
        self.send({"type": "hello", "protocol": PROTOCOL_VERSION, "credits": int(credits)})
        ready = self.receive(timeout_s=self.timeout_s)
        if not ready or ready.get("type") != "ready":
            raise RuntimeError(f"bridge did not send ready: {ready}")
        return ready

    def send(self, payload: dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("bridge client is not connected")
        with self._send_lock:
            self._socket.sendall(_frame(payload))

    def receive(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        if self._socket is None:
            raise RuntimeError("bridge client is not connected")
        if timeout_s is not None:
            self._socket.settimeout(timeout_s)
        try:
            return read_frame(self._socket)
        except socket.timeout:
            return None

    def credit(
        self,
        *,
        acks: Iterable[str] = (),
        quarantine: Iterable[dict[str, str]] = (),
        count: int = 0,
    ) -> None:
        self.send({"type": "credit", "count": int(count), "acks": list(acks), "quarantine": list(quarantine)})

    def pause(self) -> None:
        self.send({"type": "pause"})

    def resume(self) -> None:
        self.send({"type": "resume"})

    def drain(self) -> None:
        self.send({"type": "drain"})

    def status(self) -> dict[str, Any]:
        self.send({"type": "status"})
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            message = self.receive(timeout_s=min(1.0, self.timeout_s))
            if message and message.get("type") == "status":
                return message.get("metrics", {})
        return {}

    def goodbye(self) -> None:
        try:
            self.send({"type": "goodbye"})
        except OSError:
            pass

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None


class ReplayBuffer:
    """Durable, idempotent replay buffer for accepted records."""

    def __init__(self, path: str | Path, *, max_keys: int = 100_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_keys = max_keys
        self._keys: set[str] = set()
        self._order: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = record.get("idempotency_key")
                if isinstance(key, str):
                    self._remember(key)

    def _remember(self, key: str) -> None:
        if key not in self._keys:
            self._keys.add(key)
            self._order.append(key)
            while len(self._order) > self.max_keys:
                self._keys.discard(self._order.pop(0))

    def contains(self, key: str) -> bool:
        return key in self._keys

    def accept(self, envelope: dict[str, Any], *, accepted_at: str | None = None) -> bool:
        """Durably append the record. Returns False when already accepted."""
        key = idempotency_key(envelope)
        if key in self._keys:
            return False
        record = {
            "idempotency_key": key,
            "record_id": envelope.get("record_id"),
            "kind": envelope.get("kind"),
            "schema_digest": (envelope.get("schema") or {}).get("digest"),
            "accepted_at": accepted_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "envelope": envelope,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._remember(key)
        return True

    def count(self) -> int:
        return len(self._keys)


def locate_bridge_binary() -> Path | None:
    override = os.environ.get("JEV_MQ_BRIDGE_BIN")
    if override:
        candidate = Path(override)
        return candidate if candidate.exists() else None
    root = Path(__file__).resolve().parents[1] / "mq"
    for profile in ("release", "debug"):
        candidate = root / "target" / profile / "jev-mq-bridge"
        if candidate.exists():
            return candidate
    found = shutil.which("jev-mq-bridge")
    return Path(found) if found else None


def render_bridge_config(config: JevConfig, *, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render the Rust bridge JSON config from the validated YAML contract."""
    mq = config.mq
    if mq is None or not mq.enabled:
        raise ValueError("config.mq must be enabled to render a bridge config")
    allowed_models: list[str] = []
    if config.tool_task is not None:
        for item in config.tool_task.allowed_models:
            allowed_models.append(item.model_id)
            allowed_models.extend(item.aliases)
    rendered: dict[str, Any] = {
        "adapter": mq.adapter,
        "endpoints": list(mq.endpoints),
        "stream": mq.stream,
        "subject": mq.subject,
        "durable_name": mq.durable_name,
        "consumer_group": mq.consumer_group,
        "start_position": mq.start_position,
        "start_offset": mq.start_offset,
        "start_timestamp": mq.start_timestamp,
        "batch": mq.batch,
        "max_bytes": mq.max_bytes,
        "ack_wait_seconds": mq.ack_wait_seconds,
        "max_ack_pending": mq.max_ack_pending,
        "prefetch": mq.prefetch,
        "max_deliver": mq.max_deliver,
        "dlq_subject": mq.dlq_subject,
        "dlq_path": mq.dlq_path,
        "quarantine_path": mq.quarantine_path,
        "wal_path": mq.wal_path,
        "socket_path": mq.socket_path,
        "dedup_window": mq.dedup_window,
        "loop": bool(mq.loop),
        "max_cycles": int(mq.max_cycles),
        "tls_required": mq.tls_required,
        "ca_cert": mq.ca_cert,
        "client_cert": mq.client_cert,
        "client_key": mq.client_key,
        "secret_ref": mq.secret_ref,
        "schema_id": config.schema.name,
        "schema_version": config.schema.version,
        "schema_digest": config.schema.digest(),
        "allowed_models": allowed_models,
        "metrics_path": mq.metrics_path,
        "source_path": mq.source_path,
        "state_path": mq.state_path,
        "exit_after_drain": bool(mq.exit_after_drain),
    }
    if overrides:
        rendered.update(overrides)
    return rendered


@dataclass
class IngestMetrics:
    received: int = 0
    accepted: int = 0
    duplicates: int = 0
    quarantined: int = 0
    dlq_rejected: int = 0
    tool_task_pairs: int = 0
    documents: int = 0
    labelled_examples: int = 0
    bridge: dict[str, Any] = field(default_factory=dict)
    ack_latency_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "quarantined": self.quarantined,
            "dlq_rejected": self.dlq_rejected,
            "tool_task_pairs": self.tool_task_pairs,
            "documents": self.documents,
            "labelled_examples": self.labelled_examples,
            "bridge": self.bridge,
            "ack_latency_ms": self.ack_latency_ms,
        }


class MqIngestor:
    """Consume bridge records, accept durably, then grant credits/acks."""

    def __init__(
        self,
        config: JevConfig,
        *,
        replay_path: str | Path | None = None,
        labelled_handler: Callable[[dict[str, Any]], None] | None = None,
        tool_task_handler: Callable[[dict[str, Any]], None] | None = None,
        document_handler: Callable[[dict[str, Any]], None] | None = None,
        bridge_binary: str | Path | None = None,
        poll_seconds: float = 0.25,
        bridge_config_overrides: dict[str, Any] | None = None,
    ):
        mq = config.mq
        if mq is None:
            raise ValueError("config.mq is required for MQ ingestion")
        self.config = config
        self.mq = mq
        self.bridge_config_overrides = dict(bridge_config_overrides or {})
        self.replay = ReplayBuffer(replay_path or mq.replay_buffer)
        self.metrics = IngestMetrics()
        self.handlers = {
            "labelled_example": labelled_handler,
            "tool_task_pair": tool_task_handler,
            "raw_document": document_handler,
        }
        self.bridge_binary = Path(bridge_binary) if bridge_binary else locate_bridge_binary()
        self.poll_seconds = poll_seconds
        self._paused = threading.Event()
        self._stop = threading.Event()
        self._bridge: subprocess.Popen[str] | None = None

    def _write_bridge_config(self) -> Path:
        path = Path(self.mq.bridge_config)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = render_bridge_config(self.config, overrides=self.bridge_config_overrides or None)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return path

    def start_bridge(self) -> subprocess.Popen[str]:
        if self.bridge_binary is None or not self.bridge_binary.exists():
            raise FileNotFoundError(
                "jev-mq-bridge binary not found; run `cargo build --locked` in mq/ or set JEV_MQ_BRIDGE_BIN"
            )
        config_path = self._write_bridge_config()
        self._bridge = subprocess.Popen(
            [str(self.bridge_binary), "run", "--config", str(config_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if self._bridge.poll() is not None:
                stderr = self._bridge.stderr.read() if self._bridge.stderr else ""
                raise RuntimeError(f"bridge exited early: {stderr.strip()}")
            if Path(self.mq.socket_path).exists():
                return self._bridge
            time.sleep(0.05)
        raise TimeoutError("bridge did not create its socket in time")

    def stop_bridge(self) -> None:
        if self._bridge is None:
            return
        if self._bridge.poll() is None and self.mq.exit_after_drain:
            try:
                self._bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self._bridge.poll() is None:
            self._bridge.terminate()
            try:
                self._bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._bridge.kill()
                self._bridge.wait(timeout=5)
        self._bridge = None

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> None:
        self._stop.set()

    def _handle(self, envelope: dict[str, Any]) -> None:
        kind = envelope.get("kind")
        if kind == "labelled_example":
            self.metrics.labelled_examples += 1
        elif kind == "tool_task_pair":
            self.metrics.tool_task_pairs += 1
        elif kind == "raw_document":
            self.metrics.documents += 1
        handler = self.handlers.get(str(kind))
        if handler is not None:
            handler(envelope)

    def _accept_record(self, message: dict[str, Any], client: BridgeClient) -> None:
        """Validate, durably accept, then ack one delivered record frame."""
        self.metrics.received += 1
        received_at = time.perf_counter()
        delivery_id = str(message.get("delivery_id"))
        envelope = message.get("envelope")
        if not isinstance(envelope, dict):
            client.credit(acks=[delivery_id], quarantine=[{"delivery_id": delivery_id, "reason": "client_invalid_envelope"}])
            self.metrics.quarantined += 1
            return
        key = idempotency_key(envelope)
        if self.replay.contains(key):
            self.metrics.duplicates += 1
            client.credit(acks=[delivery_id], count=1)
            return
        self.replay.accept(envelope)
        self.metrics.accepted += 1
        self._handle(envelope)
        client.credit(acks=[delivery_id], count=1)
        self.metrics.ack_latency_ms.append(round((time.perf_counter() - received_at) * 1000.0, 3))

    def _settle_inflight(self, client: BridgeClient, *, quiet_s: float = 0.5) -> None:
        """Pause new pulls, then accept already-sent frames so drain completes."""
        try:
            client.pause()
        except OSError:
            return
        deadline = time.monotonic() + quiet_s
        while time.monotonic() < deadline:
            message = client.receive(timeout_s=0.1)
            if message is None:
                continue
            if message.get("type") == "record":
                self._accept_record(message, client)
                deadline = time.monotonic() + quiet_s

    def run(
        self,
        *,
        manage_bridge: bool = True,
        max_records: int | None = None,
        idle_timeout_s: float | None = None,
        duration_s: float | None = None,
    ) -> dict[str, Any]:
        if manage_bridge:
            self.start_bridge()
        credits = min(self.mq.prefetch, self.mq.max_ack_pending)
        client = BridgeClient(self.mq.socket_path)
        idle_started = time.monotonic()
        started = time.monotonic()
        duration_reached = False
        try:
            client.connect(credits=credits)
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(self.poll_seconds)
                    continue
                if duration_s is not None and time.monotonic() - started >= duration_s:
                    duration_reached = True
                    break
                message = client.receive(timeout_s=self.poll_seconds)
                if message is None:
                    if idle_timeout_s is not None and time.monotonic() - idle_started >= idle_timeout_s:
                        break
                    continue
                if message.get("type") != "record":
                    continue
                idle_started = time.monotonic()
                self._accept_record(message, client)
                if max_records is not None and self.metrics.accepted >= max_records:
                    break
            if duration_reached:
                self._settle_inflight(client)
            self.metrics.bridge = client.status()
            client.drain()
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                drained = client.receive(timeout_s=1.0)
                if drained and drained.get("type") == "drained":
                    break
            client.goodbye()
        finally:
            client.close()
            if manage_bridge:
                self.stop_bridge()
        return self.metrics.as_dict()
