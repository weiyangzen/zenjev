#!/usr/bin/env python3
"""Perpetual ZenJev loop (§1.6, §1.8).

One process owns the single training writer and the only bridge client:

* starts the Rust ``dir-spool`` bridge over ``runs/jev/feed/*.ndjson``;
* consumes envelopes with credit-based backpressure from ``MqIngestor``;
* extracts every real request with the accepted EMA snapshot, routes the
  deterministic decision, and admits high-confidence self-labelled examples to
  the LoRA trainer (RSI self-training);
* publishes immutable EMA snapshots and appends a monotonic generation ledger;
* serves inference and health over a loopback HTTP port;
* writes heartbeats and a bounded live event feed for the console.

It never exits on an idle queue. A dead bridge, dead trainer thread, or an
unrecoverable stall exits nonzero so the supervisor restarts the process, which
resumes from the last checkpoint and the bridge watermark.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from zenjev_lib import (
    ARTIFACTS,
    EVENTS_KEEP_LINES,
    EVENTS_MAX_BYTES,
    EVENTS_PATH,
    FEED_DIR,
    PERPETUAL_RUNS,
    append_jsonl,
    atomic_write_json,
    ensure_dirs,
    percentile,
    read_json,
    rotate_jsonl,
    tail_jsonl,
    sha256_hex,
    utc_now,
    write_heartbeat,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/zenjev_perpetual.yaml"
CHECKPOINT_DIR = PERPETUAL_RUNS / "checkpoints"
LEDGER_PATH = PERPETUAL_RUNS / "generations.jsonl"
INFERENCE_PORT = 8788



JUDGE_CHARS = 1200
# One inference pass turns a raw capture into a supervised pair: the instructed
# request is the input and the model-generated schema is the target. Raw
# request/response text is never trained on directly.
INSTRUCTION = (
    "Extract the schema for this engineering request: task_type, decision_choice, "
    "task_detail, tool, programming_language, framework, technology, model. "
    "Return only schema fields grounded in the request."
)
INSTRUCTION_SHA16 = __import__("hashlib").sha256(INSTRUCTION.encode("utf-8")).hexdigest()[:16]
PAIRS_PATH = PERPETUAL_RUNS / "pairs.jsonl"
PAIRS_MAX_BYTES = 512 * 1024 * 1024


def _batched_loss(model: Any, config: Any, pairs: list[tuple[str, dict[str, Any]]]) -> Any:
    """One optimizer step over a micro-batch; per-example fallback on failure."""
    try:
        from gliner2.training import ExtractorCollator

        collator = ExtractorCollator(
            model.processor,
            is_training=True,
            max_len=config.runtime.inference_max_length,
            architecture="span",
            on_capacity_exceeded="raise",
        )
        result = model(collator(list(pairs)))
        if isinstance(result, dict) and result.get("total_loss") is not None:
            return result["total_loss"]
    except Exception:
        pass
    losses = [
        gliner2_step(model, config, {"input": text, "output": output})
        for text, output in pairs
    ]
    return sum(losses) / len(losses)


def _top_labels(distribution: dict, labels: list[str], *, k: int, threshold: float) -> list[str]:
    ranked = sorted(
        ((label, float(distribution.get(label, 0.0))) for label in labels),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [label for label, probability in ranked[:k] if probability >= threshold]
    return selected or [ranked[0][0]]


def _synthesize(config: Any, text: str, judgment: dict[str, Any]) -> dict[str, Any]:
    """Canonical judge -> instruction-style training record."""
    policy = config.tool_task
    task_probabilities = (judgment.get("task_type") or {}).get("probabilities", {}) or {}
    choice_probabilities = (judgment.get("decision_choice") or {}).get("probabilities", {}) or {}
    task_labels = _top_labels(task_probabilities, list(policy.task_types), k=1, threshold=0.0)
    choice_labels = _top_labels(
        choice_probabilities, list(policy.decision_choices), k=3, threshold=0.25
    )
    return {
        "input": text,
        "output": {
            "classifications": [
                {
                    "task": "task_type",
                    "labels": list(policy.task_types),
                    "true_label": task_labels,
                },
                {
                    "task": "decision_choice",
                    "labels": list(policy.decision_choices),
                    "true_label": choice_labels,
                },
            ]
        },
    }


def shape_pair_target(
    config: Any, judgment: dict[str, Any], extraction: dict[str, Any]
) -> dict[str, Any]:
    """Shape a model judgment + extraction into the GLiNER2 training target.

    The raw capture is never a training target: entities come from grounded
    extraction spans and classifications come from the judged probability
    distribution, both restricted to the configured schema.
    """
    entities: dict[str, list[str]] = {}
    raw_entities = extraction.get("entities") if isinstance(extraction.get("entities"), dict) else {}
    for name, values in raw_entities.items():
        if name not in config.schema.entities:
            continue
        texts: list[str] = []
        for item in values if isinstance(values, list) else [values]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        if texts:
            entities[name] = texts[:12]
    classifications: list[dict[str, Any]] = []
    schema_classifications = dict(getattr(config.schema, "classifications", {}) or {})
    for task, labels in schema_classifications.items():
        payload = judgment.get(task) or {}
        probabilities = payload.get("probabilities") or {}
        chosen: list[str] = []
        if probabilities:
            ranked = sorted(
                ((label, float(value)) for label, value in probabilities.items()),
                key=lambda item: (-item[1], item[0]),
            )
            top_k = 1 if task == "task_type" else 3
            threshold = 0.0 if task == "task_type" else 0.25
            chosen = [label for label, probability in ranked[:top_k] if probability >= threshold]
            if not chosen:
                chosen = [ranked[0][0]]
        elif isinstance(payload.get("label"), str):
            chosen = [payload["label"]]
        if chosen:
            classifications.append({"task": task, "labels": list(labels), "true_label": chosen})
    target: dict[str, Any] = {"entities": entities, "classifications": classifications}
    if config.schema.entity_descriptions:
        target["entity_descriptions"] = dict(config.schema.entity_descriptions)
    return target


class Runtime:
    def __init__(self, config: Any) -> None:
        from jev.model import gliner2_step
        from jev.mq import MqIngestor
        from jev.runtime import JevRuntime
        from jev.tool_task import resolve_tool_task
        from jev.training import ContinuousLoRATrainer, ContinuousLoRAStream

        self.config = config
        self.admission = os.environ.get("ZENJEV_FABRICATOR_ADMISSION", "inference_only")
        self.batch_size = max(1, int(os.environ.get("ZENJEV_TRAIN_BATCH", "4")))
        self.batch_buffer: list[tuple[str, dict[str, Any]]] = []
        if self.admission not in ("inference_only", "canary_scored", "training_admitted"):
            raise SystemExit("invalid ZENJEV_FABRICATOR_ADMISSION")
        self.resolve_tool_task = resolve_tool_task
        self.lock = threading.Lock()
        self.seq = 0
        self.counters: dict[str, int] = collections.Counter()
        self.latencies: collections.deque[float] = collections.deque(maxlen=500)
        self.losses: collections.deque[float] = collections.deque(maxlen=200)
        self.last_record_at = time.time()
        self.last_loss: float | None = None
        self.started = time.time()

        self.runtime = JevRuntime(config)
        resume = CHECKPOINT_DIR / "latest.pt"
        self.trainer = ContinuousLoRATrainer(
            config,
            self.runtime,
            checkpoint_dir=CHECKPOINT_DIR,
            resume_from=resume if resume.exists() else None,
            resume_config_drift=True,
        )
        # Generation ids are global and monotonic across restarts: resume the
        # counter from the durable ledger before anything publishes again.
        ledger_max = 0
        for row in tail_jsonl(LEDGER_PATH, 5000):
            try:
                ledger_max = max(ledger_max, int(row.get("generation_id", 0)))
            except (TypeError, ValueError):
                continue
        if ledger_max:
            self.runtime.stats.model_generation = ledger_max
            last = tail_jsonl(LEDGER_PATH, 1)
            if last:
                # Consumed-record counters are cumulative across restarts so the
                # console never appears to lose training data on a restart.
                self.counters["real_pairs"] = int(last[-1].get("records_consumed_total") or 0)
                self.counters["training_examples"] = int(last[-1].get("training_examples_total") or 0)
            self.runtime.publish(
                self.trainer.model,
                dict(self.runtime.ema.values),
                ema_step=self.runtime.ema.updates,
            )

        original_save = self.trainer.save_checkpoint

        def save_with_manifest(*args: Any, **kwargs: Any) -> Any:
            path = original_save(*args, **kwargs)
            if not kwargs.get("archive_reason"):
                latest = CHECKPOINT_DIR / "latest.pt"
                atomic_write_json(
                    CHECKPOINT_DIR / "latest.json",
                    {
                        "generation_id": int(self.runtime.stats.model_generation),
                        "training_steps": int(self.runtime.stats.training_steps),
                        "ema_step": int(self.runtime.ema.updates),
                        "reset_id": int(self.runtime.stats.reset_id),
                        "adapter_digest": sha256_hex(latest.read_bytes())
                        if latest.exists()
                        else None,
                        "checkpoint": str(latest),
                        "created_at": utc_now(),
                    },
                )
            return path

        self.trainer.save_checkpoint = save_with_manifest

        base_step = self.trainer.step_fn

        def step_with_loss(model: Any, example: dict[str, Any]) -> Any:
            if isinstance(example, dict) and "batch" in example:
                loss = _batched_loss(model, self.config, example["batch"])
            else:
                loss = base_step(model, example)
            try:
                self.last_loss = float(loss.detach().item())
            except Exception:
                pass
            return loss

        self.trainer.step_fn = step_with_loss
        self.stream = ContinuousLoRAStream(self.trainer, max_queue_size=64).start()
        self.ingestor = MqIngestor(config)
        self.ingestor.handlers["tool_task_pair"] = self.handle_tool_task
        self.ingestor.handlers["labelled_example"] = self.handle_labelled


    # ------------------------------------------------------------------ events
    def event(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.seq += 1
            body = dict(payload)
            body["seq"] = self.seq
            body["ts"] = utc_now()
        append_jsonl(EVENTS_PATH, body)

    # ---------------------------------------------------------------- handlers
    def handle_labelled(self, envelope: dict[str, Any]) -> None:
        example = envelope.get("example")
        if not isinstance(example, dict):
            labels = envelope.get("labels")
            text = envelope.get("text") or (envelope.get("source") or {}).get("uri")
            if isinstance(labels, dict) and isinstance(text, str):
                example = {"input": text, "output": labels}
        if not isinstance(example, dict) or not example.get("input"):
            self.counters["labelled_rejected"] += 1
            return
        self.stream.submit(example)
        self.counters["labelled_trained"] += 1

    def handle_tool_task(self, envelope: dict[str, Any]) -> None:
        status = envelope.get("response_status")
        if envelope.get("capture_error") or (isinstance(status, int) and status != 200):
            self.counters["skipped_bad_capture"] += 1
            return
        synthetic = bool(envelope.get("synthetic"))
        observed = envelope.get("observed_model") or {}
        model_id = str(observed.get("model_id") or observed.get("id") or "")
        request_text = str((envelope.get("request") or {}).get("text") or "")[:2000]
        response_text = str((envelope.get("response") or {}).get("text") or "")[:800]
        if not request_text:
            self.counters["missing_request"] += 1
            return
        self.last_record_at = time.time()
        self.counters["synthetic_pairs" if synthetic else "real_pairs"] += 1

        import torch

        started = time.perf_counter()
        try:
            result = self.runtime.infer_result(request_text)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.counters["inference_oom"] += 1
            self.event({"kind": "oom_backoff", "stage": "inference", "synthetic": synthetic})
            return
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.latencies.append(latency_ms)
        output = result.get("output")
        if not isinstance(output, dict):
            self.counters["non_dict_output"] += 1
            return

        decision: dict[str, Any] = {"action": "review", "reasons": ["missing_policy"]}
        if self.config.tool_task is not None and model_id:
            try:
                decision = self.resolve_tool_task(output, self.config.tool_task, model_id=model_id)
            except Exception as error:  # noqa: BLE001 - policy failures must not kill the loop
                self.counters["decision_error"] += 1
                decision = {"action": "review", "reasons": [f"policy_error:{type(error).__name__}"]}
        action = str(decision.get("action") or "review")
        self.counters[f"decision_{action}"] += 1

        paired = False
        if not synthetic or self.admission == "training_admitted":
            paired = self._distill_pair(output, request_text)

        self.event(
            {
                "kind": "task",
                "synthetic": synthetic,
                "canary": bool(envelope.get("canary")),
                "model": model_id,
                "task_type": decision.get("task_type"),
                "action": action,
                "reasons": decision.get("reasons", []),
                "choices": decision.get("decision_choices", [])[:4],
                "latency_ms": round(latency_ms, 2),
                "generation": result.get("generation"),
                "ema_step": result.get("ema_step"),
                "paired": paired,
                "instruction_sha16": INSTRUCTION_SHA16,
                "request_preview": request_text[:220],
                "response_preview": response_text[:160],
                "record_id": envelope.get("record_id"),
                "expected": envelope.get("expected"),
                "uri": (envelope.get("source") or {}).get("uri"),
            }
        )

    def _judge(self, text: str) -> dict[str, Any]:
        """Judge the text with the live LoRA model (canonical batch_classify)."""
        from gliner2.classification import (
            ClassificationConfig,
            ClassificationSchema,
            Classifier,
        )

        policy = self.config.tool_task
        task_schema = ClassificationSchema().single(
            "task_type", list(policy.task_types), threshold=policy.min_confidence
        )
        choice_schema = ClassificationSchema().multi(
            "decision_choice",
            list(policy.decision_choices),
            min_labels=0,
            max_labels=len(policy.decision_choices),
            threshold=policy.min_confidence,
        )
        classifier = Classifier(self.trainer.model)
        classification_config = ClassificationConfig(include_confidence=True)
        task_result = classifier.batch_classify(
            [text], task_schema, config=classification_config
        )[0]
        choice_result = classifier.batch_classify(
            [text], choice_schema, config=classification_config
        )[0]
        task_payload = task_result.to_dict() if hasattr(task_result, "to_dict") else task_result
        choice_payload = (
            choice_result.to_dict() if hasattr(choice_result, "to_dict") else choice_result
        )
        return {
            "task_type": task_payload.get("task_type", task_payload),
            "decision_choice": choice_payload.get("decision_choice", choice_payload),
        }

    def _distill_pair(self, output: dict[str, Any], request_text: str) -> bool:
        """Distil one instructed pair: (instruction + request) -> schema target.

        The raw capture is never trained on directly. The live LoRA model judges
        the instructed request, the EMA snapshot contributes grounded entity
        spans, and the two form a schema-shaped training pair that is persisted
        for offline distillation/RL before it enters the training stream.
        """
        if self.config.tool_task is None:
            return False
        instructed = f"{INSTRUCTION}\n\n{request_text}"[:1600]
        try:
            judgment = self._judge(instructed[:JUDGE_CHARS])
        except Exception:
            self.counters["judge_failed"] += 1
            return False

        target = shape_pair_target(self.config, judgment, output)
        if not target["entities"] and not target["classifications"]:
            self.counters["pair_empty"] += 1
            return False
        pair = {"input": instructed, "output": target}
        self.counters["pairs_built"] += 1
        self._append_pair(pair, request_text)

        self.batch_buffer.append((instructed, target))
        if len(self.batch_buffer) >= self.batch_size:
            pending = self.batch_buffer
            self.batch_buffer = []
            try:
                self.stream.submit({"batch": pending})
                self.counters["pairs_trained"] += len(pending)
            except RuntimeError:
                self.counters["trainer_closed"] += 1
                return False
        return True

    def _append_pair(self, pair: dict[str, Any], request_text: str) -> None:
        try:
            import os as _os

            if PAIRS_PATH.exists() and PAIRS_PATH.stat().st_size > PAIRS_MAX_BYTES:
                _os.replace(PAIRS_PATH, PAIRS_PATH.with_suffix(".1.jsonl"))
            record = {
                "created_at": utc_now(),
                "instruction_sha16": INSTRUCTION_SHA16,
                "input_chars": len(pair["input"]),
                "entities": {k: len(v) for k, v in pair["output"]["entities"].items()},
                "classifications": {c["task"]: c["true_label"] for c in pair["output"]["classifications"]},
                "generation": int(self.runtime.stats.model_generation),
                "pair": pair,
            }
            append_jsonl(PAIRS_PATH, record)
        except OSError:
            self.counters["pair_write_failed"] += 1

    # ------------------------------------------------------------------ ledger
    def ledger_loop(self) -> None:
        last_generation = 0
        parent: str | None = None
        while True:
            generation = int(self.runtime.stats.model_generation)
            if generation > last_generation:
                checkpoint = CHECKPOINT_DIR / "latest.pt"
                digest = sha256_hex(checkpoint.read_bytes()) if checkpoint.exists() else None
                events = read_json(CHECKPOINT_DIR / "metrics.json", default={}) or {}
                recent = events.get("events") or []
                loss = recent[-1].get("loss") if recent else self.last_loss
                record = {
                    "session_pid": os.getpid(),
                    "generation_id": generation,
                    "parent_digest": parent,
                    "adapter_digest": digest,
                    "ema_step": int(self.runtime.ema.updates),
                    "training_steps": int(self.runtime.stats.training_steps),
                    "reset_id": int(self.runtime.stats.reset_id),
                    "records_consumed_total": int(self.counters["real_pairs"]),
                    "training_examples_total": int(self.counters["training_examples"]),
                    "train_batch": self.batch_size,
                    "instruction_sha16": INSTRUCTION_SHA16,
                    "pairs_built": int(self.counters["pairs_built"]),
                    "queue_cursor_bytes": self._spool_committed(),
                    "queue_backlog_bytes": self._spool_backlog(),
                    "loss": loss,
                    "created_at": utc_now(),
                }
                append_jsonl(LEDGER_PATH, record)
                self.event(
                    {
                        "kind": "generation",
                        "generation_id": generation,
                        "adapter_digest": digest,
                        "training_steps": record["training_steps"],
                        "reset_id": record["reset_id"],
                        "loss": loss,
                    }
                )
                parent = digest or sha256_hex(json.dumps(record, sort_keys=True))
                last_generation = generation
            time.sleep(0.5)

    # --------------------------------------------------------------- heartbeat
    def metrics_loop(self) -> None:
        last_count = 0
        last_at = time.time()
        while True:
            now = time.time()
            with self.lock:
                snapshot = dict(self.counters)
                seq = self.seq
            rate = (snapshot.get("real_pairs", 0) - last_count) / max(0.5, now - last_at)
            last_count = snapshot.get("real_pairs", 0)
            last_at = now
            bridge_status: dict[str, Any] = {}
            bridge = getattr(self.ingestor, "_bridge", None)
            bridge_alive = bool(bridge is not None and bridge.poll() is None)
            live = self.ingestor.metrics.as_dict()
            bridge_status = {
                "received": live.get("received"),
                "accepted": live.get("accepted"),
                "delivered": live.get("accepted"),
                "acked": live.get("accepted"),
                "inflight": max(
                    0,
                    int(live.get("received") or 0) - int(live.get("accepted") or 0),
                ),
                "duplicates_suppressed": live.get("duplicates"),
                "quarantined": live.get("quarantined"),
                "dlq": live.get("dlq_rejected"),
                "consumer_lag": self._spool_backlog(),
                "offset_checkpoint": self._spool_committed(),
                "last_error_class": None,
            }
            mq_metrics = read_json(REPO / "artifacts/mq/metrics.json", default={}) or {}
            if isinstance(mq_metrics, dict) and mq_metrics:
                bridge_status["last_error_class"] = mq_metrics.get("last_error_class")
                if mq_metrics.get("consumer_lag") is not None:
                    bridge_status["consumer_lag"] = mq_metrics.get("consumer_lag")
            try:
                import torch

                vram = {
                    "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
                    "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
                }
            except Exception:
                vram = {}
            write_heartbeat(
                "loop",
                {
                    "state": "running" if bridge_alive else "degraded",
                    "uptime_seconds": round(now - self.started, 1),
                    "bridge_alive": bridge_alive,
                    "bridge": bridge_status,
                    "counters": snapshot,
                    "events": seq,
                    "train_batch": self.batch_size,
                    "instruction_sha16": INSTRUCTION_SHA16,
                    "pairs_built": int(self.counters["pairs_built"]),
                    "queue_cursor_bytes": self._spool_committed(),
                    "queue_backlog_bytes": self._spool_backlog(),
                    "records_per_minute": round(rate * 60.0, 2),
                    "seconds_since_last_record": round(now - self.last_record_at, 1),
                    "generation": int(self.runtime.stats.model_generation),
                    "training_steps": int(self.runtime.stats.training_steps),
                    "ema_step": int(self.runtime.ema.updates),
                    "reset_id": int(self.runtime.stats.reset_id),
                    "loss": self.last_loss,
                    "latency_p50_ms": percentile(self.latencies, 0.5),
                    "latency_p95_ms": percentile(self.latencies, 0.95),
                    "vram": vram,
                    "trainer_running": bool(self.stream.running),
                },
            )
            atomic_write_json(
                REPO / "runs/jev/status.json",
                {
                    "schema_version": 1,
                    "updated_at_unix_ms": int(now * 1000),
                    "pid": os.getpid(),
                    "phase": "mq-rsi-train-serve",
                    "generation": int(self.runtime.stats.model_generation),
                    "ema_step": int(self.runtime.ema.updates),
                    "training_steps": int(self.runtime.stats.training_steps),
                    "reset_id": int(self.runtime.stats.reset_id),
                    "loss": self.last_loss,
                    "learning_rate": self.config.runtime.learning_rate,
                    "checkpoint": str(CHECKPOINT_DIR / "latest.pt"),
                    "config_digest": self.config.digest(),
                    "model_id": self.config.model_id,
                    "device": self.config.runtime.device,
                    "note": "zenjev perpetual MQ+RSI loop",
                },
            )
            rotate_jsonl(EVENTS_PATH, max_bytes=EVENTS_MAX_BYTES, keep_lines=EVENTS_KEEP_LINES)
            time.sleep(2.0)

    def _spool_state(self) -> dict[str, Any]:
        state = read_json(PERPETUAL_RUNS / "dir-spool-state.json", default={}) or {}
        files = state.get("files") if isinstance(state, dict) else None
        return files if isinstance(files, dict) else {}

    def _spool_backlog(self) -> int:
        backlog = 0
        for rel, cursor in self._spool_state().items():
            if not isinstance(cursor, dict):
                continue
            offset = int(cursor.get("offset", 0))
            try:
                size = (FEED_DIR / rel).stat().st_size
            except OSError:
                size = int(cursor.get("size", 0))
            backlog += max(0, size - offset)
        return int(backlog)

    def _spool_committed(self) -> int:
        return int(
            sum(
                int(cursor.get("offset", 0))
                for cursor in self._spool_state().values()
                if isinstance(cursor, dict)
            )
        )

    def watchdog_loop(self) -> None:
        stall_limit = 3600.0
        while True:
            time.sleep(10.0)
            bridge = getattr(self.ingestor, "_bridge", None)
            if bridge is not None and bridge.poll() is not None:
                print("watchdog: bridge process exited", file=sys.stderr)
                os._exit(11)
            if not self.stream.running:
                print("watchdog: trainer thread stopped", file=sys.stderr)
                os._exit(12)
            if time.time() - self.last_record_at > stall_limit:
                print("watchdog: no accepted records for too long", file=sys.stderr)
                os._exit(13)

    # ------------------------------------------------------------------- serve
    def serve_loop(self) -> None:
        parent = self

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
                    self._send(
                        200,
                        {
                            "state": "ok",
                            "generation": int(parent.runtime.stats.model_generation),
                            "training_steps": int(parent.runtime.stats.training_steps),
                            "counters": dict(parent.counters),
                            "uptime_seconds": round(time.time() - parent.started, 1),
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
                if self.path == "/extract":
                    result = parent.runtime.infer_result(text)
                    self._send(200, result)
                elif self.path == "/analyze":
                    model_id = str(payload.get("model_id") or "gpt-5.6-sol")
                    result = parent.runtime.infer_result(text)
                    output = result.get("output") if isinstance(result.get("output"), dict) else {}
                    try:
                        decision = parent.resolve_tool_task(output, parent.config.tool_task, model_id=model_id)
                    except Exception as error:  # noqa: BLE001
                        decision = {"action": "reject", "reasons": [f"{type(error).__name__}:{error}"]}
                    self._send(200, {**result, "decision": decision, "model_id": model_id})
                else:
                    self._send(404, {"error": "not_found"})

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        server = ThreadingHTTPServer(("127.0.0.1", INFERENCE_PORT), Handler)
        server.serve_forever()

    # -------------------------------------------------------------------- main
    def run(self) -> int:
        self.event({"kind": "boot", "generation": int(self.runtime.stats.model_generation)})
        threading.Thread(target=self.ledger_loop, name="ledger", daemon=True).start()
        threading.Thread(target=self.metrics_loop, name="metrics", daemon=True).start()
        threading.Thread(target=self.watchdog_loop, name="watchdog", daemon=True).start()
        threading.Thread(target=self.serve_loop, name="serve", daemon=True).start()

        def watch_loss() -> None:
            while True:
                events = read_json(CHECKPOINT_DIR / "metrics.json", default={}) or {}
                recent = events.get("events") or []
                if recent:
                    self.last_loss = recent[-1].get("loss")
                time.sleep(2.0)

        threading.Thread(target=watch_loss, name="loss", daemon=True).start()
        self.ingestor.run(manage_bridge=True)
        return 0


def main() -> int:
    os.environ.setdefault("JEV_MODEL_PATH", str(pathlib.Path.home() / "jev-model-base"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--idle-timeout", type=float, default=None)
    args = parser.parse_args()
    ensure_dirs()
    from jev.config import load_config

    config = load_config(CONFIG_PATH)
    runtime = Runtime(config)
    if args.max_records or args.idle_timeout:
        runtime.ingestor.run(
            manage_bridge=True,
            max_records=args.max_records,
            idle_timeout_s=args.idle_timeout,
        )
        return 0
    return runtime.run()


if __name__ == "__main__":
    raise SystemExit(main())
