#!/usr/bin/env python3
"""Bounded proof that the train/infer/update service can run continuously.

The MQ mock source replays a labelled dataset head-to-tail forever (each cycle
gets a distinct loop_cycle and content hash). While the trainer consumes the
endless stream, a serving thread keeps reading immutable EMA snapshots, so the
run exercises the full loop: infinite ingestion -> LoRA update -> EMA publish ->
live inference. The run is intentionally time-bounded for evidence; remove
``--duration`` for an unbounded soak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.config import JevConfig, load_config
from jev.mq import MqIngestor, build_envelope, sha256_hex, training_example_from_envelope
from jev.runtime import JevRuntime
from jev.status import write_status
from jev.tool_task import classify_tool_task
from jev.training import ContinuousLoRAStream, ContinuousLoRATrainer

TRAINING_ONLY_KEYS = ("provenance", "entity_descriptions")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def training_output(record: dict) -> dict:
    return {key: value for key, value in record["output"].items() if key not in TRAINING_ONLY_KEYS}


def build_records(config: JevConfig, dataset_path: Path, records_path: Path) -> int:
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            text = record["input"]
            output = training_output(record)
            envelope = build_envelope(
                record_id="soak-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                kind="labelled_example",
                source={
                    "uri": f"soak://{sha256_hex(text)[:16]}",
                    "content_sha256": sha256_hex(text),
                    "retrieved_at": "2026-09-21T00:00:00Z",
                    "license": "project-authored",
                },
                schema_id=config.schema.name,
                schema_version=config.schema.version,
                schema_digest=config.schema.digest(),
                labels=output,
                text=text,
            )
            handle.write(json.dumps(envelope, ensure_ascii=False) + "\n")
    return len(records)


def _submit_training_example(stream: ContinuousLoRAStream, envelope: dict) -> None:
    example = training_example_from_envelope(envelope)
    if example is not None:
        stream.submit(example)


def serving_loop(
    runtime: JevRuntime,
    config: JevConfig,
    probes: list[str],
    interval: float,
    stop: threading.Event,
    observations: list,
    errors: list,
) -> None:
    index = 0
    started = time.perf_counter()
    while not stop.is_set():
        try:
            begin = time.perf_counter()
            text = probes[index % len(probes)]
            extraction = runtime.infer_result(text)
            snapshot = runtime.snapshot_model()
            classification = (
                classify_tool_task(snapshot, text, config.tool_task)
                if snapshot is not None and config.tool_task is not None
                else {}
            )
            observations.append(
                {
                    "at_seconds": round(time.perf_counter() - started, 3),
                    "seconds": time.perf_counter() - begin,
                    "generation": extraction["generation"],
                    "ema_step": extraction["ema_step"],
                    "task_type": (classification.get("task_type") or {}).get("value"),
                }
            )
        except Exception as exc:  # pragma: no cover - GPU-only failure path
            errors.append(repr(exc))
        index += 1
        stop.wait(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--dataset", default="artifacts/datasets/stage0_distilled.jsonl")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--duration", type=float, default=60.0, help="0 = unbounded")
    parser.add_argument("--probe-interval", type=float, default=2.0)
    parser.add_argument("--checkpoint-dir", default="runs/jev/soak")
    parser.add_argument("--output", default="artifacts/soak/infinite_soak.json")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("provide --model-path or JEV_MODEL_PATH")
    os.environ["JEV_MODEL_PATH"] = args.model_path
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the soak")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    work_dir = Path(args.checkpoint_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    records_path = work_dir / "records.jsonl"
    config = load_config(args.config)
    record_count = build_records(config, Path(args.dataset), records_path)
    mq = replace(
        config.mq,
        enabled=True,
        adapter="mock",
        source_path=str(records_path),
        state_path=str(work_dir / "state.json"),
        wal_path=str(work_dir / "wal.bin"),
        socket_path=str(work_dir / "bridge.sock"),
        bridge_config=str(work_dir / "bridge.json"),
        dlq_path=str(work_dir / "dlq.jsonl"),
        quarantine_path=str(work_dir / "quarantine.jsonl"),
        metrics_path="artifacts/mq/soak_bridge_metrics.json",
        replay_buffer=str(work_dir / "replay-buffer.jsonl"),
        loop=True,
        max_cycles=0,
        batch=4,
        max_ack_pending=4,
        prefetch=4,
        max_deliver=3,
        exit_after_drain=True,
    )
    config = replace(
        config,
        mq=mq,
        runtime=replace(
            config.runtime,
            device="cuda",
            publish_every_steps=1,
            checkpoint_every_steps=5,
            max_checkpoints=2,
            ema_decay=0.9,
        ),
    )

    started = time.perf_counter()
    runtime = JevRuntime(config)
    trainer = ContinuousLoRATrainer(config, runtime, checkpoint_dir=args.checkpoint_dir)
    stream = ContinuousLoRAStream(trainer, max_queue_size=64).start()
    ingestor = MqIngestor(
        config,
        labelled_handler=lambda envelope: _submit_training_example(stream, envelope),
    )
    write_status({"phase": "serving", "note": "infinite soak starting", "model_id": config.model_id, "device": "cuda"})

    probes = [
        "The 2026 stack uses Python and PostgreSQL.",
        "Deploy the Rust bridge with Kubernetes and Docker.",
        "Summarize the LoRA benchmark results.",
    ]
    observations: list[dict] = []
    serving_errors: list[str] = []
    stop = threading.Event()
    thread = threading.Thread(
        target=serving_loop,
        args=(runtime, config, probes, args.probe_interval, stop, observations, serving_errors),
        daemon=True,
    )
    thread.start()
    ingest_error: str | None = None
    try:
        ingest = ingestor.run(manage_bridge=True, duration_s=(args.duration if args.duration > 0 else None))
    except Exception as exc:  # pragma: no cover - runtime failure path
        ingest_error = repr(exc)
        ingest = {"accepted": 0, "duplicates": 0, "bridge": {}}
    stop.set()
    thread.join(timeout=30)
    training_events = stream.stop(timeout=120)
    checkpoint = trainer.save_checkpoint()
    write_status({"phase": "idle", "note": "infinite soak finished", "model_id": config.model_id, "device": "cuda"})

    generations = sorted({item["generation"] for item in observations})
    latencies = sorted(item["seconds"] for item in observations)
    bridge = ingest.get("bridge") or {}
    evidence = {
        "pass": (
            ingest_error is None
            and not serving_errors
            and ingest.get("accepted", 0) >= record_count * 2
            and int(bridge.get("loop_cycles", 0)) >= 1
            and len(training_events) >= 2
            and len(generations) >= 2
        ),
        "mode": "infinite_loop" if args.duration <= 0 else "bounded_loop",
        "duration_seconds": args.duration,
        "dataset_records": record_count,
        "accepted": ingest.get("accepted"),
        "duplicates": ingest.get("duplicates"),
        "loop_cycles": bridge.get("loop_cycles"),
        "bridge": {
            "pulled": bridge.get("pulled"),
            "delivered": bridge.get("delivered"),
            "acked": bridge.get("acked"),
            "duplicates_suppressed": bridge.get("duplicates_suppressed"),
            "dlq": bridge.get("dlq"),
            "quarantined": bridge.get("quarantined"),
        },
        "training_steps": runtime.stats.training_steps,
        "training_events": len(training_events),
        "ema_updates": runtime.ema.updates,
        "model_generation": runtime.stats.model_generation,
        "losses_tail": [event.loss for event in training_events[-8:]],
        "serving_calls": len(observations),
        "serving_generations": generations,
        "serving_errors": serving_errors,
        "serving_p50_seconds": latencies[len(latencies) // 2] if latencies else None,
        "serving_p95_seconds": latencies[min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))] if latencies else None,
        "checkpoint": str(checkpoint),
        "peak_vram_bytes": max(torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()),
        "total_seconds": time.perf_counter() - started,
        "ingest_error": ingest_error,
        "note": "mock loop source; no live broker. Remove --duration for an unbounded soak.",
    }
    atomic_json(Path(args.output), evidence)
    print(json.dumps(evidence, ensure_ascii=False, default=str))
    return 0 if evidence["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
