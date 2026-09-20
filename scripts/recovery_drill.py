#!/usr/bin/env python3
"""Run a real GPU collapse-and-recovery drill for the Stage 0 drift guard.

A deliberately inflated loss window trips the configured loss guard, the
trainer archives the failing adapter and resets to a fresh zeroed LoRA, the old
serving snapshot keeps answering requests through warm-up, and warm-up
completion advances the published model generation. No teacher API key, broker,
or external service is required.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_reset_events(checkpoint_dir: Path) -> list[dict[str, Any]]:
    events_path = checkpoint_dir / "reset-events.jsonl"
    if not events_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def archived_files(checkpoint_dir: Path) -> list[dict[str, Any]]:
    archives = sorted((checkpoint_dir / "archives").glob("reset-*.pt"))
    return [{"path": str(path), "name": path.name, "bytes": path.stat().st_size} for path in archives]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--output", default="artifacts/recovery/collapse_recovery.json")
    parser.add_argument("--checkpoint-dir", default="/tmp/jev-recovery-drill")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)

    import torch

    from jev.config import load_config
    from jev.model import apply_lora, load_gliner
    from jev.runtime import JevRuntime
    from jev.training import ContinuousLoRATrainer

    if not torch.cuda.is_available():
        print("CUDA is required for the collapse-and-recovery drill", file=sys.stderr)
        return 2
    model_path = args.model_path or os.environ.get("JEV_MODEL_PATH")
    if not model_path:
        print("provide --model-path or JEV_MODEL_PATH for the real GPU drill", file=sys.stderr)
        return 2

    config = load_config(args.config)
    config = replace(
        config,
        drift=replace(config.drift, eval_window=2, min_windows=2, reset_cooldown_steps=1),
        runtime=replace(
            config.runtime,
            device="cuda",
            model_path=model_path,
            reset_warmup_steps=2,
            checkpoint_every_steps=1,
            publish_every_steps=1,
        ),
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    reset_fired = False
    reset_id = 0
    warmup_required = False
    warmup_completed = False
    serving_continuity = {
        "output_present": False,
        "generation_before_reset": None,
        "generation_during_warmup": None,
        "generation_after_warmup": None,
        "reset_id_during_warmup": None,
        "reset_id_after_warmup": None,
        "held": False,
    }
    setup_seconds = None
    load_seconds: list[float] = []
    train_seconds = 0.0
    phases: dict[str, dict[str, Any]] = {}
    events: list[Any] = []
    trainer = None
    runtime = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    def model_factory() -> Any:
        begin = time.perf_counter()
        model = apply_lora(load_gliner(config), config)
        load_seconds.append(time.perf_counter() - begin)
        return model

    calls = {"n": 0}

    def step_fn(live_model: Any, _example: dict[str, Any]) -> Any:
        calls["n"] += 1
        loss = torch.ones((), device=next(live_model.parameters()).device)
        for parameter in live_model.parameters():
            if parameter.requires_grad:
                loss = loss + parameter.float().pow(2).mean()
        if 3 < calls["n"] <= 6:
            loss = loss * 1000.0
        return loss

    started = time.perf_counter()
    try:
        runtime = JevRuntime(config)
        setup_begin = time.perf_counter()
        trainer = ContinuousLoRATrainer(
            config,
            runtime,
            step_fn,
            model_factory=model_factory,
            checkpoint_dir=checkpoint_dir,
        )
        setup_seconds = time.perf_counter() - setup_begin

        text = "2026 best tech stack includes Python and PostgreSQL"

        def train_phase(name: str, steps: int) -> dict[str, Any]:
            begin = time.perf_counter()
            phase_events = trainer.train([{} for _ in range(steps)], max_steps=steps)
            events.extend(phase_events)
            return {
                "steps": steps,
                "seconds": time.perf_counter() - begin,
                "losses": [event.loss for event in phase_events],
                "reset_required": [event.reset_required for event in phase_events],
                "reset_reason": next((event.reset_reason for event in phase_events if event.reset_required), None),
                "model_generation": runtime.stats.model_generation,
            }

        phases["normal_baseline"] = train_phase("normal_baseline", 3)
        phases["inflated_lead_in"] = train_phase("inflated_lead_in", 2)
        before = runtime.infer_result(text)
        serving_continuity["generation_before_reset"] = before["generation"]
        known_generation = before["generation"]

        phases["collapse"] = train_phase("collapse", 1)
        reset_fired = any(event.reset_required for event in events)
        reset_id = runtime.stats.reset_id
        warmup_required = bool(trainer._warmup_pending)
        during = runtime.infer_result(text)
        serving_continuity["generation_during_warmup"] = during["generation"]
        serving_continuity["reset_id_during_warmup"] = during["reset_id"]
        serving_continuity["output_present"] = during["output"] is not None
        serving_continuity["held"] = (
            during["generation"] == known_generation
            and during["output"] is not None
            and runtime.stats.model_generation == known_generation
        )

        phases["warmup"] = train_phase("warmup", 4)
        warmup_completed = not trainer._warmup_pending
        after = runtime.infer_result(text)
        serving_continuity["generation_after_warmup"] = after["generation"]
        serving_continuity["reset_id_after_warmup"] = after["reset_id"]
        train_seconds = sum(phase["seconds"] for phase in phases.values())
    except Exception as exc:  # keep the drill artifact even on hardware failure
        errors.append(repr(exc))

    reset_events = read_reset_events(checkpoint_dir)
    archives = archived_files(checkpoint_dir)
    archived_reasons = []
    for record in reset_events:
        archive = record.get("archive")
        archived_reasons.append(
            {
                "reset_id": record.get("reset_id"),
                "reason": record.get("reason"),
                "step": record.get("step"),
                "warmup_steps": record.get("warmup_steps"),
                "archive": archive,
                "archive_exists": bool(archive) and Path(str(archive)).exists(),
            }
        )
    checks = {
        "reset_required_fired": reset_fired,
        "reset_id_incremented": reset_id == 1,
        "archive_written": len(archives) >= 1,
        "reset_event_logged": len(reset_events) >= 1,
        "reset_event_archive_exists": bool(archived_reasons) and all(item["archive_exists"] for item in archived_reasons),
        "serving_continuity_held": serving_continuity["held"],
        "warmup_required": warmup_required,
        "warmup_completed": warmup_completed
        and isinstance(serving_continuity["generation_after_warmup"], int)
        and isinstance(serving_continuity["generation_before_reset"], int)
        and serving_continuity["generation_after_warmup"] > serving_continuity["generation_before_reset"],
        "no_errors": not errors,
    }
    payload = {
        "pass": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "config": {
            "path": str(args.config),
            "digest": config.digest(),
            "schema_digest": config.schema.digest(),
            "drift": vars(config.drift),
            "checkpoint_every_steps": config.runtime.checkpoint_every_steps,
            "publish_every_steps": config.runtime.publish_every_steps,
            "reset_warmup_steps": config.runtime.reset_warmup_steps,
        },
        "model_path": str(model_path),
        "checkpoint_dir": str(checkpoint_dir),
        "reset": {
            "fired": reset_fired,
            "reset_id": reset_id,
            "events": archived_reasons,
            "archives": archives,
        },
        "serving_continuity": serving_continuity,
        "warmup": {
            "required": warmup_required,
            "completed": warmup_completed,
            "model_generation": runtime.stats.model_generation if runtime else None,
        },
        "phases": phases,
        "events": [vars(event) for event in events],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": ".".join(str(part) for part in torch.cuda.get_device_capability(0)),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
        "timings": {
            "setup_seconds": setup_seconds,
            "model_load_seconds": load_seconds,
            "training_seconds": train_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("pass", "checks", "reset", "serving_continuity", "warmup")}, ensure_ascii=False, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
