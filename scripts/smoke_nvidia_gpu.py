#!/usr/bin/env python3
"""Run a real GLiNER2/LoRA/EMA snapshot smoke on an NVIDIA GPU.

This is an integration gate, not a quality-training recipe: the step function
uses an adapter regularizer so the optimizer, EMA, publication and concurrent
inference path can be exercised without a teacher dataset or API key.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="artifacts/nvidia_gpu_train_ema_smoke.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--inferences", type=int, default=4)
    args = parser.parse_args()

    import torch

    from jev.config import load_config
    from jev.model import apply_lora, load_gliner
    from jev.runtime import JevRuntime
    from jev.training import ContinuousLoRATrainer

    config = load_config(args.config)
    model_path = args.model_path or os.environ.get("JEV_MODEL_PATH") or config.runtime.model_path
    if not model_path:
        raise SystemExit("provide --model-path or JEV_MODEL_PATH for an offline smoke")
    config = replace(config, runtime=replace(config.runtime, device="cuda", model_path=model_path, publish_every_steps=1))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the NVIDIA GPU gate")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = apply_lora(load_gliner(config), config)
    base_load_seconds = time.perf_counter() - started
    runtime = JevRuntime(config)

    def step_fn(live_model, _example):
        return sum(parameter.float().pow(2).mean() for parameter in live_model.parameters() if parameter.requires_grad)

    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        step_fn,
        model_factory=lambda: model,
        checkpoint_dir=Path("/tmp/jev-nvidia-gpu-smoke-checkpoints"),
    )
    observations: list[dict[str, int | float]] = []
    errors: list[str] = []

    def infer_loop() -> None:
        try:
            for _ in range(args.inferences):
                begin = time.perf_counter()
                result = runtime.infer_result("2026最佳技术栈包括 Python 和 PostgreSQL")
                observations.append({
                    "seconds": time.perf_counter() - begin,
                    "generation": result["generation"],
                    "ema_step": result["ema_step"],
                })
        except Exception as exc:  # pragma: no cover - hardware-only failure path
            errors.append(repr(exc))

    thread = threading.Thread(target=infer_loop)
    thread.start()
    train_started = time.perf_counter()
    events = trainer.train([{} for _ in range(args.steps)], max_steps=args.steps)
    train_seconds = time.perf_counter() - train_started
    thread.join()
    final = runtime.infer_result("2026最佳技术栈包括 Python 和 PostgreSQL")
    latencies = sorted(item["seconds"] for item in observations)
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    output = {
        "pass": not errors and len(observations) == args.inferences and runtime.ema.updates == args.steps,
        "gpu": "NVIDIA GPU",
        "gpu_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "base_load_seconds": base_load_seconds,
        "steps": len(events),
        "losses": [event.loss for event in events],
        "model_generation": runtime.stats.model_generation,
        "ema_updates": runtime.ema.updates,
        "inference_calls": len(observations),
        "inference_observations": observations,
        "inference_p50_seconds": latencies[len(latencies) // 2] if latencies else None,
        "inference_p95_seconds": latencies[min(len(latencies) - 1, max(0, int(len(latencies) * 0.95) - 1))] if latencies else None,
        "inference_throughput_per_second": (len(latencies) / sum(latencies)) if latencies and sum(latencies) > 0 else None,
        "training_seconds": train_seconds,
        "training_steps_per_second": (len(events) / train_seconds) if train_seconds > 0 else None,
        "nvidia_smi_temperature_power": nvidia_smi,
        "errors": errors,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "result": final["output"],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False))
    return 0 if output["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
