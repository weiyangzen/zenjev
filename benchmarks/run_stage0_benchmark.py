#!/usr/bin/env python3
"""Collect the Stage 0 GPU train/infer benchmark from the NVIDIA smoke run.

The runner wraps ``scripts/smoke_nvidia_gpu.py`` with a short deterministic
step/inference count and merges its observations into one benchmark artifact.
Only pure helpers are defined at module scope so the CPU test suite can import
this file without CUDA or torch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linearly interpolated percentile with a deterministic definition."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    rank = (p / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_smoke_command(
    *,
    python: str,
    smoke_script: str | Path,
    config: str | Path,
    output: str | Path,
    steps: int,
    inferences: int,
    model_path: str | Path | None = None,
) -> list[str]:
    command = [
        python,
        str(smoke_script),
        "--config",
        str(config),
        "--output",
        str(output),
        "--steps",
        str(int(steps)),
        "--inferences",
        str(int(inferences)),
    ]
    if model_path:
        command += ["--model-path", str(model_path)]
    return command


def evaluate_pass_criteria(
    smoke: dict[str, Any],
    *,
    expected_steps: int,
    expected_inferences: int,
    vram_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Recompute the benchmark gate from the smoke report instead of trusting it."""
    observations = smoke.get("inference_observations") or []
    generations = [item.get("generation") for item in observations if isinstance(item, dict)]
    peak = smoke.get("peak_vram_bytes")
    total = smoke.get("gpu_memory_bytes")
    checks: dict[str, bool] = {
        "smoke_pass": bool(smoke.get("pass")),
        "steps_completed": int(smoke.get("steps") or 0) == expected_steps,
        "inferences_completed": int(smoke.get("inference_calls") or 0) == expected_inferences,
        "observations_recorded": len(observations) == expected_inferences,
        "no_errors": not smoke.get("errors"),
        "ema_updates": int(smoke.get("ema_updates") or 0) >= expected_steps,
        "generation_advanced": int(smoke.get("model_generation") or 0) >= expected_steps,
        "observed_generation_positive": bool(generations) and all(
            isinstance(value, int) and value > 0 for value in generations
        ),
        "vram_recorded": isinstance(peak, int) and peak > 0,
        "vram_within_device": isinstance(peak, int) and isinstance(total, int) and 0 < peak <= total,
        "temperature_power_recorded": bool(smoke.get("nvidia_smi_temperature_power")),
    }
    if vram_limit_bytes is not None:
        checks["vram_within_budget"] = isinstance(peak, int) and peak <= vram_limit_bytes
    return {"pass": all(checks.values()), "checks": checks}


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


def _read_smoke_output(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("smoke output must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--output", default="artifacts/benchmarks/stage0_gpu_benchmark.json")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--inferences", type=int, default=4)
    parser.add_argument("--vram-limit-bytes", type=int, default=None)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("CUDA is required for the Stage 0 GPU benchmark", file=sys.stderr)
        return 2
    if args.steps < 1 or args.inferences < 1:
        parser.error("--steps and --inferences must be positive")

    started = time.perf_counter()
    smoke_script = ROOT / "scripts" / "smoke_nvidia_gpu.py"
    with tempfile.TemporaryDirectory(prefix="jev-stage0-benchmark-") as directory:
        smoke_output = Path(directory) / "smoke.json"
        command = build_smoke_command(
            python=sys.executable,
            smoke_script=smoke_script,
            config=args.config,
            output=smoke_output,
            steps=args.steps,
            inferences=args.inferences,
            model_path=args.model_path,
        )
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        smoke: dict[str, Any] = {}
        errors: list[str] = []
        if smoke_output.exists():
            try:
                smoke = _read_smoke_output(smoke_output)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"smoke output unreadable: {exc!r}")
        else:
            errors.append("smoke did not write its output artifact")

    gate = evaluate_pass_criteria(
        smoke,
        expected_steps=args.steps,
        expected_inferences=args.inferences,
        vram_limit_bytes=args.vram_limit_bytes,
    )
    latencies = [
        float(item["seconds"])
        for item in smoke.get("inference_observations", [])
        if isinstance(item, dict) and isinstance(item.get("seconds"), (int, float))
    ]
    capability = torch.cuda.get_device_capability(0)
    peak = smoke.get("peak_vram_bytes")
    total = smoke.get("gpu_memory_bytes")
    payload: dict[str, Any] = {
        "pass": bool(gate["pass"] and not errors),
        "checks": gate["checks"],
        "errors": errors + [line for line in (completed.stderr or "").splitlines() if line.strip()][-10:],
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": f"{capability[0]}.{capability[1]}",
            "memory_bytes": total,
        },
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "smoke_exit_code": completed.returncode,
        "benchmark_seconds": time.perf_counter() - started,
        "base_load_seconds": smoke.get("base_load_seconds"),
        "training_seconds": smoke.get("training_seconds"),
        "training_steps_per_second": smoke.get("training_steps_per_second"),
        "inference_throughput_per_second": (
            len(latencies) / sum(latencies) if latencies and sum(latencies) > 0 else None
        ),
        "inference_p50_seconds": percentile(latencies, 50),
        "inference_p95_seconds": percentile(latencies, 95),
        "peak_vram_bytes": peak,
        "vram_headroom_bytes": (total - peak) if isinstance(peak, int) and isinstance(total, int) else None,
        "nvidia_smi_temperature_power": smoke.get("nvidia_smi_temperature_power"),
        "concurrent_train_infer": {
            "inferences": smoke.get("inference_calls"),
            "model_generation": smoke.get("model_generation"),
            "ema_updates": smoke.get("ema_updates"),
            "errors": smoke.get("errors"),
        },
        "steps": smoke.get("steps"),
        "inferences": smoke.get("inference_calls"),
        "losses": smoke.get("losses"),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
