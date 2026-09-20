from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "benchmarks" / "run_stage0_benchmark.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("run_stage0_benchmark", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_benchmark_module()


def smoke_payload(**overrides):
    payload = {
        "pass": True,
        "steps": 2,
        "inference_calls": 4,
        "inference_observations": [
            {"generation": 1, "seconds": 0.1},
            {"generation": 2, "seconds": 0.2},
            {"generation": 3, "seconds": 0.3},
            {"generation": 3, "seconds": 0.4},
        ],
        "errors": [],
        "ema_updates": 2,
        "model_generation": 3,
        "peak_vram_bytes": 512,
        "gpu_memory_bytes": 1024,
        "nvidia_smi_temperature_power": "55, 120.00 W",
    }
    payload.update(overrides)
    return payload


def test_percentile_empty_and_single():
    assert bench.percentile([], 50) is None
    assert bench.percentile([0.25], 50) == 0.25


def test_percentile_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert bench.percentile(values, 0) == 1.0
    assert bench.percentile(values, 50) == 2.5
    assert bench.percentile(values, 100) == 4.0
    assert bench.percentile(values, 95) == pytest.approx(3.85)
    assert bench.percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5


def test_evaluate_pass_criteria_accepts_healthy_smoke():
    gate = bench.evaluate_pass_criteria(smoke_payload(), expected_steps=2, expected_inferences=4)
    assert gate["pass"] is True
    assert all(gate["checks"].values())


def test_evaluate_pass_criteria_rejects_errors_and_missing_work():
    gate = bench.evaluate_pass_criteria(
        smoke_payload(**{"pass": False}, errors=["boom"], inference_calls=1, model_generation=0),
        expected_steps=2,
        expected_inferences=4,
    )
    assert gate["pass"] is False
    assert gate["checks"]["smoke_pass"] is False
    assert gate["checks"]["no_errors"] is False
    assert gate["checks"]["inferences_completed"] is False
    assert gate["checks"]["generation_advanced"] is False


def test_evaluate_pass_criteria_vram_budget():
    gate = bench.evaluate_pass_criteria(
        smoke_payload(peak_vram_bytes=2048),
        expected_steps=2,
        expected_inferences=4,
        vram_limit_bytes=1024,
    )
    assert gate["pass"] is False
    assert gate["checks"]["vram_within_budget"] is False
    assert gate["checks"]["vram_within_device"] is False


def test_build_smoke_command_is_deterministic(tmp_path):
    command = bench.build_smoke_command(
        python="/usr/bin/python3",
        smoke_script=tmp_path / "smoke_nvidia_gpu.py",
        config="configs/example.yaml",
        output=tmp_path / "out.json",
        steps=2,
        inferences=4,
    )
    assert command[:2] == ["/usr/bin/python3", str(tmp_path / "smoke_nvidia_gpu.py")]
    assert command[command.index("--steps") + 1] == "2"
    assert command[command.index("--inferences") + 1] == "4"
    assert "--model-path" not in command
    with_model = bench.build_smoke_command(
        python="python",
        smoke_script="smoke.py",
        config="c.yaml",
        output="o.json",
        steps=1,
        inferences=1,
        model_path="/staged/model",
    )
    assert with_model[-2:] == ["--model-path", "/staged/model"]


def test_atomic_write_json_replaces_and_parses(tmp_path):
    target = tmp_path / "nested" / "artifact.json"
    bench.atomic_write_json(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    bench.atomic_write_json(target, {"b": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2}
    assert not list(target.parent.glob("*.tmp"))
