from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev.config import JevConfig
from jev.runtime import JevRuntime
from jev.training import ContinuousLoRATrainer

torch = pytest.importorskip("torch")


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor(0.0))


def make_config(tmp_path: Path) -> JevConfig:
    return JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {"name": "x", "entities": ["technology"]},
        "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
        "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
        "drift": {"eval_window": 2, "min_windows": 2, "reset_cooldown_steps": 1},
        "runtime": {"reset_warmup_steps": 2, "checkpoint_every_steps": 1, "publish_every_steps": 1},
    })


def make_trainer(tmp_path: Path, config: JevConfig, step_fn, **kwargs):
    runtime = JevRuntime(config, infer=lambda model, text: float(model.adapter.detach().item()))
    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        step_fn,
        model_factory=TinyAdapter,
        checkpoint_dir=tmp_path / "run",
        **kwargs,
    )
    return runtime, trainer


def drift_step_fn(calls: dict[str, int]):
    def step_fn(model, _example):
        calls["n"] += 1
        loss = (model.adapter - 0.0).pow(2) + 1.0
        return loss * (1000.0 if 3 < calls["n"] <= 6 else 1.0)

    return step_fn


def test_synthetic_drift_reset_archives_state_and_warmup_keeps_old_snapshot(tmp_path):
    config = make_config(tmp_path)
    runtime, trainer = make_trainer(tmp_path, config, drift_step_fn({"n": 0}))
    trainer.train([{"target": 1.0}] * 3, max_steps=3)
    assert runtime.stats.reset_id == 0
    assert not trainer._warmup_pending
    trainer.train([{"target": 1.0}] * 2, max_steps=2)
    assert runtime.stats.reset_id == 0
    old_generation = runtime.stats.model_generation
    old_value = runtime.infer("x")

    events = trainer.train([{"target": 1.0}], max_steps=1)
    assert any(event.reset_required for event in events)
    assert runtime.stats.reset_id == 1
    assert trainer._warmup_pending
    assert runtime.stats.model_generation == old_generation
    assert runtime.infer("x") == pytest.approx(old_value)

    archives = sorted((tmp_path / "run" / "archives").glob("reset-*.pt"))
    assert len(archives) == 1
    archived = torch.load(archives[0], map_location="cpu", weights_only=False)
    assert archived["failure_reason"] == "loss_window_exceeded"
    assert archived["kind"] == "adapter-only"

    reset_events = [
        json.loads(line)
        for line in (tmp_path / "run" / "reset-events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(reset_events) == 1
    assert reset_events[0]["reset_id"] == 1
    assert reset_events[0]["reason"] == "loss_window_exceeded"
    assert reset_events[0]["warmup_steps"] == 2
    assert Path(reset_events[0]["archive"]).exists()

    trainer.train([{"target": 1.0}] * 2, max_steps=2)
    assert not trainer._warmup_pending
    assert runtime.stats.model_generation == old_generation + 1
    assert runtime.infer("x") == pytest.approx(0.0, abs=1e-6)


def test_warmup_evaluator_can_veto_promotion(tmp_path):
    config = make_config(tmp_path)
    audit: list[bool] = []

    def warmup_evaluator(_model):
        audit.append(True)
        return len(audit) >= 2

    runtime, trainer = make_trainer(
        tmp_path,
        config,
        drift_step_fn({"n": 0}),
        warmup_evaluator=warmup_evaluator,
    )
    trainer.train([{"target": 1.0}] * 6, max_steps=6)
    assert runtime.stats.reset_id == 1
    assert trainer._warmup_pending
    generation = runtime.stats.model_generation
    stale_value = runtime.infer("x")

    trainer.train([{"target": 1.0}] * 2, max_steps=2)
    assert audit == [True]
    assert trainer._warmup_pending
    assert trainer._warmup_remaining == 2
    assert runtime.stats.model_generation == generation
    assert runtime.infer("x") == pytest.approx(stale_value)

    trainer.train([{"target": 1.0}] * 2, max_steps=2)
    assert audit == [True, True]
    assert not trainer._warmup_pending
    assert runtime.stats.model_generation == generation + 1
