import json
from pathlib import Path

import pytest

from jev.config import ConfigError, DriftPolicy, JevConfig, load_config
from jev.distill import DistilledExample, distill
from jev.drift import DriftMonitor
from jev.runtime import JevRuntime
from jev.training import ContinuousLoRAStream, ContinuousLoRATrainer


def cfg(tmp_path: Path) -> JevConfig:
    return JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {"name": "x", "entities": ["technology"]},
        "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
        "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
    })


def test_config_digest_and_required_source(tmp_path):
    (tmp_path / "in.txt").write_text("hello", encoding="utf-8")
    value = cfg(tmp_path)
    assert len(value.digest()) == 64
    with pytest.raises(ConfigError):
        JevConfig.from_dict({"schema": {"name": "x"}, "sources": [], "teacher": {}})


def test_distill_records_provenance(tmp_path):
    (tmp_path / "in.txt").write_text("Python is a technology.", encoding="utf-8")
    value = cfg(tmp_path)
    class FakeTeacher:
        def label(self, document):
            return {"technology": ["Python"]}
    out = tmp_path / "out.jsonl"
    assert distill(value, out, FakeTeacher()) == 1
    record = json.loads(out.read_text())
    assert record["source"]["content_sha256"]
    assert record["teacher"]["model"] == "teacher"


def test_drift_requires_reset_after_nonfinite():
    monitor = DriftMonitor(DriftPolicy(eval_window=2, min_windows=1, reset_cooldown_steps=1))
    assert monitor.observe(float("nan"), 1).reset_required


def test_drift_resets_after_three_bad_windows():
    monitor = DriftMonitor(DriftPolicy(eval_window=2, min_windows=3, loss_ratio=2.0, reset_cooldown_steps=1))
    for _ in range(2):
        monitor.observe(1.0, 1.0)
    for window in range(3):
        state = None
        for _ in range(2):
            state = monitor.observe(3.0, 1.0)
    assert state is not None and state.reset_required and state.reason == "loss_window_exceeded"
    assert state.resets == 1


def test_runtime_publishes_and_infers():
    tmp = Path("/tmp/jev-test")
    tmp.mkdir(exist_ok=True)
    value = cfg(tmp)
    runtime = JevRuntime(value, infer=lambda model, text: {"model": model, "text": text})
    runtime.publish("adapter")
    assert runtime.infer("x")["model"] == "adapter"


def test_training_ema_clip_checkpoint_and_resume(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.tensor(0.0))

    value = JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {"name": "x", "entities": ["technology"]},
        "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
        "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
        "runtime": {"ema_decay": 0.5, "grad_clip_norm": 0.1, "checkpoint_every_steps": 1, "publish_every_steps": 1},
    })
    runtime = JevRuntime(value, infer=lambda model, text: float(model.adapter.detach().item()))
    trainer = ContinuousLoRATrainer(
        value,
        runtime,
        lambda model, example: (model.adapter - example["target"]).pow(2),
        model_factory=TinyAdapter,
        checkpoint_dir=tmp_path / "run",
    )
    events = trainer.train([{"target": 10.0}], max_steps=1)
    assert events[0].checkpoint
    latest = tmp_path / "run" / "latest.pt"
    assert latest.exists()
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert payload["kind"] == "adapter-only"
    assert set(payload["adapter_state"]) == {"adapter"}
    assert payload["ema_updates"] == 1
    resumed_runtime = JevRuntime(value, infer=lambda model, text: float(model.adapter.detach().item()))
    resumed = ContinuousLoRATrainer(
        value,
        resumed_runtime,
        lambda model, example: (model.adapter - example["target"]).pow(2),
        model_factory=TinyAdapter,
        checkpoint_dir=tmp_path / "resumed",
        resume_from=latest,
    )
    assert resumed.runtime.stats.training_steps == 1
    assert resumed.runtime.ema.updates == 1
    assert resumed.runtime.infer("x") == pytest.approx(runtime.infer("x"), rel=1e-5)


def test_collapse_archives_state_and_warmup_keeps_old_snapshot(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.tensor(0.0))

    value = JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {"name": "x", "entities": ["technology"]},
        "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
        "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
        "runtime": {"reset_warmup_steps": 2, "checkpoint_every_steps": 1, "publish_every_steps": 1},
    })
    runtime = JevRuntime(value, infer=lambda model, text: float(model.adapter.detach().item()))
    trainer = ContinuousLoRATrainer(
        value,
        runtime,
        lambda model, example: (model.adapter - example["target"]).pow(2),
        model_factory=TinyAdapter,
        checkpoint_dir=tmp_path / "run",
    )
    trainer.train([{"target": 1.0}], max_steps=1)
    old_generation = runtime.stats.model_generation
    old_value = runtime.infer("x")
    trainer._reset_model("synthetic_collapse")
    assert runtime.stats.reset_id == 1
    assert runtime.stats.model_generation == old_generation
    assert runtime.infer("x") == pytest.approx(old_value)
    archives = list((tmp_path / "run" / "archives").glob("reset-*.pt"))
    assert len(archives) == 1
    archived = torch.load(archives[0], map_location="cpu", weights_only=False)
    assert archived["failure_reason"] == "synthetic_collapse"
    trainer.train([{"target": 0.0}, {"target": 0.0}], max_steps=2)
    assert runtime.stats.model_generation == old_generation + 1
    assert not trainer._warmup_pending


def test_bounded_stream_trains_while_runtime_remains_available(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.tensor(2.0), requires_grad=False)
            self.adapter = torch.nn.Parameter(torch.tensor(0.0))

    value = JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {"name": "x", "entities": ["technology"]},
        "sources": [{"name": "s", "kind": "text", "location": str(tmp_path / "in.txt")}],
        "teacher": {"provider": "test", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
        "runtime": {"checkpoint_every_steps": 100, "publish_every_steps": 1},
    })
    runtime = JevRuntime(value, infer=lambda model, text: float(model.adapter.detach().item()))
    trainer = ContinuousLoRATrainer(
        value,
        runtime,
        lambda model, example: (model.adapter - example["target"]).pow(2),
        model_factory=TinyAdapter,
        checkpoint_dir=tmp_path / "run",
    )
    stream = ContinuousLoRAStream(trainer, max_queue_size=1).start()
    stream.submit({"target": 1.0}, timeout=1)
    stream.submit({"target": 0.0}, timeout=1)
    events = stream.stop(timeout=5)
    assert len(events) == 2
    assert runtime.stats.training_steps == 2
    assert runtime.infer("x") is not None
