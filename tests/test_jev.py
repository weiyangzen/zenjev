import json
from pathlib import Path

import pytest

from jev.config import ConfigError, DriftPolicy, JevConfig, load_config
from jev.distill import DistilledExample, distill
from jev.drift import DriftMonitor
from jev.runtime import JevRuntime


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
