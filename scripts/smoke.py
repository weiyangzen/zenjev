#!/usr/bin/env python3
"""Dependency-free Jev control-plane smoke test."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from jev.config import JevConfig
from jev.distill import distill
from jev.drift import DriftMonitor
from jev.runtime import JevRuntime


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source.txt"
        source.write_text("Python and PostgreSQL are evaluated for 2026.", encoding="utf-8")
        config = JevConfig.from_dict({"schema": {"name": "stack", "entities": ["technology"]}, "sources": [{"name": "local", "kind": "text", "location": str(source)}], "teacher": {"provider": "smoke", "model": "fixture", "base_url": "https://invalid", "api_key_env": "UNUSED"}})
        class Teacher:
            def label(self, document):
                return {"technology": ["Python", "PostgreSQL"]}
        output = root / "distilled.jsonl"
        assert distill(config, output, Teacher()) == 1
        assert json.loads(output.read_text())["schema_digest"] == config.digest()
        monitor = DriftMonitor(config.drift)
        assert monitor.observe(float("nan"), 1).reset_required
        runtime = JevRuntime(config, infer=lambda model, text: {"text": text, "generation": runtime.stats.model_generation})
        runtime.publish("fixture")
        assert runtime.infer("hello")["text"] == "hello"
    print("jev_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

