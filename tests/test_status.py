"""Status publisher contract tests for the zenjev-monitor consumer."""

from __future__ import annotations

import json

from jev.status import SCHEMA_VERSION, clear_status, resolve_status_path, write_status


def test_write_status_is_atomic_and_contract_shaped(tmp_path):
    target = tmp_path / "status.json"
    write_status({"phase": "training", "generation": 3, "loss": 1.25}, target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["phase"] == "training"
    assert payload["generation"] == 3
    assert payload["loss"] == 1.25
    assert isinstance(payload["updated_at_unix_ms"], int)
    assert isinstance(payload["pid"], int)
    assert list(tmp_path.iterdir()) == [target]


def test_status_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_STATUS_PATH", str(tmp_path / "env-status.json"))
    assert resolve_status_path() == tmp_path / "env-status.json"
    assert resolve_status_path("explicit.json") == resolve_status_path("explicit.json")
    write_status({"phase": "idle"})
    assert (tmp_path / "env-status.json").exists()


def test_clear_status_is_idempotent(tmp_path):
    target = tmp_path / "status.json"
    write_status({"phase": "idle"}, target)
    clear_status(target)
    clear_status(target)
    assert not target.exists()
