"""Governance tests for the authoritative blueprint and its Gantt projection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def governance():
    spec = importlib.util.spec_from_file_location("validate_blueprint", ROOT / "scripts" / "validate_blueprint.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_text() -> str:
    return (ROOT / "Docs" / "stage0_zenjev_blueprint.md").read_text(encoding="utf-8")


def test_companion_naming_is_case_sensitive(governance):
    assert governance.companion_path(Path("Docs/xBlueprint.md")).name == "xGantt.md"
    assert governance.companion_path(Path("Docs/x_blueprint.md")).name == "x_blueprint_Gantt.md"
    assert governance.GANTT.name == "stage0_zenjev_blueprint_Gantt.md"


def test_current_blueprint_parses_with_unique_ids_and_disabled_controller(governance):
    items, spec_digest, event = governance.parse(source_text())
    assert len(items) == len({item.id for item in items})
    assert len(spec_digest) == 64
    assert event["event_id"] == "governance-policy-correction"
    prerequisites = governance.json_block(source_text(), "execution-prerequisites")
    assert prerequisites["controller_enabled"] is False
    assert all(prerequisites[key] is None for key in governance.POLICY_KEYS)
    assert all(value is None for value in prerequisites["concurrency"].values())


def test_projection_roundtrip_and_tamper_detection(governance):
    source = source_text()
    rendered = governance.render(source, "2026-09-20T12:00:00Z")
    governance.check_projection(source, rendered)
    tampered = rendered.replace("| Checklist items |", "| Checklist items (tampered) |")
    with pytest.raises(governance.GovernanceError):
        governance.check_projection(source, tampered)


def test_unknown_dependency_fails_closed(governance):
    broken = source_text().replace("**Depends on:** ZJ-001. **Owner:** Master", "**Depends on:** ZJ-999. **Owner:** Master", 1)
    with pytest.raises(governance.GovernanceError, match="unknown dependency"):
        governance.parse(broken)


def test_dependency_cycle_fails_closed(governance):
    source = source_text()
    marker = "**Depends on:** ZJ-002. **Owner:** Master"
    assert marker in source
    broken = source.replace(marker, "**Depends on:** ZJ-002,ZJ-001. **Owner:** Master", 1)
    # ZJ-003 already depends on ZJ-002, so this remains acyclic; introduce a real cycle.
    broken = source.replace("**Depends on:** ZJ-002. **Owner:** Master", "**Depends on:** ZJ-003. **Owner:** Master", 1)
    with pytest.raises(governance.GovernanceError, match="cycle"):
        governance.parse(broken)


def test_accepted_item_with_unfinished_dependency_fails_closed(governance):
    broken = source_text().replace("- [x] **ZJ-001**", "- [ ] **ZJ-001**", 1)
    with pytest.raises(governance.GovernanceError):
        governance.parse(broken)


def test_invalid_checklist_mark_fails_closed(governance):
    broken = source_text().replace("- [x] **ZJ-001**", "- [?] **ZJ-001**", 1)
    with pytest.raises(governance.GovernanceError, match="invalid checklist row"):
        governance.parse(broken)
