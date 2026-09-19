import json
import sys
import types

from jev.config import JevConfig
from jev.distill import _validate_output
from jev.model import apply_lora, extract, load_gliner


def make_config():
    return JevConfig.from_dict({
        "model_id": "fastino/gliner2-base-v1",
        "schema": {
            "name": "tech",
            "version": 2,
            "entities": {"technology": "A software technology"},
            "fields": [{"name": "year", "dtype": "int", "choices": [2026]}],
            "classifications": {"maturity": ["stable", "experimental"]},
            "relations": ["uses"],
        },
        "sources": [{"name": "fixture", "kind": "text", "location": "fixture.txt"}],
        "teacher": {"provider": "fixture", "model": "teacher", "base_url": "https://example.invalid/v1", "api_key_env": "KEY"},
    })


def test_schema_revision_and_training_conversion():
    config = make_config()
    converted = _validate_output(config, {"technology": ["Python"], "year": 2026, "maturity": "stable", "_evidence": {}})
    assert converted["entities"]["technology"] == ["Python"]
    assert converted["classifications"][0]["true_label"] == ["stable"]
    assert converted["json_structures"][0]["tech"]["year"] == "2026"
    assert config.model_revision == "79c3a777abc572b4767922f3916cf63fb5754df2"


def test_apply_lora_returns_wrapped_model():
    class Fake:
        def apply_lora(self, **kwargs):
            return (self, kwargs)

    model, kwargs = apply_lora(Fake(), make_config())
    assert kwargs["r"] == 8 and kwargs["alpha"] == 16


def test_extract_builds_gliner_schema(monkeypatch):
    class Schema:
        def __init__(self):
            self.calls = []
        def entities(self, value): self.calls.append(("entities", value)); return self
        def classification(self, task, labels): self.calls.append(("classification", task, labels)); return self
        def structure(self, name):
            self.calls.append(("structure", name))
            parent = self
            class Builder:
                def field(self, name, **kwargs): parent.calls.append(("field", name, kwargs)); return self
            return Builder()
        def relations(self, value): self.calls.append(("relations", value)); return self
    fake_module = types.SimpleNamespace(Schema=Schema)
    monkeypatch.setitem(sys.modules, "gliner2", fake_module)
    class Model:
        def extract(self, text, schema, **kwargs): return schema.calls, kwargs
    calls, kwargs = extract(Model(), make_config(), "text")
    assert any(c[0] == "entities" for c in calls)
    assert any(c[0] == "classification" for c in calls)
    assert any(c[0] == "relations" for c in calls)
    assert kwargs["include_spans"] is True


def test_extract_normalizes_decoder_structure_types():
    class Model:
        def extract(self, text, schema, **kwargs):
            return {"tech": [{"year": "2026"}]}

    result = extract(Model(), make_config(), "text")
    assert result["tech"][0]["year"] == 2026


def test_load_gliner_uses_staged_snapshot_without_hub_revision(monkeypatch, tmp_path):
    staged = tmp_path / "gliner2-base"
    staged.mkdir()
    calls = {}

    class AutoExtractor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["source"] = source
            calls["kwargs"] = kwargs
            return "model"

    monkeypatch.setitem(sys.modules, "gliner2", types.SimpleNamespace(AutoExtractor=AutoExtractor))
    config = make_config()
    config = JevConfig.from_dict({**config.canonical(), "runtime": {**config.canonical()["runtime"], "model_path": str(staged)}})
    assert load_gliner(config) == "model"
    assert calls["source"] == str(staged)
    assert "revision" not in calls["kwargs"]
