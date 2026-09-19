import json
import sys
import types

from jev.config import JevConfig
from jev.distill import OpenAICompatibleTeacher, _validate_output
from jev.model import apply_lora, extract, gliner2_step, load_gliner
from jev.sources import SourceDocument
from jev.tool_task import classify_decision_choices, classify_task_type, resolve_tool_task


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


def test_gliner2_step_uses_official_collator(monkeypatch):
    calls = {}

    class Collator:
        def __init__(self, processor, **kwargs):
            calls["kwargs"] = kwargs
        def __call__(self, batch):
            calls["batch"] = batch
            return "batch"

    monkeypatch.setitem(sys.modules, "gliner2.training", types.SimpleNamespace(ExtractorCollator=Collator))
    class Model:
        processor = object()
        def __call__(self, batch):
            assert batch == "batch"
            return {"total_loss": "loss"}
    assert gliner2_step(Model(), make_config(), {"input": "Python", "output": {"entities": {"technology": ["Python"]}}}) == "loss"
    assert calls["kwargs"]["is_training"] is True


def test_teacher_cache_reuses_strict_json_response(monkeypatch, tmp_path):
    config = JevConfig.from_dict({
        **make_config().canonical(),
        "teacher": {
            **make_config().canonical()["teacher"],
            "cache_dir": str(tmp_path / "teacher-cache"),
            "max_retries": 0,
        },
    })
    payload = {"choices": [{"message": {"content": '{"technology":["Python"]}'}}], "usage": {"total_tokens": 3}}
    calls = {"count": 0}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(payload).encode()

    def fake_urlopen(request, timeout):
        calls["count"] += 1
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setenv("KEY", "test-secret")
    teacher = OpenAICompatibleTeacher(config)
    document = SourceDocument("fixture", "fixture://1", "Python", "test")
    assert teacher.label(document) == {"technology": ["Python"]}
    assert teacher.last_metadata["cache_hit"] is False
    assert teacher.label(document) == {"technology": ["Python"]}
    assert teacher.last_metadata["cache_hit"] is True
    assert calls["count"] == 1
    assert list((tmp_path / "teacher-cache").glob("*.json"))


def test_tool_task_policy_allowlist_routes_without_executing_tools(tmp_path):
    config = JevConfig.from_dict({
        **make_config().canonical(),
        "tool_task": {
            "name": "jev-tool-task",
            "task_types": ["coding", "other"],
            "task_actions": {"coding": "allow", "other": "reject"},
            "allowed_models": [{"id": "gpt-6-astra", "provider": "openai-compatible"}],
            "allowed_tools": ["python"],
            "decision_choices": ["PyTorch", "Python", "SQL"],
            "decision_actions": ["allow", "review", "reject"],
            "min_confidence": 0.75,
        },
    })
    assert config.tool_task is not None
    output = {"task_type": {"label": "coding", "confidence": 0.9}, "entities": {"tool": [{"text": "python", "start": 0, "end": 6}]}}
    decision = resolve_tool_task(output, config.tool_task, model_id="gpt-6-astra")
    assert decision["allowed"] is True and decision["action"] == "allow"
    choice_output = {"decision_choice": {"value": ["PyTorch", "Python"], "confidence": 0.91, "probabilities": {"PyTorch": 0.91, "Python": 0.82, "SQL": 0.11}}}
    choices = resolve_tool_task({**output, **choice_output}, config.tool_task, model_id="gpt-6-astra")["decision_choices"]
    assert choices == [{"choice": "PyTorch", "probability": 0.91}, {"choice": "Python", "probability": 0.82}, {"choice": "SQL", "probability": 0.11}]
    extracted = resolve_tool_task({**output, "entities": {"technology": [{"text": "PyTorch", "confidence": 0.88}]}}, config.tool_task, model_id="gpt-6-astra")
    assert extracted["decision_choices"] == [{"choice": "PyTorch", "probability": 0.88}]
    rejected = resolve_tool_task(output, config.tool_task, model_id="unknown-model")
    assert rejected["allowed"] is False and "model_not_allowlisted" in rejected["reasons"]


def test_tool_task_example_config_loads():
    config = JevConfig.from_dict(__import__("jev.config", fromlist=["_load_document"])._load_document("configs/jev_tool_task.yaml"))
    assert config.tool_task is not None
    assert config.tool_task.name == "jev-tool-task"


def test_classify_task_type_uses_finite_policy_labels(monkeypatch):
    config = JevConfig.from_dict({
        **make_config().canonical(),
        "tool_task": {
            "task_types": ["coding", "other"],
            "task_actions": {"coding": "allow", "other": "reject"},
            "allowed_models": [{"id": "gpt-6-astra", "provider": "openai-compatible"}],
        },
    })
    calls = {}
    class Schema:
        def single(self, name, labels, **kwargs):
            calls["schema"] = (name, labels, kwargs)
            return self
    class Classifier:
        def __init__(self, model): pass
        def classify(self, text, schema, config):
            calls["text"] = text
            return {"task_type": {"label": "coding", "confidence": 0.9}}
    monkeypatch.setitem(sys.modules, "gliner2.classification", types.SimpleNamespace(
        ClassificationConfig=lambda **kwargs: kwargs, ClassificationSchema=Schema, Classifier=Classifier,
    ))
    result = classify_task_type(object(), "fix bug", config.tool_task)
    assert result["task_type"]["label"] == "coding"
    assert calls["schema"][1] == ["coding", "other"]


def test_classify_decision_choices_preserves_probability_map(monkeypatch):
    policy = JevConfig.from_dict({
        **make_config().canonical(),
        "tool_task": {
            "task_types": ["coding"],
            "task_actions": {"coding": "allow"},
            "allowed_models": [{"id": "gpt-6-astra", "provider": "openai-compatible"}],
            "decision_choices": ["PyTorch", "Python"],
        },
    }).tool_task
    calls = {}
    class Schema:
        def multi(self, name, labels, **kwargs):
            calls["schema"] = (name, labels, kwargs)
            return self
    class Result:
        def to_dict(self):
            return {"decision_choice": {"value": ["PyTorch"], "confidence": 0.9, "probabilities": {"PyTorch": 0.9, "Python": 0.2}}}
    class Classifier:
        def __init__(self, model): pass
        def classify(self, text, schema, config): return Result()
    monkeypatch.setitem(sys.modules, "gliner2.classification", types.SimpleNamespace(
        ClassificationConfig=lambda **kwargs: kwargs, ClassificationSchema=Schema, Classifier=Classifier,
    ))
    result = classify_decision_choices(object(), "train with PyTorch", policy)
    assert result["decision_choice"]["probabilities"]["PyTorch"] == 0.9
    assert calls["schema"][1] == ["PyTorch", "Python"]
