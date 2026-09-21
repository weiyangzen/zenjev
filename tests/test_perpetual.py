from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from jev.config import ConfigError, load_config  # noqa: E402
from jev.mq import render_bridge_config, validate_envelope  # noqa: E402
from zenjev_fabricator import build_record  # noqa: E402
from zenjev_feeder import convert_line  # noqa: E402

CONFIG_PATH = REPO / "configs/zenjev_perpetual.yaml"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def allowed_models(config) -> set[str]:
    allowed = {item.model_id for item in config.tool_task.allowed_models}
    allowed |= {alias for item in config.tool_task.allowed_models for alias in item.aliases}
    return allowed


def test_dir_spool_config_renders_source_root(config):
    assert config.mq is not None and config.mq.adapter == "dir-spool"
    rendered = render_bridge_config(config)
    assert rendered["adapter"] == "dir-spool"
    assert rendered["source_root"] == "/home/sansha/data/jevraw/_zenjev"
    assert rendered["patterns"] == ["*.ndjson"]
    assert rendered["poll_interval_ms"] == 2000
    assert rendered["schema_digest"] == config.schema.digest()


def test_dir_spool_requires_source_root():
    from jev.config import MqConfig

    with pytest.raises(ConfigError):
        MqConfig.from_dict({"enabled": True, "adapter": "dir-spool"})


def test_feeder_normalizes_both_capture_formats(config):
    allowed = allowed_models(config)
    format_b = json.dumps(
        {
            "schema_version": 1,
            "model": "gpt-5.6-sol",
            "created_at": "2026-09-01T00:00:00Z",
            "input_body": json.dumps(
                {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix the Python parser and add PyTorch regression coverage."}]}]}
            ),
            "output_body": 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Use pytest with Python and PyTorch."}\n',
        }
    ).encode()
    envelope, reason = convert_line(format_b, rel="dump/a.ndjson", offset=0, config=config, allowed_models=allowed)
    assert reason is None and envelope is not None
    assert envelope["kind"] == "tool_task_pair"
    assert "Python" in envelope["request"]["text"]
    assert "pytest" in envelope["response"]["text"]
    assert envelope["observed_model"]["model_id"] == "gpt-5.6-sol"
    assert envelope["schema"]["digest"] == config.schema.digest()
    assert validate_envelope(json.dumps(envelope).encode(), config, max_bytes=64 * 1024 * 1024) == envelope

    format_a = json.dumps(
        {
            "version": "full-trace-spool-v2",
            "receivedAt": "2026-09-02T00:00:00Z",
            "requestBody": {"kind": "json", "value": {"model": "gpt-5.6-sol", "input": [{"content": [{"text": "Refactor the Docker compose file and validate with bash."}]}]}},
            "responseBody": {"kind": "json", "value": {"output": [{"content": [{"type": "output_text", "text": "Use docker compose v2 and a bash smoke script."}]}]}},
        }
    ).encode()
    envelope_a, reason_a = convert_line(format_a, rel="dump/b.ndjson", offset=16, config=config, allowed_models=allowed)
    assert reason_a is None and envelope_a is not None
    assert envelope_a["source"]["uri"].endswith("dump/b.ndjson#16")
    assert envelope_a["content_sha256"]


def test_feeder_fails_closed_on_unknown_model(config):
    line = json.dumps(
        {
            "model": "unlisted-model",
            "input_body": json.dumps({"input": [{"content": [{"text": "A sufficiently long request text for the parser."}]}]}),
            "output_body": "data: " + json.dumps({"output": [{"content": [{"text": "A sufficiently long response text for the parser."}]}]}),
        }
    ).encode()
    envelope, reason = convert_line(line, rel="x.ndjson", offset=0, config=config, allowed_models=allowed_models(config))
    assert envelope is None and reason.startswith("model_not_allowlisted")


def test_fabricator_emits_valid_synthetic_canary(config):
    from random import Random

    record = build_record(Random(7), 1)
    assert record["synthetic"] is True and record["canary"] is True
    assert record["expected"]["task_type"] in config.tool_task.task_types
    assert record["expected"]["decision_choice"] in config.tool_task.decision_choices
    from jev.mq import build_envelope
    from zenjev_lib import sha256_hex

    envelope = build_envelope(
        record_id=record["record_id"],
        kind="tool_task_pair",
        source={
            "uri": f"synthetic://dashboard/{record['record_id']}",
            "content_sha256": sha256_hex(json.dumps(record, sort_keys=True)),
            "retrieved_at": record["created_at"],
            "license": "project-authored-synthetic",
        },
        schema_id=config.schema.name,
        schema_version=config.schema.version,
        schema_digest=config.schema.digest(),
        observed_model=record["observed_model"],
        request=record["request"],
        response=record["response"],
        synthetic=True,
        canary=True,
        expected=record["expected"],
    )
    assert validate_envelope(json.dumps(envelope).encode(), config, max_bytes=1024 * 1024) == envelope


def test_feeder_skips_failed_and_error_captures(config):
    allowed = allowed_models(config)
    base = {
        "model": "gpt-5.6-sol",
        "responseStatus": 413,
        "input_body": json.dumps(
            {"input": [{"content": [{"text": "A sufficiently long request text for the parser."}]}]}
        ),
        "output_body": "data: " + json.dumps(
            {"output": [{"content": [{"text": "body too large"}]}]}
        ),
    }
    envelope, reason = convert_line(
        json.dumps(base).encode(), rel="x.ndjson", offset=0, config=config, allowed_models=allowed
    )
    assert envelope is None and reason == "non_200_response:413"

    errored = {**base, "responseStatus": 200, "captureError": "truncated"}
    envelope, reason = convert_line(
        json.dumps(errored).encode(), rel="x.ndjson", offset=0, config=config, allowed_models=allowed
    )
    assert envelope is None and reason == "capture_error"

    healthy = {**base, "responseStatus": 200}
    envelope, reason = convert_line(
        json.dumps(healthy).encode(), rel="x.ndjson", offset=0, config=config, allowed_models=allowed
    )
    assert reason is None and envelope is not None
    assert envelope["response_status"] == 200 and envelope["capture_error"] is False


def test_soft_reasons_route_to_review_not_reject(config):
    from jev.tool_task import resolve_tool_task

    output = {
        "task_type": {"label": "coding", "confidence": 0.9},
        "entities": {"technology": [{"text": "SomeUnknownTech", "confidence": 0.6}]},
        "decision_choice": {"label": "Git", "confidence": 0.2},
    }
    decision = resolve_tool_task(output, config.tool_task, model_id="gpt-5.6-sol")
    assert decision["action"] == "review"
    assert "model_not_allowlisted" not in decision["reasons"]
    rejected = resolve_tool_task(output, config.tool_task, model_id="unknown-model")
    assert rejected["action"] == "reject"
    assert "model_not_allowlisted" in rejected["reasons"]


def test_calibrated_policy_allows_known_stack(config):
    from jev.tool_task import resolve_tool_task

    output = {
        "task_type": {"label": "coding", "confidence": 0.9},
        "entities": {
            "tool": [{"text": "Docker", "start": 0, "end": 6}],
            "technology": [{"text": "PyTorch", "confidence": 0.9}],
        },
        "decision_choice": {"label": "PyTorch", "confidence": 0.8},
    }
    decision = resolve_tool_task(output, config.tool_task, model_id="gpt-5.6-sol")
    assert decision["allowed"] is True and decision["action"] == "allow"


def test_shape_pair_target_uses_schema_entities_and_judgment(config):
    from zenjev_loop import shape_pair_target

    judgment = {
        "task_type": {"probabilities": {"coding": 0.8, "tool_use": 0.2}},
        "decision_choice": {"probabilities": {"PyTorch": 0.9, "Python": 0.6, "Docker": 0.1}},
    }
    extraction = {
        "entities": {
            "tool": [{"text": "Docker", "start": 0, "end": 6}],
            "unknown_entity": [{"text": "ignored", "start": 7, "end": 14}],
        }
    }
    target = shape_pair_target(config, judgment, extraction)
    assert target["entities"] == {"tool": ["Docker"]}
    tasks = {item["task"]: item["true_label"] for item in target["classifications"]}
    assert tasks["task_type"] == ["coding"]
    assert tasks["decision_choice"] == ["PyTorch", "Python"]


def test_shape_pair_target_rejects_empty_judgment(config):
    from zenjev_loop import shape_pair_target

    target = shape_pair_target(config, {}, {})
    assert target["entities"] == {} and target["classifications"] == []
