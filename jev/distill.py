from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import JevConfig
from .sources import SourceDocument, iter_source


@dataclass(frozen=True)
class DistilledExample:
    input: str
    output: dict[str, Any]
    source: dict[str, Any]
    teacher: dict[str, Any]
    schema_digest: str
    created_at: str

    def as_json(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("teacher response must be one strict JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    return value


class OpenAICompatibleTeacher:
    """Minimal OpenAI-compatible chat client; provider/model stay configurable."""

    def __init__(self, config: JevConfig):
        self.config = config
        self.last_metadata: dict[str, Any] = {}

    def label(self, document: SourceDocument) -> dict[str, Any]:
        key = os.environ.get(self.config.teacher.api_key_env)
        if not key:
            raise RuntimeError(f"missing teacher API key environment variable: {self.config.teacher.api_key_env}")
        prompt = {"schema": self.config.schema.as_teacher_contract(), "text": document.text}
        body = json.dumps({"model": self.config.teacher.model, "temperature": self.config.teacher.temperature, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": "Extract the requested schema. Return exactly one JSON object and no markdown."}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.config.teacher.base_url + "/chat/completions", body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=self.config.teacher.timeout_s) as response:
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]
        self.last_metadata = {
            "provider": self.config.teacher.provider,
            "model": self.config.teacher.model,
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "usage": payload.get("usage", {}),
        }
        return _json_object(content)


def _validate_output(config: JevConfig, output: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("teacher output must be a JSON object")
    allowed = set(config.schema.entities) | {field.name for field in config.schema.fields} | set(config.schema.classifications) | set(config.schema.relations) | {"_evidence", "schema_version"}
    unknown = set(output) - allowed
    if unknown:
        raise ValueError(f"teacher output contains unknown schema keys: {sorted(unknown)}")
    if "schema_version" in output and output["schema_version"] != config.schema.version:
        raise ValueError("teacher schema_version does not match configured schema")
    for field in config.schema.fields:
        if field.required and field.name not in output:
            raise ValueError(f"required schema field missing: {field.name}")
        if field.name in output and not _matches_dtype(output[field.name], field.dtype):
            raise ValueError(f"field {field.name!r} does not match dtype {field.dtype}")
        if field.name in output and field.choices:
            vals = output[field.name] if isinstance(output[field.name], list) else [output[field.name]]
            if any(x not in field.choices for x in vals):
                raise ValueError(f"field {field.name!r} contains a value outside choices")
    for entity in config.schema.entities:
        if entity in output and (not isinstance(output[entity], list) or any(not isinstance(x, str) for x in output[entity])):
            raise ValueError(f"entity {entity!r} must be a list of strings")
    for task, labels in config.schema.classifications.items():
        if task in output:
            vals = output[task] if isinstance(output[task], list) else [output[task]]
            if any(x not in labels for x in vals):
                raise ValueError(f"classification {task!r} contains an unknown label")
    for relation in config.schema.relations:
        if relation in output:
            values = output[relation]
            if not isinstance(values, list) or any(not isinstance(item, dict) or not isinstance(item.get("head"), str) or not isinstance(item.get("tail"), str) for item in values):
                raise ValueError(f"relation {relation!r} must be a list of {{head, tail}} objects")
    _validate_evidence(config, output)
    return _to_training_output(config, output)


def _matches_dtype(value: Any, dtype: str) -> bool:
    return {"str": isinstance(value, str), "int": isinstance(value, int) and not isinstance(value, bool), "float": isinstance(value, (int, float)) and not isinstance(value, bool), "bool": isinstance(value, bool), "list": isinstance(value, list)}[dtype]


def _validate_evidence(config: JevConfig, output: dict[str, Any]) -> None:
    evidence = output.get("_evidence", {})
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        raise ValueError("_evidence must be an object")
    for key, spans in evidence.items():
        if key not in output and key not in config.schema.classifications:
            raise ValueError(f"evidence key {key!r} is not a schema key")
        if not isinstance(spans, list):
            raise ValueError(f"evidence for {key!r} must be a list")
        for span in spans:
            if not isinstance(span, dict) or not isinstance(span.get("text"), str):
                raise ValueError("evidence span requires text/start/end")
            start, end = span.get("start"), span.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
                raise ValueError("evidence offsets must be nonnegative start <= end")


def _attach_or_validate_evidence(output: dict[str, Any], text: str) -> dict[str, Any]:
    evidence = output.get("_evidence")
    if evidence is None:
        evidence = {}
        for key, value in output.items():
            if key.startswith("_"):
                continue
            values = value if isinstance(value, list) else [value]
            spans = []
            for item in values:
                if not isinstance(item, str):
                    continue
                start = text.find(item)
                if start >= 0:
                    spans.append({"text": item, "start": start, "end": start + len(item)})
            if spans:
                evidence[key] = spans
        output = dict(output)
        output["_evidence"] = evidence
    else:
        for key, spans in evidence.items():
            for span in spans:
                if text[span["start"]:span["end"]] != span["text"]:
                    raise ValueError(f"evidence span for {key!r} does not match source text")
    return output


def _to_training_output(config: JevConfig, output: dict[str, Any]) -> dict[str, Any]:
    """Convert the teacher's schema-shaped object to GLiNER2 InputExample output."""
    result: dict[str, Any] = {}
    entities = {k: v for k, v in output.items() if k in config.schema.entities}
    if entities:
        result["entities"] = entities
        if config.schema.entity_descriptions:
            result["entity_descriptions"] = config.schema.entity_descriptions
    classifications = []
    for task, labels in config.schema.classifications.items():
        if task in output:
            value = output[task]
            classifications.append({"task": task, "labels": list(labels), "true_label": value if isinstance(value, list) else [value]})
    if classifications:
        result["classifications"] = classifications
    structures = {}
    for field in config.schema.fields:
        if field.name not in output:
            continue
        value = output[field.name]
        if isinstance(value, list):
            value = [str(item) if item is not None else item for item in value]
        elif not isinstance(value, str):
            value = str(value)
        structures[field.name] = value
    if structures:
        result["json_structures"] = [{config.schema.name: structures}]
        result["json_descriptions"] = {config.schema.name: {f.name: f.description for f in config.schema.fields if f.description}}
    relations = {k: output[k] for k in config.schema.relations if k in output}
    if relations:
        result["relations"] = [{name: value} for name, value in relations.items()]
    result["provenance"] = {"schema_digest": config.schema.digest(), "evidence": output.get("_evidence", {})}
    return result


def distill(config: JevConfig, output_path: str | Path, teacher: OpenAICompatibleTeacher | None = None) -> int:
    teacher = teacher or OpenAICompatibleTeacher(config)
    count = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for source in config.sources:
            for document in iter_source(source):
                output = _validate_output(config, _attach_or_validate_evidence(teacher.label(document), document.text))
                teacher_meta = getattr(teacher, "last_metadata", None) or {"provider": config.teacher.provider, "model": config.teacher.model}
                record = DistilledExample(document.text, output, {"name": document.source_name, "uri": document.uri, "license_note": document.license_note, "content_sha256": hashlib.sha256(document.text.encode()).hexdigest()}, teacher_meta, config.digest(), datetime.now(timezone.utc).isoformat())
                fh.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
                count += 1
    return count
