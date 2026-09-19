from __future__ import annotations

import hashlib
import json
import os
import re
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
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("teacher did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    return value


class OpenAICompatibleTeacher:
    """Minimal OpenAI-compatible chat client; provider/model stay configurable."""

    def __init__(self, config: JevConfig):
        self.config = config

    def label(self, document: SourceDocument) -> dict[str, Any]:
        key = os.environ.get(self.config.teacher.api_key_env)
        if not key:
            raise RuntimeError(f"missing teacher API key environment variable: {self.config.teacher.api_key_env}")
        prompt = {"schema": self.config.schema.as_teacher_contract(), "text": document.text}
        body = json.dumps({"model": self.config.teacher.model, "temperature": self.config.teacher.temperature, "messages": [{"role": "system", "content": "Extract the requested schema. Return JSON only."}, {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}]}, ensure_ascii=False).encode()
        request = urllib.request.Request(self.config.teacher.base_url + "/chat/completions", body, {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=self.config.teacher.timeout_s) as response:
            payload = json.loads(response.read().decode())
        content = payload["choices"][0]["message"]["content"]
        return _json_object(content)


def _validate_output(config: JevConfig, output: dict[str, Any]) -> dict[str, Any]:
    allowed = set(config.schema.entities) | {field.name for field in config.schema.fields} | set(config.schema.classifications) | set(config.schema.relations)
    unknown = set(output) - allowed
    if unknown:
        raise ValueError(f"teacher output contains unknown schema keys: {sorted(unknown)}")
    for field in config.schema.fields:
        if field.required and field.name not in output:
            raise ValueError(f"required schema field missing: {field.name}")
    return output


def distill(config: JevConfig, output_path: str | Path, teacher: OpenAICompatibleTeacher | None = None) -> int:
    teacher = teacher or OpenAICompatibleTeacher(config)
    count = 0
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for source in config.sources:
            for document in iter_source(source):
                output = _validate_output(config, teacher.label(document))
                record = DistilledExample(document.text, output, {"name": document.source_name, "uri": document.uri, "license_note": document.license_note, "content_sha256": hashlib.sha256(document.text.encode()).hexdigest()}, {"provider": config.teacher.provider, "model": config.teacher.model}, config.digest(), datetime.now(timezone.utc).isoformat())
                fh.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
                count += 1
    return count

