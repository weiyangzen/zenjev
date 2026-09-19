from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _load_document(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        value = json.loads(raw)
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            # Accept JSON as a YAML subset when the optional parser is absent.
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ConfigError("YAML config requires PyYAML (or use JSON)") from exc
        else:
            value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be an object")
    return value


@dataclass(frozen=True)
class SchemaField:
    name: str
    dtype: str = "str"
    description: str = ""
    required: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SchemaField":
        name = str(value.get("name", "")).strip()
        if not name or any(c in name for c in "\n\r\t"):
            raise ConfigError("schema field name must be non-empty and single-line")
        dtype = str(value.get("dtype", "str"))
        if dtype not in {"str", "int", "float", "bool", "list", "object"}:
            raise ConfigError(f"unsupported field dtype: {dtype}")
        return cls(name, dtype, str(value.get("description", "")), bool(value.get("required", False)))


@dataclass(frozen=True)
class UserSchema:
    name: str
    description: str
    entities: tuple[str, ...] = ()
    fields: tuple[SchemaField, ...] = ()
    classifications: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserSchema":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ConfigError("schema.name is required")
        entities = tuple(str(x).strip() for x in value.get("entities", []) if str(x).strip())
        fields = tuple(SchemaField.from_dict(x) for x in value.get("fields", []))
        raw_cls = value.get("classifications", {}) or {}
        if not isinstance(raw_cls, dict):
            raise ConfigError("schema.classifications must be an object")
        classifications = {str(k): tuple(str(x) for x in v) for k, v in raw_cls.items()}
        relations = tuple(str(x).strip() for x in value.get("relations", []) if str(x).strip())
        if not entities and not fields and not classifications and not relations:
            raise ConfigError("schema must define entities, fields, classifications or relations")
        return cls(name, str(value.get("description", "")), entities, fields, classifications, relations)

    def as_teacher_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "entities": list(self.entities),
            "fields": [vars(x) for x in self.fields],
            "classifications": {k: list(v) for k, v in self.classifications.items()},
            "relations": list(self.relations),
            "output_rule": "Return JSON only. Do not invent values; omit absent optional values.",
        }


@dataclass(frozen=True)
class SourceConfig:
    name: str
    kind: str
    location: str
    enabled: bool = True
    license_note: str = ""
    max_documents: int = 1000

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceConfig":
        name, kind, location = (str(value.get(k, "")).strip() for k in ("name", "kind", "location"))
        if not name or kind not in {"jsonl", "text", "url", "rss"} or not location:
            raise ConfigError("source requires name, kind=jsonl|text|url|rss, and location")
        limit = int(value.get("max_documents", 1000))
        if limit < 1 or limit > 1_000_000:
            raise ConfigError("source.max_documents must be in [1, 1000000]")
        return cls(name, kind, location, bool(value.get("enabled", True)), str(value.get("license_note", "")), limit)


@dataclass(frozen=True)
class TeacherConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    timeout_s: float = 60.0
    temperature: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherConfig":
        provider, model, base_url, api_key_env = (str(value.get(k, "")).strip() for k in ("provider", "model", "base_url", "api_key_env"))
        if not provider or not model or not base_url or not api_key_env:
            raise ConfigError("teacher.provider/model/base_url/api_key_env are required")
        timeout = float(value.get("timeout_s", 60))
        if timeout <= 0:
            raise ConfigError("teacher.timeout_s must be positive")
        return cls(provider, model, base_url.rstrip("/"), api_key_env, timeout, float(value.get("temperature", 0)))


@dataclass(frozen=True)
class DriftPolicy:
    eval_window: int = 256
    min_windows: int = 3
    loss_ratio: float = 2.0
    loss_margin: float = 0.0
    ema_norm_ratio: float = 3.0
    max_non_finite: int = 0
    reset_cooldown_steps: int = 100

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DriftPolicy":
        out = cls(**{k: value[k] for k in value if k in cls.__dataclass_fields__})
        if out.eval_window < 1 or out.min_windows < 1 or out.loss_ratio <= 1 or out.loss_margin < 0 or out.ema_norm_ratio <= 1:
            raise ConfigError("invalid drift policy thresholds")
        return out


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    inference_max_length: int = 512
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    publish_every_steps: int = 10
    seed: int = 42


@dataclass(frozen=True)
class JevConfig:
    model_id: str
    schema: UserSchema
    sources: tuple[SourceConfig, ...]
    teacher: TeacherConfig
    drift: DriftPolicy = field(default_factory=DriftPolicy)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JevConfig":
        model_id = str(value.get("model_id", "fastino/gliner2-base-v1")).strip()
        if not model_id:
            raise ConfigError("model_id is required")
        sources = tuple(SourceConfig.from_dict(x) for x in value.get("sources", []))
        if not sources:
            raise ConfigError("at least one configurable source is required")
        return cls(model_id, UserSchema.from_dict(value.get("schema", {})), sources, TeacherConfig.from_dict(value.get("teacher", {})), DriftPolicy.from_dict(value.get("drift", {})), RuntimeConfig(**{k: value.get("runtime", {}).get(k, v) for k, v in vars(RuntimeConfig()).items()}))

    def canonical(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "schema": self.schema.as_teacher_contract(), "sources": [vars(x) for x in self.sources], "teacher": vars(self.teacher), "drift": vars(self.drift), "runtime": vars(self.runtime)}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path: str | Path) -> JevConfig:
    return JevConfig.from_dict(_load_document(path))
