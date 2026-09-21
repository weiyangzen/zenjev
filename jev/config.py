from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


MODEL_REVISIONS = {
    "fastino/gliner2-base-v1": "79c3a777abc572b4767922f3916cf63fb5754df2",
    "fastino/gliner2-multi-v1": "edd4f6efd8611a4c29632fa8034b181ee8da9ebc",
}


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
    aliases: tuple[str, ...] = ()
    choices: tuple[Any, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SchemaField":
        name = str(value.get("name", "")).strip()
        if not name or any(c in name for c in "\n\r\t"):
            raise ConfigError("schema field name must be non-empty and single-line")
        dtype = str(value.get("dtype", "str"))
        if dtype not in {"str", "int", "float", "bool", "list"}:
            raise ConfigError(f"unsupported field dtype: {dtype}")
        aliases = tuple(str(x).strip() for x in value.get("aliases", []) if str(x).strip())
        choices = tuple(value.get("choices", []))
        if any(x is None or (isinstance(x, str) and not x) for x in choices):
            raise ConfigError("field choices must be nonempty values")
        return cls(name, dtype, str(value.get("description", "")), bool(value.get("required", False)), aliases, choices)


@dataclass(frozen=True)
class UserSchema:
    name: str
    description: str
    entities: tuple[str, ...] = ()
    fields: tuple[SchemaField, ...] = ()
    classifications: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relations: tuple[str, ...] = ()
    version: int = 1
    language: str = "auto"
    unknown_label_policy: str = "reject"
    entity_descriptions: dict[str, str] = field(default_factory=dict)
    word_splitter: str = "whitespace"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UserSchema":
        name = str(value.get("name", "")).strip()
        if not name:
            raise ConfigError("schema.name is required")
        raw_entities = value.get("entities", [])
        if not isinstance(raw_entities, (list, dict)):
            raise ConfigError("schema.entities must be a list or description mapping")
        entities = tuple(str(x).strip() for x in raw_entities if str(x).strip())
        descriptions = {str(k): str(v) for k, v in raw_entities.items()} if isinstance(raw_entities, dict) else {}
        explicit_descriptions = value.get("entity_descriptions", {}) or {}
        if not isinstance(explicit_descriptions, dict):
            raise ConfigError("schema.entity_descriptions must be an object")
        descriptions.update({str(k): str(v) for k, v in explicit_descriptions.items()})
        fields = tuple(SchemaField.from_dict(x) for x in value.get("fields", []))
        raw_cls = value.get("classifications", {}) or {}
        if not isinstance(raw_cls, dict):
            raise ConfigError("schema.classifications must be an object")
        if any(not isinstance(v, list) or not v or any(not isinstance(x, str) or not x for x in v) for v in raw_cls.values()):
            raise ConfigError("classification labels must be nonempty string lists")
        classifications = {str(k): tuple(v) for k, v in raw_cls.items()}
        relations = tuple(str(x).strip() for x in value.get("relations", []) if str(x).strip())
        if not entities and not fields and not classifications and not relations:
            raise ConfigError("schema must define entities, fields, classifications or relations")
        version = int(value.get("version", 1))
        language = str(value.get("language", "auto"))
        unknown = str(value.get("unknown_label_policy", "reject"))
        if version < 1 or unknown not in {"reject", "unmapped"}:
            raise ConfigError("schema.version must be positive and unknown_label_policy must be reject|unmapped")
        labels = list(entities) + [x.name for x in fields] + list(classifications) + list(relations)
        if len(labels) != len(set(labels)) or "_evidence" in labels:
            raise ConfigError("schema labels must be unique across tasks and may not be _evidence")
        if set(descriptions) - set(entities):
            raise ConfigError("schema.entity_descriptions contains an unknown entity")
        splitter = str(value.get("word_splitter", "whitespace"))
        if splitter not in {"whitespace", "char"}:
            raise ConfigError("schema.word_splitter must be whitespace|char")
        return cls(name, str(value.get("description", "")), entities, fields, classifications, relations, version, language, unknown, descriptions, splitter)

    def as_teacher_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "language": self.language,
            "unknown_label_policy": self.unknown_label_policy,
            "description": self.description,
            "entities": list(self.entities),
            "entity_descriptions": self.entity_descriptions,
            "word_splitter": self.word_splitter,
            "fields": [vars(x) for x in self.fields],
            "classifications": {k: list(v) for k, v in self.classifications.items()},
            "relations": list(self.relations),
            "output_rule": "Return one strict JSON object with flat schema keys. Entities are lists of exact source substrings; fields use their declared types; classification is one allowed label; relations are lists of {head, tail} exact source substrings. Include _evidence mapping classification/field keys to lists of {text, start, end} with exact exclusive-end source offsets. Do not invent values or include unknown keys; omit absent optional values.",
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_teacher_contract(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


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
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    cache_dir: str | None = "runs/jev/teacher-cache"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherConfig":
        provider, model, base_url, api_key_env = (str(value.get(k, "")).strip() for k in ("provider", "model", "base_url", "api_key_env"))
        if not provider or not model or not base_url or not api_key_env:
            raise ConfigError("teacher.provider/model/base_url/api_key_env are required")
        timeout = float(value.get("timeout_s", 60))
        if timeout <= 0:
            raise ConfigError("teacher.timeout_s must be positive")
        retries = int(value.get("max_retries", 2))
        backoff = float(value.get("retry_backoff_s", 1.0))
        if retries < 0 or backoff < 0:
            raise ConfigError("teacher retries/backoff must be nonnegative")
        cache = value.get("cache_dir", "runs/jev/teacher-cache")
        cache = None if cache is None else str(cache)
        return cls(provider, model, base_url.rstrip("/"), api_key_env, timeout, float(value.get("temperature", 0)), retries, backoff, cache)


@dataclass(frozen=True)
class AllowedModel:
    """A model route that the tool-task policy is allowed to select."""

    model_id: str
    provider: str
    aliases: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AllowedModel":
        model_id = str(value.get("id", value.get("model_id", ""))).strip()
        provider = str(value.get("provider", "")).strip()
        aliases = tuple(str(item).strip() for item in value.get("aliases", []) if str(item).strip())
        if not model_id or not provider:
            raise ConfigError("tool_task.allowed_models entries require id and provider")
        return cls(model_id, provider, aliases)


@dataclass(frozen=True)
class ToolTaskPolicy:
    """Standard Jev tool-task extraction and deterministic routing policy."""

    name: str = "jev-tool-task"
    version: int = 1
    task_types: tuple[str, ...] = ()
    task_actions: dict[str, str] = field(default_factory=dict)
    allowed_models: tuple[AllowedModel, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    allowed_programming_languages: tuple[str, ...] = ()
    allowed_technologies: tuple[str, ...] = ()
    # Finite technology-stack labels used by the decision classifier.  Keeping
    # this list in policy makes the output auditable and prevents free-form
    # model text from becoming an executable route.
    decision_choices: tuple[str, ...] = ()
    decision_actions: tuple[str, ...] = ("allow", "review", "reject")
    min_confidence: float = 0.75
    unknown_policy: str = "reject"
    require_evidence: bool = True

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ToolTaskPolicy | None":
        if not value:
            return None
        name = str(value.get("name", "jev-tool-task")).strip()
        version = int(value.get("version", 1))
        task_types = tuple(str(item).strip() for item in value.get("task_types", []) if str(item).strip())
        actions = {str(key).strip(): str(item).strip() for key, item in (value.get("task_actions", {}) or {}).items()}
        models = tuple(AllowedModel.from_dict(item) for item in value.get("allowed_models", []))
        tools = tuple(str(item).strip() for item in value.get("allowed_tools", []) if str(item).strip())
        languages = tuple(str(item).strip() for item in value.get("allowed_programming_languages", []) if str(item).strip())
        technologies = tuple(str(item).strip() for item in value.get("allowed_technologies", []) if str(item).strip())
        raw_choices = value.get("decision_choices")
        if raw_choices is None:
            raw_choices = [*technologies, *tools, *languages]
        decision_choices = tuple(str(item).strip() for item in raw_choices if str(item).strip())
        decision_actions = tuple(str(item).strip() for item in value.get("decision_actions", ["allow", "review", "reject"]) if str(item).strip())
        confidence = float(value.get("min_confidence", 0.75))
        unknown = str(value.get("unknown_policy", "reject"))
        if not name or version < 1 or not task_types or not models:
            raise ConfigError("tool_task requires name, positive version, task_types and allowed_models")
        if len(set(task_types)) != len(task_types) or len({model.model_id for model in models}) != len(models):
            raise ConfigError("tool_task task types and allowed model IDs must be unique")
        if set(actions) - set(task_types) or any(action not in decision_actions for action in actions.values()):
            raise ConfigError("tool_task.task_actions must map known task types to decision actions")
        if not 0 <= confidence <= 1 or unknown not in {"reject", "unmapped"}:
            raise ConfigError("tool_task min_confidence must be in [0,1] and unknown_policy reject|unmapped")
        if len(set(decision_actions)) != len(decision_actions) or not {"allow", "reject"}.issubset(decision_actions):
            raise ConfigError("tool_task.decision_actions must include allow and reject")
        if len(set(decision_choices)) != len(decision_choices):
            raise ConfigError("tool_task.decision_choices must be unique")
        return cls(name, version, task_types, actions, models, tools, languages, technologies, decision_choices, decision_actions, confidence, unknown, bool(value.get("require_evidence", True)))

    def as_contract(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task_types": list(self.task_types),
            "task_actions": self.task_actions,
            "allowed_models": [{"id": item.model_id, "provider": item.provider, "aliases": list(item.aliases)} for item in self.allowed_models],
            "allowed_tools": list(self.allowed_tools),
            "allowed_programming_languages": list(self.allowed_programming_languages),
            "allowed_technologies": list(self.allowed_technologies),
            "decision_choices": list(self.decision_choices),
            "decision_actions": list(self.decision_actions),
            "min_confidence": self.min_confidence,
            "unknown_policy": self.unknown_policy,
            "require_evidence": self.require_evidence,
        }


MQ_ADAPTERS = ("nats-jetstream", "kafka", "iggy", "dir-spool", "mock")
MQ_START_POSITIONS = ("new", "first", "last", "by_offset", "by_timestamp")


def _remote_endpoint(endpoint: str) -> bool:
    host = endpoint.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0].strip("[]").lower()
    return host not in {"", "localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class MqConfig:
    """Configurable external-MQ ingestion contract (§1.5). Values, never secrets."""

    enabled: bool = False
    adapter: str = "nats-jetstream"
    endpoints: tuple[str, ...] = ()
    stream: str = "JEV_RECORDS"
    subject: str = "jev.records"
    durable_name: str = "jev-trainer"
    consumer_group: str | None = None
    start_position: str = "new"
    start_offset: int | None = None
    start_timestamp: int | None = None
    batch: int = 64
    max_bytes: int = 33_554_432
    ack_wait_seconds: int = 30
    max_ack_pending: int = 16
    prefetch: int = 32
    max_deliver: int = 5
    dlq_subject: str = "jev.dlq"
    dlq_path: str = "runs/jev/mq/dlq.jsonl"
    quarantine_path: str = "runs/jev/mq/quarantine.jsonl"
    wal_path: str = "runs/jev/mq/wal.bin"
    socket_path: str = "runs/jev/mq/bridge.sock"
    bridge_config: str = "runs/jev/mq/bridge.json"
    metrics_path: str = "artifacts/mq/metrics.json"
    replay_buffer: str = "runs/jev/mq/replay-buffer.jsonl"
    dedup_window: int = 10_000
    # Infinite testing mode (mock adapter): replay the same dataset head-to-tail
    # with a distinct loop_cycle per pass; max_cycles=0 means unbounded.
    loop: bool = False
    max_cycles: int = 0
    tls_required: bool = True
    ca_cert: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    secret_ref: str | None = None
    source_path: str | None = None
    state_path: str | None = None
    # Local directory spool (`dir-spool` adapter, §1.7): a perpetual,
    # byte-offset-watermarked NDJSON source that needs no broker.
    source_root: str | None = None
    patterns: tuple[str, ...] = ("*.ndjson",)
    poll_interval_ms: int = 2000
    exit_after_drain: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "MqConfig | None":
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ConfigError("mq must be an object")
        out = cls(**{key: value[key] for key in value if key in cls.__dataclass_fields__})
        if isinstance(out.patterns, list):
            out = replace(out, patterns=tuple(out.patterns))
        if out.adapter not in MQ_ADAPTERS:
            raise ConfigError(f"mq.adapter must be one of {MQ_ADAPTERS}")
        if out.start_position not in MQ_START_POSITIONS:
            raise ConfigError(f"mq.start_position must be one of {MQ_START_POSITIONS}")
        if out.start_position == "by_offset" and out.start_offset is None:
            raise ConfigError("mq.start_position=by_offset requires start_offset")
        if out.start_position == "by_timestamp" and out.start_timestamp is None:
            raise ConfigError("mq.start_position=by_timestamp requires start_timestamp")
        positive = {
            "batch": out.batch,
            "max_bytes": out.max_bytes,
            "ack_wait_seconds": out.ack_wait_seconds,
            "max_ack_pending": out.max_ack_pending,
            "prefetch": out.prefetch,
            "max_deliver": out.max_deliver,
            "dedup_window": out.dedup_window,
            "poll_interval_ms": out.poll_interval_ms,
        }
        if any(number <= 0 for number in positive.values()):
            raise ConfigError("mq batch/max_bytes/ack_wait/max_ack_pending/prefetch/max_deliver/dedup_window must be positive")
        if out.max_cycles < 0:
            raise ConfigError("mq.max_cycles must be nonnegative")
        if out.secret_ref is not None and not (out.secret_ref.startswith("env:") or out.secret_ref.startswith("file:")):
            raise ConfigError("mq.secret_ref must be an env:/file: reference, never a secret value")
        if out.client_cert and not out.client_key or out.client_key and not out.client_cert:
            raise ConfigError("mq.client_cert and mq.client_key must be provided together")
        paths = (out.wal_path, out.socket_path, out.quarantine_path, out.replay_buffer, out.bridge_config)
        if any(not path for path in paths):
            raise ConfigError("mq paths must be non-empty")
        if not out.enabled:
            return out
        if out.adapter == "mock":
            if not out.source_path:
                raise ConfigError("mq.adapter=mock requires source_path (test/smoke use only)")
        elif out.adapter == "dir-spool":
            if not out.source_root:
                raise ConfigError("mq.adapter=dir-spool requires source_root")
            if not out.patterns:
                raise ConfigError("mq.adapter=dir-spool requires at least one pattern")
        else:
            if not out.endpoints:
                raise ConfigError(f"mq.adapter={out.adapter} requires at least one endpoint")
            if any(_remote_endpoint(endpoint) and not out.tls_required for endpoint in out.endpoints):
                raise ConfigError("non-TLS remote MQ endpoints fail closed")
        if out.adapter == "nats-jetstream" and (not out.stream or not out.durable_name):
            raise ConfigError("mq.adapter=nats-jetstream requires stream and durable_name")
        if out.adapter == "kafka" and not out.consumer_group:
            raise ConfigError("mq.adapter=kafka requires consumer_group")
        return out

    def as_contract(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "adapter": self.adapter,
            "endpoints": list(self.endpoints),
            "stream": self.stream,
            "subject": self.subject,
            "durable_name": self.durable_name,
            "consumer_group": self.consumer_group,
            "start_position": self.start_position,
            "start_offset": self.start_offset,
            "start_timestamp": self.start_timestamp,
            "batch": self.batch,
            "max_bytes": self.max_bytes,
            "ack_wait_seconds": self.ack_wait_seconds,
            "max_ack_pending": self.max_ack_pending,
            "prefetch": self.prefetch,
            "max_deliver": self.max_deliver,
            "dlq_subject": self.dlq_subject,
            "tls_required": self.tls_required,
            "secret_ref": self.secret_ref,
            "dedup_window": self.dedup_window,
            "loop": self.loop,
            "source_root": self.source_root,
            "patterns": list(self.patterns),
            "poll_interval_ms": self.poll_interval_ms,
            "max_cycles": self.max_cycles,
        }


@dataclass(frozen=True)
class DriftPolicy:
    eval_window: int = 256
    min_windows: int = 3
    loss_ratio: float = 2.0
    loss_margin: float = 0.0
    ema_norm_ratio: float = 3.0
    max_non_finite: int = 0
    reset_cooldown_steps: int = 100
    f1_absolute_drop: float = 0.15
    f1_relative_floor: float = 0.80
    validation_loss_ratio: float = 2.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DriftPolicy":
        out = cls(**{k: value[k] for k in value if k in cls.__dataclass_fields__})
        if (out.eval_window < 1 or out.min_windows < 1 or out.loss_ratio <= 1 or out.loss_margin < 0
                or out.ema_norm_ratio <= 1 or out.max_non_finite < 0
                or out.reset_cooldown_steps < 0 or out.f1_absolute_drop < 0
                or not 0 < out.f1_relative_floor <= 1 or out.validation_loss_ratio <= 1):
            raise ConfigError("invalid drift policy thresholds")
        return out


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    # Optional local snapshot path. JEV_MODEL_PATH overrides this for hosts
    # where the pinned Hub revision is staged outside the repository.
    model_path: str | None = None
    inference_max_length: int = 512
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    # EMA and optimizer safeguards are explicit runtime policy.  Keeping them
    # in the config makes a checkpoint reproducible and prevents a host-local
    # default from silently changing an adapter lineage.
    ema_decay: float = 0.999
    learning_rate: float = 1e-4
    grad_clip_norm: float | None = 1.0
    publish_every_steps: int = 10
    checkpoint_every_steps: int = 10
    max_checkpoints: int = 3
    reset_warmup_steps: int = 0
    seed: int = 42


@dataclass(frozen=True)
class JevConfig:
    model_id: str
    schema: UserSchema
    sources: tuple[SourceConfig, ...]
    teacher: TeacherConfig
    drift: DriftPolicy = field(default_factory=DriftPolicy)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model_revision: str = MODEL_REVISIONS["fastino/gliner2-base-v1"]
    tool_task: ToolTaskPolicy | None = None
    mq: MqConfig | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "JevConfig":
        model_id = str(value.get("model_id", "fastino/gliner2-base-v1")).strip()
        if not model_id:
            raise ConfigError("model_id is required")
        sources = tuple(SourceConfig.from_dict(x) for x in value.get("sources", []))
        if not sources:
            raise ConfigError("at least one configurable source is required")
        revision = str(value.get("model_revision", MODEL_REVISIONS.get(model_id, "")))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ConfigError("model_revision must be a frozen 40-character Hub commit SHA")
        runtime = RuntimeConfig(**{k: value.get("runtime", {}).get(k, v) for k, v in vars(RuntimeConfig()).items()})
        if not 0 < float(runtime.ema_decay) < 1:
            raise ConfigError("runtime.ema_decay must be between 0 and 1")
        if runtime.learning_rate <= 0:
            raise ConfigError("runtime.learning_rate must be positive")
        if runtime.grad_clip_norm is not None and runtime.grad_clip_norm <= 0:
            raise ConfigError("runtime.grad_clip_norm must be positive or null")
        if runtime.publish_every_steps < 1 or runtime.checkpoint_every_steps < 1:
            raise ConfigError("runtime publish/checkpoint cadence must be positive")
        if runtime.max_checkpoints < 1 or runtime.reset_warmup_steps < 0:
            raise ConfigError("runtime max_checkpoints must be positive and reset_warmup_steps nonnegative")
        return cls(model_id, UserSchema.from_dict(value.get("schema", {})), sources, TeacherConfig.from_dict(value.get("teacher", {})), DriftPolicy.from_dict(value.get("drift", {})), runtime, revision, ToolTaskPolicy.from_dict(value.get("tool_task")), MqConfig.from_dict(value.get("mq")))

    def canonical(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "model_revision": self.model_revision, "schema": self.schema.as_teacher_contract(), "sources": [vars(x) for x in self.sources], "teacher": vars(self.teacher), "drift": vars(self.drift), "runtime": vars(self.runtime), "tool_task": self.tool_task.as_contract() if self.tool_task else None, "mq": self.mq.as_contract() if self.mq else None}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path: str | Path) -> JevConfig:
    return JevConfig.from_dict(_load_document(path))
