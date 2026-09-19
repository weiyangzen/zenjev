from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import JevConfig


class ModelUnavailable(RuntimeError):
    pass


def load_gliner(config: JevConfig) -> Any:
    """Load GLiNER2 and apply the current adapter when optional ML deps exist."""
    try:
        from gliner2 import AutoExtractor
    except ImportError as exc:
        raise ModelUnavailable("install jev[runtime] on the 5090 host") from exc
    kwargs: dict[str, Any] = {}
    if config.runtime.device != "auto":
        kwargs["map_location"] = config.runtime.device
    kwargs["word_splitter"] = config.schema.word_splitter
    model_source = os.environ.get("JEV_MODEL_PATH") or config.runtime.model_path or config.model_id
    # A local staged snapshot is already immutable; passing a Hub revision to a
    # filesystem path is rejected by some Transformers versions.
    if not Path(model_source).is_dir():
        kwargs["revision"] = config.model_revision
    model = AutoExtractor.from_pretrained(model_source, **kwargs)
    return model


def apply_lora(model: Any, config: JevConfig) -> Any:
    if not hasattr(model, "apply_lora"):
        raise ModelUnavailable("installed GLiNER2 does not expose apply_lora")
    return model.apply_lora(targets=["encoder"], r=config.runtime.lora_r, alpha=config.runtime.lora_alpha, dropout=config.runtime.lora_dropout)


def extract(model: Any, config: JevConfig, text: str) -> Any:
    if hasattr(model, "extract"):
        try:
            from gliner2 import Schema
        except ImportError as exc:
            raise ModelUnavailable("GLiNER2 Schema is unavailable") from exc
        schema = Schema()
        if config.schema.entities:
            schema.entities({name: config.schema.entity_descriptions.get(name, "") for name in config.schema.entities})
        for task, labels in config.schema.classifications.items():
            schema.classification(task, list(labels))
        if config.schema.fields:
            builder = schema.structure(config.schema.name)
            for field in config.schema.fields:
                # GLiNER2's decoder accepts only str/list. Preserve the
                # user's richer type contract at the control plane and cast
                # decoded structure values below.
                builder.field(field.name, dtype="list" if field.dtype == "list" else "str",
                              choices=[str(choice) for choice in field.choices] or None,
                              description=field.description)
        if config.schema.relations:
            schema.relations(list(config.schema.relations))
        result = model.extract(text, schema, max_len=config.runtime.inference_max_length, include_spans=True, include_confidence=True)
        return _normalize_structure_types(result, config)
    return model.extract_entities(text, list(config.schema.entities))


def _normalize_structure_types(result: Any, config: JevConfig) -> Any:
    """Cast GLiNER2's str/list decoder output to the configured field types."""
    if not isinstance(result, dict) or not config.schema.fields:
        return result
    records = result.get(config.schema.name)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return result
    fields = {field.name: field.dtype for field in config.schema.fields}
    for record in records:
        if not isinstance(record, dict):
            continue
        for name, dtype in fields.items():
            if name not in record or record[name] is None:
                continue
            value = record[name]
            try:
                if dtype == "int" and isinstance(value, str):
                    record[name] = int(value)
                elif dtype == "float" and isinstance(value, str):
                    record[name] = float(value)
                elif dtype == "bool" and isinstance(value, str) and value.lower() in {"true", "false"}:
                    record[name] = value.lower() == "true"
                elif dtype == "list" and not isinstance(value, list):
                    record[name] = [value]
            except (TypeError, ValueError):
                # Keep an unparseable decoder value visible for validation and
                # caller-side handling instead of silently inventing a value.
                pass
    return result
