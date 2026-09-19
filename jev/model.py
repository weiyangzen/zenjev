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
                builder.field(field.name, dtype="list" if field.dtype == "list" else "str", choices=list(field.choices) or None, description=field.description)
        if config.schema.relations:
            schema.relations(list(config.schema.relations))
        return model.extract(text, schema, max_len=config.runtime.inference_max_length, include_spans=True, include_confidence=True)
    return model.extract_entities(text, list(config.schema.entities))
