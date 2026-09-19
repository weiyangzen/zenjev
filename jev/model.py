from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from .config import JevConfig


class ModelUnavailable(RuntimeError):
    pass


def load_gliner(config: JevConfig, manifest_path: str | Path | None = None) -> Any:
    """Load GLiNER2 and apply the current adapter when optional ML deps exist."""
    try:
        from gliner2 import AutoExtractor
    except ImportError as exc:
        raise ModelUnavailable("install jev[runtime] on an NVIDIA GPU host") from exc
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
    _validate_model_manifest(config, model, manifest_path)
    return model


def _validate_model_manifest(config: JevConfig, model: Any, manifest_path: str | Path | None = None) -> None:
    """Fail closed when a staged checkpoint is not the declared base model."""
    candidate = manifest_path or os.environ.get("JEV_MODEL_MANIFEST")
    if candidate is None:
        default = Path(__file__).resolve().parents[1] / "artifacts/model_manifest.json"
        candidate = default if default.exists() else None
    if candidate is None:
        return
    try:
        manifest = json.loads(Path(candidate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailable(f"cannot read model manifest: {candidate}") from exc
    if manifest.get("model_id") != config.model_id or manifest.get("revision") != config.model_revision:
        raise ModelUnavailable("loaded model does not match the configured model manifest")
    if hasattr(model, "parameters"):
        actual = sum(parameter.numel() for parameter in model.parameters())
        expected = int(manifest.get("parameter_count", 0))
        tolerance = float(manifest.get("parameter_count_tolerance", 0.05))
        if expected <= 0 or abs(actual - expected) > expected * tolerance:
            raise ModelUnavailable(f"model parameter count {actual} is outside manifest tolerance")
        if "gliner2" not in model.__class__.__module__.lower():
            raise ModelUnavailable("loaded model class is not a GLiNER2 extractor")


def apply_lora(model: Any, config: JevConfig) -> Any:
    if not hasattr(model, "apply_lora"):
        raise ModelUnavailable("installed GLiNER2 does not expose apply_lora")
    wrapped = model.apply_lora(targets=["encoder"], r=config.runtime.lora_r, alpha=config.runtime.lora_alpha, dropout=config.runtime.lora_dropout)
    if hasattr(wrapped, "named_parameters"):
        trainable = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
        if not trainable or any("lora_" not in name.lower() for name in trainable):
            raise ModelUnavailable("LoRA application left non-adapter parameters trainable")
    return wrapped


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


def gliner2_step(model: Any, config: JevConfig, example: dict[str, Any]) -> Any:
    """Compute the official GLiNER2 supervised loss for one distilled record.

    ``distill`` emits the GLiNER2 ``{"input", "output"}`` JSONL contract. The
    official processor/collator turns that contract into a training batch and
    the model's ``total_loss`` covers entity, classification, structure, and
    relation tasks. The imports stay lazy so the control plane remains usable
    without PyTorch or GLiNER2 installed.
    """
    try:
        from gliner2.training import ExtractorCollator
    except ImportError as exc:
        raise ModelUnavailable("install jev[train] for the GLiNER2 training loss") from exc
    if not isinstance(example, dict):
        raise TypeError("training example must be a JSON object")
    text = example.get("input", example.get("text"))
    output = example.get("output", example.get("schema"))
    if not isinstance(text, str) or not text.strip() or not isinstance(output, dict):
        raise ValueError("training example requires string input and object output")
    processor = getattr(model, "processor", None)
    if processor is None:
        raise ModelUnavailable("GLiNER2 model does not expose its training processor")
    collator = ExtractorCollator(
        processor,
        is_training=True,
        max_len=config.runtime.inference_max_length,
        architecture="span",
        on_capacity_exceeded="raise",
    )
    batch = collator([(text, output)])
    result = model(batch)
    if not isinstance(result, dict) or result.get("total_loss") is None:
        raise RuntimeError("GLiNER2 forward did not return total_loss")
    return result["total_loss"]


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
