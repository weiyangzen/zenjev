from __future__ import annotations

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
    model = AutoExtractor.from_pretrained(config.model_id, **kwargs)
    return model


def apply_lora(model: Any, config: JevConfig) -> Any:
    if not hasattr(model, "apply_lora"):
        raise ModelUnavailable("installed GLiNER2 does not expose apply_lora")
    model.apply_lora(targets=["encoder"], r=config.runtime.lora_r, lora_alpha=config.runtime.lora_alpha, lora_dropout=config.runtime.lora_dropout)
    return model


def extract(model: Any, config: JevConfig, text: str) -> Any:
    if hasattr(model, "extract"):
        return model.extract(text, config.schema.as_teacher_contract())
    return model.extract_entities(text, list(config.schema.entities))

