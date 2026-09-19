"""Jev tool-task analysis and deterministic allowlist routing.

GLiNER2 supplies schema-conditioned labels, spans and confidence. This module
keeps the safety-critical decision outside the model: an extracted task type
can select a policy action only after model, tool and confidence checks pass.
"""

from __future__ import annotations

from typing import Any

from .config import JevConfig, ToolTaskPolicy


def classify_task_type(model: Any, text: str, policy: ToolTaskPolicy) -> dict[str, Any]:
    """Run GLiNER2's document classifier over the finite task-type labels."""
    try:
        from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier
    except ImportError as exc:
        raise RuntimeError("GLiNER2 classification support is unavailable") from exc
    schema = ClassificationSchema().single("task_type", list(policy.task_types), threshold=policy.min_confidence)
    classifier = Classifier(model)
    return classifier.classify(text, schema, config=ClassificationConfig(include_confidence=True))


def _classification_value(output: dict[str, Any], name: str) -> tuple[str | None, float | None]:
    value = output.get(name)
    if isinstance(value, dict):
        label = value.get("label")
        confidence = value.get("confidence")
        return (label if isinstance(label, str) else None, float(confidence) if isinstance(confidence, (int, float)) else None)
    if isinstance(value, str):
        return value, None
    for item in output.get("classifications", []):
        if isinstance(item, dict) and item.get("task") == name:
            labels = item.get("true_label", item.get("labels", []))
            if isinstance(labels, list) and labels and isinstance(labels[0], str):
                return labels[0], None
    return None, None


def _entity_values(output: dict[str, Any], name: str) -> list[str]:
    entities = output.get("entities", output)
    value = entities.get(name, []) if isinstance(entities, dict) else []
    if not isinstance(value, list):
        value = [value]
    values: list[str] = []
    for item in value:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            values.append(item["text"])
    return values


def _has_evidence(output: dict[str, Any]) -> bool:
    if output.get("_evidence") or (isinstance(output.get("provenance"), dict) and output["provenance"].get("evidence")):
        return True
    entities = output.get("entities")
    if isinstance(entities, dict):
        return any(isinstance(item, dict) and isinstance(item.get("start"), int) and isinstance(item.get("end"), int) for values in entities.values() if isinstance(values, list) for item in values)
    return False


def resolve_tool_task(
    output: dict[str, Any],
    policy: ToolTaskPolicy,
    *,
    model_id: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Resolve an extraction into an auditable policy result.

    The returned action is a policy label. This function never invokes a tool,
    model endpoint, shell command, or programming language runtime.
    """
    allowed_models = {item.model_id for item in policy.allowed_models}
    model_allowed = model_id in allowed_models or any(model_id in item.aliases for item in policy.allowed_models)
    task_type, extracted_confidence = _classification_value(output, "task_type")
    score = extracted_confidence if extracted_confidence is not None else confidence
    reasons: list[str] = []
    if not model_allowed:
        reasons.append("model_not_allowlisted")
    if task_type is None or task_type not in policy.task_types:
        reasons.append("task_type_unknown")
    if score is not None and score < policy.min_confidence:
        reasons.append("confidence_below_threshold")
    if policy.require_evidence and not _has_evidence(output):
        reasons.append("evidence_missing")
    for field_name, allowed in (("tool", policy.allowed_tools), ("programming_language", policy.allowed_programming_languages), ("technology", policy.allowed_technologies)):
        if allowed:
            unknown = set(_entity_values(output, field_name)) - set(allowed)
            if unknown:
                reasons.append(f"{field_name}_not_allowlisted")
    action = policy.task_actions.get(task_type or "")
    if action is None:
        action = "review" if policy.unknown_policy == "unmapped" else "reject"
    if reasons:
        action = "review" if "confidence_below_threshold" in reasons and "model_not_allowlisted" not in reasons else "reject"
    return {
        "policy": policy.name,
        "policy_version": policy.version,
        "allowed": not reasons and action == "allow",
        "action": action,
        "task_type": task_type,
        "model_id": model_id,
        "confidence": score,
        "reasons": reasons,
    }


def analyze_tool_task(model: Any, config: JevConfig, request: str, response: str, *, model_id: str) -> dict[str, Any]:
    """Extract request and response separately, then apply the policy router."""
    if config.tool_task is None:
        raise ValueError("config.tool_task is required for tool-task analysis")
    from .model import extract

    request_output = extract(model, config, request)
    response_output = extract(model, config, response)
    return {
        "request": request_output,
        "response": response_output,
        "decision": resolve_tool_task(request_output, config.tool_task, model_id=model_id),
        "schema_digest": config.schema.digest(),
        "policy_digest": config.digest(),
    }
