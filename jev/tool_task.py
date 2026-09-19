"""ZenJev tool-task analysis and deterministic allowlist routing.

GLiNER2 supplies schema-conditioned labels, spans and confidence. This module
keeps the safety-critical decision outside the model: an extracted task type
can select a policy action only after model, tool and confidence checks pass.
"""

from __future__ import annotations

from typing import Any, Iterable

from .config import JevConfig, ToolTaskPolicy


def _as_classifier(model: Any, classifier_type: Any) -> Any:
    """Accept either a GLiNER2 base model or an already-built Classifier."""
    return model if hasattr(model, "classify") else classifier_type(model)


def classify_task_type(model: Any, text: str, policy: ToolTaskPolicy) -> dict[str, Any]:
    """Run GLiNER2's document classifier over the finite task-type labels."""
    try:
        from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier
    except ImportError as exc:
        raise RuntimeError("GLiNER2 classification support is unavailable") from exc
    schema = ClassificationSchema().single("task_type", list(policy.task_types), threshold=policy.min_confidence)
    classifier = _as_classifier(model, Classifier)
    result = classifier.classify(text, schema, config=ClassificationConfig(include_confidence=True))
    return result.to_dict() if hasattr(result, "to_dict") else result


def classify_decision_choices(model: Any, text: str, policy: ToolTaskPolicy) -> dict[str, Any]:
    """Predict a finite technology-stack choice list with probabilities.

    GLiNER2 is constrained to the policy's labels.  The classifier returns a
    probability map for every candidate; callers can preserve that map as an
    auditable choice list instead of treating one free-form model string as a
    decision.  This function never invokes a tool or model endpoint.
    """
    if not policy.decision_choices:
        return {"decision_choice": {"value": [], "confidence": None, "probabilities": {}}}
    try:
        from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier
    except ImportError as exc:
        raise RuntimeError("GLiNER2 classification support is unavailable") from exc
    schema = ClassificationSchema().multi(
        "decision_choice",
        list(policy.decision_choices),
        min_labels=0,
        max_labels=len(policy.decision_choices),
        threshold=policy.min_confidence,
    )
    classifier = _as_classifier(model, Classifier)
    result = classifier.classify(text, schema, config=ClassificationConfig(include_confidence=True))
    return result.to_dict() if hasattr(result, "to_dict") else result


def classify_tool_task(model: Any, text: str, policy: ToolTaskPolicy) -> dict[str, Any]:
    """Return task type plus a probability-bearing technology choice list."""
    output = classify_task_type(model, text, policy)
    choices = classify_decision_choices(model, text, policy)
    output.update(choices)
    return output


def _classification_value(output: dict[str, Any], name: str) -> tuple[str | None, float | None]:
    value = output.get(name)
    if isinstance(value, dict):
        label = value.get("label", value.get("value"))
        if isinstance(label, list):
            label = label[0] if label else None
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


def _probability_choices(output: dict[str, Any], names: Iterable[str]) -> list[dict[str, Any]]:
    """Normalize GLiNER2 probability maps to sorted ``choice`` records."""
    records: dict[str, float] = {}
    for name in names:
        value = output.get(name)
        if isinstance(value, dict):
            probabilities = value.get("probabilities", {})
            if isinstance(probabilities, dict):
                for choice, probability in probabilities.items():
                    if isinstance(choice, str) and isinstance(probability, (int, float)):
                        records[choice] = max(0.0, min(1.0, float(probability)))
            selected = value.get("value", value.get("label"))
            if isinstance(selected, str) and selected not in records and isinstance(value.get("confidence"), (int, float)):
                records[selected] = max(0.0, min(1.0, float(value["confidence"])))
            elif isinstance(selected, list) and isinstance(value.get("confidence"), (int, float)):
                for choice in selected:
                    if isinstance(choice, str) and choice not in records:
                        records[choice] = max(0.0, min(1.0, float(value["confidence"])))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("choice"), str) and isinstance(item.get("probability"), (int, float)):
                    records[item["choice"]] = max(0.0, min(1.0, float(item["probability"])))
    # Extraction schemas can expose the stack as entity spans rather than a
    # classifier task.  Preserve those confidence scores in the same stable
    # choice contract when no explicit decision task was returned.
    if not records:
        entities = output.get("entities")
        if isinstance(entities, dict):
            for field_name in ("technology", "framework", "programming_language", "tool"):
                values = entities.get(field_name, [])
                if not isinstance(values, list):
                    values = [values]
                for item in values:
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and isinstance(item.get("confidence"), (int, float)):
                        records[item["text"]] = max(0.0, min(1.0, float(item["confidence"])))
    return [
        {"choice": choice, "probability": probability}
        for choice, probability in sorted(records.items(), key=lambda item: (-item[1], item[0]))
    ]


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
    choices = _probability_choices(output, ("decision_choice", "decision_choices", "decision_route"))
    if choices and max(item["probability"] for item in choices) < policy.min_confidence:
        reasons.append("decision_confidence_below_threshold")
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
        "decision_choices": choices,
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
