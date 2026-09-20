"""ZenJev tool-task analysis and deterministic allowlist routing.

GLiNER2 supplies schema-conditioned labels, spans and confidence. This module
keeps the safety-critical decision outside the model: an extracted task type
can select a policy action only after model, tool and confidence checks pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Iterable

from .config import JevConfig, ToolTaskPolicy

_SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|password|secret|passwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
)


def redact_secrets(text: str) -> str:
    """Remove credential-shaped substrings before a landing-pool record is stored."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}: [REDACTED]" if match.lastindex else "[REDACTED]", redacted)
    return redacted


def admit_tool_task_record(record: dict[str, Any], config: JevConfig) -> dict[str, Any]:
    """Validate a §1.4 landing-pool record before extraction.

    The observed model is matched exactly against the policy allowlist and may
    never be inferred from response text. Request/response text is redacted and
    bound to source provenance and schema/policy digests.
    """
    if config.tool_task is None:
        raise ValueError("config.tool_task is required for landing-pool admission")
    if not isinstance(record, dict):
        raise ValueError("landing-pool record must be an object")
    if int(record.get("schema_version", 0)) != 1:
        raise ValueError("landing-pool record schema_version must be 1")
    request_id = record.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("landing-pool record requires a non-empty request_id")
    source = record.get("source")
    if not isinstance(source, dict):
        raise ValueError("landing-pool record requires a source object")
    for key in ("uri", "content_sha256", "retrieved_at"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"landing-pool source requires {key}")
    observed = record.get("observed_model")
    if not isinstance(observed, dict):
        raise ValueError("landing-pool record requires observed_model")
    model_id = observed.get("model_id", observed.get("id"))
    allowed_ids = {item.model_id for item in config.tool_task.allowed_models}
    allowed_ids |= {alias for item in config.tool_task.allowed_models for alias in item.aliases}
    if not isinstance(model_id, str) or model_id not in allowed_ids:
        raise ValueError(f"observed model is not allowlisted: {model_id!r}")
    sections: dict[str, str] = {}
    for section in ("request", "response"):
        payload = record.get(section)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str) or not payload["text"]:
            raise ValueError(f"landing-pool record requires {section}.text")
        sections[section] = payload["text"]
    admitted = {
        "request_id": request_id,
        "source": {
            "uri": source["uri"],
            "content_sha256": source["content_sha256"],
            "retrieved_at": source["retrieved_at"],
            "license": str(source.get("license", "")),
        },
        "observed_model": {
            "provider": str(observed.get("provider", "")),
            "model_id": model_id,
        },
        "request": {"text": redact_secrets(sections["request"])},
        "response": {"text": redact_secrets(sections["response"])},
        "provenance": {
            "schema_digest": config.schema.digest(),
            "policy_digest": hashlib.sha256(
                json.dumps(config.tool_task.as_contract(), sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
            "admitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "record_digest": hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest(),
        },
    }
    return admitted


def pair_tool_task_records(records: Iterable[dict[str, Any]], config: JevConfig) -> list[dict[str, Any]]:
    """Pair section-aware landing-pool entries (request/response) by request_id.

    Records may already be combined, or arrive as separate single-section
    entries with a ``section`` field. Incomplete pairs are rejected, never
    silently dropped.
    """
    combined: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("landing-pool record must be an object")
        request_id = str(record.get("request_id", ""))
        if not request_id:
            raise ValueError("landing-pool record requires a non-empty request_id")
        section = record.get("section")
        if section in ("request", "response"):
            if request_id not in combined:
                base = {key: value for key, value in record.items() if key != section and key not in ("request", "response")}
                combined[request_id] = base
                order.append(request_id)
            payload = record.get(section, record.get("text"))
            combined[request_id][section] = payload if isinstance(payload, dict) else {"text": payload}
        else:
            if request_id not in combined:
                order.append(request_id)
            combined[request_id] = record
    paired: list[dict[str, Any]] = []
    for request_id in order:
        record = combined[request_id]
        if not isinstance(record.get("request"), dict) or not isinstance(record.get("response"), dict):
            raise ValueError(f"incomplete request/response pair for {request_id}")
        paired.append(admit_tool_task_record(record, config))
    return paired


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
