#!/usr/bin/env python3
"""Golden routing evaluation for the jev-tool-task decision contract.

The deterministic router is always evaluated with synthetic extraction outputs.
The real GLiNER2 classification path is attempted only when CUDA and a staged
model path are available; its absence never fails the run, but the chosen path
is recorded. Every router call runs under a guard that fails if a subprocess or
socket call is attempted, so a policy record can never execute a tool.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jev.config import load_config
from jev.tool_task import resolve_tool_task

SIDE_EFFECT_MODULES = {
    "subprocess",
    "socket",
    "pty",
    "shlex",
    "multiprocessing",
    "ctypes",
    "urllib",
    "http",
    "requests",
    "asyncio",
}
POLICY_RECORD_KEYS = {
    "policy",
    "policy_version",
    "allowed",
    "action",
    "task_type",
    "model_id",
    "confidence",
    "decision_choices",
    "reasons",
}
GUARD_TARGETS = (
    (subprocess, "Popen"),
    (subprocess, "run"),
    (subprocess, "call"),
    (subprocess, "check_call"),
    (subprocess, "check_output"),
    (os, "system"),
    (os, "popen"),
    (socket, "socket"),
    (socket, "create_connection"),
)


def audit_tool_task_module_imports() -> dict[str, Any]:
    """Static module-level proof that the router has no side-effect imports."""
    module = importlib.import_module("jev.tool_task")
    tree = ast.parse(inspect.getsource(module))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    offending = sorted(imports & SIDE_EFFECT_MODULES)
    return {
        "module": "jev.tool_task",
        "imports": sorted(imports),
        "side_effect_imports": offending,
        "subprocess_imported": "subprocess" in offending,
        "socket_imported": "socket" in offending,
        "side_effect_free": not offending,
    }


MODULE_IMPORT_AUDIT = audit_tool_task_module_imports()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def no_tool_execution_guard() -> Iterator[None]:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("resolve_tool_task attempted to execute a tool")

    saved = [(obj, name, getattr(obj, name)) for obj, name in GUARD_TARGETS]
    for obj, name, _original in saved:
        setattr(obj, name, blocked)
    try:
        yield
    finally:
        for obj, name, original in saved:
            setattr(obj, name, original)


def synthetic_output(
    task_type: str,
    confidence: float,
    entities: dict[str, list[str]] | None = None,
    choices: dict[str, float] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {"task_type": {"label": task_type, "confidence": confidence}}
    if entities:
        output["entities"] = {
            name: [{"text": text, "start": 0, "end": len(text), "confidence": 0.9} for text in values]
            for name, values in entities.items()
        }
    if choices:
        output["decision_choice"] = {
            "value": list(choices),
            "confidence": max(choices.values()),
            "probabilities": dict(choices),
        }
    return output


ALLOWED_OUTPUT = synthetic_output(
    "coding",
    0.92,
    {"tool": ["python"], "technology": ["PyTorch"]},
    {"PyTorch": 0.91, "Python": 0.82},
)

GOLDEN_CASES: list[dict[str, Any]] = [
    {
        "name": "allowlisted_model_and_tool",
        "request": "Fix the Python parser bug and add a regression test.",
        "response": "Use Python with PyTorch and run the regression test.",
        "observed_model_id": "gpt-6-astra",
        "expected_action": "allow",
        "expected_reasons": [],
        "synthetic_output": ALLOWED_OUTPUT,
    },
    {
        "name": "unknown_model_rejected",
        "request": "Summarize the incident plan.",
        "response": "Python and PyTorch are recommended for the recovery tool.",
        "observed_model_id": "mystery-9",
        "expected_action": "reject",
        "expected_reasons": ["model_not_allowlisted"],
        "synthetic_output": ALLOWED_OUTPUT,
    },
    {
        "name": "low_confidence_reviewed",
        "request": "Maybe refactor the parser.",
        "response": "Python could help here.",
        "observed_model_id": "deepseek-4.1f",
        "expected_action": "review",
        "expected_reasons": ["confidence_below_threshold"],
        "synthetic_output": synthetic_output("coding", 0.40, {"tool": ["python"]}),
    },
    {
        "name": "unknown_task_type_rejected",
        "request": "Do the quantum compilation pass.",
        "response": "Python and PyTorch can approximate it.",
        "observed_model_id": "gpt-6-astra",
        "expected_action": "reject",
        "expected_reasons": ["task_type_unknown"],
        "synthetic_output": synthetic_output("quantum_compilation", 0.95, {"tool": ["python"]}),
    },
    {
        "name": "no_tool_execution",
        "request": "Fix the Python parser bug and add a regression test.",
        "response": "Python and PyTorch are allowlisted for this task.",
        "observed_model_id": "gpt-6-astra",
        "expected_action": "allow",
        "expected_reasons": [],
        "synthetic_output": ALLOWED_OUTPUT,
    },
]


def classification_outputs(model: Any, config: Any, case: dict[str, Any]) -> dict[str, Any]:
    from jev.model import extract
    from jev.tool_task import classify_tool_task

    merged: dict[str, Any] = {}
    entities: dict[str, list[Any]] = {}
    for text in (case["request"], case["response"]):
        output = extract(model, config, text)
        if isinstance(output, dict):
            merged.update(output)
            if isinstance(output.get("entities"), dict):
                for name, values in output["entities"].items():
                    if isinstance(values, list):
                        entities.setdefault(name, []).extend(values)
    if entities:
        merged["entities"] = entities
    classification = classify_tool_task(model, case["response"], config.tool_task)
    if isinstance(classification, dict):
        merged.update(classification)
    return merged


def run_model_backed(args: argparse.Namespace, config: Any) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    import torch

    if not torch.cuda.is_available():
        return False, {"reason": "cuda_unavailable"}, []
    model_path = args.model_path or os.environ.get("JEV_MODEL_PATH") or config.runtime.model_path
    if not model_path:
        return False, {"reason": "model_path_missing"}, []
    from dataclasses import replace

    from jev.model import load_gliner

    model_config = replace(config, runtime=replace(config.runtime, device="cuda", model_path=model_path))
    try:
        model = load_gliner(model_config)
        if hasattr(model, "eval"):
            model.eval()
        results: list[dict[str, Any]] = []
        with torch.inference_mode(), no_tool_execution_guard():
            for case in GOLDEN_CASES:
                output = classification_outputs(model, model_config, case)
                decision = resolve_tool_task(output, config.tool_task, model_id=case["observed_model_id"])
                results.append(
                    {
                        "name": case["name"],
                        "expected_action": case["expected_action"],
                        "actual_action": decision["action"],
                        "match": decision["action"] == case["expected_action"],
                        "task_type": decision["task_type"],
                        "reasons": decision["reasons"],
                        "decision_choices": decision["decision_choices"],
                    }
                )
        return True, {"path": str(model_path)}, results
    except Exception as exc:  # optional path must never fail the deterministic gate
        return False, {"reason": "model_error", "error": repr(exc)}, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jev_tool_task.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--output", default="artifacts/evaluations/tool_task_golden.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    config = load_config(args.config)
    if config.tool_task is None:
        print("config does not define a tool_task policy", file=sys.stderr)
        return 2

    guard_violations: list[str] = []
    case_results: list[dict[str, Any]] = []
    for case in GOLDEN_CASES:
        try:
            with no_tool_execution_guard():
                decision = resolve_tool_task(case["synthetic_output"], config.tool_task, model_id=case["observed_model_id"])
        except AssertionError as exc:
            guard_violations.append(f"{case['name']}: {exc}")
            case_results.append({"name": case["name"], "expected_action": case["expected_action"], "match": False, "policy_record": False})
            continue
        record_ok = POLICY_RECORD_KEYS <= set(decision)
        reasons_match = set(case["expected_reasons"]) <= set(decision["reasons"])
        case_results.append(
            {
                "name": case["name"],
                "request": case["request"],
                "response": case["response"],
                "observed_model_id": case["observed_model_id"],
                "expected_action": case["expected_action"],
                "expected_reasons": case["expected_reasons"],
                "actual_action": decision["action"],
                "actual_reasons": decision["reasons"],
                "task_type": decision["task_type"],
                "decision_choices": decision["decision_choices"],
                "configured_action": config.tool_task.task_actions.get(decision["task_type"] or ""),
                "policy_record": record_ok,
                "match": decision["action"] == case["expected_action"],
                "reasons_match": reasons_match,
            }
        )

    deterministic_pass = all(
        result.get("match") and result.get("reasons_match") and result.get("policy_record") for result in case_results
    )
    model_backed, model_meta, model_results = run_model_backed(args, config)
    model_map = {item["name"]: item for item in model_results}
    for result in case_results:
        if result["name"] in model_map:
            result["model_backed"] = model_map[result["name"]]
    model_backed_pass = (
        all(item.get("match") for item in model_results) if model_results else None
    )

    payload = {
        "pass": bool(deterministic_pass and not guard_violations),
        "deterministic_pass": deterministic_pass,
        "model_backed": model_backed,
        "model_backed_pass": model_backed_pass,
        "model_backed_meta": model_meta,
        "model_path": args.model_path or os.environ.get("JEV_MODEL_PATH") or config.runtime.model_path,
        "policy": config.tool_task.as_contract(),
        "module_import_audit": MODULE_IMPORT_AUDIT,
        "no_tool_execution": {
            "guard_enabled": True,
            "guard_held": not guard_violations,
            "violations": guard_violations,
            "assertion": "resolve_tool_task returns a policy record and never executes a tool",
        },
        "cases": case_results,
        "seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pass": payload["pass"],
                "model_backed": model_backed,
                "model_backed_meta": model_meta,
                "actions": {result["name"]: result.get("actual_action") for result in case_results},
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
