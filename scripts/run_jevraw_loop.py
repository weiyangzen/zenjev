#!/usr/bin/env python3
"""Judge -> synthesize -> train -> serve loop over a 0.5% jevraw subset.

For each round the currently served LoRA snapshot judges raw LLM interactions
into schema probability distributions, those distributions are distilled into
instruction-style training records (instruct + prompt -> schema distribution),
the LoRA adapter is trained on them, the EMA snapshot is republished, and the
next round judges with the updated adapter while inference keeps serving.

Only aggregate evidence is written to artifacts/; raw prompts and responses stay
under runs/ (gitignored) because jevraw contains user data.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import itertools
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.config import JevConfig, load_config
from jev.model import apply_lora, load_gliner
from jev.runtime import JevRuntime
from jev.tool_task import classify_tool_task
from jev.training import ContinuousLoRATrainer

JUDGE_CHARS = 1200
INSTRUCTION = "Given the prompt, return the task_type and the technology-stack probability distribution."


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def relative_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sample_files(root: Path, fraction: float) -> list[Path]:
    """Deterministic 0.5% file sample by SHA-256 of the relative path."""
    threshold = int(round(fraction * 100_000))
    selected: list[Path] = []
    for pattern in ("**/*.ndjson", "**/*.jsonl"):
        for name in glob.glob(str(root / pattern), recursive=True):
            path = Path(name)
            digest = hashlib.sha256(relative_id(path, root).encode("utf-8")).hexdigest()
            if int(digest[:16], 16) % 100_000 < threshold:
                selected.append(path)
    return sorted(set(selected))


def _content_parts(content: object) -> list[str]:
    texts: list[str] = []
    if isinstance(content, str):
        if content.strip():
            texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                texts.append(part["text"])
    return texts


def prompt_from_value(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    messages: list[str] = []
    for item in value.get("input", []) or []:
        if isinstance(item, dict) and item.get("role") == "user":
            messages.extend(_content_parts(item.get("content")))
    if not messages:
        return None
    return "\n".join(messages)[-4000:]


def prompt_text(request_body: object) -> str | None:
    if not isinstance(request_body, dict):
        return None
    return prompt_from_value(request_body.get("value"))


def response_text(response_body: object) -> str | None:
    if not isinstance(response_body, dict):
        return None
    if response_body.get("kind") != "text":
        value = response_body.get("value")
        return json.dumps(value, ensure_ascii=False)[:4000] if value is not None else None
    stream = response_body.get("value")
    if not isinstance(stream, str):
        return None
    parts: list[str] = []
    for line in stream.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if data.get("type") != "response.completed":
            continue
        response = data.get("response", {})
        for item in response.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content", []) or []:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
            elif item.get("type") in ("function_call", "custom_tool_call"):
                arguments = item.get("arguments") or item.get("input")
                if isinstance(arguments, str):
                    parts.append(arguments)
        break
    if not parts:
        for line in stream.splitlines():
            if line.startswith("data: ") and "custom_tool_call_input.done" in line:
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if isinstance(data.get("input"), str):
                    parts.append(data["input"])
    return "\n".join(parts)[:4000] or None


def normalize_record(record: dict) -> tuple[str, str] | None:
    """Extract (prompt, response) from either jevraw capture schema."""
    if "requestBody" in record or "responseBody" in record:
        if record.get("responseStatus") != 200:
            return None
        prompt = prompt_text(record.get("requestBody"))
        response = response_text(record.get("responseBody"))
        return (prompt, response) if prompt and response else None
    if "input_body" in record or "output_body" in record:
        raw_input = record.get("input_body")
        if isinstance(raw_input, str):
            try:
                request_value = json.loads(raw_input)
            except json.JSONDecodeError:
                return None
        else:
            request_value = raw_input
        prompt = prompt_from_value(request_value)
        raw_output = record.get("output_body")
        response = response_text({"kind": "text", "value": raw_output}) if isinstance(raw_output, str) else response_text(raw_output)
        return (prompt, response) if prompt and response else None
    return None


def iter_subset(root: Path, files: list[Path], per_file: int, max_records: int, scan_limit: int = 30):
    """Yield bounded (path, line, prompt, response) records from sampled files."""
    found = 0
    for path in files:
        taken = 0
        try:
            handle = path.open(encoding="utf-8", errors="replace")
        except OSError:
            continue
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number > scan_limit or taken >= per_file or found >= max_records:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                normalized = normalize_record(record)
                if normalized is None:
                    continue
                prompt, response = normalized
                taken += 1
                found += 1
                yield path, line_number, prompt, response
        if found >= max_records:
            break


def top_labels(distribution: dict, labels: list[str], *, k: int, threshold: float) -> list[str]:
    ranked = sorted(
        ((label, float(distribution.get(label, 0.0))) for label in labels),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [label for label, probability in ranked[:k] if probability >= threshold]
    return selected or [ranked[0][0]]


def entropy(distribution: dict, labels: list[str]) -> float:
    total = sum(float(distribution.get(label, 0.0)) for label in labels)
    if total <= 0:
        return 0.0
    value = 0.0
    for label in labels:
        probability = float(distribution.get(label, 0.0)) / total
        if probability > 0:
            value -= probability * math.log(probability)
    return value


def judge_batch(model: object, config: JevConfig, texts: list[str], *, chunk: int = 16) -> list[dict]:
    """Batched schema-probability judging on the currently served adapter."""
    from gliner2.classification import ClassificationConfig, ClassificationSchema, Classifier

    policy = config.tool_task
    task_schema = ClassificationSchema().single("task_type", list(policy.task_types), threshold=policy.min_confidence)
    choice_schema = ClassificationSchema().multi(
        "decision_choice",
        list(policy.decision_choices),
        min_labels=0,
        max_labels=len(policy.decision_choices),
        threshold=policy.min_confidence,
    )
    classification_config = ClassificationConfig(include_confidence=True)
    classifier = Classifier(model)
    results: list[dict] = []
    for start in range(0, len(texts), chunk):
        window = texts[start : start + chunk]
        task_results = classifier.batch_classify(window, task_schema, config=classification_config)
        choice_results = classifier.batch_classify(window, choice_schema, config=classification_config)
        for task_result, choice_result in zip(task_results, choice_results):
            task_payload = task_result.to_dict() if hasattr(task_result, "to_dict") else task_result
            choice_payload = choice_result.to_dict() if hasattr(choice_result, "to_dict") else choice_result
            results.append(
                {
                    "task_type": task_payload.get("task_type", task_payload),
                    "decision_choice": choice_payload.get("decision_choice", choice_payload),
                }
            )
    return results


def synthesize(config: JevConfig, text: str, judgment: dict) -> tuple[dict, dict]:
    policy = config.tool_task
    task_probabilities = judgment.get("task_type", {}).get("probabilities", {}) or {}
    choice_probabilities = judgment.get("decision_choice", {}).get("probabilities", {}) or {}
    task_labels = top_labels(task_probabilities, list(policy.task_types), k=1, threshold=0.0)
    choice_labels = top_labels(choice_probabilities, list(policy.decision_choices), k=3, threshold=0.25)
    training = {
        "input": text,
        "output": {
            "classifications": [
                {"task": "task_type", "labels": list(policy.task_types), "true_label": task_labels},
                {"task": "decision_choice", "labels": list(policy.decision_choices), "true_label": choice_labels},
            ]
        },
    }
    instruct = {
        "instruction": INSTRUCTION,
        "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_chars": len(text),
        "schema": {"task_type": list(policy.task_types), "decision_choice": list(policy.decision_choices)},
        "distribution": {"task_type": task_probabilities, "decision_choice": choice_probabilities},
        "selected": {"task_type": task_labels, "decision_choice": choice_labels},
        "training": training,
    }
    return training, instruct


def summarize(distributions: list[dict], labels: list[str]) -> dict:
    if not distributions:
        return {"count": 0}
    top_probabilities = [max((float(item.get(label, 0.0)) for label in labels), default=0.0) for item in distributions]
    entropies = [entropy(item, labels) for item in distributions]
    return {
        "count": len(distributions),
        "mean_top_probability": sum(top_probabilities) / len(top_probabilities),
        "mean_entropy": sum(entropies) / len(entropies),
    }


def serve_probes(runtime: JevRuntime, config: JevConfig, probes: list[str], rounds: int, observations: list, errors: list) -> None:
    try:
        for index in range(rounds):
            text = probes[index % len(probes)]
            begin = time.perf_counter()
            extraction = runtime.infer_result(text)
            snapshot = runtime.snapshot_model()
            classification = classify_tool_task(snapshot, text, config.tool_task) if snapshot is not None else {}
            observations.append(
                {
                    "seconds": time.perf_counter() - begin,
                    "generation": extraction["generation"],
                    "ema_step": extraction["ema_step"],
                    "task_type": (classification.get("task_type") or {}).get("value"),
                }
            )
    except Exception as exc:  # pragma: no cover - GPU-only failure path
        errors.append(repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/home/sansha/data/jevraw")
    parser.add_argument("--fraction", type=float, default=0.005)
    parser.add_argument("--per-file", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps-per-round", type=int, default=8)
    parser.add_argument("--probe-inferences", type=int, default=4)
    parser.add_argument("--config", default="configs/jev_tool_task.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--checkpoint-dir", default="runs/jev/jevraw")
    parser.add_argument("--output", default="artifacts/jevraw/loop.json")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("provide --model-path or JEV_MODEL_PATH")
    os.environ["JEV_MODEL_PATH"] = args.model_path
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the jevraw loop")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    config = load_config(args.config)
    config = replace(
        config,
        runtime=replace(config.runtime, device="cuda", publish_every_steps=1, checkpoint_every_steps=1, max_checkpoints=2, ema_decay=0.9),
    )
    raw_root = Path(args.raw_root)
    work_dir = Path(args.checkpoint_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    files = sample_files(raw_root, args.fraction)
    selection_started = time.perf_counter()
    records = list(iter_subset(raw_root, files, args.per_file, args.max_records))
    selection_seconds = time.perf_counter() - selection_started
    if not records:
        raise SystemExit("no usable records in the jevraw sample")
    subset_path = work_dir / "subset.jsonl"
    with subset_path.open("w", encoding="utf-8") as handle:
        for path, line, prompt, response in records:
            handle.write(
                json.dumps(
                    {
                        "file": relative_id(path, raw_root),
                        "line": line,
                        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                        "prompt_chars": len(prompt),
                        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                        "response_chars": len(response),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    load_started = time.perf_counter()
    runtime = JevRuntime(config)
    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        checkpoint_dir=args.checkpoint_dir,
        model_factory=lambda: apply_lora(load_gliner(config), config),
    )
    base_load_seconds = time.perf_counter() - load_started

    probes = [
        "Fix the Python parser bug and add a regression test.",
        "Design a Kubernetes deployment for the Rust bridge with PostgreSQL.",
        "Translate the operator runbook into Chinese.",
        "Analyze the benchmark results and summarize the trend.",
    ]
    round_evidence: list[dict] = []
    serving_observations: list[dict] = []
    serving_errors: list[str] = []
    total_training_examples = 0

    for round_index in range(1, args.rounds + 1):
        round_started = time.perf_counter()
        # Judge with the live trained LoRA adapter; serve through the EMA
        # snapshot, so each round both continues training and keeps inference up.
        judging_model = trainer.model
        if hasattr(judging_model, "eval"):
            judging_model.eval()
        texts: list[str] = []
        for _path, _line, prompt, response in records:
            texts.append(prompt[:JUDGE_CHARS])
            texts.append(response[:JUDGE_CHARS])
        judged = judge_batch(judging_model, config, texts)
        task_distributions = [((item.get("task_type") or {}).get("probabilities", {}) or {}) for item in judged]
        choice_distributions = [((item.get("decision_choice") or {}).get("probabilities", {}) or {}) for item in judged]
        judgments = [{"training": training, "instruct": instruct} for text, item in zip(texts, judged) for training, instruct in [synthesize(config, text, item)]]
        judging_seconds = time.perf_counter() - round_started
        pre_generation = runtime.stats.model_generation
        pre_ema = runtime.ema.updates

        synth_path = work_dir / f"round-{round_index}-synthesized.jsonl"
        with synth_path.open("w", encoding="utf-8") as handle:
            for item in judgments:
                handle.write(json.dumps(item["instruct"], ensure_ascii=False) + "\n")
        training_examples = [item["training"] for item in judgments]
        total_training_examples += len(training_examples)

        serving_observations.clear()
        serving_errors.clear()
        thread = threading.Thread(
            target=serve_probes,
            args=(runtime, config, probes, args.probe_inferences, serving_observations, serving_errors),
        )
        thread.start()
        train_started = time.perf_counter()
        events = trainer.train(itertools.cycle(training_examples), max_steps=args.steps_per_round)
        training_seconds = time.perf_counter() - train_started
        thread.join()
        checkpoint = trainer.save_checkpoint()

        round_evidence.append(
            {
                "round": round_index,
                "judging_model": "live_lora",
                "served_generation_entering_round": pre_generation,
                "served_ema_step_entering_round": pre_ema,
                "judged_texts": len(judgments),
                "judging_seconds": judging_seconds,
                "task_type": summarize(task_distributions, list(config.tool_task.task_types)),
                "decision_choice": summarize(choice_distributions, list(config.tool_task.decision_choices)),
                "synthesized": len(training_examples),
                "training_steps": len(events),
                "losses": [event.loss for event in events],
                "training_seconds": training_seconds,
                "ema_updates": runtime.ema.updates,
                "model_generation": runtime.stats.model_generation,
                "serving_observations": list(serving_observations),
                "serving_errors": list(serving_errors),
                "checkpoint": str(checkpoint),
            }
        )

    latencies = [item["seconds"] for item in serving_observations]
    evidence = {
        "pass": (
            not serving_errors
            and len(round_evidence) == args.rounds
            and all(round_item["synthesized"] > 0 for round_item in round_evidence)
            and all(round_item["training_steps"] == args.steps_per_round for round_item in round_evidence)
            and all(round_item["task_type"].get("mean_top_probability", 0.0) > 0 for round_item in round_evidence)
            and all(round_item["decision_choice"].get("mean_top_probability", 0.0) > 0 for round_item in round_evidence)
        ),
        "raw_root": str(raw_root),
        "fraction": args.fraction,
        "sampled_files": len(files),
        "per_file": args.per_file,
        "max_records": args.max_records,
        "selected_records": len(records),
        "selection_seconds": selection_seconds,
        "rounds": args.rounds,
        "steps_per_round": args.steps_per_round,
        "total_training_examples": total_training_examples,
        "config_digest": config.digest(),
        "schema_digest": config.schema.digest(),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "ema_updates": runtime.ema.updates,
        "model_generation": runtime.stats.model_generation,
        "final_checkpoint": str(work_dir / "latest.pt"),
        "base_load_seconds": base_load_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_vram_bytes": max(torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated()),
        "serving_p50_seconds": sorted(latencies)[len(latencies) // 2] if latencies else None,
        "rounds_evidence": round_evidence,
        "privacy": "aggregate evidence only; raw prompts/responses and synthesized distributions stay under runs/",
    }
    atomic_json(Path(args.output), evidence)
    print(json.dumps(evidence, ensure_ascii=False, default=str))
    return 0 if evidence["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
