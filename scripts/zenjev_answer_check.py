#!/usr/bin/env python3
"""Answer-check audit: latest LoRA vs stored pair targets (§1.8, ZJ-082).

Read-only verifier prototype. Loads the latest adapter checkpoint, re-judges a
sample of stored distillation pairs with the newest model, and compares the
fresh judgment against the stored target schema. Emits an agreement report and
a scalar reward to ``artifacts/perpetual/answer_check.json``. It never trains
and never mutates model state.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import time
from typing import Any

from zenjev_lib import PERPETUAL_RUNS, atomic_write_json, percentile, sha256_hex, utc_now

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/zenjev_perpetual.yaml"
PAIRS_PATH = PERPETUAL_RUNS / "pairs.jsonl"
CHECKPOINT_PATH = PERPETUAL_RUNS / "checkpoints/latest.pt"
OUT_PATH = REPO / "artifacts/perpetual/answer_check.json"


def tail_lines(path: pathlib.Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        size = handle.tell()
        block = min(size, 4 * 1024 * 1024 * max(1, limit // 50))
        handle.seek(max(0, size - block))
        data = handle.read().decode("utf-8", errors="replace").splitlines()
    return [line for line in data[-limit:] if line.strip()]


def load_pairs(limit: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for line in tail_lines(PAIRS_PATH, limit):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        pair = record.get("pair")
        if isinstance(pair, dict) and isinstance(pair.get("input"), str):
            pairs.append({"pair": pair, "created_at": record.get("created_at")})
    return pairs


def load_latest_model(config: Any) -> tuple[Any, str | None]:
    import torch

    from jev.model import apply_lora, load_gliner

    model = apply_lora(load_gliner(config), config)
    digest = None
    if CHECKPOINT_PATH.exists():
        payload = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state = {name: p for name, p in model.named_parameters() if p.requires_grad}
        adapter = payload.get("adapter_state", {})
        if set(adapter) == set(state):
            with torch.no_grad():
                for key, parameter in state.items():
                    parameter.copy_(adapter[key].to(device=parameter.device, dtype=parameter.dtype))
            digest = sha256_hex(CHECKPOINT_PATH.read_bytes())[:16]
    if hasattr(model, "eval"):
        model.eval()
    return model, digest


def target_labels(target: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for item in target.get("classifications", []) or []:
        if isinstance(item, dict) and isinstance(item.get("task"), str):
            labels = item.get("true_label")
            out[item["task"]] = [str(x) for x in labels] if isinstance(labels, list) else [str(labels)]
    return out


def extracted_labels(output: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for task in ("task_type", "decision_choice"):
        payload = output.get(task)
        if isinstance(payload, dict):
            probabilities = payload.get("probabilities") or {}
            if probabilities:
                ranked = sorted(probabilities.items(), key=lambda kv: (-float(kv[1]), kv[0]))
                out[task] = [str(ranked[0][0])]
            elif isinstance(payload.get("label"), str):
                out[task] = [payload["label"]]
        elif isinstance(payload, str):
            out[task] = [payload]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--path", choices=("judge", "extract"), default="judge",
                        help="judge matches the head that produced the stored targets")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="cpu keeps the audit off the busy serving GPU")
    args = parser.parse_args()

    from dataclasses import replace

    from jev.config import load_config

    config = load_config(CONFIG_PATH)
    if args.device != "auto":
        config = replace(config, runtime=replace(config.runtime, device=args.device))
    pairs = load_pairs(args.sample)
    if not pairs:
        raise SystemExit("no stored pairs to check")
    model, digest = load_latest_model(config)

    def judge_labels(text: str) -> dict[str, list[str]]:
        from gliner2.classification import (
            ClassificationConfig,
            ClassificationSchema,
            Classifier,
        )

        policy = config.tool_task
        task_schema = ClassificationSchema().single(
            "task_type", list(policy.task_types), threshold=policy.min_confidence
        )
        choice_schema = ClassificationSchema().multi(
            "decision_choice",
            list(policy.decision_choices),
            min_labels=0,
            max_labels=len(policy.decision_choices),
            threshold=policy.min_confidence,
        )
        classifier = Classifier(model)
        classification_config = ClassificationConfig(include_confidence=True)
        task_result = classifier.batch_classify([text], task_schema, config=classification_config)[0]
        choice_result = classifier.batch_classify([text], choice_schema, config=classification_config)[0]
        task_payload = task_result.to_dict() if hasattr(task_result, "to_dict") else task_result
        choice_payload = choice_result.to_dict() if hasattr(choice_result, "to_dict") else choice_result
        out: dict[str, list[str]] = {}
        for task, payload in (("task_type", task_payload), ("decision_choice", choice_payload)):
            block = payload.get(task, payload) if isinstance(payload, dict) else {}
            if isinstance(block, dict):
                probabilities = block.get("probabilities") or {}
                if probabilities:
                    ranked = sorted(probabilities.items(), key=lambda kv: (-float(kv[1]), kv[0]))
                    out[task] = [str(ranked[0][0])]
                elif isinstance(block.get("label"), str):
                    out[task] = [block["label"]]
        return out

    stats = collections.Counter()
    errors: collections.Counter = collections.Counter()
    entity_hits = entity_total = entity_pred = 0
    latencies: list[float] = []
    samples: list[dict[str, Any]] = []
    for item in pairs:
        pair = item["pair"]
        target = pair.get("output") if isinstance(pair.get("output"), dict) else {}
        expected = target_labels(target)
        started = time.perf_counter()
        try:
            if args.path == "judge":
                actual = judge_labels(pair["input"][: args.max_chars])
                output: dict[str, Any] = {}
            else:
                output = extract(model, config, pair["input"][: args.max_chars])
                actual = extracted_labels(output)
        except Exception as error:  # noqa: BLE001
            stats["inference_failed"] += 1
            errors[type(error).__name__] += 1
            continue
        latencies.append((time.perf_counter() - started) * 1000.0)
        stats["checked"] += 1
        for task in ("task_type", "decision_choice"):
            want = set(expected.get(task) or [])
            got = set(actual.get(task) or [])
            if not want:
                continue
            stats[f"{task}_checked"] += 1
            if want & got:
                stats[f"{task}_agree"] += 1
            if want == got:
                stats[f"{task}_exact"] += 1
        want_entities = target.get("entities") or {}
        got_entities = {
            name: [str(entry.get("text")) for entry in values if isinstance(entry, dict)]
            for name, values in ((output.get("entities") or {}) if isinstance(output.get("entities"), dict) else {}).items()
        }
        if args.path == "judge":
            want_entities = {}
        for name, values in want_entities.items():
            want_set = {str(v) for v in values}
            got_set = set(got_entities.get(name, []))
            entity_total += len(want_set)
            entity_pred += len(got_set)
            entity_hits += len(want_set & got_set)
        if len(samples) < 8:
            samples.append(
                {
                    "expected": expected,
                    "actual": actual,
                    "input_sha16": sha256_hex(pair["input"])[:16],
                    "input_chars": len(pair["input"]),
                }
            )

    def rate(hit: str, total: str) -> float | None:
        denominator = stats.get(total, 0)
        return round(stats.get(hit, 0) / denominator, 4) if denominator else None

    entity_precision = round(entity_hits / entity_pred, 4) if entity_pred else None
    entity_recall = round(entity_hits / entity_total, 4) if entity_total else None
    task_rate = rate("task_type_agree", "task_type_checked")
    choice_rate = rate("decision_choice_agree", "decision_choice_checked")
    components = [value for value in (task_rate, choice_rate, entity_recall) if value is not None]
    reward = round(sum(components) / len(components), 4) if components else None

    report = {
        "recorded_at": utc_now(),
        "checkpoint_sha256_16": digest,
        "config_digest_16": config.digest()[:16],
        "sampled": len(pairs),
        "path": args.path,
        "device": args.device,
        "errors": dict(errors),
        "checked": stats.get("checked", 0),
        "agreement": {
            "task_type": task_rate,
            "decision_choice": choice_rate,
            "task_type_exact": rate("task_type_exact", "task_type_checked"),
            "decision_choice_exact": rate("decision_choice_exact", "decision_choice_checked"),
            "entity_precision": entity_precision,
            "entity_recall": entity_recall,
        },
        "reward": reward,
        "latency_ms": {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)},
        "note": "read-only self-consistency verifier prototype; reward = mean(task_type, decision_choice, entity_recall)",
        "samples": samples,
    }
    atomic_write_json(OUT_PATH, report)
    print(json.dumps({k: report[k] for k in ("sampled", "checked", "path", "device", "agreement", "reward", "latency_ms", "errors")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
