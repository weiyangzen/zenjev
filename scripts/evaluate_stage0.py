#!/usr/bin/env python3
"""Evaluate base / live-LoRA / EMA-snapshot extraction on Stage 0 held-out labels.

The base GLiNER2 predictions are computed once, then a resumed trainer publishes
the accepted EMA snapshot through the runtime. No training or checkpoint
mutation happens here; the script only measures exact-match span quality for the
labels that appear in the human-authored held-out file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


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


def normalize_span(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def spans_for_labels(output: Any, labels: list[str]) -> dict[str, set[str]]:
    """Collect exact predicted span strings per label from a GLiNER2 result."""
    result: dict[str, set[str]] = {label: set() for label in labels}
    if not isinstance(output, dict):
        return result
    entities = output.get("entities")
    if not isinstance(entities, dict):
        entities = output
    for label in labels:
        values = entities.get(label)
        if isinstance(values, dict):
            values = [values]
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        for item in values:
            text = item.get("text") if isinstance(item, dict) else item
            normalized = normalize_span(text)
            if normalized:
                result[label].add(normalized)
    return result


def sorted_predictions(predictions: list[dict[str, set[str]]]) -> list[dict[str, list[str]]]:
    return [{label: sorted(values) for label, values in entry.items()} for entry in predictions]


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(
    predictions: list[dict[str, set[str]]],
    gold: list[dict[str, set[str]]],
    labels: list[str],
) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for label in labels:
        tp = fp = fn = 0
        for predicted, expected in zip(predictions, gold):
            pred = predicted.get(label, set())
            truth = expected.get(label, set())
            tp += len(pred & truth)
            fp += len(pred - truth)
            fn += len(truth - pred)
        per_label[label] = precision_recall_f1(tp, fp, fn)
        total["tp"] += tp
        total["fp"] += fp
        total["fn"] += fn
    overall = precision_recall_f1(total["tp"], total["fp"], total["fn"])
    return {"per_label": per_label, "overall": overall}


def load_heldout(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise ValueError("held-out records require text and labels")
        records.append(value)
    if not records:
        raise ValueError("held-out file is empty")
    return records


def target_labels(records: list[dict[str, Any]], allowed: tuple[str, ...]) -> list[str]:
    labels: list[str] = []
    for record in records:
        labels_map = record.get("labels", {})
        if not isinstance(labels_map, dict):
            continue
        for name in labels_map:
            if name in allowed and name not in labels:
                labels.append(name)
    return labels


def collect_gold(records: list[dict[str, Any]], labels: list[str]) -> list[dict[str, set[str]]]:
    gold = []
    for record in records:
        labels_map = record.get("labels", {})
        entry: dict[str, set[str]] = {}
        for label in labels:
            values = labels_map.get(label, []) if isinstance(labels_map, dict) else []
            if not isinstance(values, list):
                values = [values]
            entry[label] = {normalized for normalized in (normalize_span(value) for value in values) if normalized}
        gold.append(entry)
    return gold


def predict(model: Any, config: Any, texts: list[str], labels: list[str]) -> tuple[list[dict[str, set[str]]], float]:
    import torch

    from jev.model import extract

    predictions: list[dict[str, set[str]]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for text in texts:
            predictions.append(spans_for_labels(extract(model, config, text), labels))
    return predictions, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--checkpoint", default="runs/jev/stage0/latest.pt")
    parser.add_argument("--heldout", default=str(ROOT / "data/stage0_heldout.jsonl"))
    parser.add_argument("--output", default="artifacts/evaluations/stage0_evaluation.json")
    args = parser.parse_args(argv)

    import torch

    from jev.config import load_config
    from jev.model import load_gliner
    from jev.runtime import JevRuntime
    from jev.training import ContinuousLoRATrainer

    if not torch.cuda.is_available():
        print("CUDA is required for the Stage 0 GPU evaluation", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "latest.pt"
    if not checkpoint.exists():
        print(f"checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    started = time.perf_counter()
    config = load_config(args.config)
    model_path = args.model_path or os.environ.get("JEV_MODEL_PATH") or config.runtime.model_path
    if model_path and not Path(model_path).is_dir():
        print(f"model path is not a staged directory: {model_path}", file=sys.stderr)
        return 2
    if model_path:
        os.environ["JEV_MODEL_PATH"] = str(model_path)
    # The runtime config is used exactly as loaded so the checkpoint's
    # config digest matches; the snapshot path comes from JEV_MODEL_PATH.

    records = load_heldout(args.heldout)
    labels = target_labels(records, config.schema.entities)
    if not labels:
        print("held-out labels do not intersect the configured schema entities", file=sys.stderr)
        return 2
    texts = [record["text"] for record in records]
    gold = collect_gold(records, labels)

    load_started = time.perf_counter()
    base = load_gliner(config)
    if hasattr(base, "eval"):
        base.eval()
    base_load_seconds = time.perf_counter() - load_started
    base_predictions, base_seconds = predict(base, config, texts, labels)
    del base
    torch.cuda.empty_cache()

    try:
        runtime = JevRuntime(config)
        trainer = ContinuousLoRATrainer(
            config,
            runtime,
            checkpoint_dir=checkpoint.parent,
            resume_from=checkpoint,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot resume checkpoint {checkpoint}: {exc}", file=sys.stderr)
        return 2
    if hasattr(trainer.model, "eval"):
        trainer.model.eval()
    resume = {
        "path": str(checkpoint),
        "training_steps": runtime.stats.training_steps,
        "ema_updates": runtime.ema.updates,
        "reset_id": runtime.stats.reset_id,
    }
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("training_steps", "ema_updates", "reset_id", "config_digest", "schema_digest", "model_id", "model_revision", "created_at"):
        if isinstance(payload, dict) and key in payload:
            resume[key] = payload[key]

    live_predictions, live_seconds = predict(trainer.model, config, texts, labels)

    ema_predictions: list[dict[str, set[str]]] = []
    ema_seconds_started = time.perf_counter()
    ema_metadata: dict[str, Any] = {}
    with torch.inference_mode():
        for text in texts:
            result = runtime.infer_result(text)
            if not ema_metadata:
                ema_metadata = {
                    "generation": result["generation"],
                    "ema_step": result["ema_step"],
                    "reset_id": result["reset_id"],
                    "checkpoint": result["checkpoint"],
                    "schema_digest": result["schema_digest"],
                    "config_digest": result["config_digest"],
                    "model_id": result["model_id"],
                }
            ema_predictions.append(spans_for_labels(result["output"], labels))
    ema_seconds = time.perf_counter() - ema_seconds_started

    capability = torch.cuda.get_device_capability(0)
    output = {
        "config": {
            "path": str(args.config),
            "digest": config.digest(),
            "schema_digest": config.schema.digest(),
            "model_id": config.model_id,
            "model_revision": config.model_revision,
        },
        "checkpoint": resume,
        "heldout": {
            "path": str(args.heldout),
            "sha256": hashlib.sha256(Path(args.heldout).read_bytes()).hexdigest(),
            "records": len(records),
            "labels": labels,
        },
        "variants": {
            "base": {"metrics": compute_metrics(base_predictions, gold, labels), "seconds": base_seconds, "predictions": sorted_predictions(base_predictions)},
            "live_lora": {"metrics": compute_metrics(live_predictions, gold, labels), "seconds": live_seconds, "predictions": sorted_predictions(live_predictions)},
            "ema_snapshot": {"metrics": compute_metrics(ema_predictions, gold, labels), "seconds": ema_seconds, "predictions": sorted_predictions(ema_predictions), "snapshot": ema_metadata},
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "capability": f"{capability[0]}.{capability[1]}",
        },
        "timing": {
            "base_load_seconds": base_load_seconds,
            "base_eval_seconds": base_seconds,
            "live_eval_seconds": live_seconds,
            "ema_eval_seconds": ema_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    atomic_write_json(args.output, output)
    summary = {name: output["variants"][name]["metrics"]["overall"]["f1"] for name in output["variants"]}
    print(json.dumps({"output": str(args.output), "labels": labels, "f1": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
