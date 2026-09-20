#!/usr/bin/env python3
"""Run the Stage 0 closed loop on an NVIDIA GPU.

offline bootstrap distillation -> content-addressed dataset/provenance ->
LoRA continual training -> EMA publication -> concurrent inference ->
checkpoint -> resume -> post-resume inference.

No teacher API key is required: the local pinned GLiNER2 base model acts as the
bootstrap teacher for the project-authored corpus, and every record records that
provenance explicitly. A live external teacher replaces it by changing the
config when credentials exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev.config import JevConfig, load_config
from jev.distill import distill
from jev.model import apply_lora, extract, load_gliner
from jev.runtime import JevRuntime
from jev.training import ContinuousLoRATrainer

TRAINING_ONLY_KEYS = ("provenance", "entity_descriptions")


class LocalBaseTeacher:
    """Deterministic bootstrap teacher backed by the local pinned base model."""

    def __init__(self, config: JevConfig, model: object):
        self.config = config
        self.model = model
        self.last_metadata: dict[str, object] = {}

    def label(self, document) -> dict:
        result = extract(self.model, self.config, document.text)
        output = extraction_to_teacher_output(self.config, result)
        self.last_metadata = {
            "provider": "local-gliner2-base-bootstrap",
            "model": self.config.model_id,
            "revision": self.config.model_revision,
            "schema_digest": self.config.schema.digest(),
            "source_uri": document.uri,
            "external_teacher": False,
        }
        return output


def _entity_spans(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)]


def _span_evidence(spans: list[dict]) -> list[dict]:
    return [
        {"text": span["text"], "start": int(span["start"]), "end": int(span["end"])}
        for span in spans
        if isinstance(span.get("start"), int) and isinstance(span.get("end"), int)
    ]


def extraction_to_teacher_output(config: JevConfig, result: object) -> dict:
    """Map a GLiNER2 extraction result to the teacher schema-shaped object."""
    if not isinstance(result, dict):
        raise ValueError("extraction result must be an object")
    output: dict = {}
    evidence: dict[str, list[dict]] = {}
    entities = result.get("entities", {})
    if isinstance(entities, dict):
        for name in config.schema.entities:
            spans = _entity_spans(entities.get(name, []))
            texts = [span["text"] for span in spans if span["text"]]
            if texts:
                output[name] = texts
                evidence[name] = _span_evidence(spans)
    for task in config.schema.classifications:
        value = result.get(task)
        if isinstance(value, dict) and isinstance(value.get("label"), str):
            if value["label"] in config.schema.classifications[task]:
                output[task] = value["label"]
    records = result.get(config.schema.name)
    if isinstance(records, list) and records and isinstance(records[0], dict):
        for field in config.schema.fields:
            raw = records[0].get(field.name)
            if raw in (None, "", []):
                continue
            try:
                if field.dtype == "str":
                    value = raw if isinstance(raw, str) else ", ".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
                elif field.dtype == "int" and not isinstance(raw, bool):
                    value = int(raw)
                elif field.dtype == "float" and not isinstance(raw, bool):
                    value = float(raw)
                elif field.dtype == "bool":
                    value = bool(raw)
                else:
                    value = raw if isinstance(raw, list) else [raw]
            except (TypeError, ValueError):
                continue
            output[field.name] = value
    relations = result.get("relation_extraction", {})
    if isinstance(relations, dict):
        for relation in config.schema.relations:
            pairs = []
            relation_evidence = []
            for item in relations.get(relation, []) or []:
                if not isinstance(item, dict):
                    continue
                head = item.get("head", {})
                tail = item.get("tail", {})
                if isinstance(head, dict) and isinstance(tail, dict) and head.get("text") and tail.get("text"):
                    pairs.append({"head": str(head["text"]), "tail": str(tail["text"])})
                    relation_evidence.extend(_span_evidence([head, tail]))
            if pairs:
                output[relation] = pairs
                evidence[relation] = relation_evidence
    if evidence:
        output["_evidence"] = evidence
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def training_example(record: dict) -> dict:
    output = {key: value for key, value in record["output"].items() if key not in TRAINING_ONLY_KEYS}
    return {"input": record["input"], "output": output}


def infer_loop(runtime: JevRuntime, texts: list[str], count: int, observations: list, errors: list) -> None:
    try:
        for index in range(count):
            begin = time.perf_counter()
            result = runtime.infer_result(texts[index % len(texts)])
            observations.append(
                {
                    "seconds": time.perf_counter() - begin,
                    "generation": result["generation"],
                    "ema_step": result["ema_step"],
                    "reset_id": result["reset_id"],
                }
            )
    except Exception as exc:  # pragma: no cover - GPU-only failure path
        errors.append(repr(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--model-path", default=os.environ.get("JEV_MODEL_PATH"))
    parser.add_argument("--dataset", default="artifacts/datasets/stage0_distilled.jsonl")
    parser.add_argument("--provenance", default="artifacts/provenance/stage0_dataset_manifest.json")
    parser.add_argument("--checkpoint-dir", default="runs/jev/stage0")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--inferences", type=int, default=4)
    parser.add_argument("--output", default="artifacts/pipeline/stage0_pipeline.json")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("provide --model-path or JEV_MODEL_PATH")
    os.environ["JEV_MODEL_PATH"] = args.model_path
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Stage 0 pipeline")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    config = load_config(args.config)
    dataset_path = Path(args.dataset)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    base_model = load_gliner(config)
    load_seconds = time.perf_counter() - started
    teacher = LocalBaseTeacher(config, base_model)
    distill_started = time.perf_counter()
    examples = distill(config, dataset_path, teacher)
    distill_seconds = time.perf_counter() - distill_started
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    provenance = {
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "records": len(records),
        "config_digest": config.digest(),
        "schema_digest": config.schema.digest(),
        "teacher": {
            "provider": "local-gliner2-base-bootstrap",
            "external_teacher": False,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "note": "offline bootstrap for the environment demonstration; configure teacher.* for a live trusted route",
        },
        "sources": [
            {
                "name": record["source"]["name"],
                "uri": record["source"]["uri"],
                "content_sha256": record["source"]["content_sha256"],
                "license_note": record["source"]["license_note"],
            }
            for record in records
        ],
        "record_hashes": [
            hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest() for record in records
        ],
    }
    atomic_json(Path(args.provenance), provenance)

    runtime = JevRuntime(config)
    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        checkpoint_dir=args.checkpoint_dir,
        model_factory=lambda: apply_lora(load_gliner(config), config),
    )
    train_records = [training_example(record) for record in records]
    observations: list[dict] = []
    errors: list[str] = []
    inference_texts = [
        "The 2026 stack uses Python and PostgreSQL.",
        "Use Docker and Kubernetes for deployment.",
        "GLiNER2 and PyTorch form the training stack.",
        "Rust implements the bridge.",
    ]

    thread = threading.Thread(target=infer_loop, args=(runtime, inference_texts, args.inferences, observations, errors))
    thread.start()
    train_started = time.perf_counter()
    events = trainer.train(train_records, max_steps=args.steps)
    train_seconds = time.perf_counter() - train_started
    thread.join()
    checkpoint = trainer.save_checkpoint()
    pre_resume = runtime.infer_result(inference_texts[0])
    pre_resume_ema = runtime.ema.updates

    resumed_runtime = JevRuntime(config)
    resumed = ContinuousLoRATrainer(
        config,
        resumed_runtime,
        checkpoint_dir=args.checkpoint_dir,
        model_factory=lambda: apply_lora(load_gliner(config), config),
        resume_from=checkpoint,
    )
    post_resume = resumed_runtime.infer_result(inference_texts[0])

    latencies = [item["seconds"] for item in observations]
    evidence = {
        "pass": (
            not errors
            and len(events) == args.steps
            and len(observations) == args.inferences
            and runtime.ema.updates == args.steps
            and resumed.runtime.stats.training_steps == runtime.stats.training_steps
            and resumed_runtime.ema.updates == pre_resume_ema
            and bool(post_resume["output"]) == bool(pre_resume["output"])
        ),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}",
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "config_digest": config.digest(),
        "schema_digest": config.schema.digest(),
        "base_load_seconds": load_seconds,
        "distill_seconds": distill_seconds,
        "distilled_records": examples,
        "training_steps": runtime.stats.training_steps,
        "ema_updates": runtime.ema.updates,
        "model_generation": runtime.stats.model_generation,
        "losses": [event.loss for event in events],
        "events": [vars(event) for event in events],
        "inference_observations": observations,
        "inference_p50_seconds": percentile(latencies, 0.50),
        "inference_p95_seconds": percentile(latencies, 0.95),
        "training_seconds": train_seconds,
        "training_steps_per_second": (len(events) / train_seconds) if train_seconds > 0 else None,
        "checkpoint": str(checkpoint),
        "checkpoint_resume": {
            "training_steps": resumed.runtime.stats.training_steps,
            "ema_updates": resumed_runtime.ema.updates,
            "reset_id": resumed_runtime.stats.reset_id,
        },
        "resumed_schema_digest": post_resume["schema_digest"],
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "errors": errors,
        "dataset_sha256": provenance["dataset_sha256"],
    }
    atomic_json(Path(args.output), evidence)
    print(json.dumps(evidence, ensure_ascii=False, default=str))
    return 0 if evidence["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
