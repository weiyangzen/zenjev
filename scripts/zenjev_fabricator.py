#!/usr/bin/env python3
"""Synthetic schema-task fabricator (§1.9).

Generates schema-conforming ``tool_task_pair`` envelopes from a seeded grammar
and appends them to ``runs/jev/feed/fabricator.ndjson``. Every record carries
``synthetic: true``, a ``synthetic://`` source URI, and ``canary: true`` with an
expected extraction/route so the console can show live accuracy. Synthetic
records are counted separately and are never admitted to LoRA training (the
loop only trains on records without the synthetic flag).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
from typing import Any

from zenjev_lib import FEED_DIR, append_jsonl, ensure_dirs, sha256_hex, utc_now, write_heartbeat

from jev.config import load_config
from jev.mq import build_envelope

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/zenjev_perpetual.yaml"
SPOOL_PATH = FEED_DIR / "fabricator.ndjson"

INSTRUCTS = [
    "You are a senior engineer. Answer concisely and name the exact tools.",
    "Follow the project conventions and prefer the smallest safe change.",
    "Explain the tradeoffs, then give the concrete implementation steps.",
    "Review the request and produce a minimal reproducible plan.",
]
PROMPTS = [
    "Fix the flaky test in the {language} parser and add a regression test.",
    "Refactor the {framework} data loader so batch retries are idempotent.",
    "Add a health check endpoint backed by {technology} and document it.",
    "Instrument the {language} worker with structured logs and a {tool} script.",
    "Migrate the schema to {technology} and keep backward compatibility.",
    "Write a benchmark for the {framework} hot path and compare against {tool}.",
]
STACKS: list[dict[str, Any]] = [
    {"language": "Python", "framework": "PyTorch", "technology": "GLiNER2", "tool": "python", "choice": "PyTorch"},
    {"language": "Python", "framework": "PEFT", "technology": "Transformers", "tool": "python", "choice": "PEFT"},
    {"language": "SQL", "framework": "PostgreSQL", "technology": "PostgreSQL", "tool": "sql", "choice": "PostgreSQL"},
    {"language": "Go", "framework": "Kubernetes", "technology": "Docker", "tool": "docker", "choice": "Kubernetes"},
    {"language": "TypeScript", "framework": "JavaScript", "technology": "Docker", "tool": "browser", "choice": "Docker"},
    {"language": "Rust", "framework": "GLiNER2", "technology": "PyTorch", "tool": "git", "choice": "Rust"},
    {"language": "bash", "framework": "Docker", "technology": "Kubernetes", "tool": "bash", "choice": "Kubernetes"},
    {"language": "JavaScript", "framework": "Transformers", "technology": "GLiNER2", "tool": "git", "choice": "GLiNER2"},
]
TASK_TYPES = {
    "Fix the flaky test": "debugging",
    "Refactor the": "coding",
    "Add a health check": "coding",
    "Instrument the": "operations",
    "Migrate the schema": "coding",
    "Write a benchmark": "data_analysis",
}
RESPONSE_SHAPES = [
    "Use {tool} to reproduce, then patch the {framework} module in {language}. Run the suite before and after.",
    "The smallest safe change touches the {technology} adapter. Validate with {tool} and keep the old path behind a flag.",
    "Recommended stack: {language} + {framework}. Use {tool} for the regression and record p95 latency.",
    "Do it in two steps: reproduce with {tool}, then update the {language} code to use {technology}.",
]


def build_record(rng: random.Random, sequence: int) -> dict[str, Any]:
    instruct = rng.choice(INSTRUCTS)
    prompt_template = rng.choice(PROMPTS)
    stack = rng.choice(STACKS)
    prompt = prompt_template.format(**stack)
    request_text = f"{instruct}\n\nTask: {prompt}"
    response_text = rng.choice(RESPONSE_SHAPES).format(**stack)
    task_type = next((kind for prefix, kind in TASK_TYPES.items() if prompt.startswith(prefix)), "coding")
    request_id = f"fab-{sequence:08d}-{rng.randrange(16**6):06x}"
    return {
        "record_id": request_id,
        "observed_model": {"provider": "zenjev-fabricator", "model_id": "gpt-5.6-sol"},
        "request": {"text": request_text},
        "response": {"text": response_text},
        "synthetic": True,
        "canary": True,
        "expected": {
            "task_type": task_type,
            "decision_choice": stack["choice"],
            "tool": stack["tool"],
            "technology": stack["technology"],
        },
        "stack": stack,
        "created_at": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--per-tick", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--count", type=int, default=0, help="emit N records then exit (smoke)")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config(CONFIG_PATH)
    rng = random.Random(args.seed)
    sequence = 0
    emitted = 0
    started = time.time()
    spool = SPOOL_PATH.open("a", encoding="utf-8")
    try:
        while True:
            tick_started = time.time()
            for _ in range(max(1, args.per_tick)):
                sequence += 1
                record = build_record(rng, sequence)
                envelope = build_envelope(
                    record_id=record["record_id"],
                    kind="tool_task_pair",
                    source={
                        "uri": f"synthetic://dashboard/{record['record_id']}",
                        "content_sha256": sha256_hex(json.dumps(record, sort_keys=True)),
                        "retrieved_at": record["created_at"],
                        "license": "project-authored-synthetic",
                    },
                    schema_id=config.schema.name,
                    schema_version=config.schema.version,
                    schema_digest=config.schema.digest(),
                    observed_model=record["observed_model"],
                    request=record["request"],
                    response=record["response"],
                    synthetic=True,
                    canary=True,
                    expected=record["expected"],
                    fabricator={"build": "zenjev_fabricator/1", "seed": args.seed, "sequence": sequence},
                )
                spool.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                emitted += 1
            spool.flush()
            write_heartbeat(
                "fabricator",
                {
                    "state": "running",
                    "uptime_seconds": round(time.time() - started, 1),
                    "emitted_total": emitted,
                    "interval_seconds": args.interval,
                    "per_tick": max(1, args.per_tick),
                    "tick_seconds": round(time.time() - tick_started, 3),
                    "spool": str(SPOOL_PATH.relative_to(REPO)),
                },
            )
            if args.once or (args.count and emitted >= args.count):
                print(json.dumps({"emitted": emitted, "spool": str(SPOOL_PATH)}))
                return 0
            time.sleep(max(0.2, args.interval))
    finally:
        spool.close()


if __name__ == "__main__":
    raise SystemExit(main())
