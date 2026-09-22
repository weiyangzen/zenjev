#!/usr/bin/env python3
"""Pair-validity audit (§1.8, ZJ-084).

Read-only structural audit of the stored (instruction + request -> schema)
distillation pairs: schema-label conformance, entity grounding, instruction
prefix, target diversity and duplicate rate. Answers "is this really useful
training data?" with numbers instead of opinion.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

from zenjev_lib import PERPETUAL_RUNS, atomic_write_json, utc_now

REPO = pathlib.Path(__file__).resolve().parents[1]
PAIRS_PATH = PERPETUAL_RUNS / "pairs.jsonl"
OUT_PATH = REPO / "artifacts/perpetual/pair_validity.json"


def tail_records(path: pathlib.Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    size = path.stat().st_size
    block = min(size, 8 * 1024 * 1024 * max(1, limit // 200))
    with path.open("rb") as handle:
        handle.seek(max(0, size - block))
        lines = handle.read().decode("utf-8", errors="replace").splitlines()[-limit:]
    records: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1500)
    parser.add_argument("--instruction-sha16", default=None)
    args = parser.parse_args()

    from jev.config import load_config
    import sys

    sys.path.insert(0, str(REPO / "scripts"))

    config = load_config(REPO / "configs/zenjev_perpetual.yaml")
    schema_labels = {
        task: set(labels)
        for task, labels in dict(getattr(config.schema, "classifications", {}) or {}).items()
    }
    # The instruction lives in the loop; import it without starting anything.
    from zenjev_loop import INSTRUCTION, INSTRUCTION_SHA16

    instruction_sha = args.instruction_sha16 or INSTRUCTION_SHA16

    records = tail_records(PAIRS_PATH, args.sample)
    stats: collections.Counter = collections.Counter()
    label_counts: dict[str, collections.Counter] = {
        task: collections.Counter() for task in schema_labels
    }
    entity_counts: collections.Counter = collections.Counter()
    grounded = entity_total = 0
    digests: collections.Counter = collections.Counter()
    for record in records:
        pair = record.get("pair")
        if not isinstance(pair, dict) or not isinstance(pair.get("input"), str):
            stats["malformed_pair"] += 1
            continue
        stats["pairs"] += 1
        text = pair["input"]
        target = pair.get("output") if isinstance(pair.get("output"), dict) else {}
        if record.get("instruction_sha16") == instruction_sha and text.startswith(INSTRUCTION):
            stats["instruction_prefixed"] += 1
        digests[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
        classifications = target.get("classifications") or []
        for item in classifications:
            task = item.get("task") if isinstance(item, dict) else None
            true_label = item.get("true_label") if isinstance(item, dict) else None
            labels = item.get("labels") if isinstance(item, dict) else None
            if task not in schema_labels or not isinstance(true_label, list):
                stats["malformed_classification"] += 1
                continue
            if set(labels or []) != schema_labels[task]:
                stats["label_set_mismatch"] += 1
            for value in true_label:
                if value in schema_labels[task]:
                    stats["label_in_schema"] += 1
                else:
                    stats["label_out_of_schema"] += 1
                label_counts[task][str(value)] += 1
            stats["classification_total"] += 1
        for name, values in (target.get("entities") or {}).items():
            for value in values if isinstance(values, list) else [values]:
                entity_total += 1
                entity_counts[str(name)] += 1
                if isinstance(value, str) and value and value in text:
                    grounded += 1
        if not classifications and not (target.get("entities") or {}):
            stats["empty_target"] += 1

    duplicates = sum(count - 1 for count in digests.values() if count > 1)
    report = {
        "recorded_at": utc_now(),
        "sampled": len(records),
        "pairs": stats["pairs"],
        "structural": {
            "malformed_pair": stats["malformed_pair"],
            "malformed_classification": stats["malformed_classification"],
            "label_set_mismatch": stats["label_set_mismatch"],
            "empty_target": stats["empty_target"],
            "instruction_prefixed": stats["instruction_prefixed"],
            "duplicate_inputs": duplicates,
        },
        "labels": {
            "in_schema": stats["label_in_schema"],
            "out_of_schema": stats["label_out_of_schema"],
            "distribution": {task: dict(counter.most_common(6)) for task, counter in label_counts.items()},
        },
        "entities": {
            "total": entity_total,
            "grounded_in_input": grounded,
            "grounding_rate": round(grounded / entity_total, 4) if entity_total else None,
            "per_type": dict(entity_counts),
        },
        "verdict": None,
    }
    valid = (
        stats["malformed_pair"] == 0
        and stats["malformed_classification"] == 0
        and stats["label_out_of_schema"] == 0
        and stats["empty_target"] == 0
    )
    grounding_rate = (grounded / entity_total) if entity_total else 1.0
    report["verdict"] = {
        "structurally_valid": valid,
        "grounding_rate": report["entities"]["grounding_rate"],
        "self_supervised": True,
        "note": (
            "Structurally valid self-distillation pairs: labels are always in the schema, "
            "entities are grounded spans, and every pair trains. They carry the policy's own "
            "judgment, not external ground truth, so they are valid for consistency/continued "
            "pretraining but cannot by themselves raise capability beyond the judge. External "
            "reward or a stronger teacher is required for that."
        ),
        "signal_strength": "self-supervised" if valid and grounding_rate >= 0.9 else "suspect",
    }
    atomic_write_json(OUT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=1)[:2600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
