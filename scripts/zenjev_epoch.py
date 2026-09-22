#!/usr/bin/env python3
"""One-time ledger epoch alignment (ZJ-081).

LoRA generation ids were originally issued every few steps, so the historical
ratio between consumed records and generations is far below the operating
target. This tool archives the current ledger and starts a fresh epoch whose
base generation id is aligned to the consumed-record count:

    base = floor(records_consumed / records_per_generation)

The running loop seeds its counter from the ledger maximum, so after the epoch
the ratio stays at ``records_per_generation`` (default 10000) as long as the
publish cadence matches (``publish_every_steps * train_batch == records_per_generation``).

The archive keeps full audit history; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from zenjev_lib import ARTIFACTS, PERPETUAL_RUNS, append_jsonl, atomic_write_json, utc_now

REPO = pathlib.Path(__file__).resolve().parents[1]
LEDGER_PATH = PERPETUAL_RUNS / "generations.jsonl"


def load_rows() -> list[dict]:
    rows: list[dict] = []
    try:
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    except OSError:
        pass
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-per-generation", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.records_per_generation < 1:
        raise SystemExit("--records-per-generation must be positive")

    rows = load_rows()
    if not rows:
        raise SystemExit("no ledger to align")
    heartbeat = json.loads((ARTIFACTS / "loop.json").read_text()) if (ARTIFACTS / "loop.json").exists() else {}
    live = heartbeat.get("counters") or {}
    records = max(int(row.get("records_consumed_total") or 0) for row in rows)
    steps = max(int(row.get("training_steps") or 0) for row in rows)
    ema_step = max(int(row.get("ema_step") or 0) for row in rows)
    old_max = max(int(row.get("generation_id") or 0) for row in rows)
    base = records // args.records_per_generation
    aligned_records = base * args.records_per_generation

    result = {
        "ledger_entries": len(rows),
        "old_max_generation": old_max,
        "records_consumed_total": records,
        "records_per_generation": args.records_per_generation,
        "new_base_generation": base,
        "aligned_records": aligned_records,
        "actual_records": records,
        "ratio_after_epoch": args.records_per_generation if base else None,
    }
    if args.dry_run:
        print(json.dumps(result, ensure_ascii=False))
        return 0

    archive = LEDGER_PATH.with_name(
        f"generations.epoch-{base}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    )
    if LEDGER_PATH.exists():
        LEDGER_PATH.replace(archive)
    append_jsonl(
        LEDGER_PATH,
        {
            "session_pid": None,
            "generation_id": base,
            "parent_digest": None,
            "adapter_digest": None,
            "ema_step": ema_step,
            "training_steps": steps,
            "reset_id": 0,
            "records_consumed_total": records,
            "next_publish_at": (base + 1) * args.records_per_generation,
            "training_examples_total": 0,
            "pairs_built_total": int(live.get("pairs_built") or 0),
            "pairs_trained_total": int(live.get("pairs_trained") or 0),
            "loss": None,
            "loss_ema": None,
            "created_at": utc_now(),
            "note": "epoch alignment",
            "archived_ledger": str(archive),
        },
    )
    atomic_write_json(
        PERPETUAL_RUNS / "epoch.json",
        {
            **result,
            "aligned_at": utc_now(),
            "archived_ledger": str(archive),
            "ledger": str(LEDGER_PATH),
        },
    )
    print(json.dumps({**result, "archived_ledger": str(archive)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
