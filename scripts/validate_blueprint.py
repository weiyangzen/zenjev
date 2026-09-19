#!/usr/bin/env python3
"""Fail-closed checks for Jev's authoritative Stage0 checklist and Gantt."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = ROOT / "Docs/stage0_zenjev_blueprint.md"
GANTT = ROOT / "Docs/stage0_zenjev_gantt.md"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    source = BLUEPRINT.read_text(encoding="utf-8")
    rows = re.findall(r"^- \[([ _x])\] \*\*(ZJ-\d{3})\*\* .*?\*\*Depends on:\*\* ([^.]*)\.", source, re.M)
    if not rows:
        raise SystemExit("no authoritative checklist rows")
    ids = [item_id for _, item_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate checklist IDs")
    deps = {item_id: [x.strip() for x in raw.split(",") if x.strip() != "—"] for _, item_id, raw in rows}
    unknown = sorted({dep for values in deps.values() for dep in values if dep not in deps})
    if unknown:
        raise SystemExit(f"unknown dependencies: {unknown}")
    state = {item_id: 0 for item_id in ids}
    def visit(item_id: str) -> None:
        if state[item_id] == 1:
            raise SystemExit(f"dependency cycle at {item_id}")
        if state[item_id] == 2:
            return
        state[item_id] = 1
        for dep in deps[item_id]:
            visit(dep)
        state[item_id] = 2
    for item_id in ids:
        visit(item_id)
    gantt = GANTT.read_text(encoding="utf-8")
    if gantt.count("Source SHA-256:") != 1 or digest(BLUEPRINT) not in gantt:
        raise SystemExit("Gantt source digest is stale or missing")
    projected = re.findall(r"^\| (ZJ-\d{3}) \|", gantt, re.M)
    if projected != ids:
        raise SystemExit("Gantt monitoring index is not a one-to-one checklist projection")
    print(f"blueprint_ok ids={len(ids)} digest={digest(BLUEPRINT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
