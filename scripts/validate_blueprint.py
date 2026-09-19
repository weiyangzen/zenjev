#!/usr/bin/env python3
"""Generate/check the Stage 0 projection. Never starts an execution controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT_REL = Path("Docs/stage0_zenjev_blueprint.md")
BLUEPRINT = ROOT / BLUEPRINT_REL
CONCURRENCY_KEYS = {
    "logical_claim_cap", "persistent_service_record_cap", "agent_execution_cap",
    "startup_reservation_cap", "launch_fanout_per_wave", "live_transport_cap",
    "authenticated_goal_cap", "running_turn_cap", "outbound_request_rate",
    "outbound_request_window_seconds", "in_flight_request_cap",
    "max_outstanding_requests_per_execution", "integration_cap", "validator_cap",
    "exact_path_conflict_cap", "desired_live_target", "hard_worker_cap",
    "accelerator_validator_leases",
}
POLICY_KEYS = {
    "operator_prompt", "operator_prompt_sha256", "platform", "route_policy",
    "worker_lifecycle", "nested_agent_policy", "continuation_policy",
    "replacement_policy", "terminal_stop_policy", "scheduler_cadence",
    "lease_policy", "budgets", "cron_marker", "cooldown_policy",
    "breaker_reset_policy",
}
MARKS = {" ": "todo", "_": "self_tested", "x": "accepted"}
ROW = re.compile(
    r"^- \[([ _x])\] \*\*(ZJ-\d{3})\*\* (.+?) "
    r"\*\*Depends on:\*\* (.+?)\. \*\*Owner:\*\* (.+?)\. "
    r"\*\*Owned paths:\*\* (.+?)\. \*\*Gate:\*\* (.+?)\.$"
)


class GovernanceError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    mark: str
    id: str
    title: str
    dependencies: tuple[str, ...]
    owner: str
    paths: tuple[str, ...]
    gates: str


def companion_path(blueprint: Path) -> Path:
    """Skill naming is case-sensitive; preserve the entire preceding stem."""
    stem = blueprint.stem
    stem = stem[:-len("Blueprint")] + "Gantt" if stem.endswith("Blueprint") else stem + "_Gantt"
    return blueprint.with_name(stem + blueprint.suffix)


GANTT = companion_path(BLUEPRINT)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_block(source: str, name: str) -> dict:
    pattern = rf"<!-- {re.escape(name)}:start -->\n```json\n(.*?)\n```\n<!-- {re.escape(name)}:end -->"
    matches = re.findall(pattern, source, re.S)
    if len(matches) != 1:
        raise GovernanceError(f"expected one {name} JSON block")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise GovernanceError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise GovernanceError(f"{name} must be an object")
    return value


def timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise GovernanceError("timestamp must be UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise GovernanceError("invalid timestamp") from exc


def parse(source: str) -> tuple[list[Item], str, dict]:
    items: list[Item] = []
    for line in source.splitlines():
        # Any task-like row must be recognized; invalid marks cannot disappear.
        if re.match(r"^\s*[-*+]\s+\[", line):
            match = ROW.fullmatch(line)
            if not match:
                raise GovernanceError(f"invalid checklist row: {line[:100]}")
            mark, item_id, title, raw_deps, owner, raw_paths, gates = match.groups()
            deps = () if raw_deps == "—" else tuple(raw_deps.split(","))
            if any(not re.fullmatch(r"ZJ-\d{3}", dep) for dep in deps) or len(set(deps)) != len(deps):
                raise GovernanceError(f"invalid/duplicate dependency for {item_id}")
            paths = tuple(re.findall(r"`([^`]+)`", raw_paths))
            if not paths:
                raise GovernanceError(f"missing ownership for {item_id}")
            for path in paths:
                if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or "\\" in path or "|" in path:
                    raise GovernanceError(f"unsafe owned path for {item_id}")
            items.append(Item(mark, item_id, title, deps, owner, paths, gates))
    if not items or len({item.id for item in items}) != len(items):
        raise GovernanceError("missing or duplicate checklist IDs")
    lookup = {item.id: item for item in items}
    visited, active = set(), set()

    def visit(item_id: str) -> None:
        if item_id in active:
            raise GovernanceError(f"dependency cycle at {item_id}")
        if item_id in visited:
            return
        active.add(item_id)
        for dependency in lookup[item_id].dependencies:
            if dependency not in lookup:
                raise GovernanceError(f"unknown dependency {dependency}")
            visit(dependency)
        active.remove(item_id)
        visited.add(item_id)

    for item in items:
        visit(item.id)
        if item.mark == "x" and any(lookup[dep].mark != "x" for dep in item.dependencies):
            raise GovernanceError(f"accepted item {item.id} has unfinished dependencies")
    spec_start = "## 1. Frozen repository and execution specification\n"
    spec_end = "## 2. Acceptance gates\n"
    if source.count(spec_start) != 1 or source.count(spec_end) != 1:
        raise GovernanceError("missing or duplicated specification boundary")
    specification = source.split(spec_start, 1)[1].split(spec_end, 1)[0]
    spec_digest = sha256(spec_start + specification)
    prerequisites = json_block(source, "execution-prerequisites")
    if set(prerequisites) != POLICY_KEYS | {"schema_version", "controller_enabled", "concurrency"}:
        raise GovernanceError("execution prerequisite schema differs from the disabled policy")
    if prerequisites["schema_version"] != 1 or prerequisites["controller_enabled"] is not False:
        raise GovernanceError("controller activation needs a separately reviewed policy migration")
    if any(prerequisites[key] is not None for key in POLICY_KEYS):
        raise GovernanceError("unresolved operator policy may not contain invented defaults")
    concurrency = prerequisites["concurrency"]
    if not isinstance(concurrency, dict) or set(concurrency) != CONCURRENCY_KEYS or any(value is not None for value in concurrency.values()):
        raise GovernanceError("all missing operator concurrency dimensions must remain unresolved")
    expected = companion_path(BLUEPRINT_REL).as_posix()
    if f"| Same-name Gantt companion | `{expected}`" not in source:
        raise GovernanceError("incorrect companion filename in specification")
    obsolete_companion = BLUEPRINT_REL.with_name("stage0_zenjev_gantt.md").as_posix()
    if obsolete_companion in source:
        raise GovernanceError("obsolete companion reference")
    event = json_block(source, "governance-event")
    if set(event) != {"event_id", "recorded_at", "description"} or not re.fullmatch(r"[a-z0-9-]+", event["event_id"]):
        raise GovernanceError("invalid recorded governance event")
    if timestamp(event["recorded_at"]) > datetime.now(timezone.utc):
        raise GovernanceError("recorded event cannot be in the future")
    return items, spec_digest, event


def render(source: str, generated_at: str) -> str:
    items, spec_digest, event = parse(source)
    if timestamp(generated_at) < timestamp(event["recorded_at"]):
        raise GovernanceError("generation precedes source event")
    counts = {mark: sum(item.mark == mark for item in items) for mark in MARKS}
    lookup = {item.id: item for item in items}
    frontier: dict[str, str] = {}
    for item in items:
        ready = all(lookup[dep].mark == "x" for dep in item.dependencies)
        frontier[item.id] = "complete" if item.mark == "x" else (
            "integration" if ready and item.mark == "_" else "implementation" if ready else "waiting_dependencies"
        )
    lines = [
        "# Stage 0 ZenJev Gantt / Kanban projection", "",
        "Generated read-only projection. The sole authority is [stage0_zenjev_blueprint.md](stage0_zenjev_blueprint.md).",
        "No implementation acceptance is implied by generating this file.", "",
        f"**Source blueprint:** `{BLUEPRINT_REL.as_posix()}`",
        f"**Source SHA-256:** `{sha256(source)}`",
        f"**Specification SHA-256:** `{spec_digest}`",
        f"**Generated at (UTC):** `{generated_at}`",
        "**Generation command:** `python3 scripts/validate_blueprint.py --generate`",
        "**Check command:** `python3 scripts/validate_blueprint.py --check`", "",
        "## Monitoring summary", "", "| Surface | Value |", "|---|---|",
        f"| Checklist items | {len(items)} |",
        f"| Todo / self-tested / Master-accepted | {counts[' ']} / {counts['_']} / {counts['x']} |",
        "| Controller | disabled; operator concurrency/lifecycle/route unresolved |",
        "| Logical claims / service records / admitted executions | not applicable; no controller ledger |",
        "| Startup reservations / live transports / authenticated goals / running turns | not applicable; no controller ledger |",
        "| Request starts/window / in-flight / outstanding requests | not applicable; no controller ledger |",
        "| Validator / integration leases; handoff / integration / repair backlog | not applicable; no controller ledger |",
        "| Breaker / saturation / underfill | not applicable; activation disabled, no capacity target authorized |",
        "| Telemetry scope | future repository controller only; does not count the current interactive session |",
        "| Timing | every checklist item is unscheduled |", "",
        "## Recorded event Gantt", "",
        "Only the recorded governance edit is plotted. It is a zero-duration event, not a task estimate or completion date.", "",
        "```mermaid", "gantt", "    title Recorded governance event (UTC)",
        "    dateFormat YYYY-MM-DDTHH:mm:ss[Z]", "    axisFormat %Y-%m-%d %H:%M",
        "    section Recorded events",
        f"    Governance policy correction :milestone, {event['event_id']}, {event['recorded_at']}, 0d",
        "```", "", "## Unscheduled monitoring index", "",
        "Each stable checklist ID has exactly one row. Owners are planned roles; no row claims a live execution.",
        "The validation-preparation frontier consists of unfinished items; it authorizes preparing checks, never early acceptance.", "",
        "| ID | State | Depends on | Planned owner | Owned paths | Claim / run | Frontier | Validation preparation | Startup / live / handoff / integration / repair | Blocked | Timing | Gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in items:
        unfinished = [dep for dep in item.dependencies if lookup[dep].mark != "x"]
        blocked = "unfinished dependencies: " + ",".join(unfinished) if unfinished else "none from DAG"
        lines.append(
            f"| {item.id} | {MARKS[item.mark]} | {','.join(item.dependencies) or '—'} | {item.owner} | "
            f"{', '.join('`' + path + '`' for path in item.paths)} | unclaimed / none | {frontier[item.id]} | "
            f"{'complete' if item.mark == 'x' else 'pending'} | not applicable (controller disabled) | {blocked} | Unscheduled | {item.gates} |"
        )
    lines += ["", "## Reconciliation", "",
        "The checker reconstructs this entire file from the authoritative source and recorded generation time. Any stale digest, count, dependency, owner, owned path, state, frontier, omitted/duplicate ID or invented timing fails verification. Generation uses an atomic replacement and never edits authoritative checklist state. Future controller activation requires a policy migration and ledger-aware projection; this tool cannot activate it.", ""]
    return "\n".join(lines)


def check_projection(source: str, projection: str) -> None:
    times = re.findall(r"^\*\*Generated at \(UTC\):\*\* `(.*?)`", projection, re.M)
    if len(times) != 1:
        raise GovernanceError("missing or duplicate generation timestamp")
    if timestamp(times[0]) > datetime.now(timezone.utc):
        raise GovernanceError("generation timestamp is in the future")
    expected = render(source, times[0])
    if projection != expected:
        raise GovernanceError("Gantt projection differs from authoritative source; regenerate")


def atomic_write(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="atomically regenerate the read-only companion")
    mode.add_argument("--check", action="store_true", help="validate without writes (default)")
    parser.add_argument(
        "--generated-at",
        help="UTC generation timestamp (YYYY-MM-DDTHH:MM:SSZ); defaults to now for --generate",
    )
    args = parser.parse_args(argv)
    try:
        source = BLUEPRINT.read_text(encoding="utf-8")
        items, _, _ = parse(source)
        if args.generate:
            generated_at = args.generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            timestamp(generated_at)
            atomic_write(GANTT, render(source, generated_at))
        check_projection(source, GANTT.read_text(encoding="utf-8"))
        if (ROOT / BLUEPRINT_REL.with_name("stage0_zenjev_gantt.md")).exists():
            raise GovernanceError("obsolete misnamed Gantt still exists")
    except (GovernanceError, OSError) as exc:
        parser.exit(1, f"governance_error: {exc}\n")
    print(f"blueprint_ok ids={len(items)} digest={sha256(source)} controller=disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
