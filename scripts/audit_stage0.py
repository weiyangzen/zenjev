#!/usr/bin/env python3
"""Fail-closed Stage 0 secret, license, provenance, and path audit (ZJ-034)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "audits" / "stage0_audit.json"
BLUEPRINT = "Docs/stage0_zenjev_blueprint.md"
CONFIG_FILES = ("configs/example.yaml", "configs/jev_tool_task.yaml")
DATA_FILES = ("data/source.jsonl", "data/stage0_corpus.jsonl", "data/stage0_heldout.jsonl")
PROJECT_AUTHORED = {"data/stage0_corpus.jsonl", "data/stage0_heldout.jsonl"}
MODEL_MANIFEST = "artifacts/model_manifest.json"
MQ_PATH_FIELDS = ("dlq_path", "quarantine_path", "metrics_path")
TRUSTED_PREFIXES = ("runs/", "artifacts/")
EXCLUDED_PARTS = {".git", ".venv", "runs", "__pycache__"}
EXCLUDED_FILES = {"scripts/audit_stage0.py", "artifacts/audits/stage0_audit.json"}
EXTERNAL_PROVENANCE_KEYS = {"source", "uri", "url", "retrieved_at", "license", "license_note"}
ENV_REFERENCE = re.compile(r"^(?:env:|file:)")
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")

SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{20,}={0,3}")),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?P<key>[A-Za-z0-9_\-]*(?:api[_-]?key|secret|token|password|passwd|credential)[A-Za-z0-9_\-]*)"
    r"\s*[:=]\s*[\"'](?P<value>[^\"']{12,})[\"']"
)
PLACEHOLDER_MARKERS = (
    "plaintext",
    "example",
    "fixture",
    "redacted",
    "dummy",
    "fake",
    "changeme",
    "placeholder",
    "sample",
)


class Audit:
    def __init__(self) -> None:
        self.findings: list[dict] = []

    def record(self, check: str, status: str, detail: object) -> None:
        self.findings.append({"check": check, "status": status, "detail": detail})

    def ok(self, check: str, detail: object = "ok") -> None:
        self.record(check, "pass", detail)

    def fail(self, check: str, detail: object) -> None:
        self.record(check, "fail", detail)

    def info(self, check: str, detail: object) -> None:
        self.record(check, "info", detail)

    @property
    def failures(self) -> list[dict]:
        return [finding for finding in self.findings if finding["status"] == "fail"]


def _excluded(rel: Path) -> bool:
    if rel.as_posix() in EXCLUDED_FILES:
        return True
    if any(part in EXCLUDED_PARTS for part in rel.parts):
        return True
    return len(rel.parts) >= 2 and rel.parts[0] == "mq" and "target" in rel.parts


def _text_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if _excluded(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        yield rel.as_posix(), text


def _redact(value: str) -> str:
    return value[:4] + "..." if len(value) > 4 else "..."


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_secrets(audit: Audit) -> int:
    hits: list[dict] = []
    placeholders: list[dict] = []
    scanned = 0
    for rel, text in _text_files():
        scanned += 1
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                hits.append({"path": rel, "pattern": label, "line": _line(text, match.start()), "match": _redact(match.group(0))})
        for match in SECRET_ASSIGNMENT.finditer(text):
            value = match.group("value")
            entry = {
                "path": rel,
                "pattern": "assigned_secret",
                "key": match.group("key"),
                "line": _line(text, match.start()),
                "match": _redact(value),
            }
            if ENV_REFERENCE.match(value) or ENV_NAME.match(value):
                continue
            if any(marker in value.lower() for marker in PLACEHOLDER_MARKERS):
                placeholders.append(entry)
                continue
            hits.append(entry)
    if hits:
        audit.fail("secret_scan", {"hit_count": len(hits), "scanned_files": scanned, "hits": hits[:50]})
    else:
        audit.ok("secret_scan", {"hit_count": 0, "scanned_files": scanned})
    if placeholders:
        audit.info("secret_scan_placeholders", {"count": len(placeholders), "entries": placeholders})
    return scanned


def check_configs(audit: Audit) -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from jev.config import load_config

    loaded: dict = {}
    for rel in CONFIG_FILES:
        try:
            loaded[rel] = load_config(ROOT / rel)
        except Exception as exc:
            audit.fail("config_load", f"{rel}: {exc}")
    reference_failures: list[str] = []
    for rel, config in loaded.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", config.teacher.api_key_env):
            reference_failures.append(f"{rel}: teacher.api_key_env is not an environment variable name")
        mq = config.mq
        if mq is not None and mq.secret_ref is not None and not ENV_REFERENCE.match(mq.secret_ref):
            reference_failures.append(f"{rel}: mq.secret_ref is not an env:/file: reference")
    if reference_failures:
        audit.fail("config_secret_references", reference_failures)
    elif loaded:
        audit.ok("config_secret_references", sorted(loaded))
    return loaded


def check_mq_paths(audit: Audit, configs: dict) -> None:
    failures: list[str] = []
    checked = 0
    for rel, config in configs.items():
        mq = config.mq
        if mq is None:
            continue
        for field in MQ_PATH_FIELDS:
            value = str(getattr(mq, field, "") or "")
            checked += 1
            normalized = value.replace("\\", "/")
            if not value or Path(value).is_absolute() or not normalized.startswith(TRUSTED_PREFIXES):
                failures.append(f"{rel}: mq.{field}={value!r} is not under runs/ or artifacts/")
    if failures:
        audit.fail("mq_artifact_paths", failures)
    else:
        audit.ok("mq_artifact_paths", {"checked": checked})


def _read_jsonl(path: Path) -> list:
    records: list = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def check_data(audit: Audit, configs: dict) -> None:
    config_notes: dict[str, str] = {}
    for config in configs.values():
        for source in config.sources:
            note = source.license_note.strip()
            if note:
                config_notes[Path(source.location).name] = note
    results: list[dict] = []
    failures: list[str] = []
    for rel in DATA_FILES:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"{rel} is missing")
            results.append({"path": rel, "status": "missing"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
            records = _read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{rel} is not valid JSONL: {exc}")
            continue
        if not records:
            failures.append(f"{rel} is empty")
            continue
        keys = {key for record in records if isinstance(record, dict) for key in record}
        internal_license = any(
            isinstance(record, dict) and (record.get("license") or record.get("license_note")) for record in records
        )
        note = config_notes.get(path.name)
        external = sorted(keys & EXTERNAL_PROVENANCE_KEYS)
        if internal_license or note:
            status, evidence = "documented", note or "record-level license note"
        elif rel in PROJECT_AUTHORED and not external and not re.search(r"https?://", text):
            status, evidence = "project_authored", "synthetic Stage 0 corpus authored in-repo; no external source/URI/license metadata"
        else:
            status, evidence = "unlicensed", "no license note in records or config and not declared project-authored"
            failures.append(f"{rel} has no license note and is not project-authored")
        results.append(
            {"path": rel, "status": status, "records": len(records), "evidence": evidence, "external_keys": external}
        )
    if failures:
        audit.fail("data_files", {"failures": failures, "files": results})
    else:
        audit.ok("data_files", results)


def check_model_manifest(audit: Audit, configs: dict) -> None:
    path = ROOT / MODEL_MANIFEST
    if not path.exists():
        audit.fail("model_manifest", f"{MODEL_MANIFEST} is missing")
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        audit.fail("model_manifest", f"cannot parse {MODEL_MANIFEST}: {exc}")
        return
    missing = [key for key in ("model_id", "revision", "parameter_count", "license") if not manifest.get(key)]
    if missing:
        audit.fail("model_manifest", {"missing_fields": missing})
        return
    mismatched = [
        rel
        for rel, config in configs.items()
        if config.model_id != manifest["model_id"] or config.model_revision != manifest["revision"]
    ]
    if mismatched:
        audit.fail("model_manifest", {"mismatched_configs": mismatched, "manifest": MODEL_MANIFEST})
    else:
        audit.ok(
            "model_manifest",
            {"path": MODEL_MANIFEST, "model_id": manifest["model_id"], "revision": manifest["revision"]},
        )


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def check_checklist(audit: Audit) -> dict | None:
    source = (ROOT / BLUEPRINT).read_text(encoding="utf-8")
    try:
        parser = _load_module("validate_blueprint", ROOT / "scripts" / "validate_blueprint.py")
        items, spec_digest, _event = parser.parse(source)
    except Exception as exc:
        audit.fail("blueprint_checklist", f"cannot parse {BLUEPRINT}: {exc}")
        return None
    counts = {
        name: sum(item.mark == mark for item in items)
        for mark, name in ((" ", "todo"), ("_", "self_tested"), ("x", "accepted"))
    }
    report = {
        "counts": counts,
        "total": len(items),
        "spec_digest": spec_digest,
        "blueprint_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "blueprint_modified": False,
        "note": "counts are reported from the authoritative parser; this audit never mutates checklist state",
    }
    audit.info("blueprint_checklist", report)
    return report


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="destination JSON path")
    args = parser.parse_args(argv)
    audit = Audit()
    scanned = scan_secrets(audit)
    configs = check_configs(audit)
    check_mq_paths(audit, configs)
    check_data(audit, configs)
    check_model_manifest(audit, configs)
    checklist = check_checklist(audit)
    failures = audit.failures
    report = {
        "schema_version": 1,
        "pass": not failures,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repository_root": str(ROOT),
        "scanned_files": scanned,
        "findings": audit.findings,
        "checklist": checklist,
        "failure_count": len(failures),
    }
    atomic_write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
