#!/usr/bin/env python3
"""Write the Stage 0 reproducibility manifest to artifacts/repro/ (ZJ-041)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "repro" / "repro_manifest.json"

HASHED_FILES = (
    "Docs/stage0_zenjev_blueprint.md",
    "configs/example.yaml",
    "configs/jev_tool_task.yaml",
    "artifacts/model_manifest.json",
)
LOCKFILES = ("uv.lock", "mq/Cargo.lock")
CONFIG_FILES = ("configs/example.yaml", "configs/jev_tool_task.yaml")

VALIDATION_COMMANDS = (
    {"command": "python3 -m compileall -q jev scripts", "cwd": ".", "evidence": "exit code 0"},
    {"command": "python3 -m pytest -q", "cwd": ".", "evidence": "pytest summary"},
    {"command": "python3 scripts/validate_blueprint.py --check", "cwd": ".", "evidence": "stdout blueprint_ok"},
    {"command": "python3 scripts/smoke.py", "cwd": ".", "evidence": "stdout jev_smoke_ok"},
    {"command": "cargo fmt --check", "cwd": "mq", "evidence": "exit code 0"},
    {"command": "cargo clippy --locked -- -D warnings", "cwd": "mq", "evidence": "exit code 0"},
    {"command": "cargo test --locked", "cwd": "mq", "evidence": "cargo test summary"},
    {"command": "python3 scripts/smoke_mq.py", "cwd": ".", "evidence": "artifacts/mq/smoke.json"},
    {"command": "python3 -m jev.cli validate --config configs/example.yaml", "cwd": ".", "evidence": "stdout JSON"},
    {"command": "python3 -m jev.cli validate --config configs/jev_tool_task.yaml", "cwd": ".", "evidence": "stdout JSON"},
    {"command": "python3 -m jev.cli extract --config configs/example.yaml --text '2026最佳技术栈包括 Python 和 PostgreSQL'", "cwd": ".", "evidence": "stdout inference JSON"},
    {"command": "python3 scripts/recovery_drill.py --config configs/example.yaml --model-path /home/sansha/jev-model-base --output artifacts/recovery/collapse_recovery.json", "cwd": ".", "evidence": "artifacts/recovery/collapse_recovery.json"},
    {"command": "python3 scripts/validate_nvidia_gpu.py", "cwd": ".", "evidence": "stdout JSON pass field"},
    {"command": "JEV_MODEL_PATH=/home/sansha/jev-model-base python3 scripts/smoke_nvidia_gpu.py --output artifacts/nvidia_gpu_train_ema_smoke.json", "cwd": ".", "evidence": "artifacts/nvidia_gpu_train_ema_smoke.json"},
    {"command": "python3 scripts/audit_stage0.py", "cwd": ".", "evidence": "artifacts/audits/stage0_audit.json"},
    {"command": "python3 scripts/build_repro_bundle.py", "cwd": ".", "evidence": "artifacts/repro/repro_manifest.json"},
)

DEFERRED_COMMANDS = (
    {
        "command": "python3 scripts/smoke_mq_nvidia_gpu.py",
        "cwd": ".",
        "reason": "requires a live external broker on the designated NVIDIA GPU host; G11 soak not performed in this environment",
    },
)


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    fallback = Path.home() / ".cargo" / "bin" / name
    return str(fallback) if fallback.exists() else None


def command_version(name: str, *args: str) -> str | None:
    path = executable(name)
    if path is None:
        return None
    try:
        result = subprocess.run([path, *args], capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    return first[0] if first else None


def git_output(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def gather_versions() -> dict:
    versions: dict = {
        "python": platform.python_version(),
        "python_full": " ".join(sys.version.split()),
        "torch": None,
        "torch_cuda": None,
        "cuda_available": None,
        "rustc": command_version("rustc", "--version"),
        "cargo": command_version("cargo", "--version"),
    }
    try:
        import torch
    except Exception:
        return versions
    versions["torch"] = getattr(torch, "__version__", None)
    versions["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        versions["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        versions["cuda_available"] = None
    return versions


def gather_configs() -> dict:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from jev.config import load_config
    except Exception as exc:
        return {rel: {"error": f"import failed: {exc}"} for rel in CONFIG_FILES}
    digests: dict = {}
    for rel in CONFIG_FILES:
        try:
            config = load_config(ROOT / rel)
            digests[rel] = {
                "config_digest": config.digest(),
                "schema_digest": config.schema.digest(),
                "model_id": config.model_id,
                "model_revision": config.model_revision,
            }
        except Exception as exc:
            digests[rel] = {"error": str(exc)}
    return digests


def build_manifest() -> dict:
    configs = gather_configs()
    primary = configs.get("configs/example.yaml", {})
    files: dict = {}
    for rel in HASHED_FILES:
        path = ROOT / rel
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size if path.exists() else None}
    status_lines = [line for line in git_output("status", "--porcelain").splitlines() if line]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repository_root": str(ROOT),
        "git": {
            "commit": git_output("rev-parse", "HEAD").strip(),
            "status_porcelain": status_lines,
        },
        "files": files,
        "config_digest": primary.get("config_digest"),
        "schema_digest": primary.get("schema_digest"),
        "configs": configs,
        "lockfiles": {rel: sha256_file(ROOT / rel) for rel in LOCKFILES},
        "versions": gather_versions(),
        "validation_commands": list(VALIDATION_COMMANDS),
        "deferred_commands": list(DEFERRED_COMMANDS),
    }


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
    manifest = build_manifest()
    atomic_write_json(Path(args.output), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
