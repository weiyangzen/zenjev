#!/usr/bin/env python3
"""Fail-closed NVIDIA GPU gate for the actual model runtime.

This intentionally does not treat ``nvidia-smi`` alone as a successful model
gate: PyTorch, CUDA visibility, and the GLiNER2 import must all work.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _emit(payload: dict, output: str | None) -> None:
    body = json.dumps(payload, ensure_ascii=False)
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(body + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="optional JSON evidence path")
    args = parser.parse_args()
    required = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "gliner2", "peft")}
    if not all(required.values()):
        _emit({"pass": False, "reason": "missing_runtime_dependencies", "dependencies": required}, args.output)
        return 2
    import torch

    if not torch.cuda.is_available():
        _emit({"pass": False, "reason": "cuda_unavailable", "torch": torch.__version__}, args.output)
        return 2
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_memory": props.total_memory, "capability": f"{props.major}.{props.minor}"})
    if not devices:
        _emit({"pass": False, "reason": "nvidia_gpu_not_found", "devices": devices}, args.output)
        return 2
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        _emit({"pass": False, "reason": "nvidia_smi_unavailable", "devices": devices, "stderr": result.stderr.strip()}, args.output)
        return 2
    _emit({"pass": True, "torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices, "nvidia_smi": result.stdout.strip()}, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
