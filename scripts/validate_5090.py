#!/usr/bin/env python3
"""Fail-closed RTX 5090 gate for the actual model runtime.

This intentionally does not treat ``nvidia-smi`` alone as a successful model
gate: PyTorch, CUDA visibility, and the GLiNER2 import must all work.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys


def main() -> int:
    required = {name: bool(importlib.util.find_spec(name)) for name in ("torch", "gliner2", "peft")}
    if not all(required.values()):
        print(json.dumps({"pass": False, "reason": "missing_runtime_dependencies", "dependencies": required}))
        return 2
    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"pass": False, "reason": "cuda_unavailable", "torch": torch.__version__}))
        return 2
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_memory": props.total_memory, "capability": f"{props.major}.{props.minor}"})
    if not any("5090" in item["name"] for item in devices):
        print(json.dumps({"pass": False, "reason": "RTX_5090_not_found", "devices": devices}))
        return 2
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False)
    print(json.dumps({"pass": True, "torch": torch.__version__, "cuda": torch.version.cuda, "devices": devices, "nvidia_smi": result.stdout.strip()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
