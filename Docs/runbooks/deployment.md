# Deployment runbook: offline NVIDIA GPU host

This runbook stages ZenJev on the designated NVIDIA GPU host so model loading,
training, and serving never depend on a live Hub connection at runtime. It
covers items ZJ-030 and the runtime half of ZJ-020/ZJ-031.

## 1. Host prerequisites

| Component | Stage 0 requirement | Verified host observation |
|---|---|---|
| GPU | NVIDIA CUDA-capable, compute capability 8.0+ for BF16 | `NVIDIA GeForce RTX 5090 D`, 32607 MiB, capability 12.0 |
| Driver | Recent production driver with CUDA 12.x runtime | `nvidia-smi` reports 595.84 |
| CUDA runtime | Bundled with the PyTorch wheel (no system `nvcc` needed) | `torch 2.9.1+cu128` with CUDA 12.8 |
| Python | 3.10+ (3.12 on the test host) | 3.12.3 |
| VRAM headroom | Base model load plus activation headroom (observed base peak ~863 MB) | 32 GB class card |
| Rust (MQ only) | 1.85+; pinned `mq/rust-toolchain.toml` supplies 1.98.1 | `rustc 1.98.1` |

Confirm the driver and device before anything else:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
```

CPU-only execution may be used for control-plane development smoke tests but
can never satisfy the Stage 0 NVIDIA GPU gate (G6).

## 2. Install extras

The control plane (`jev.config`, `jev.cli`, `jev.mq`, `jev.drift`, `jev.ema`)
imports only the standard library. PyTorch, GLiNER2, and PEFT are optional:

```bash
# Serving-only host
python3 -m pip install -e '.[runtime]'
# or
python3 -m pip install -r requirements.txt

# Training host (adds GLiNER2 training collator and PyYAML)
python3 -m pip install -e '.[train]'
# or
python3 -m pip install -r requirements-train.txt

# Test/validation tooling
python3 -m pip install -r requirements-test.txt
```

## 3. Stage the offline model snapshot

The pinned base is GLiNER2 205M, model id `fastino/gliner2-base-v1`, revision
`79c3a777abc572b4767922f3916cf63fb5754df2` (see
`artifacts/model_manifest.json`). Stage one immutable snapshot directory
outside the repository; the designated path is `/home/sansha/jev-model-base`:

```text
/home/sansha/jev-model-base/
  added_tokens.json
  config.json
  encoder_config/config.json
  model.safetensors
  special_tokens_map.json
  spm.model
  tokenizer_config.json
  tokenizer.json
```

Provision it once with a pinned download, then treat the directory as
read-only. Verify the revision recorded in the manifest before first use:

```bash
python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("artifacts/model_manifest.json").read_text())
print(manifest["model_id"], manifest["revision"], manifest["parameter_count"])
PY
sha256sum /home/sansha/jev-model-base/model.safetensors
```

Keep the snapshot path outside the source tree so checkpoints, caches, and
ignored run data never mix with base weights. `JEV_MODEL_PATH` (or
`runtime.model_path`) is a local directory, so no Hub revision query is issued;
startup still validates the loaded parameter count against the manifest and
fails closed on a mismatch.

## 4. Environment variables

| Variable | Purpose | Example |
|---|---|---|
| `JEV_MODEL_PATH` | Local immutable snapshot directory; overrides `runtime.model_path` and the Hub id. | `JEV_MODEL_PATH=/home/sansha/jev-model-base` |
| `JEV_MODEL_MANIFEST` | Manifest checked at load time; defaults to `artifacts/model_manifest.json`. | `JEV_MODEL_MANIFEST=/home/sansha/Github/zenjev/artifacts/model_manifest.json` |
| `JEV_MQ_BRIDGE_BIN` | Absolute path to a prebuilt `jev-mq-bridge`; otherwise the repo `mq/target/{release,debug}` build is used. | `JEV_MQ_BRIDGE_BIN=/home/sansha/Github/zenjev/mq/target/release/jev-mq-bridge` |
| `JEV_TEACHER_API_KEY` | Secret *reference target* for `teacher.api_key_env` (distillation only); value lives in the process environment. | `export JEV_TEACHER_API_KEY=...` |
| `JEV_MQ_TOKEN` | Secret *reference target* named by `mq.secret_ref: env:JEV_MQ_TOKEN`. | `export JEV_MQ_TOKEN=...` |

Credential values never appear in YAML, JSON, logs, or receipts. Validate that
before deployment with:

```bash
.venv/bin/python scripts/audit_stage0.py
```

## 5. Validate the GPU gate

Run the fail-closed GPU validator: it requires `torch`, `gliner2`, and `peft`
to import, CUDA to be visible, at least one device to exist, and `nvidia-smi`
to answer. It prints one JSON object and exits `2` on any failure.

```bash
.venv/bin/python scripts/validate_nvidia_gpu.py
```

Expected shape on success:

```json
{"pass": true, "torch": "2.9.1+cu128", "cuda": "12.8", "devices": [...], "nvidia_smi": "..."}
```

Then run the real model smoke on the staged snapshot (base load, LoRA, EMA,
training step, concurrent inference, p50/p95 latency, peak VRAM):

```bash
JEV_MODEL_PATH=/home/sansha/jev-model-base \
  .venv/bin/python scripts/smoke_nvidia_gpu.py \
  --output artifacts/nvidia_gpu_train_ema_smoke.json
```

Evidence lands in `artifacts/nvidia_gpu_train_ema_smoke.json`. A CPU-only or
simulated result does not satisfy G6. See
[`acceptance.md`](acceptance.md) for the full profile and
[`training_serving.md`](training_serving.md) for the checkpoint layout this
smoke exercises.

## 6. Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `missing_runtime_dependencies` | extras not installed in the interpreter | install `.[runtime]`/`.[train]` |
| `cuda_unavailable` | CPU wheel or driver mismatch | install the CUDA wheel matching `nvidia-smi` |
| `nvidia_smi_unavailable` | driver not reachable from the shell | fix the driver/toolkit before serving |
| `loaded model does not match the configured model manifest` | snapshot revision or model id differs | re-stage the pinned revision |
| `model parameter count ... outside manifest tolerance` | wrong checkpoint staged | re-stage the 205M base |
| `install jev[runtime] on an NVIDIA GPU host` | `gliner2` missing | install the runtime extra |
