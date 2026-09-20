# Stage 0 GPU benchmarks (ZJ-031)

`run_stage0_benchmark.py` wraps `scripts/smoke_nvidia_gpu.py` in a short,
deterministic configuration (`--steps` / `--inferences`), then merges the smoke
report into one benchmark artifact. It is GPU-required and exits `2` when CUDA
is unavailable, when the smoke run fails, or when the recomputed gate fails.

## Usage

```bash
.venv/bin/python benchmarks/run_stage0_benchmark.py \
  --config configs/example.yaml \
  --model-path "$JEV_MODEL_PATH" \
  --steps 2 --inferences 4 \
  --output artifacts/benchmarks/stage0_gpu_benchmark.json
```

## Recorded fields

* throughput: training steps/s and inference calls/s
* latency: p50/p95 over the inference observations (linear interpolation)
* peak VRAM and headroom against the device total
* `nvidia-smi` temperature/power line captured by the smoke
* base model load seconds
* GPU name/capability, torch and CUDA versions
* concurrent train/infer stability: inference ran on a background thread while
  training published EMA snapshots; generation advanced and `errors` is empty

## Pass criteria

`evaluate_pass_criteria` recomputes the gate from the smoke payload:
`smoke_pass`, `steps_completed`, `inferences_completed`,
`observations_recorded`, `no_errors`, `ema_updates`, `generation_advanced`,
`observed_generation_positive`, `vram_recorded`, `vram_within_device`,
`temperature_power_recorded`, plus `vram_within_budget` when
`--vram-limit-bytes` is supplied. The artifact is written atomically to
`--output` with a top-level `pass` boolean and is also printed to stdout.

## Determinism

The runner is a single process without background scheduling beyond the smoke's
own inference thread. Step and inference counts are fixed by flags; the smoke
uses a fixed adapter regularizer instead of a teacher dataset, so no API key or
broker is required.

## Tests

```bash
.venv/bin/python -m pytest tests/benchmarks -q
```

The helper tests are CPU-only: the module is imported by path, imports torch or
CUDA only inside `main`, and never loads a model.
