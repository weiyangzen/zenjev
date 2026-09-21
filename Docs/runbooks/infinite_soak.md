# Infinite soak (train / infer / update loop)

Proves the service can run continuously: the mock MQ source replays the dataset
head-to-tail forever (`loop: true`) while the trainer keeps updating LoRA,
publishing EMA snapshots and serving inference concurrently. A bounded run is
used as evidence; omit `--duration` for an unbounded soak.

## How the loop works

- Each cycle injects `loop_cycle` and recomputes `content_sha256`, so the
  idempotency key is unique per pass and dedup never suppresses the replay.
- `max_cycles: 0` = unbounded; `max_cycles: N` stops after N passes.
- The bridge state file (`state.json`) persists `cycle`/`cursor`; a restart
  resumes the soak.
- The trainer writes `runs/jev/status.json` every step; the Rust monitor reads
  `generation`, `ema_step`, `loss`, `training_steps` from it.

## Bounded soak (produce evidence)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
JEV_MODEL_PATH=/home/sansha/jev-model-base \
  .venv/bin/python scripts/run_infinite_soak.py \
    --duration 60 \
    --output artifacts/soak/infinite_soak.json
```

Expected evidence fields: `accepted`, `loop_cycles`, `training_steps`,
`serving_generations`, `serving_p50/p95_seconds`, `peak_vram_bytes`,
`serving_errors: []`.

## Unbounded run

```bash
JEV_MODEL_PATH=/home/sansha/jev-model-base \
  .venv/bin/python scripts/run_infinite_soak.py --duration 0
```

Stop with `Ctrl-C`; `runs/jev/soak/latest.pt` can then be served with
`python -m jev.cli serve --config ... --checkpoint runs/jev/soak/latest.pt`.

## With monitoring

```bash
monitor/target/release/zenjev-monitor \
  --interval 1 --duration 90 \
  --jsonl artifacts/monitor/soak_metrics.jsonl
```

The monitor degrades without a GPU or status file (`gpu: null`, `stale: true`)
instead of crashing. `artifacts/monitor/soak_summary.json` aggregates
CPU/GPU/VRAM and the observed generation range.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `Jev training stream failed` | training record without `input`/`output` | send an envelope with `example`, or `labels` + `text` (`training_example_from_envelope`) |
| `status_age_seconds` grows, `stale: true` | trainer hung/exited | inspect logs and `runs/jev/status.json`, restart the worker |
| no GPU values | nvidia-smi missing/driver issue | expected degradation; install/repair `nvidia-smi` |
| `loop_cycles` stays 0 | `loop` not enabled on the bridge | pass `--loop` or set `mq.loop: true` |
