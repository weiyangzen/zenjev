# Perpetual operations runbook (Stage 0.1)

All services are user systemd units and run until the operator stops them.
They resume from durable state after a restart or reboot; no one-shot mode is
used in production.

## Services

| Unit | Script | Role |
|---|---|---|
| `zenjev-feeder` | `scripts/zenjev_feeder.py` | Reads `/home/sansha/data/jevraw` read-only, normalizes captures into §1.5 `tool_task_pair` envelopes appended to `runs/jev/feed/rawspool.ndjson`. Own watermark: `runs/jev/perpetual/feeder-state.json`. Throttled (`--max-records 10 --tick-sleep 2.0`) so the spool stays near training capacity. |
| `zenjev-loop` | `scripts/zenjev_loop.py` | Starts the Rust `dir-spool` bridge over `runs/jev/feed/*.ndjson`, consumes with credit backpressure, extracts with the EMA snapshot, routes the deterministic decision, self-labels with the live LoRA judge (canonical `run_jevraw_loop.py` recipe), trains LoRA, publishes generations, serves inference on `127.0.0.1:8788`, and writes `runs/jev/status.json` for `zenjev-monitor`. |
| `zenjev-fabricator` | `scripts/zenjev_fabricator.py` | Emits synthetic canary `tool_task_pair` envelopes to `runs/jev/feed/fabricator.ndjson`. Never trained unless `ZENJEV_FABRICATOR_ADMISSION=training_admitted`. |
| `zenjev-console` | `scripts/zenjev_console.py` | Read-only web panel on `0.0.0.0:8790` (SSE), serves `scripts/console/index.html`, proxies `/api/extract` to the loop. |

Open the panel at `http://192.168.20.214:8790/`. The canonical `zenjev-monitor`
(`monitor/`) remains an independent read-only host/GPU observer that reads
`runs/jev/status.json`.

## Install, start, stop, status

```bash
bash scripts/install_zenjev_services.sh          # idempotent install + restart
sudo loginctl enable-linger sansha               # reboot persistence (once)
systemctl --user status zenjev-loop
systemctl --user restart zenjev-loop             # resume from checkpoint + watermark
systemctl --user stop zenjev-feeder zenjev-loop zenjev-fabricator zenjev-console
journalctl --user -u zenjev-loop -n 100 --no-pager
tail -f runs/jev/logs/zenjev-loop.log
```

## State and evidence paths

| Path | Contents |
|---|---|
| `runs/jev/perpetual/generations.jsonl` | Monotonic LoRA generation ledger with parent hash chain, records consumed, EMA step, reset id, `session_pid`. Generation ids resume from the ledger maximum, so a restart never reissues an id. |
| `runs/jev/perpetual/checkpoints/latest.pt` | Adapter + EMA + optimizer checkpoint (resume source). |
| `runs/jev/perpetual/dir-spool-state.json` | Committed per-file byte watermarks for the envelope spool. |
| `runs/jev/perpetual/wal.bin`, `dlq.jsonl`, `quarantine.jsonl` | Bridge spool, dead letters, quarantine. |
| `artifacts/mq/metrics.json` | Bridge metrics written when a connection closes; live counters are reported by the loop heartbeat. |
| `artifacts/perpetual/{loop,feeder,fabricator,console}.json` | Heartbeats, uptime, counters, latency, VRAM. |
| `artifacts/perpetual/events.jsonl` | Bounded live task feed (rotates at 8 MB). |
| `artifacts/perpetual/acceptance.json` | Frozen G12–G15 evidence produced by `scripts/perpetual_evidence.py`. |
| `runs/jev/status.json` | Monitor-compatible status document (generation, ema_step, training_steps, loss). |

## Fabricator admission

`ZENJEV_FABRICATOR_ADMISSION` on `zenjev-loop` is one of `inference_only`
(default), `canary_scored`, or `training_admitted`. Only `training_admitted`
lets synthetic records into LoRA training; synthetic records are always counted
in a separate series and never enter the collapse-guard baseline.

## Operator checks

```bash
python -m jev.cli health                      # heartbeats, gaps, sessions; exit!=0 when stale
python scripts/perpetual_evidence.py          # freeze G12-G15 evidence into artifacts/perpetual/acceptance.json
scripts/zenjev_services.sh restart            # start/stop/restart/status/logs/rotate wrapper
python scripts/restart_injection.py           # SIGKILL drill: supervisor restart + resume proof
curl -s http://127.0.0.1:8788/health | python3 -m json.tool
curl -s http://127.0.0.1:8790/api/snapshot | python3 -m json.tool | head -40
curl -s -X POST http://127.0.0.1:8790/api/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"用 Python 和 GLiNER2 准备 2026 技术栈"}'
python3 -m pytest -q tests/test_perpetual.py
```

## Recovery

* Bridge restart: the spool watermark is only advanced by downstream durable
  acceptance, so unacked records are redelivered and deduplicated by
  `record_id` plus content hash.
* Loop restart: resumes `latest.pt` (adapter/EMA/optimizer), the spool
  watermark, and the ledger generation counter; `reset_id` only grows.
* Raw feeder restart: resumes at the persisted byte offset of every dump file;
  a trailing partial line is retried, never consumed twice.
* To replay the raw corpus from the beginning, stop `zenjev-feeder` and delete
  `runs/jev/perpetual/feeder-state.json` (spool + bridge dedup still protect the
  trainer), then start the unit again.

## Known bounds (Stage 0.1)

* The feeder runs slightly below training capacity so the 200 MB-class spool
  backlog drains instead of growing without bound; backlog is on the console.
* Generation ids were reissued by early restarts before the ledger-seeding fix;
  the historical duplicate window is recorded in `acceptance.json`, and the
  current session is strictly monotonic.
* A multi-hour soak with reboot injection (ZJ-071) is not yet recorded.
