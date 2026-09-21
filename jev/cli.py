"""ZenJev operator CLI.

Commands cover schema/source validation, distillation, adapter training,
EMA-snapshot serving, metrics/reset inspection, manifest inspection and the
external-MQ operator surfaces (bridge config, run, replay, DLQ/quarantine).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, JevConfig, load_config
from .distill import distill
from .model import apply_lora, extract, load_gliner

EXIT_ERROR = 2


def _echo(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, default=str))


def _load_examples(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _cmd_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _echo(
        {
            "valid": True,
            "config_digest": config.digest(),
            "model_id": config.model_id,
            "schema_digest": config.schema.digest(),
            "mq_enabled": bool(config.mq and config.mq.enabled),
        }
    )
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest_path = Path(__file__).resolve().parents[1] / "artifacts/model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    from .mq import render_bridge_config

    _echo(
        {
            "config": config.canonical(),
            "config_digest": config.digest(),
            "schema_digest": config.schema.digest(),
            "model_manifest": manifest,
            "mq_bridge_config": render_bridge_config(config) if config.mq and config.mq.enabled else None,
        }
    )
    return 0


def _cmd_distill(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _echo({"examples": distill(config, args.output)})
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    model = apply_lora(load_gliner(config), config)
    _echo(extract(model, config, args.text))
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .runtime import JevRuntime
    from .training import ContinuousLoRATrainer

    config = load_config(args.config)
    runtime = JevRuntime(config)
    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )
    examples = _load_examples(args.data)
    events = trainer.train(examples, max_steps=args.steps)
    _echo(
        {
            "events": [vars(event) for event in events],
            "steps": runtime.stats.training_steps,
            "ema_updates": runtime.ema.updates,
            "model_generation": runtime.stats.model_generation,
            "checkpoint": str(Path(args.checkpoint_dir) / "latest.pt"),
        }
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .runtime import JevRuntime
    from .training import ContinuousLoRATrainer

    config = load_config(args.config)
    runtime = JevRuntime(config)
    ContinuousLoRATrainer(
        config,
        runtime,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.checkpoint or args.checkpoint_dir,
    )
    if args.input_jsonl:
        with Path(args.input_jsonl).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                result = runtime.infer_result(record.get("text", ""))
                _echo({"record_id": record.get("record_id"), "result": result})
    else:
        _echo(runtime.infer_result(args.text or ""))
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    root = Path(args.checkpoint_dir)
    metrics = root / "metrics.json"
    events = root / "reset-events.jsonl"
    payload = {
        "metrics": json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else None,
        "reset_events": [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line.strip()] if events.exists() else [],
        "archives": sorted(path.name for path in (root / "archives").glob("*.pt")) if (root / "archives").exists() else [],
        "latest_checkpoint": str(root / "latest.pt") if (root / "latest.pt").exists() else None,
    }
    _echo(payload)
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    from .runtime import JevRuntime
    from .training import ContinuousLoRATrainer

    config = load_config(args.config)
    runtime = JevRuntime(config)
    trainer = ContinuousLoRATrainer(
        config,
        runtime,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.checkpoint,
    )
    trainer._reset_model(args.reason)
    _echo(
        {
            "reset_id": runtime.stats.reset_id,
            "reason": args.reason,
            "archive_dir": str(Path(args.checkpoint_dir) / "archives"),
        }
    )
    return 0


def _mq_config(args: argparse.Namespace) -> JevConfig:
    config = load_config(args.config)
    if config.mq is None:
        raise ConfigError("config has no mq section")
    return config


def _cmd_mq_validate(args: argparse.Namespace) -> int:
    from .mq import locate_bridge_binary, render_bridge_config

    config = _mq_config(args)
    payload: dict = {"mq": config.mq.as_contract(), "fail_closed": True}
    binary = locate_bridge_binary()
    if binary is not None and config.mq.enabled:
        import subprocess
        import tempfile

        rendered = render_bridge_config(config)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(rendered, handle)
            temporary = handle.name
        result = subprocess.run(
            [str(binary), "validate", "--config", temporary], capture_output=True, text=True, check=False
        )
        Path(temporary).unlink(missing_ok=True)
        payload["bridge_validate"] = {"returncode": result.returncode, "stdout": result.stdout.strip()}
        if result.returncode != 0:
            _echo(payload)
            return EXIT_ERROR
    _echo(payload)
    return 0


def _cmd_mq_bridge_config(args: argparse.Namespace) -> int:
    from .mq import render_bridge_config

    config = _mq_config(args)
    rendered = render_bridge_config(config)
    destination = Path(args.output or config.mq.bridge_config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _echo({"path": str(destination), "adapter": rendered["adapter"], "schema_digest": rendered["schema_digest"]})
    return 0


def _cmd_mq_run(args: argparse.Namespace) -> int:
    from .mq import MqIngestor, training_example_from_envelope

    config = _mq_config(args)
    stream = None
    trainer = None
    if args.train:
        from .runtime import JevRuntime
        from .training import ContinuousLoRAStream, ContinuousLoRATrainer

        runtime = JevRuntime(config)
        trainer = ContinuousLoRATrainer(config, runtime, checkpoint_dir=args.checkpoint_dir)
        stream = ContinuousLoRAStream(trainer, max_queue_size=args.queue_size).start()

    def labelled_handler(envelope: dict) -> None:
        if stream is None:
            return
        example = training_example_from_envelope(envelope)
        if example is not None:
            stream.submit(example)

    overrides: dict = {}
    if args.loop:
        overrides["loop"] = True
    if args.max_cycles is not None:
        overrides["max_cycles"] = args.max_cycles
    ingestor = MqIngestor(
        config,
        labelled_handler=labelled_handler if args.train else None,
        bridge_config_overrides=overrides or None,
    )
    if args.pause_seconds:
        ingestor.pause()

        def resume_later() -> None:
            import threading

            timer = threading.Timer(float(args.pause_seconds), ingestor.resume)
            timer.daemon = True
            timer.start()

        resume_later()
    metrics = ingestor.run(
        manage_bridge=not args.attach,
        max_records=args.max_records,
        idle_timeout_s=args.idle_timeout,
        duration_s=args.duration,
    )
    if stream is not None:
        metrics["training_events"] = len(stream.stop(timeout=120))
        metrics["training_steps"] = trainer.runtime.stats.training_steps if trainer else None
        metrics["ema_updates"] = trainer.runtime.ema.updates if trainer else None
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _echo(metrics)
    return 0


def _cmd_mq_lag(args: argparse.Namespace) -> int:
    from .mq import BridgeClient

    config = _mq_config(args)
    socket_path = Path(config.mq.socket_path)
    if socket_path.exists():
        client = BridgeClient(socket_path)
        try:
            client.connect(credits=0)
            _echo({"source": "bridge", "metrics": client.status()})
        finally:
            client.close()
        return 0
    metrics_path = Path(config.mq.metrics_path)
    if metrics_path.exists():
        _echo({"source": "metrics_file", "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))})
        return 0
    _echo({"source": "unavailable", "metrics": None, "hint": "start `jev mq run` first"})
    return 0


def _mq_review(path: Path, output: str | None) -> dict:
    entries: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    reasons: dict[str, int] = {}
    for entry in entries:
        reason = str(entry.get("reason", "unknown"))
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"path": str(path), "count": len(entries), "reasons": reasons, "exported": output}


def _cmd_mq_dlq(args: argparse.Namespace) -> int:
    config = _mq_config(args)
    _echo(_mq_review(Path(config.mq.dlq_path), args.output))
    return 0


def _cmd_mq_quarantine(args: argparse.Namespace) -> int:
    config = _mq_config(args)
    _echo(_mq_review(Path(config.mq.quarantine_path), args.output))
    return 0


def _cmd_mq_replay(args: argparse.Namespace) -> int:
    from .mq import MqIngestor, render_bridge_config

    config = _mq_config(args)
    overrides = {"exit_after_drain": True}
    if args.from_offset is not None:
        overrides.update({"start_position": "by_offset", "start_offset": args.from_offset})
    elif args.from_timestamp is not None:
        overrides.update({"start_position": "by_timestamp", "start_timestamp": args.from_timestamp})
    else:
        overrides["start_position"] = "first"
    rendered = render_bridge_config(config, overrides=overrides)
    replay_path = Path(args.output or (config.mq.bridge_config + ".replay.json"))
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(rendered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        _echo({"path": str(replay_path), "overrides": overrides, "dry_run": True})
        return 0
    ingestor = MqIngestor(config, bridge_config_overrides=overrides)
    metrics = ingestor.run(manage_bridge=True, max_records=args.max_records, idle_timeout_s=args.idle_timeout)
    _echo({"path": str(replay_path), "overrides": overrides, "metrics": metrics})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jev")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("manifest")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("distill")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_distill)

    p = sub.add_parser("extract")
    p.add_argument("--config", required=True)
    p.add_argument("--text", required=True)
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("serve")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", default="runs/jev")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--input-jsonl", default=None)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("metrics")
    p.add_argument("--checkpoint-dir", default="runs/jev")
    p.set_defaults(func=_cmd_metrics)

    p = sub.add_parser("reset")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", default="runs/jev")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--reason", default="operator_requested")
    p.set_defaults(func=_cmd_reset)

    mq = sub.add_parser("mq")
    mq_sub = mq.add_subparsers(dest="mq_command", required=True)

    p = mq_sub.add_parser("validate")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_mq_validate)

    p = mq_sub.add_parser("bridge-config")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_mq_bridge_config)

    p = mq_sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--idle-timeout", type=float, default=None)
    p.add_argument("--checkpoint-dir", default="runs/jev")
    p.add_argument("--queue-size", type=int, default=128)
    p.add_argument("--train", action="store_true")
    p.add_argument("--attach", action="store_true", help="use an already running bridge")
    p.add_argument("--pause-seconds", type=float, default=0.0)
    p.add_argument("--loop", action="store_true", help="force infinite loop mode on for this run")
    p.add_argument("--duration", type=float, default=None, help="stop after this many wall seconds and drain")
    p.add_argument("--max-cycles", type=int, default=None, help="stop the loop after this many cycles")
    p.set_defaults(func=_cmd_mq_run)

    p = mq_sub.add_parser("lag")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_mq_lag)

    p = mq_sub.add_parser("dlq")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_mq_dlq)

    p = mq_sub.add_parser("quarantine")
    p.add_argument("--config", required=True)
    p.add_argument("--output", default=None)
    p.set_defaults(func=_cmd_mq_quarantine)

    p = mq_sub.add_parser("replay")
    p.add_argument("--config", required=True)
    p.add_argument("--from-offset", type=int, default=None)
    p.add_argument("--from-timestamp", type=int, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--idle-timeout", type=float, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_mq_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"jev: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
