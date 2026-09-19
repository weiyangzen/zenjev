from __future__ import annotations

import argparse
import json
import sys

from .config import ConfigError, load_config
from .distill import distill
from .model import apply_lora, extract, load_gliner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "distill", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "distill":
            p.add_argument("--output", required=True)
        if name == "extract":
            p.add_argument("--text", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print(json.dumps({"valid": True, "config_digest": config.digest(), "model_id": config.model_id}, ensure_ascii=False))
        elif args.command == "distill":
            print(json.dumps({"examples": distill(config, args.output)}, ensure_ascii=False))
        else:
            model = apply_lora(load_gliner(config), config)
            print(json.dumps(extract(model, config, args.text), ensure_ascii=False))
    except (ConfigError, ValueError, RuntimeError, OSError) as exc:
        print(f"jev: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

