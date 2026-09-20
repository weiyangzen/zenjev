#!/usr/bin/env bash
set -euo pipefail
root="${1:-.}"
find "$root" -type f \
  -not -path '*/.git/*' \
  -not -path '*/.zenjev/runtime/*' \
  -not -path '*/.venv/*' \
  -not -path '*/.pytest_cache/*' \
  -not -path '*/artifacts/delivery/*_receipt.json' \
  -not -path '*/__pycache__/*' \
  -not -path '*/mq/target/*' \
  -not -path '*/runs/*' \
  -not -path '*/benchmarks/*.pt' \
  -print0 | sort -z | xargs -0 sha256sum
